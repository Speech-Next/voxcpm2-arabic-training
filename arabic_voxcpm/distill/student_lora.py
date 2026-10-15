"""Per-adapter LoRAs for a distilled LocDiT student (#12): every dialect adapter's LocDiT LoRA tuned for the student.

    python -m arabic_voxcpm.distill.student_lora --teacher <model dir> --data <adapter teacher data> --out <dir> \
        --stage mse
    ... --stage spectral --init <dir>/mse-best
    ... --stage dmd --init <dir>/spectral-best

The adapters' LocDiT LoRAs were trained with the original 10-step LocDiT; the 2-step student (--student, default the
engine's Arabic student) runs under them unchanged, and how much that costs differs by adapter. Here the student's own
weights stay frozen (the LocDiT the engine shares between adapters) and, for every adapter of the teacher data (rows
routed through no adapter are left out), a LoRA of that adapter's LocDiT layout (rank, alpha, targets) is trained,
initialized from the adapter's own LocDiT LoRA (--init adapters) or from a previous stage (--init <dir>). A batch holds
--rows rows of every adapter; each part runs the student under its adapter's trainable LoRA, so an adapter's LoRA
learns from its own rows only (independent problems sharing a GPU step: the loss is a mean per adapter, the gradient
clipped per adapter). The engine puts these tensors in place of an adapter's LocDiT LoRA when it runs this student
(nanovllm_voxcpm DialectTTS: <student stem>.adapters/ next to the student): the same shapes, so no cost in speed.

The student's sampler is its checkpoint's own (schedule, mean_mode), as the engine runs it. Stages, as dit.py:
- mse: the student's K-step output regresses the teacher's 10-step CFG output for the same noise.
- spectral: a multi-resolution STFT loss through the frozen AudioVAE decoder against the teacher's patch, plus
  --mse-weight times the mse loss.
- dmd: DMD2. Real score: the frozen teacher under the adapter's original LoRA, CFG --cfg (CFG-zero*). Fake score: a
  LocDiT initialized from the teacher whose shared weights learn by flow matching on every adapter's student samples,
  plus its own trainable LoRA per adapter (initialized from the adapter's), so it can follow each adapter's student
  distribution; --fake-steps updates per student update. The student follows x0_fake - x0_real, normalized, plus --reg
  times the regression to the teacher.

Validation: the last --val-rows rows of every adapter (in data order: rows of that adapter's last batches of the plan).
Every adapter keeps its own best (mse, dmd: val_mse_teacher; spectral: val_spec), the starting LoRA included.

Output, per stage: <stage>-best/ and <stage>-last/, one directory per adapter: lora_config.json (the adapter's own,
plus how this LoRA was made) and lora_weights.safetensors (the LocDiT LoRA tensors under their full names,
feat_decoder.estimator.<module>.lora_A / lora_B, fp32), the layout the engine reads; state.json (per adapter: the step
and validation of the saved LoRA).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from arabic_voxcpm.distill import dit
from arabic_voxcpm.train import adapters as A

PREFIX = dit.PREFIX
ARABIC_STUDENT = "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k2_dmd.pt"
KEYS = ("mu", "cond", "z", "x")


class AdapterRows:
    """The teacher rows routed through an adapter, in memory (stored dtype), split per adapter: the last ``val_rows``
    rows of each (at most half) are validation. ``names``: the adapters kept (default: every adapter with rows)."""

    def __init__(self, path: str, names: list[str] | None = None, val_rows: int = 2000, device="cpu"):
        m = json.load(open(os.path.join(path, "manifest.json")))
        self.all_adapters: list[str] = m["adapters"]
        cols: dict[str, list[torch.Tensor]] = {k: [] for k in (*KEYS, "adapter")}
        for s in m["shards"]:
            d = torch.load(os.path.join(path, s["file"]), map_location="cpu", weights_only=True, mmap=True)
            keep = (d["adapter"] >= 0).nonzero().flatten()
            for k in KEYS:
                cols[k].append(d[k][keep])
            cols["adapter"].append(d["adapter"][keep].long())
        d = {k: torch.cat(v) for k, v in cols.items()}
        present = [self.all_adapters[a] for a in sorted(set(d["adapter"].tolist()))]
        if names is None:
            names = present
        missing = sorted(set(names) - set(present))
        assert not missing, f"no teacher rows for adapters {missing} (rows for: {present})"
        self.names = sorted(names)
        self.train: dict[str, dict[str, torch.Tensor]] = {}
        self.val: dict[str, dict[str, torch.Tensor]] = {}
        for name in self.names:
            idx = (d["adapter"] == self.all_adapters.index(name)).nonzero().flatten()
            v = min(val_rows, len(idx) // 2)
            self.train[name] = {k: d[k][idx[: len(idx) - v]] for k in KEYS}
            self.val[name] = {k: d[k][idx[len(idx) - v :]] for k in KEYS}
        self.device = torch.device(device)

    def rows(self, split: str = "train") -> dict[str, int]:
        return {n: len((self.train if split == "train" else self.val)[n]["mu"]) for n in self.names}

    def _take(self, d: dict, idx) -> dict:
        return {k: v[idx].to(self.device, non_blocking=True).float() for k, v in d.items()}

    def batch(self, n: int, gen: torch.Generator | None = None) -> dict[str, dict]:
        """``n`` random training rows of every adapter."""
        return {a: self._take(self.train[a], torch.randint(0, len(self.train[a]["mu"]), (n,), generator=gen))
                for a in self.names}

    def val_parts(self, name: str, n: int, limit: int | None = None):
        d = self.val[name]
        m = len(d["mu"]) if limit is None else min(limit, len(d["mu"]))
        for s in range(0, m, n):
            yield self._take(d, torch.arange(s, min(s + n, m)))


def parts(b: dict, micro: int):
    """(part of the batch, its share of the batch), <= micro rows each."""
    n = len(b["mu"])
    for s in range(0, n, micro):
        p = {k: v[s : s + micro] for k, v in b.items()}
        yield p, len(p["mu"]) / n


def read_lora(adapter_dir: str) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """An adapter directory's LocDiT LoRA: {module (relative to the LocDiT): (A, B)} in fp32, and its config file."""
    from safetensors import safe_open

    raw = {}
    with safe_open(os.path.join(adapter_dir, A.WEIGHTS), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(PREFIX) and k.endswith((".lora_A", ".lora_B")):
                raw[k[len(PREFIX):]] = f.get_tensor(k).float()
    mods = {k[: -len(".lora_A")]: (raw[k], raw[k[: -len(".lora_A")] + ".lora_B"]) for k in raw if k.endswith(".lora_A")}
    assert mods, f"no LocDiT LoRA tensors in {adapter_dir}"
    return mods, json.load(open(os.path.join(adapter_dir, A.CONFIG), encoding="utf-8"))


class LoRASet(nn.Module):
    """Trainable LocDiT LoRAs, one per adapter, initialized from adapter directories ({name: dir}): parameters
    lora[name] = [A_0, B_0, A_1, B_1, ...] in the order of mods[name]; scale[name] = alpha / r."""

    def __init__(self, dirs: dict[str, str]):
        super().__init__()
        self.lora = nn.ModuleDict()
        self.mods: dict[str, list[str]] = {}
        self.scale: dict[str, float] = {}
        self.config: dict[str, dict] = {}
        for name in sorted(dirs):
            mods, cfg = read_lora(dirs[name])
            self.mods[name] = sorted(mods)
            self.scale[name] = A.scaling(cfg.get("lora_config", cfg))
            self.config[name] = cfg
            self.lora[name] = nn.ParameterList([nn.Parameter(t.clone()) for m in self.mods[name] for t in mods[m]])
        self._layers: dict[int, dict[str, A.AdapterLinear]] = {}

    @property
    def names(self) -> list[str]:
        return sorted(self.mods)

    def params(self, name: str) -> list[nn.Parameter]:
        return list(self.lora[name])

    def tensors(self, name: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        p = self.lora[name]
        return {m: (p[2 * i], p[2 * i + 1]) for i, m in enumerate(self.mods[name])}

    def bind(self, root: nn.Module, name: str) -> None:
        """Point root's AdapterLinear layers at adapter ``name``'s trainable tensors. They go into the layers'
        __dict__, as AdapterBank's tensors do, so they stay out of root's parameters and state_dict."""
        key = id(root)
        if key not in self._layers:
            self._layers[key] = {n: m for n, m in root.named_modules() if isinstance(m, A.AdapterLinear)}
        layers = self._layers[key]
        t = self.tensors(name)
        if set(layers) != set(t):
            raise ValueError(f"adapter {name!r} does not match the LocDiT's LoRA layers: "
                             f"missing {sorted(set(layers) - set(t))[:3]}, unused {sorted(set(t) - set(layers))[:3]}")
        for n, m in layers.items():
            m.__dict__["lora_A"], m.__dict__["lora_B"] = t[n]
            m.scaling = self.scale[name]

    def save(self, out: str, name: str, meta: dict) -> None:
        """Adapter ``name``'s LoRA as an adapter directory of LocDiT tensors (out/<name>/), fp32."""
        from safetensors.torch import save_file

        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        sd = {}
        for m, (a, b) in self.tensors(name).items():
            sd[f"{PREFIX}{m}.lora_A"] = a.detach().float().cpu().contiguous()
            sd[f"{PREFIX}{m}.lora_B"] = b.detach().float().cpu().contiguous()
        save_file(sd, os.path.join(d, A.WEIGHTS + ".tmp"))
        os.replace(os.path.join(d, A.WEIGHTS + ".tmp"), os.path.join(d, A.WEIGHTS))
        cfg = {k: v for k, v in self.config[name].items() if k != "student_lora"}
        cfg["student_lora"] = meta
        json.dump(cfg, open(os.path.join(d, A.CONFIG), "w", encoding="utf-8"), indent=1)


def init_dirs(init: str, model_dir: str, names: list[str]) -> dict[str, str]:
    """Where each adapter's LoRA starts: ``adapters`` = the model's adapter directories, else a stage directory with
    <adapter>/ subdirectories."""
    root = os.path.join(model_dir, "adapters") if init == "adapters" else init
    dirs = {n: os.path.join(root, n) for n in names}
    missing = [n for n, d in dirs.items() if not os.path.isfile(os.path.join(d, A.WEIGHTS))]
    assert not missing, f"no LoRA to start from for {missing} under {root}"
    return dirs


@torch.no_grad()
def validate(student, loras: LoRASet, data: AdapterRows, grid, bs: int, mean_mode: bool, vae=None,
             spec_rows: int = 0) -> dict[str, dict[str, float]]:
    """Per adapter: val_mse_teacher (and val_spec over its first spec_rows validation rows when vae is given)."""
    student.eval()
    out: dict[str, dict[str, float]] = {}
    for name in data.names:
        loras.bind(student, name)
        se = n = 0.0
        for b in data.val_parts(name, bs):
            with torch.autocast(data.device.type, dtype=torch.bfloat16):
                x = dit.student_solve(student, b["z"], b["mu"], b["cond"], grid, mean_mode)
            se += float(((x - b["x"]) ** 2).mean(dim=(1, 2)).sum())
            n += len(x)
        out[name] = {"val_mse_teacher": se / n}
        if vae is not None and spec_rows:
            sp = m = 0.0
            for b in data.val_parts(name, min(bs, 32), limit=spec_rows):
                with torch.autocast(data.device.type, dtype=torch.bfloat16):
                    x = dit.student_solve(student, b["z"], b["mu"], b["cond"], grid, mean_mode)
                    ls = dit.stft_loss(dit.decode_patch(vae, b["cond"], x), dit.decode_patch(vae, b["cond"], b["x"]))
                sp += float(ls) * len(x)
                m += len(x)
            out[name]["val_spec"] = sp / m
    student.train()
    return out


def main(args) -> None:
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"log-{args.stage}.jsonl"), "a")

    def emit(rec):
        rec["time"] = round(time.time(), 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    data = AdapterRows(args.data, args.adapters.split(",") if args.adapters else None, args.val_rows, dev)
    names = data.names
    adapter_dirs = A.find(args.teacher)
    missing = sorted(set(names) - set(adapter_dirs))
    assert not missing, f"adapters {missing} are not in {args.teacher}/adapters"
    chosen = {n: adapter_dirs[n] for n in names}

    ck = dit.load_checkpoint(args.student)
    schedule = [tuple(s) for s in ck["schedule"]]
    mean_mode = bool(ck.get("mean_mode", False))
    grid = dit.engine_grid(schedule)

    teacher = dit.load_estimator(args.teacher, dev)
    for p in teacher.parameters():
        p.requires_grad = False
    n_lora = A.inject_dit(teacher, chosen)
    bank = A.AdapterBank(chosen, prefix=PREFIX, device=dev, dtype=torch.bfloat16 if cuda else torch.float32)
    # copies before any LoRA is bound, so no copy holds another model's tensors
    student = copy.deepcopy(teacher).float()
    if ck.get("dit_layers"):
        dit.drop_layers(student, int(ck["dit_layers"]))
    student.load_state_dict({k[len(PREFIX):]: v for k, v in ck["state_dict"].items()})
    for p in student.parameters():
        p.requires_grad = False
    student.train()
    loras = LoRASet(init_dirs(args.init, args.teacher, names)).to(dev)
    fake = fake_loras = None
    if args.stage == "dmd":
        fake = copy.deepcopy(teacher).float().train()
        for p in fake.parameters():
            p.requires_grad = True
        fake_loras = LoRASet(chosen).to(dev)
    vae = None
    if args.stage == "spectral":
        vae = dit.load_vae(args.teacher, dev)
        for p in vae.parameters():
            p.requires_grad = False
    if args.checkpointing:  # after the copies: a checkpointed layer's forward refers to its own module
        for m in (student, fake, vae):
            if m is not None:
                dit.checkpoint_layers(m)

    emit({"event": "setup", "adapters": names, "lora_layers": n_lora, "train_rows": data.rows("train"),
          "val_rows": data.rows("val"), "student": args.student, "init": args.init, "schedule": schedule,
          "grid": grid, "mean_mode": mean_mode,
          "lora_params_per_adapter": sum(p.numel() for p in loras.params(names[0]))})
    opt = torch.optim.AdamW(loras.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, fused=cuda)
    fake_opt = None
    if fake is not None:
        fake_opt = torch.optim.AdamW([{"params": list(fake.parameters()), "lr": args.lr_fake},
                                      {"params": list(fake_loras.parameters()), "lr": args.lr_fake_lora}],
                                     betas=(0.9, 0.99), weight_decay=0.0, fused=cuda)
        base_fake_lrs = [g["lr"] for g in fake_opt.param_groups]
    gen = torch.Generator().manual_seed(args.seed)
    ac = dict(device_type=dev.type, dtype=torch.bfloat16)
    key = "val_spec" if args.stage == "spectral" else "val_mse_teacher"
    meta = {"student": args.student, "stage": args.stage, "init": args.init, "data": os.path.abspath(args.data)}
    best: dict[str, tuple[float, int]] = {}

    def checkpoint(step: int, v: dict[str, dict[str, float]]) -> None:
        last = os.path.join(args.out, f"{args.stage}-last")
        for name in names:
            loras.save(last, name, {**meta, "step": step, **v[name]})
            if v[name][key] < best.get(name, (float("inf"), -1))[0]:
                best[name] = (v[name][key], step)
                loras.save(os.path.join(args.out, f"{args.stage}-best"), name, {**meta, "step": step, **v[name]})
        json.dump({"step": step, "val": v}, open(os.path.join(last, "state.json"), "w"), indent=1)
        json.dump({n: {"step": s, key: x} for n, (x, s) in best.items()},
                  open(os.path.join(args.out, f"{args.stage}-best", "state.json"), "w"), indent=1)

    def val():
        return validate(student, loras, data, grid, args.val_batch, mean_mode, vae, args.spec_val_rows)

    v = val()
    emit({"step": 0, "event": "val", "mean": {k: sum(x[k] for x in v.values()) / len(v) for k in v[names[0]]}, **v})
    checkpoint(0, v)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(1, args.iters + 1):
        f = min(1.0, step / args.warmup) * (0.5 * (1 + math.cos(math.pi * step / args.iters)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        rec: dict[str, float] = {}
        if fake is not None and args.fake_steps > 0:
            for g, lr0 in zip(fake_opt.param_groups, base_fake_lrs):
                g["lr"] = lr0 * f
            fl_sum = 0.0
            for _ in range(args.fake_steps):
                fb = data.batch(args.rows, gen=gen)
                for name in names:  # flow matching on each adapter's (detached) student samples
                    loras.bind(student, name)
                    fake_loras.bind(fake, name)
                    for mb, w in parts(fb[name], args.micro):
                        with torch.no_grad(), torch.autocast(**ac):
                            xs = dit.student_solve(student, mb["z"], mb["mu"], mb["cond"], grid, mean_mode)
                        t = torch.sigmoid(torch.randn(len(xs), device=dev) - 0.4)
                        n = torch.randn_like(xs)
                        xt = (1 - t[:, None, None]) * xs + t[:, None, None] * n
                        with torch.autocast(**ac):
                            vf = fake(xt, mb["mu"], t, mb["cond"], torch.zeros_like(t)).float()
                        fl = F.mse_loss(vf, n - xs)
                        (fl * w).backward()
                        fl_sum += float(fl.detach()) * w / len(names)
                torch.nn.utils.clip_grad_norm_([*fake.parameters(), *fake_loras.parameters()], 1.0)
                fake_opt.step()
                fake_opt.zero_grad(set_to_none=True)
            rec["loss_fake"] = fl_sum / args.fake_steps
        b = data.batch(args.rows, gen=gen)
        sums: dict[str, float] = {}
        gnorms = {}
        for name in names:
            loras.bind(student, name)
            bank.use(teacher, name)
            for mb, w in parts(b[name], args.micro):
                with torch.autocast(**ac):
                    x = dit.student_solve(student, mb["z"], mb["mu"], mb["cond"], grid, mean_mode)
                mse = F.mse_loss(x, mb["x"])
                if args.stage == "mse":
                    terms = {"loss": mse}
                elif args.stage == "spectral":
                    with torch.autocast(**ac):
                        with torch.no_grad():
                            ref = dit.decode_patch(vae, mb["cond"], mb["x"])
                        spec = dit.stft_loss(dit.decode_patch(vae, mb["cond"], x), ref)
                    terms = {"loss_spec": spec, "loss_mse": mse, "loss": args.spec_weight * spec + args.mse_weight * mse}
                else:
                    fake_loras.bind(fake, name)
                    t = torch.rand(len(x), device=dev) * 0.96 + 0.02
                    n = torch.randn_like(x)
                    xt = (1 - t[:, None, None]) * x.detach() + t[:, None, None] * n
                    with torch.no_grad(), torch.autocast(**ac):
                        x0_real = xt - t[:, None, None] * dit.teacher_velocity(teacher, xt, mb["mu"], mb["cond"], t,
                                                                               args.cfg)
                        x0_fake = xt - t[:, None, None] * fake(xt, mb["mu"], t, mb["cond"], torch.zeros_like(t)).float()
                        wn = (x.detach() - x0_real).abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-4)
                        grad = (x0_fake - x0_real) / wn
                    dmd = 0.5 * F.mse_loss(x, (x - grad).detach())
                    terms = {"loss_dmd": dmd, "loss_reg": mse, "loss": dmd + args.reg * mse}
                (terms["loss"] * w).backward()
                for k, x_ in terms.items():
                    sums[k] = sums.get(k, 0.0) + float(x_.detach()) * w / len(names)
            gnorms[name] = float(torch.nn.utils.clip_grad_norm_(loras.params(name), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec.update(sums)
        if step % args.log_every == 0:
            mem = {"mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)} if cuda else {}
            emit({"step": step, **rec, "grad_norm": {k: round(x, 4) for k, x in gnorms.items()}, "lr": args.lr * f,
                  "it_per_s": round(args.log_every / (time.time() - t0), 3), **mem})
            t0 = time.time()
        if step % args.val_every == 0 or step == args.iters:
            v = val()
            emit({"step": step, "event": "val", "mean": {k: sum(x[k] for x in v.values()) / len(v)
                                                        for k in v[names[0]]}, **v})
            checkpoint(step, v)
    done = {"event": "done", "best": {n: {"step": s, key: x} for n, (x, s) in best.items()}}
    if cuda:
        done["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    emit(done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", default="/workspace/models/VoxCPM2-Arabic-Dialects",
                    help="the model directory: its LocDiT (the teacher), adapters/, the AudioVAE (spectral stage)")
    ap.add_argument("--data", required=True, help="teacher_data.py output of the adapters teacher")
    ap.add_argument("--student", default=ARABIC_STUDENT, help="the distilled LocDiT (path or repo:file), frozen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["mse", "spectral", "dmd"], default="mse")
    ap.add_argument("--init", default="adapters", help="adapters (their own LocDiT LoRAs) or a stage directory")
    ap.add_argument("--adapters", help="comma-separated adapters to train (default: every adapter with rows)")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--rows", type=int, default=64, help="rows of every adapter per step")
    ap.add_argument("--micro", type=int, default=128, help="rows per forward")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-fake", type=float, default=1e-5, help="dmd: the fake score's shared weights")
    ap.add_argument("--lr-fake-lora", type=float, default=5e-5, help="dmd: the fake score's per-adapter LoRAs")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--spec-weight", type=float, default=1.0)
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--spec-val-rows", type=int, default=256, help="spectral: validation rows decoded per adapter")
    ap.add_argument("--reg", type=float, default=0.25)
    ap.add_argument("--fake-steps", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--val-rows", type=int, default=2000, help="validation rows per adapter (its last rows)")
    ap.add_argument("--val-batch", type=int, default=512)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--no-checkpointing", dest="checkpointing", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
