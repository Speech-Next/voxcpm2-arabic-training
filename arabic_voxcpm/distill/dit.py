"""LocDiT students (issues #7, #12): a few-step, guidance-free LocDiT distilled from the teacher's sampler.

    python -m arabic_voxcpm.distill.dit --teacher <model dir> --data <teacher data> --out <dir> --stage mse --init current
    python -m arabic_voxcpm.distill.dit ... --stage spectral --init <dir>/mse-best.pt
    python -m arabic_voxcpm.distill.dit ... --stage dmd --init <dir>/spectral-best.pt

The student is a LocDiT of the teacher's architecture (--layers: fewer decoder layers), run for K Euler steps on the
conditional input only, x <- x - dt_j * u(x, mu, t_j, cond, dt_j), over a fixed grid (--schedule, default the
current student's 1 -> 0.7071 -> 0) rounded to bf16 as the engine rounds it. Mean mode (default): the step size goes
to the estimator's dt input. --init: ``current`` (the English student, kadirnar/VoxCPM2-Turbo-distilled-DiT
student_k2_dmd.pt; it was trained with dt = 0), ``teacher`` (the teacher's LocDiT) or a checkpoint.

Adapters: teacher data of an adapter teacher (teacher_data.py --teacher adapters) holds each row's adapter. A batch
is split by adapter, and every part runs with that adapter's LocDiT LoRA (frozen, train.adapters) on the student, the
teacher and the fake score, the gradients summed over the parts: the student's shared weights learn to work under
every adapter's LoRA (and without one, for the base model's dialects), as the engine runs them.

Stages:
- mse: the K-step output regresses the teacher's 10-step CFG output for the same noise; --real adds a regression to
  the real patch.
- spectral: a multi-resolution STFT loss through the frozen AudioVAE decoder, the student's patch against the
  teacher's (each decoded after its previous real patch, cond, and the patch's own audio compared), plus --mse-weight
  times the mse loss.
- dmd: distribution matching (DMD2): a fake score (a LocDiT initialized from the teacher, --fake-init, trained online
  by flow matching on the student's samples, --fake-steps updates per student update) against the real score (the
  frozen teacher with the row's adapter, CFG 2.0, CFG-zero*); the student follows x0_fake - x0_real, normalized,
  plus --reg times the regression to the teacher.
Memory: bf16 autocast, activation checkpointing per decoder layer, fused AdamW, each model's gradients freed after its
step, the batch run in parts of at most --micro rows; the peak is logged (mem_gb).

Checkpoints are what the inference engine loads (TurboConfig.dit_checkpoint, runner.py): {"state_dict":
feat_decoder.estimator.* (bf16, no LoRA tensors), "steps", "schedule": [(t, dt), ...], "mean_mode", and "dit_layers"
for a student with fewer layers}, plus the run's metadata. Every --val-every steps: <stage>-last.pt, and
<stage>-best.pt when the stage's validation loss improves (mse and dmd: val_mse_teacher; spectral: val_spec).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from arabic_voxcpm.distill import sampler

PREFIX = "feat_decoder.estimator."
CURRENT_STUDENT = "kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt"
KEYS = ("mu", "cond", "z", "x", "target")


def resolve(spec: str) -> str:
    """A local path, or ``repo:file`` on the Hugging Face Hub (downloaded once)."""
    if os.path.exists(spec):
        return spec
    from huggingface_hub import hf_hub_download

    repo, filename = spec.split(":", 1)
    return hf_hub_download(repo, filename)


def load_estimator(model_dir: str, device="cuda"):
    """The model's LocDiT (feat_decoder.estimator), bf16, the rest of the model freed."""
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    model = VoxCPM2Model.from_local(model_dir, optimize=False, device="cpu")  # only the LocDiT goes to the GPU
    est = model.feat_decoder.estimator
    del model
    return est.to(device, torch.bfloat16).eval()


def load_vae(model_dir: str, device="cuda"):
    from arabic_voxcpm import vae

    return vae.load_vae(model_dir, device)


class TeacherData:
    """Teacher rows in memory (stored dtype), batches of random rows. The last --val-shards shards are validation.
    ``adapters``: the adapter names (manifest.json), a row's ``adapter`` indexes them (-1: the base model)."""

    def __init__(self, path: str, val_shards: int = 1, device="cuda"):
        mpath = os.path.join(path, "manifest.json")
        m = json.load(open(mpath)) if os.path.exists(mpath) else None
        shards = [os.path.join(path, s["file"]) for s in m["shards"]] if m else \
            sorted(glob.glob(os.path.join(path, "shard-*.pt")))
        assert shards, f"no teacher shards in {path}"
        self.adapters: list[str] = m["adapters"] if m else []
        parts = [torch.load(p, map_location="cpu", weights_only=True, mmap=True) for p in shards]

        def cat(ps):
            d = {k: torch.cat([p[k] for p in ps]) for k in KEYS if k in ps[0]}
            d["adapter"] = torch.cat([p["adapter"].long() if "adapter" in p else torch.full((len(p["mu"]),), -1)
                                      for p in ps])
            return d

        self.train = cat(parts[:-val_shards] if len(parts) > val_shards else parts)
        self.val = cat(parts[-val_shards:])
        self.device = device

    def name(self, a: int) -> str | None:
        return self.adapters[a] if a >= 0 else None

    def _take(self, d: dict, idx) -> dict:
        out = {k: v[idx].to(self.device, non_blocking=True).float() for k, v in d.items() if k != "adapter"}
        out["adapter"] = d["adapter"][idx]
        return out

    def batch(self, n: int, split: str = "train", gen: torch.Generator | None = None) -> dict:
        d = self.train if split == "train" else self.val
        return self._take(d, torch.randint(0, len(d["mu"]), (n,), generator=gen))

    def parts(self, b: dict, micro: int):
        """(adapter name, part of the batch, its share of the batch): one adapter per part, <= micro rows."""
        n = len(b["mu"])
        for a in sorted(set(b["adapter"].tolist())):
            idx = (b["adapter"] == a).nonzero().flatten()
            for s in range(0, len(idx), micro):
                sel = idx[s : s + micro]
                dsel = sel.to(b["mu"].device)
                yield self.name(a), {k: (v[sel] if k == "adapter" else v[dsel]) for k, v in b.items()}, len(sel) / n

    def val_parts(self, n: int, limit: int | None = None):
        """The validation rows (the first ``limit``) in parts of one adapter, <= n rows."""
        m = len(self.val["mu"]) if limit is None else min(limit, len(self.val["mu"]))
        adapter = self.val["adapter"][:m]
        for a in sorted(set(adapter.tolist())):
            idx = (adapter == a).nonzero().flatten()
            for s in range(0, len(idx), n):
                yield self.name(a), self._take(self.val, idx[s : s + n])


def parse_schedule(spec: str) -> list[tuple[float, float]]:
    """'1,0.7071,0' (the t grid) -> [(t, dt), ...]."""
    ts = [float(x) for x in spec.split(",")]
    return [(a, a - b) for a, b in zip(ts, ts[1:])]


def engine_grid(schedule) -> list[tuple[float, float]]:
    """The (t, dt) the engine feeds for a checkpoint's schedule: its t span (runner.py: [t_0] + [t - dt]) cast to
    bf16, t and dt stepped in bf16 as model_utils_cfm.solve_euler does."""
    span = torch.tensor([schedule[0][0]] + [t - dt for t, dt in schedule], dtype=torch.float64).to(torch.bfloat16)
    out, t, dt = [], span[0], span[0] - span[1]
    for k in range(1, len(span)):
        out.append((float(t), float(dt)))
        t = t - dt
        if k < len(span) - 1:
            dt = t - span[k + 1]
    return out


def student_solve(est, z, mu, cond, grid, mean_mode: bool = True) -> torch.Tensor:
    x = z
    b = z.shape[0]
    for t, dt in grid:
        tt = torch.full((b,), t, device=z.device)
        dd = torch.full((b,), dt if mean_mode else 0.0, device=z.device)
        x = x - dt * est(x, mu, tt, cond, dd).float()
    return x


def teacher_velocity(est, x, mu, cond, t, cfg: float) -> torch.Tensor:
    """The teacher's guided velocity at (x, t) (CFG-zero*, as its sampler); t: [B]."""
    b = x.shape[0]
    v = est(torch.cat([x, x]), torch.cat([mu, torch.zeros_like(mu)]), torch.cat([t, t]), torch.cat([cond, cond]),
            torch.zeros(2 * b, device=x.device)).float()
    pos, neg = v[:b], v[b:]
    s = sampler._optimized_scale(pos.reshape(b, -1), neg.reshape(b, -1)).view(b, 1, 1)
    return neg * s + cfg * (pos - neg * s)


def stft_loss(a: torch.Tensor, b: torch.Tensor, ffts=(512, 1024, 2048)) -> torch.Tensor:
    """Multi-resolution STFT loss of waveforms a against b [B, T]: spectral convergence + log-magnitude L1, averaged
    over the resolutions (hop = n / 4, Hann window)."""
    total = 0.0
    a, b = a.float(), b.float()
    for n in ffts:
        win = torch.hann_window(n, device=a.device)
        sa = torch.stft(a, n, n // 4, window=win, return_complex=True).abs().clamp(min=1e-5)
        sb = torch.stft(b, n, n // 4, window=win, return_complex=True).abs().clamp(min=1e-5)
        sc = torch.linalg.norm((sa - sb).flatten(1), dim=1) / torch.linalg.norm(sb.flatten(1), dim=1).clamp(min=1e-5)
        total = total + sc.mean() + (sa.log() - sb.log()).abs().mean()
    return total / len(ffts)


def decode_patch(vae, cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The audio of patch x [B, D, P] decoded after cond (the previous real patch): [B, samples of x]."""
    wav = vae.decode(torch.cat([cond, x], dim=2))
    wav = wav.reshape(len(x), -1)
    return wav[:, wav.shape[1] // 2 :]


def save(est, path: str, schedule, extra: dict | None = None, mean_mode: bool = True) -> None:
    sd = {f"{PREFIX}{k}": v.detach().to(torch.bfloat16).cpu() for k, v in est.state_dict().items()}
    assert not any("lora" in k for k in sd), "LoRA tensors in a student checkpoint"
    torch.save({"state_dict": sd, "steps": len(schedule), "schedule": [tuple(s) for s in schedule],
                "mean_mode": mean_mode, **(extra or {})}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str) -> dict:
    return torch.load(resolve(path), map_location="cpu", weights_only=True)


def load_student(path: str) -> dict:
    ck = load_checkpoint(path)
    return {k[len(PREFIX):]: v for k, v in ck["state_dict"].items()}


def drop_layers(est, keep: int):
    """Keep every k-th decoder layer (fewer layers, same width) for a smaller student."""
    layers = est.decoder.layers
    n = len(layers)
    idx = [round(i * (n - 1) / (keep - 1)) for i in range(keep)] if keep > 1 else [n - 1]
    est.decoder.layers = torch.nn.ModuleList([layers[i] for i in idx])
    return idx


def checkpoint_layers(module) -> int:
    """Activation checkpointing per transformer layer (LocDiT decoder layers, AudioVAE decoder blocks)."""
    from arabic_voxcpm.train.model import _checkpointed

    n = 0
    for m in module.modules():
        if type(m).__name__ in ("MiniCPMDecoderLayer", "CausalDecoderBlock"):
            _checkpointed(m)
            n += 1
    return n


@torch.no_grad()
def validate(student, data, grid, bs: int, use, mean_mode: bool, vae=None, spec_rows: int = 0) -> dict:
    student.eval()
    se_t = se_r = n = 0.0
    by: dict[str, list[float]] = {}
    for name, b in data.val_parts(bs):
        use(name)
        with torch.autocast(torch.device(data.device).type, dtype=torch.bfloat16):
            x = student_solve(student, b["z"], b["mu"], b["cond"], grid, mean_mode)
        e = ((x - b["x"]) ** 2).mean(dim=(1, 2))
        se_t += float(e.sum())
        acc = by.setdefault(name or "base", [0.0, 0])
        acc[0] += float(e.sum())
        acc[1] += len(e)
        if "target" in b:
            se_r += float(((x - b["target"]) ** 2).mean(dim=(1, 2)).sum())
        n += len(x)
    out = {"val_mse_teacher": se_t / n, "val_mse_real": se_r / n,
           "val_mse_teacher_by_adapter": {k: round(v[0] / v[1], 6) for k, v in sorted(by.items())}}
    if vae is not None and spec_rows:
        sp = m = 0.0
        for name, b in data.val_parts(min(bs, 64), limit=spec_rows):
            use(name)
            with torch.autocast(torch.device(data.device).type, dtype=torch.bfloat16):
                x = student_solve(student, b["z"], b["mu"], b["cond"], grid, mean_mode)
                ls = stft_loss(decode_patch(vae, b["cond"], x), decode_patch(vae, b["cond"], b["x"]))
            sp += float(ls) * len(x)
            m += len(x)
        out["val_spec"] = sp / m
    student.train()
    return out


def main(args) -> None:
    from arabic_voxcpm.train import adapters as A

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"log-{args.stage}.jsonl"), "a")

    def emit(rec):
        rec["time"] = round(time.time(), 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    schedule = parse_schedule(args.schedule)
    grid = engine_grid(schedule)
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    data = TeacherData(args.data, args.val_shards, dev)
    teacher = load_estimator(args.teacher, dev)
    for p in teacher.parameters():
        p.requires_grad = False

    # the adapters' LocDiT LoRAs (frozen), switched per batch part on every LocDiT
    bank = None
    if data.adapters:
        available = A.find(args.adapters or args.teacher)
        missing = sorted(set(data.adapters) - set(available))
        assert not missing, f"the teacher data's adapters {missing} are not in {args.adapters or args.teacher}"
        chosen = {n: available[n] for n in data.adapters}
        n_lora = A.inject_dit(teacher, chosen)
        bank = A.AdapterBank(chosen, prefix=PREFIX, device=dev, dtype=torch.bfloat16 if cuda else torch.float32)
        emit({"event": "adapters", "adapters": data.adapters, "lora_layers": n_lora})

    init = {"current": CURRENT_STUDENT, "teacher": None}.get(args.init, args.init)
    init_ck = load_checkpoint(init) if init else {}
    layers = args.layers or int(init_ck.get("dit_layers") or 0)
    student = copy.deepcopy(teacher).float()
    meta = {"init": args.init, "stage": args.stage, "data": os.path.abspath(args.data), "adapters": data.adapters}
    if layers:
        emit({"event": "layers", "kept": drop_layers(student, layers)})
        meta["dit_layers"] = layers
    if init:
        student.load_state_dict({k[len(PREFIX):]: v for k, v in init_ck["state_dict"].items()})
    student.train()
    for p in student.parameters():
        p.requires_grad = True
    fake = None
    if args.stage == "dmd":
        fake = copy.deepcopy(teacher if args.fake_init == "teacher" else student).float().train()
        for p in fake.parameters():
            p.requires_grad = True
    vae = None
    if args.stage == "spectral":
        vae = load_vae(args.teacher, dev)
        for p in vae.parameters():
            p.requires_grad = False
    if args.checkpointing:  # after the copies: a checkpointed layer's forward refers to its own module
        for m in (student, fake, vae):
            if m is not None:
                checkpoint_layers(m)
    strict = not layers

    def use(name, *models):
        if bank is not None:
            for m in models:
                bank.use(m, name, strict=strict or m is teacher)

    emit({"event": "data", "train_rows": len(data.train["mu"]), "val_rows": len(data.val["mu"]),
          "student_params": sum(p.numel() for p in student.parameters()), "schedule": schedule, "grid": grid,
          "mean_mode": args.mean_mode, "init": args.init})
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, fused=cuda)
    fake_opt = None
    if fake is not None:
        fake_opt = torch.optim.AdamW(fake.parameters(), lr=args.lr_fake, betas=(0.9, 0.99), weight_decay=0.0,
                                     fused=cuda)
    gen = torch.Generator().manual_seed(args.seed)
    ac = dict(device_type=dev.type, dtype=torch.bfloat16)

    def val():
        return validate(student, data, grid, args.batch, lambda n: use(n, student), args.mean_mode, vae,
                        args.spec_val_rows)

    best_key = "val_spec" if args.stage == "spectral" else "val_mse_teacher"
    best = float("inf")
    v = val()
    emit({"step": 0, "event": "val", **v})
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(1, args.iters + 1):
        lr = args.lr * min(1.0, step / args.warmup) * (0.5 * (1 + math.cos(math.pi * step / args.iters)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        rec = {}
        if args.stage == "dmd" and args.fake_steps > 0:
            # fake score: flow matching on the student's (detached) samples, each part with its adapter's LoRA
            fl_sum = 0.0
            for _ in range(args.fake_steps):
                fb = data.batch(args.batch, gen=gen)
                for name, mb, w in data.parts(fb, args.micro):
                    use(name, student, fake)
                    with torch.no_grad(), torch.autocast(**ac):
                        xs = student_solve(student, mb["z"], mb["mu"], mb["cond"], grid, args.mean_mode)
                    t = torch.sigmoid(torch.randn(len(xs), device=dev) - 0.4)
                    n = torch.randn_like(xs)
                    xt = (1 - t[:, None, None]) * xs + t[:, None, None] * n
                    with torch.autocast(**ac):
                        vf = fake(xt, mb["mu"], t, mb["cond"], torch.zeros_like(t)).float()
                    fl = F.mse_loss(vf, n - xs)
                    (fl * w).backward()
                    fl_sum += float(fl.detach()) * w
                torch.nn.utils.clip_grad_norm_(fake.parameters(), 1.0)
                fake_opt.step()
                fake_opt.zero_grad(set_to_none=True)
            rec["loss_fake"] = fl_sum / args.fake_steps
        b = data.batch(args.batch, gen=gen)
        sums: dict[str, float] = {}
        for name, mb, w in data.parts(b, args.micro):
            use(name, *(m for m in (student, teacher, fake) if m is not None))
            with torch.autocast(**ac):
                x = student_solve(student, mb["z"], mb["mu"], mb["cond"], grid, args.mean_mode)
            mse = F.mse_loss(x, mb["x"])
            if args.stage == "mse":
                parts = {"loss": mse + args.real * F.mse_loss(x, mb["target"]) if args.real > 0 else mse}
            elif args.stage == "spectral":
                with torch.autocast(**ac):
                    with torch.no_grad():
                        ref = decode_patch(vae, mb["cond"], mb["x"])
                    spec = stft_loss(decode_patch(vae, mb["cond"], x), ref)
                parts = {"loss_spec": spec, "loss_mse": mse}
                parts["loss"] = args.spec_weight * spec + args.mse_weight * mse
            else:
                t = torch.rand(len(x), device=dev) * 0.96 + 0.02
                n = torch.randn_like(x)
                xt = (1 - t[:, None, None]) * x.detach() + t[:, None, None] * n
                with torch.no_grad(), torch.autocast(**ac):
                    x0_real = xt - t[:, None, None] * teacher_velocity(teacher, xt, mb["mu"], mb["cond"], t, args.cfg)
                    x0_fake = xt - t[:, None, None] * fake(xt, mb["mu"], t, mb["cond"], torch.zeros_like(t)).float()
                    wn = (x.detach() - x0_real).abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-4)
                    grad = (x0_fake - x0_real) / wn
                dmd = 0.5 * F.mse_loss(x, (x - grad).detach())
                parts = {"loss_dmd": dmd, "loss_reg": mse, "loss": dmd + args.reg * mse}
            (parts["loss"] * w).backward()
            for k, val_ in parts.items():
                sums[k] = sums.get(k, 0.0) + float(val_.detach()) * w
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec.update(sums)
        if step % args.log_every == 0:
            mem = {"mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)} if cuda else {}
            emit({"step": step, **rec, "grad_norm": float(gn), "lr": lr,
                  "it_per_s": round(args.log_every / (time.time() - t0), 3), **mem})
            t0 = time.time()
        if step % args.val_every == 0 or step == args.iters:
            v = val()
            emit({"step": step, "event": "val", **v})
            extra = {"step": step, **meta, **{k: x for k, x in v.items() if not isinstance(x, dict)}}
            save(student, os.path.join(args.out, f"{args.stage}-last.pt"), schedule, extra, args.mean_mode)
            if v.get(best_key, float("inf")) < best:
                best = v[best_key]
                save(student, os.path.join(args.out, f"{args.stage}-best.pt"), schedule, extra, args.mean_mode)
            if args.save_every and step % args.save_every == 0:
                save(student, os.path.join(args.out, f"{args.stage}-step{step}.pt"), schedule, extra, args.mean_mode)
    done = {"event": "done", "best": best, "best_key": best_key}
    if cuda:
        done["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    emit(done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", default="/workspace/models/VoxCPM2-Arabic-Dialects",
                    help="the teacher model directory (its LocDiT; the AudioVAE for the spectral stage)")
    ap.add_argument("--adapters", help="the model directory with adapters/ (default: --teacher)")
    ap.add_argument("--data", required=True, help="teacher_data.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["mse", "spectral", "dmd"], default="mse")
    ap.add_argument("--init", default="teacher",
                    help="current (the English student), teacher (its LocDiT) or a student checkpoint")
    ap.add_argument("--schedule", default="1,0.7071,0", help="the t grid, 1 -> 0 (K steps = points - 1)")
    ap.add_argument("--no-mean-mode", dest="mean_mode", action="store_false",
                    help="feed dt = 0 to the estimator (as the English student was trained)")
    ap.add_argument("--layers", type=int, default=0, help="keep this many decoder layers (0 = all)")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=256, help="rows per forward (a batch runs in parts by adapter)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-fake", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--real", type=float, default=0.0, help="mse stage: weight of a regression to the real patch")
    ap.add_argument("--spec-weight", type=float, default=1.0, help="spectral stage: weight of the STFT loss")
    ap.add_argument("--mse-weight", type=float, default=1.0, help="spectral stage: weight of the mse loss")
    ap.add_argument("--spec-val-rows", type=int, default=2048, help="spectral stage: validation rows decoded")
    ap.add_argument("--reg", type=float, default=0.25, help="dmd stage: weight of the regression to the teacher")
    ap.add_argument("--fake-steps", type=int, default=5)
    ap.add_argument("--fake-init", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=0, help="also keep <stage>-step<N>.pt every N steps")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-checkpointing", dest="checkpointing", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
