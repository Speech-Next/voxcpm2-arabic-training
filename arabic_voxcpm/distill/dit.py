"""LocDiT students (issue #7): a few-step, guidance-free LocDiT distilled from the Arabic teacher's sampler.

    python -m arabic_voxcpm.distill.dit --teacher <merged teacher> --data <teacher data> --out <dir> --stage mse
    python -m arabic_voxcpm.distill.dit ... --stage dmd --init <dir>/best.pt

The student is the teacher's LocDiT (same architecture, initialized from its weights, or fewer layers with --layers),
run for K Euler steps on the conditional input only. It takes its step size as input (the estimator's dt, otherwise
unused: mean mode), x <- x - dt_j * u(x, mu, t_j, cond, dt_j), over a fixed grid (--schedule).
- mse: the K-step output regresses the teacher's 10-step CFG output for the same noise (teacher_data.py).
- dmd: distribution matching (DMD2): a fake score (a LocDiT copy trained online by flow matching on the student's
  samples, --fake-steps updates per student update) against the real score (the frozen teacher with CFG 2.0,
  CFG-zero*); the student follows x0_fake - x0_real, normalized, plus --reg times the regression to the teacher.
Checkpoints are what the inference engine loads (TurboConfig.dit_checkpoint): {"state_dict": feat_decoder.estimator.*,
"steps", "schedule": [(t, dt), ...], "mean_mode": True}.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from arabic_voxcpm.distill import sampler


def load_estimator(teacher_dir: str, device="cuda"):
    """The teacher's LocDiT (feat_decoder.estimator), the rest of the model freed."""
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    model = VoxCPM2Model.from_local(teacher_dir, optimize=False, device="cpu")  # only the LocDiT goes to the GPU
    est = model.feat_decoder.estimator
    del model
    return est.to(device).eval()


class TeacherData:
    """Teacher shards in memory; batches of random rows. The last --val-shards shards are validation."""

    def __init__(self, path: str, val_shards: int = 1, device="cuda"):
        shards = sorted(glob.glob(os.path.join(path, "shard-*.pt")))
        assert shards, f"no teacher shards in {path}"
        parts = [torch.load(p, map_location="cpu", weights_only=False) for p in shards]
        keys = ("mu", "cond", "z", "x", "target")

        def cat(ps):
            return {k: torch.cat([p[k] for p in ps]) for k in keys if k in ps[0]}

        self.train = cat(parts[:-val_shards] if len(parts) > val_shards else parts)
        self.val = cat(parts[-val_shards:])
        self.device = device

    def batch(self, n: int, split: str = "train", gen: torch.Generator | None = None) -> dict:
        d = self.train if split == "train" else self.val
        idx = torch.randint(0, len(d["mu"]), (n,), generator=gen)
        return {k: v[idx].to(self.device, non_blocking=True).float() for k, v in d.items()}

    def val_batches(self, n: int):
        m = len(self.val["mu"])
        for a in range(0, m, n):
            yield {k: v[a : a + n].to(self.device).float() for k, v in self.val.items()}


def parse_schedule(spec: str) -> list[tuple[float, float]]:
    """'1,0.7071,0' (the t grid) -> [(t, dt), ...]."""
    ts = [float(x) for x in spec.split(",")]
    return [(a, a - b) for a, b in zip(ts, ts[1:])]


def student_solve(est, z, mu, cond, schedule) -> torch.Tensor:
    x = z
    b = z.shape[0]
    for t, dt in schedule:
        tt = torch.full((b,), t, device=z.device)
        dd = torch.full((b,), dt, device=z.device)
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


def save(est, path: str, schedule, extra: dict | None = None) -> None:
    sd = {f"feat_decoder.estimator.{k}": v.detach().to(torch.bfloat16).cpu() for k, v in est.state_dict().items()}
    torch.save({"state_dict": sd, "steps": len(schedule), "schedule": schedule, "mean_mode": True, **(extra or {})},
               path + ".tmp")
    os.replace(path + ".tmp", path)


def load_student(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {k[len("feat_decoder.estimator."):]: v for k, v in ck["state_dict"].items()}


def drop_layers(est, keep: int):
    """Keep every k-th decoder layer (fewer layers, same width) for a smaller student."""
    layers = est.decoder.layers
    n = len(layers)
    idx = [round(i * (n - 1) / (keep - 1)) for i in range(keep)] if keep > 1 else [n - 1]
    est.decoder.layers = torch.nn.ModuleList([layers[i] for i in idx])
    return idx


@torch.no_grad()
def validate(student, data, schedule, bs: int) -> dict:
    student.eval()
    se_t = se_r = n = 0.0
    for b in data.val_batches(bs):
        with torch.autocast(torch.device(data.device).type, dtype=torch.bfloat16):
            x = student_solve(student, b["z"], b["mu"], b["cond"], schedule)
        se_t += float(((x - b["x"]) ** 2).mean(dim=(1, 2)).sum())
        if "target" in b:
            se_r += float(((x - b["target"]) ** 2).mean(dim=(1, 2)).sum())
        n += len(x)
    student.train()
    return {"val_mse_teacher": se_t / n, "val_mse_real": se_r / n}


def main(args) -> None:
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"log-{args.stage}.jsonl"), "a")

    def emit(rec):
        rec["time"] = round(time.time(), 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    schedule = parse_schedule(args.schedule)
    dev = torch.device(args.device)
    teacher = load_estimator(args.teacher, dev)
    for p in teacher.parameters():
        p.requires_grad = False
    student = copy.deepcopy(teacher).float()
    if args.layers:
        emit({"event": "layers", "kept": drop_layers(student, args.layers)})
    if args.init:
        student.load_state_dict(load_student(args.init))
    student.train()
    for p in student.parameters():
        p.requires_grad = True
    data = TeacherData(args.data, args.val_shards, dev)
    emit({"event": "data", "train_rows": len(data.train["mu"]), "val_rows": len(data.val["mu"]),
          "student_params": sum(p.numel() for p in student.parameters()), "schedule": schedule})
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    fake = fake_opt = None
    if args.stage == "dmd":
        fake = copy.deepcopy(student).float().train()
        fake_opt = torch.optim.AdamW(fake.parameters(), lr=args.lr_fake, betas=(0.9, 0.99), weight_decay=0.0)
    gen = torch.Generator().manual_seed(args.seed)
    best = float("inf")
    v = validate(student, data, schedule, args.batch)
    emit({"step": 0, "event": "val", **v})
    t0 = time.time()
    for step in range(1, args.iters + 1):
        lr = args.lr * min(1.0, step / args.warmup) * (0.5 * (1 + math.cos(math.pi * step / args.iters)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        b = data.batch(args.batch, gen=gen)
        if args.stage == "mse":
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                x = student_solve(student, b["z"], b["mu"], b["cond"], schedule)
            loss = F.mse_loss(x, b["x"])
            if args.real > 0:
                loss = loss + args.real * F.mse_loss(x, b["target"])
            rec = {"loss": loss.item()}
        else:
            # fake score: flow matching on the student's (detached) samples
            for _ in range(args.fake_steps):
                fb = data.batch(args.batch, gen=gen)
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                    xs = student_solve(student, fb["z"], fb["mu"], fb["cond"], schedule)
                t = torch.sigmoid(torch.randn(len(xs), device=dev) - 0.4)
                n = torch.randn_like(xs)
                xt = (1 - t[:, None, None]) * xs + t[:, None, None] * n
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    vf = fake(xt, fb["mu"], t, fb["cond"], torch.zeros_like(t)).float()
                fl = F.mse_loss(vf, n - xs)
                fake_opt.zero_grad(set_to_none=True)
                fl.backward()
                torch.nn.utils.clip_grad_norm_(fake.parameters(), 1.0)
                fake_opt.step()
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                x = student_solve(student, b["z"], b["mu"], b["cond"], schedule)
            t = torch.rand(len(x), device=dev) * 0.96 + 0.02
            n = torch.randn_like(x)
            xt = (1 - t[:, None, None]) * x.detach() + t[:, None, None] * n
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                x0_real = xt - t[:, None, None] * teacher_velocity(teacher, xt, b["mu"], b["cond"], t, args.cfg)
                x0_fake = xt - t[:, None, None] * fake(xt, b["mu"], t, b["cond"], torch.zeros_like(t)).float()
                w = (x.detach() - x0_real).abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-4)
                grad = (x0_fake - x0_real) / w
            dmd = 0.5 * F.mse_loss(x, (x - grad).detach())
            reg = F.mse_loss(x, b["x"])
            loss = dmd + args.reg * reg
            rec = {"loss_dmd": dmd.item(), "loss_reg": reg.item(), "loss_fake": fl.item()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            emit({"step": step, **rec, "grad_norm": float(gn), "lr": lr,
                  "it_per_s": round(args.log_every / (time.time() - t0), 2)})
            t0 = time.time()
        if step % args.val_every == 0 or step == args.iters:
            v = validate(student, data, schedule, args.batch)
            emit({"step": step, "event": "val", **v})
            save(student, os.path.join(args.out, f"{args.stage}-last.pt"), schedule, {"step": step, **v})
            if args.stage == "mse" and v["val_mse_teacher"] < best:
                best = v["val_mse_teacher"]
                save(student, os.path.join(args.out, f"{args.stage}-best.pt"), schedule, {"step": step, **v})
    emit({"event": "done", "best": best})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="merged teacher model directory")
    ap.add_argument("--data", required=True, help="teacher_data.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["mse", "dmd"], default="mse")
    ap.add_argument("--init", help="a student checkpoint to start from (the dmd stage: the mse student)")
    ap.add_argument("--schedule", default="1,0.7071,0", help="the t grid, 1 -> 0 (K steps = points - 1)")
    ap.add_argument("--layers", type=int, default=0, help="keep this many decoder layers (0 = all)")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-fake", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--real", type=float, default=0.0, help="mse stage: weight of a regression to the real patch")
    ap.add_argument("--reg", type=float, default=0.25, help="dmd stage: weight of the regression to the teacher")
    ap.add_argument("--fake-steps", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
