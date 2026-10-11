"""Teacher data for the LocDiT students (issue #6): the teacher's own sampler output for real contexts.

    python -m arabic_voxcpm.distill.teacher_data --teacher /workspace/models/arabic-teacher-v1-best \\
        --out /workspace/data/arabic/teacher/v1 --patches 1500000

Training clips are packed as in training (dialect tag, 70 % cloning a voice by reference); the teacher's LMs run over
them teacher-forced and give, at every target patch, the LocDiT's conditioning mu and the previous real patch (cond).
From seeded noise z the teacher's sampler (10 Euler steps, CFG 2.0, CFG-zero*: what inference runs) solves the patch
x. Shards of 50k rows: mu (bf16 [N, C]), cond / z / x / the real patch (fp16 [N, D, P]), dialect index, clip uid and
patch index.
Dialects are sampled evenly (each clip's chance ~ 1 / its dialect's clip count), so small dialects are not drowned.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

import torch

from arabic_voxcpm.distill import sampler
from arabic_voxcpm.train import dataset as D
from arabic_voxcpm.train import dialects, pack
from arabic_voxcpm.train.forward import dit_inputs
from arabic_voxcpm.train.model import Tokenizer

SHARD_ROWS = 50_000


def load_teacher(path: str):
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    model = VoxCPM2Model.from_local(path, optimize=False)  # bf16, eval
    return model.cuda().eval()


def balanced(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    count = Counter(r["dialect"] for r in rows)
    weights = [1.0 / count[r["dialect"]] for r in rows]
    return rng.choices(rows, weights=weights, k=n)


class Writer:
    def __init__(self, out: str):
        self.out, self.k, self.buf = out, 0, {k: [] for k in ("mu", "cond", "z", "x", "target", "dialect", "uid", "patch")}
        os.makedirs(out, exist_ok=True)
        while os.path.exists(self.path(self.k)):
            self.k += 1

    def path(self, k):
        return os.path.join(self.out, f"shard-{k:05d}.pt")

    def rows(self) -> int:
        return sum(len(x) for x in self.buf["mu"])

    def add(self, **kw):
        for k, v in kw.items():
            self.buf[k].append(v)
        if self.rows() >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if not self.buf["mu"]:
            return
        d = {k: torch.cat(v) if isinstance(v[0], torch.Tensor) else sum(v, []) for k, v in self.buf.items()}
        torch.save(d, self.path(self.k) + ".tmp")
        os.replace(self.path(self.k) + ".tmp", self.path(self.k))
        self.k += 1
        self.buf = {k: [] for k in self.buf}


@torch.no_grad()
def main(args) -> None:
    torch.manual_seed(args.seed)
    model = load_teacher(args.teacher)
    tok = Tokenizer(args.teacher)
    rows = D.select_rows(heldout=None if args.heldout == "none" else args.heldout)
    store = D.LatentStore()
    avg_patches = sum(r["frames"] for r in rows) / len(rows) / 4
    picked = balanced(rows, int(args.patches / avg_patches * 1.05), args.seed)
    ds = D.TrainSet(picked, store, tok, p_ref=args.p_ref, p_no_tag=0.0, seed=args.seed)
    sampler_ = D.TokenBudgetBatches(ds, args.budget, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler_, collate_fn=pack.collate, num_workers=4)
    order = {d: i for i, d in enumerate(dialects.DIALECTS)}
    writer = Writer(args.out)
    json.dump({"teacher": args.teacher, "steps": args.steps, "cfg": args.cfg, "p_ref": args.p_ref, "seed": args.seed,
               "dialects": dialects.DIALECTS}, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    done, t0 = writer.k * SHARD_ROWS, time.time()
    batches = sampler_.batches()
    for bi, idx in enumerate(batches):
        if done >= args.patches:
            break
        batch = {k: v.cuda(non_blocking=True) for k, v in pack.collate([ds[i] for i in idx]).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, target, cond, _, lm_mask = dit_inputs(model, **batch)
        z = torch.randn(target.shape, device="cuda", generator=g, dtype=torch.float32)
        x = sampler.solve(model.feat_decoder.estimator, z, mu, cond.float(), cfg_value=args.cfg, steps=args.steps)
        # which clip and patch each row is (row-major over the batch's loss positions)
        pos = lm_mask.nonzero().tolist()
        first = {}
        meta_d, meta_u, meta_p = [], [], []
        for b, t in pos:
            r = ds.rows[idx[b]]
            first.setdefault(b, t)
            meta_d.append(order[r["dialect"]])
            meta_u.append(r["uid"])
            meta_p.append(t - first[b])
        writer.add(mu=mu.to(torch.bfloat16).cpu(), cond=cond.half().cpu(), z=z.half().cpu(), x=x.half().cpu(),
                   target=target.half().cpu(),
                   dialect=torch.tensor(meta_d, dtype=torch.int16), uid=meta_u, patch=torch.tensor(meta_p, dtype=torch.int32))
        done += len(pos)
        if bi % 20 == 0:
            print(f"{done} patches, {done / (time.time() - t0):.0f}/s", flush=True)
    writer.flush()
    print(f"done: {done} patches in {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="merged teacher model directory (train.model.export_merged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v1/heldout.json")
    ap.add_argument("--patches", type=int, default=1_500_000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--p-ref", type=float, default=0.7)
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
