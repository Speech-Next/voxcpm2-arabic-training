"""Train the Arabic teacher: one LoRA over LM, LocDiT and projections plus the dialect tag embeddings, on all corpora.

    python -m arabic_voxcpm.train.train --out /workspace/runs/teacher-v1 --steps 20000

Loss: VoxCPM2's own (LocDiT flow matching on the target patches + stop), weighted 1 : 1 as upstream. Batches hold up to
--budget padded positions; --accum batches per optimizer step. Every --val-every steps: the loss on a fixed validation
set (SADA's validation split + a sample of the other corpora's clips, never trained on); every --save-every steps a
checkpoint (trainable tensors + optimizer), resumable with --resume.

teacher-v3 (#5): --corpus-weights draws rows per epoch by corpus or corpus:dialect weight (dataset.py); --ref-aug p
degrades the reference of a cloning sample with probability p (augment.py): in the training step on the GPU, or from
variants precomputed with ``python -m arabic_voxcpm.train.augment`` (--ref-aug-dir). Validation stays unweighted and
clean.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from arabic_voxcpm.train import augment as A
from arabic_voxcpm.train import dataset as D
from arabic_voxcpm.train import model as M
from arabic_voxcpm.train import pack
from arabic_voxcpm.train.forward import forward

KEYS = ["text_tokens", "text_mask", "audio_feats", "audio_mask", "loss_mask", "position_ids", "labels"]


def split_rows(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    val, train = [], []
    for r in rows:
        (val if r.get("split") == "validation" or rng.random() < val_frac else train).append(r)
    return train, val


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def run_batch(model, batch, progress: float):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward(model, **{k: batch[k] for k in KEYS}, progress=progress)
    return out["loss/diff"], out["loss/stop"]


@torch.no_grad()
def validate(model, batches, device) -> dict:
    model.eval()
    d = s = n = 0.0
    for b in batches:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        ld, ls = run_batch(model, b, 1.0)
        w = float(b["loss_mask"].sum())
        d, s, n = d + float(ld) * w, s + float(ls) * w, n + w
    model.train()
    return {"val_diff": d / n, "val_stop": s / n}


def main(args) -> None:
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    def emit(rec):
        rec["time"] = round(time.time(), 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    model, tok = M.load(args.model, json.loads(args.lora) if args.lora else None)
    corpora = tuple(args.corpora.split(","))
    rows = D.select_rows(corpora, heldout=None if args.heldout == "none" else args.heldout, quality=args.quality,
                         max_cer=args.max_cer, min_dnsmos=args.min_dnsmos)
    train_rows, val_rows = split_rows(rows, args.val_frac, args.seed)
    store = D.LatentStore(corpora)
    variants = []
    if args.ref_aug > 0 and args.ref_aug_dir:  # precomputed degraded clips: v0, v1, ...
        variants = [D.LatentStore(corpora, root=d) for d in A.variant_dirs(args.ref_aug_dir)]
        assert variants, f"no variants under {args.ref_aug_dir}"
    train_ds = D.TrainSet(train_rows, store, tok, p_ref=args.p_ref, p_no_tag=args.p_no_tag, seed=args.seed,
                          p_ref_aug=args.ref_aug, aug_variants=variants)
    val_ds = D.TrainSet(val_rows[: args.val_max], store, tok, p_ref=args.p_ref, p_no_tag=0.0, seed=args.seed + 1)
    cw = D.parse_weights(args.corpus_weights)
    weights = D.row_weights(train_ds.rows, cw) if cw else None
    hours = sum(r["frames"] for r in train_ds.rows) / D.FPS / 3600
    emit({"event": "data", "train_clips": len(train_ds), "train_hours": round(hours, 1), "val_clips": len(val_ds),
          "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
          "hours": D.hours(train_ds.rows), "corpus_weights": cw,
          "effective_hours": D.hours(train_ds.rows, weights) if weights else None,
          "ref_aug": {"p": args.ref_aug, "mode": "precomputed" if variants else "step" if args.ref_aug > 0 else None,
                      "variants": len(variants),
                      "variant_clips": [sum(r["uid"] in v for r in train_ds.rows) for v in variants]}})
    if hours < args.min_train_hours:
        raise SystemExit(f"{hours:.1f} h of training data < --min-train-hours {args.min_train_hours}: latents missing?")
    augmenter = None
    if args.ref_aug > 0 and not variants:
        augmenter = A.RefAugmenter(model.audio_vae, seed=args.seed, chunk_frames=args.ref_aug_chunk,
                                   sr_cond=args.ref_aug_sr_cond or None)
    sampler = D.TokenBudgetBatches(train_ds, args.budget, seed=args.seed, weights=weights)
    loader = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler, collate_fn=pack.collate,
                                         num_workers=args.workers, pin_memory=True)
    val_batches = list(torch.utils.data.DataLoader(val_ds, batch_sampler=D.TokenBudgetBatches(val_ds, args.budget, seed=7),
                                                   collate_fn=pack.collate, num_workers=2))
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "tag_embed" not in n]
    tags = [p for n, p in model.named_parameters() if p.requires_grad and "tag_embed" in n]
    opt = torch.optim.AdamW([{"params": lora, "weight_decay": args.wd}, {"params": tags, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    step, epoch, best = 0, 0, float("inf")
    if args.resume:
        M.load_trainable(model, args.resume)
        st = json.load(open(os.path.join(args.resume, "train_state.json")))
        step, epoch, best = st["step"], st["epoch"], st.get("best", best)
        opt.load_state_dict(torch.load(os.path.join(args.resume, "optim.pt"), map_location="cuda"))
        emit({"event": "resume", "from": args.resume, "step": step})
    model.train()
    device = "cuda"
    t0, seen_tok, acc = time.time(), 0, 0
    agg = {"diff": 0.0, "stop": 0.0, "n": 0, "aug_s": 0.0, "n_aug": 0}
    mem_warned = False
    while step < args.steps:
        sampler.epoch = train_ds.epoch = epoch
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if augmenter is not None:  # before the forward: its memory is free again for the activations
                ta = time.time()
                agg["n_aug"] += augmenter(batch, (step, acc))
                torch.cuda.synchronize()
                agg["aug_s"] += time.time() - ta
            ld, ls = run_batch(model, batch, step / args.steps)
            ((ld * args.w_diff + ls * args.w_stop) / args.accum).backward()
            agg["diff"] += float(ld)
            agg["stop"] += float(ls)
            agg["n"] += 1
            seen_tok += int(batch["text_tokens"].numel())
            acc += 1
            if acc < args.accum:
                continue
            acc = 0
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            gnorm = torch.nn.utils.clip_grad_norm_(lora + tags, args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30
                rec = {"step": step, "epoch": epoch, "loss_diff": agg["diff"] / agg["n"], "loss_stop": agg["stop"] / agg["n"],
                       "grad_norm": float(gnorm), "lr": opt.param_groups[0]["lr"], "tok_per_s": round(seen_tok / dt),
                       "mem_gb": round(mem, 1), "mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1)}
                if augmenter is not None:  # share of the step time spent degrading references
                    rec |= {"n_aug": agg["n_aug"], "aug_frac": round(agg["aug_s"] / dt, 3)}
                emit(rec)
                if args.max_mem_gb and mem > args.max_mem_gb and not mem_warned:
                    emit({"event": "warning", "step": step, "msg": f"peak memory {mem:.1f} GB > --max-mem-gb "
                                                                   f"{args.max_mem_gb} (lower --budget)"})
                    mem_warned = True
                agg = {"diff": 0.0, "stop": 0.0, "n": 0, "aug_s": 0.0, "n_aug": 0}
                t0, seen_tok = time.time(), 0
            if step % args.val_every == 0 or step == args.steps:
                v = validate(model, val_batches, device)
                score = v["val_diff"] + v["val_stop"]
                emit({"step": step, "event": "val", **v})
                if score < best:
                    best = score
                    M.save(model, os.path.join(args.out, "best"), {"step": step, "epoch": epoch, "best": best, **v})
            if step % args.save_every == 0 or step == args.steps:
                path = os.path.join(args.out, "last")
                M.save(model, path, {"step": step, "epoch": epoch, "best": best})
                torch.save(opt.state_dict(), os.path.join(path, "optim.pt"))
            if step >= args.steps:
                break
        epoch += 1
    emit({"event": "done", "step": step, "best": best})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects", help="base VoxCPM2 (adapters ignored)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v1/heldout.json", help="'none' for smoke tests")
    ap.add_argument("--corpora", default="lahgtna,sada", help="comma-separated; synth = synthetic speech (synth/)")
    ap.add_argument("--quality", help="data/quality.py scores: real clips kept only if scored and passing")
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("--min-dnsmos", type=float, default=3.0)
    ap.add_argument("--lora", help="JSON overrides of model.LORA_DEFAULT, e.g. '{\"r\": 128, \"alpha\": 128}'")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--budget", type=int, default=12000, help="padded positions per batch")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--w-diff", type=float, default=1.0)
    ap.add_argument("--w-stop", type=float, default=1.0)
    ap.add_argument("--p-ref", type=float, default=0.7)
    ap.add_argument("--p-no-tag", type=float, default=0.1)
    ap.add_argument("--corpus-weights", help="expected draws per epoch, e.g. 'synth=2,sada=0.5,synth:hejazi=3' "
                                             "(corpus or corpus:dialect; others 1)")
    ap.add_argument("--ref-aug", type=float, default=0.0, help="probability that a cloning sample's reference is "
                                                               "degraded (noise / reverberation / low-pass)")
    ap.add_argument("--ref-aug-dir", help="precomputed degraded variants (augment.py); default: in the training step")
    ap.add_argument("--ref-aug-chunk", type=int, default=1250, help="latent frames per AudioVAE round-trip batch")
    ap.add_argument("--ref-aug-sr-cond", type=int, default=16000,
                    help="decoder bandwidth condition (16000: the encoder's band, least round-trip drift; 0: default)")
    ap.add_argument("--max-mem-gb", type=float, default=0.0, help="warn once when peak allocated memory exceeds it")
    ap.add_argument("--min-train-hours", type=float, default=0.0, help="refuse to train on less (incomplete latents)")
    ap.add_argument("--val-frac", type=float, default=0.004)
    ap.add_argument("--val-max", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume")
    main(ap.parse_args())
