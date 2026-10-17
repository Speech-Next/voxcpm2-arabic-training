"""#2 paired comparison on all clips and on the items outside the development subset (eval/devset.py) a checkpoint
was chosen on: the second is the unbiased number. Best-ASR CER / WER, UTMOS, DNSMOS, speaker sim, late stop, runaway;
paired bootstrap 95 % (4000 resamples), overall and per group. Markdown to stdout.

    python -m arabic_voxcpm.eval.heldout <candidate run> <baseline run> ... --dev <dev set>/items.jsonl [--per-group]

Runs are names under EVAL_RUNS (default /workspace/data/arabic/eval/runs).
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

RUNS = os.environ.get("EVAL_RUNS", "/workspace/data/arabic/eval/runs")
M = [("cer_min", "CER % (best ASR)", 100, 2), ("wer_min", "WER % (best ASR)", 100, 1), ("utmos", "UTMOS", 1, 2),
     ("dnsmos_ovrl", "DNSMOS", 1, 2), ("voice_sim", "speaker sim", 1, 3), ("late_stop", "late stop %", 100, 1),
     ("runaway", "runaway %", 100, 1)]


def load(run):
    out = {}
    for d in map(json.loads, open(os.path.join(RUNS, run, "scores.jsonl"))):
        d["cer_min"], d["wer_min"] = min(d["cer"], d["cer_c"]), min(d["wer"], d["wer_c"])
        out[(d["id"], d["seed"])] = d
    return out


def boot(x, rng, n=4000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def table(cand, base, keys, title, per_group):
    rng = np.random.default_rng(0)
    print(f"\n### {title}: {len(keys)} clips\n")
    print("| metric | baseline | candidate | difference | 95 % interval |\n|---|---:|---:|---:|---:|")
    for m, name, s, p in M:
        mean, lo, hi = boot([(float(cand[k][m]) - float(base[k][m])) * s for k in keys], rng)
        b = np.mean([float(base[k][m]) * s for k in keys])
        bold = "**" if lo > 0 or hi < 0 else ""
        print(f"| {name} | {b:.{p}f} | {b + mean:.{p}f} | {bold}{mean:+.{p}f}{bold} | [{lo:+.{p}f}, {hi:+.{p}f}] |")
    if per_group:
        g = collections.defaultdict(list)
        for k in keys:
            g[base[k]["group"]].append(k)
        print("\n| group | n | UTMOS diff | CER diff (pts) | runaway diff (pts) |\n|---|---:|---:|---:|---:|")
        for name, ks in sorted(g.items()):
            cells = []
            for m, s, p in (("utmos", 1, 2), ("cer_min", 100, 2), ("runaway", 100, 1)):
                mean, lo, hi = boot([(float(cand[k][m]) - float(base[k][m])) * s for k in ks], rng)
                bold = "**" if lo > 0 or hi < 0 else ""
                cells.append(f"{bold}{mean:+.{p}f}{bold}")
            print(f"| {name} | {len(ks)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("bases", nargs="+")
    ap.add_argument("--dev", default="/workspace/data/arabic/eval/v2dev/items.jsonl")
    ap.add_argument("--per-group", action="store_true")
    a = ap.parse_args()
    dev = {json.loads(l)["id"] for l in open(a.dev)}
    c = load(a.cand)
    for b in a.bases:
        base = load(b)
        keys = sorted(k for k in c if k in base)
        rest = [k for k in keys if k[0] not in dev]
        print(f"\n## `{a.cand}` vs `{b}` (development items: {len(dev)}, {len(keys) - len(rest)} clips)")
        table(c, base, keys, "all clips (comparable to earlier runs)", False)
        table(c, base, rest, "outside the development subset (unbiased: the decision number)", a.per_group)
