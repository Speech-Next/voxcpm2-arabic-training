"""Compare scored evaluation runs: per-group means and paired bootstrap intervals against a baseline run.

    python -m arabic_voxcpm.eval.report --base runs/current-teacher runs/candidate [...] [--md report.md]

Pairs are (item, seed) present in both runs. A difference is significant when its 95 % interval excludes 0.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

METRICS = [("cer", 100, "CER %", -1), ("wer", 100, "WER %", -1), ("utmos", 1, "UTMOS", 1),
           ("voice_sim", 1, "speaker sim", 1), ("dnsmos_ovrl", 1, "DNSMOS", 1)]


def load(run: str) -> dict:
    return {(r["id"], r["seed"]): r for r in map(json.loads, open(os.path.join(run, "scores.jsonl"), encoding="utf-8"))}


def boot(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(base: dict, cand: dict) -> dict:
    keys = sorted(set(base) & set(cand))
    groups = defaultdict(list)
    for k in keys:
        groups[base[k]["group"]].append(k)
    groups["ALL"] = keys
    out = {}
    for g, ks in groups.items():
        row = {"n": len(ks)}
        for m, scale, _, _ in METRICS:
            pairs = [(base[k][m], cand[k][m]) for k in ks if m in base[k] and m in cand[k]]
            if not pairs:
                continue
            b, c = (np.array(x, dtype=float) * scale for x in zip(*pairs))
            lo, hi = boot(c - b)
            row[m] = {"base": float(b.mean()), "cand": float(c.mean()), "diff": float((c - b).mean()), "ci": [lo, hi]}
        out[g] = row
    return out


def fmt(cmp: dict, name: str) -> str:
    lines = [f"### {name}", "", "| group | n | " + " | ".join(label for _, _, label, _ in METRICS) + " |",
             "|---|---:|" + "---:|" * len(METRICS)]
    for g in sorted(cmp, key=lambda g: (g == "ALL", g)):
        cells = []
        for m, _, _, better in METRICS:
            v = cmp[g].get(m)
            if not v:
                cells.append("–")
                continue
            sig = v["ci"][0] > 0 or v["ci"][1] < 0
            good = (v["diff"] * better) > 0
            mark = ("**" if sig else "") + f"{v['cand']:.2f} ({v['diff']:+.2f})" + ("**" if sig else "")
            cells.append(mark + (" ✓" if sig and good else " ✗" if sig else ""))
        lines.append(f"| {g} | {cmp[g]['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md")
    args = ap.parse_args()
    base = load(args.base)
    text = [f"Baseline: `{os.path.basename(args.base.rstrip('/'))}`. Cells: candidate mean (difference); bold = 95 % "
            "paired bootstrap interval excludes 0; ✓ better, ✗ worse."]
    for r in args.runs:
        text.append(fmt(compare(base, load(r)), os.path.basename(r.rstrip("/"))))
    out = "\n\n".join(text)
    print(out)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(out + "\n")
