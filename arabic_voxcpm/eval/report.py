"""Compare scored evaluation runs: per-group means and paired bootstrap intervals against a baseline run.

    python -m arabic_voxcpm.eval.report --base runs/current-teacher runs/candidate [...] [--md report.md] [--json r.json]

Pairs are (item, seed) present in both runs. A difference is significant when its 95 % interval excludes 0.
Each run is compared with the --base of its own set and cloning mode (--base repeats: one per mode); with none of
its mode, with the first of its set, and the heading says so. Every heading names the mode; speaker similarity is
n/a for runs without cloning (--mode none).

Neither ASR is a clean literal transcriber of dialect: Whisper sometimes drops the first words, Cohere sometimes
answers dialect with its MSA paraphrase ("بس هذا يمكن" -> "لكن هذا قد يكون") or loops on one letter. The per-clip
minimum of the two ("best ASR") is reported next to each.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from arabic_voxcpm.eval import common
from arabic_voxcpm.eval.score import LATE, RUNAWAY, run_parts, run_timing

# (key, scale, label, better: 1 higher, -1 lower, 0 neither)
METRICS = [("cer", 100, "CER %", -1), ("wer", 100, "WER %", -1), ("cer_c", 100, "CER % (Cohere)", -1),
           ("wer_c", 100, "WER % (Cohere)", -1), ("cer_min", 100, "CER % (best ASR)", -1),
           ("wer_min", 100, "WER % (best ASR)", -1), ("utmos", 1, "UTMOS", 1),
           ("voice_sim", 1, "speaker sim", 1), ("dnsmos_ovrl", 1, "DNSMOS", 1),
           ("adi_ok", 100, "dialect region %", 1), ("dur_ratio", 1, "duration ratio", 0),
           ("late_stop", 100, "late stop %", -1), ("runaway", 100, "runaway %", -1)]


def load(run: str) -> dict:
    """(item, seed) -> score row, with the best-ASR minimum; rows scored before the stop checks get them here."""
    rows, timing_of = {}, None
    for r in map(json.loads, open(os.path.join(run, "scores.jsonl"), encoding="utf-8")):
        for m in ("cer", "wer"):
            if r.get(m) is not None and r.get(m + "_c") is not None:
                r[m + "_min"] = min(r[m], r[m + "_c"])
        if "dur_ratio" not in r and "duration_s" in r:
            timing_of = _timing(run) if timing_of is None else timing_of
            if timing_of:
                r.update(timing_of(r))
        rows[r["id"], r["seed"]] = r
    return rows


def _timing(run: str):
    """The run's stop checks (score.run_timing), {} for a row they cannot be made for (no run.json or set at hand,
    or numbers without the library's normalizer)."""
    try:
        cfg, items, _ = run_parts(run)
        f = run_timing(cfg, items)
    except (OSError, KeyError):
        return False

    def safe(r):
        try:
            return f(r)
        except Exception:  # noqa: BLE001 (the normalizer not found): the report without this row's stop checks
            return {}

    return safe


def info(run: str) -> dict:
    """What a heading says about a run: its name and run.json's configuration (mode "ref" for older runs)."""
    cfg = common.run_config(run) if os.path.exists(os.path.join(run, "run.json")) else {"mode": "ref"}
    keys = ("mode", "set", "engine", "model", "dit", "precision", "seeds")
    return {"name": os.path.basename(run.rstrip("/")), "path": run} | {k: cfg.get(k) for k in keys}


def pick_base(bases: list[str], run: str) -> tuple[str, str]:
    """The baseline for ``run``: same set and mode, else the first of its set (else the first): (path, note)."""
    me = info(run)
    infos = [info(b) for b in bases if os.path.abspath(b) != os.path.abspath(run)] or [info(b) for b in bases]
    for i in infos:
        if i["set"] == me["set"] and i["mode"] == me["mode"]:
            return i["path"], ""
    same_set = [i for i in infos if i["set"] == me["set"]] or infos
    b = same_set[0]
    note = []
    if b["mode"] != me["mode"]:
        note.append(f"baseline mode {b['mode']}, not {me['mode']}")
    if b["set"] != me["set"]:
        note.append(f"baseline set {b['set']}: the items of both only")
    return b["path"], "; ".join(note)


def boot(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def groups_of(rows: dict, keys) -> dict:
    groups = defaultdict(list)
    for k in keys:
        groups[rows[k]["group"]].append(k)
    groups["ALL"] = list(keys)
    return groups


def compare(base: dict, cand: dict) -> dict:
    out = {}
    for g, ks in groups_of(base, sorted(set(base) & set(cand))).items():
        row = {"n": len(ks)}
        for m, scale, _, _ in METRICS:
            pairs = [(base[k][m], cand[k][m]) for k in ks if base[k].get(m) is not None and cand[k].get(m) is not None]
            if not pairs:
                continue
            b, c = (np.array(x, dtype=float) * scale for x in zip(*pairs))
            lo, hi = boot(c - b)
            row[m] = {"base": float(b.mean()), "cand": float(c.mean()), "diff": float((c - b).mean()), "ci": [lo, hi]}
        out[g] = row
    return out


def summary(rows: dict) -> dict:
    """Per-group means of one run (unpaired)."""
    out = {}
    for g, ks in groups_of(rows, sorted(rows)).items():
        out[g] = {"n": len(ks)}
        for m, scale, _, _ in METRICS:
            v = [rows[k][m] for k in ks if rows[k].get(m) is not None]
            if v:
                out[g][m] = float(np.mean(np.array(v, dtype=float)) * scale)
    return out


def gt_floor(set_dir: str | None) -> dict:
    """Per-group ASR error on the set's real recordings (gt_scores.jsonl), the floor of CER / WER."""
    path = os.path.join(set_dir or "", "gt_scores.jsonl")
    if not set_dir or not os.path.exists(path):
        return {}
    rows = {(r["id"], 0): r for r in map(json.loads, open(path, encoding="utf-8"))}
    for r in rows.values():
        for m in ("cer", "wer"):
            if r.get(m) is not None and r.get(m + "_c") is not None:
                r[m + "_min"] = min(r[m], r[m + "_c"])
    return summary(rows)


def heading(i: dict) -> str:
    return (f"`{i['name']}` (mode {i['mode']}, set {os.path.basename(str(i['set']).rstrip('/'))}, "
            f"{i.get('engine') or 'library'}, dit {i.get('dit') or '-'}, {i.get('precision') or '-'})")


def cell_na(m: str, mode: str) -> bool:
    return m == "voice_sim" and mode == "none"


def fmt(cmp: dict, name: str, mode: str = "ref") -> str:
    lines = [f"### {name}", "", "| group | n | " + " | ".join(label for _, _, label, _ in METRICS) + " |",
             "|---|---:|" + "---:|" * len(METRICS)]
    for g in sorted(cmp, key=lambda g: (g == "ALL", g)):
        cells = []
        for m, _, _, better in METRICS:
            v = cmp[g].get(m)
            if not v:
                cells.append("n/a" if cell_na(m, mode) else "–")
                continue
            sig = v["ci"][0] > 0 or v["ci"][1] < 0
            mark = ("**" if sig else "") + f"{v['cand']:.2f} ({v['diff']:+.2f})" + ("**" if sig else "")
            if sig and better:
                mark += " ✓" if v["diff"] * better > 0 else " ✗"
            cells.append(mark)
        lines.append(f"| {g} | {cmp[g]['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_ci(cmp: dict, mode: str = "ref") -> str:
    """All groups together: baseline, candidate, difference and its 95 % interval per metric."""
    lines = ["| metric | baseline | candidate | difference | 95 % interval |", "|---|---:|---:|---:|---:|"]
    for m, _, label, _ in METRICS:
        v = cmp.get("ALL", {}).get(m)
        if v:
            lines.append(f"| {label} | {v['base']:.2f} | {v['cand']:.2f} | {v['diff']:+.2f} | "
                         f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] |")
        elif cell_na(m, mode):
            lines.append(f"| {label} | n/a | n/a | n/a | no cloning |")
    return "\n".join(lines)


def fmt_summary(s: dict, floor: dict, mode: str = "ref") -> str:
    """One run's per-group means, with the ASR floor of the real recordings."""
    lines = ["| group | n | " + " | ".join(label for _, _, label, _ in METRICS) + " | CER % real (floor) |",
             "|---|---:|" + "---:|" * (len(METRICS) + 1)]
    for g in sorted(s, key=lambda g: (g == "ALL", g)):
        cells = [f"{s[g][m]:.2f}" if m in s[g] else ("n/a" if cell_na(m, mode) else "–") for m, _, _, _ in METRICS]
        f = floor.get(g, {}).get("cer")
        cells.append(f"{f:.2f}" if f is not None else "–")
        lines.append(f"| {g} | {s[g]['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(baselines: list[list[str]], runs: list[str], absolute: bool = False) -> tuple[str, dict]:
    """The markdown report and its JSON: each run against each baseline, a baseline being a list of alternatives
    (one per mode / set, pick_base chooses); absolute: also each run's per-group means and the ground-truth floor."""
    md = ["Cells: candidate mean (difference to the baseline); bold = 95 % paired bootstrap interval excludes 0; "
          f"✓ better, ✗ worse. Duration ratio: generated / expected at the group's real speaking rate; late stop: "
          f"ratio > {LATE} or max_patches reached, runaway: ratio > {RUNAWAY} or max_patches reached."]
    data = {"runs": []}
    for r in runs:
        me = info(r)
        entry = {"run": me, "comparisons": []}
        if absolute:
            entry["summary"] = summary(load(r))
            entry["gt_floor"] = gt_floor(me["set"])
            md.append(f"### {heading(me)}: per group\n\n" + fmt_summary(entry["summary"], entry["gt_floor"], me["mode"]))
        for alternatives in baselines:
            path, note = pick_base(alternatives, r)
            if os.path.abspath(path) == os.path.abspath(r):
                continue
            cmp = compare(load(path), load(r))
            bi = info(path)
            entry["comparisons"].append({"baseline": bi, "note": note, "compare": cmp})
            title = f"{heading(me)} vs `{bi['name']}` (mode {bi['mode']})" + (f" — {note}" if note else "")
            md.append(fmt(cmp, title, me["mode"]) + "\n\nAll groups:\n\n" + fmt_ci(cmp, me["mode"]))
        data["runs"].append(entry)
    return "\n\n".join(md), data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, action="append", help="baseline run; repeat for one per mode / set")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    args = ap.parse_args()
    out, data = build([args.base], args.runs)
    print(out)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(out + "\n")
    if args.json:
        json.dump(data, open(args.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
