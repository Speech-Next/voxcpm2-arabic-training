"""#8 phase 2's summary (the issue's Done-when items) and the release directory's card, from the runs of
scripts/run_sparse24.sh.

    python scripts/sparse24_report.py --name sparse24 --sd /workspace/runs/sparse-8/phase2 --release <dir>
"""

import argparse
import glob
import json
import os
import re
import statistics

EVAL = os.environ.get("EVAL_RUNS", "/workspace/data/arabic/eval/runs")
METRICS = [("cer_min", "CER % (best ASR)", 2), ("wer_min", "WER % (best ASR)", 2), ("utmos", "UTMOS", 2),
           ("voice_sim", "speaker sim", 3), ("dnsmos_ovrl", "DNSMOS", 2), ("adi_ok", "dialect region %", 1),
           ("late_stop", "late stop %", 2), ("runaway", "runaway %", 2)]


def compare(run: str, base: str) -> dict:
    r = json.load(open(os.path.join(EVAL, run, "report.json")))["runs"][0]
    for c in r["comparisons"]:
        if c["baseline"]["name"].startswith(base):
            return c["compare"]["ALL"]
    raise SystemExit(f"{run}: no comparison with {base}")


def ttfa(path: str) -> dict:
    """{requests: (avg, p50, p95)} of a benchmark.py table."""
    out = {}
    for line in open(path):
        m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+) ms\s*\|\s*([\d.]+) ms\s*\|\s*([\d.]+) ms\s*\|", line)
        if m:
            out[int(m[1])] = tuple(float(m[i]) for i in (2, 3, 4))
    return out


def cell(c: dict, key: str, digits: int) -> str:
    d = c[key]
    lo, hi = d["ci"]
    sig = lo > 0 or hi < 0
    s = f"{d['cand']:.{digits}f} ({d['diff']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}])"
    return f"**{s}**" if sig else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sparse24")
    ap.add_argument("--sd", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--tag", default="", help="a retry's tag (run_sparse24.sh TAG): its run, train directory, summary")
    a = ap.parse_args()
    sfx = f"-{a.tag}" if a.tag else ""
    dense, pruned, rec = f"{a.name}-dense", f"{a.name}-pruned", f"{a.name}-recovered{sfx}"
    c_rec, c_pr = compare(rec, dense), compare(pruned, dense)
    c_first = compare(f"{a.name}-recovered", dense) if a.tag else None  # the first recovery, for comparison
    base = c_rec  # 'base' fields: the dense model's means

    runs = {k: [ttfa(p) for p in sorted(glob.glob(os.path.join(a.sd, "ttfa", f"{k}-*.txt")))] for k in ("dense", "sparse")}
    levels = sorted(set.intersection(*[set(t) for v in runs.values() for t in v])) if all(runs.values()) else []

    def mean(kind, level, i):
        return statistics.mean(t[level][i] for t in runs[kind])

    train = os.path.join(a.sd, f"train{sfx}")
    log = [json.loads(l) for l in open(os.path.join(train, "log.jsonl"))]
    vals = [r for r in log if "val" in r]
    best = json.load(open(os.path.join(train, "best", "state.json")))
    man = json.load(open(os.path.join(a.sd, "targets", "manifest.json")))

    d = {k: c_rec[k]["diff"] for k, _, _ in METRICS}
    head = (mean("dense", 200, 0) - mean("sparse", 200, 0)) if 200 in levels else None
    items = [
        ("base_lm at least 5 ms shorter in the 200-request head step",
         f"TTFA at 200 requests {head:+.1f} ms on average (dense − sparse, 2 runs each)" if head is not None else "not measured",
         head is not None and head >= 5),
        ("CER / WER within 0.3 points", f"CER {d['cer_min']:+.2f}, WER {d['wer_min']:+.2f}",
         d["cer_min"] <= 0.3 and d["wer_min"] <= 0.3),
        ("speaker similarity within 0.005", f"{d['voice_sim']:+.4f}", d["voice_sim"] >= -0.005),
        ("no more runaways / late stops", f"runaway {d['runaway']:+.2f} points, late stop {d['late_stop']:+.2f} points",
         d["runaway"] <= 0 and d["late_stop"] <= 0),
    ]
    lines = [f"# #8 phase 2: 2:4 sparse base_lm FFN ({a.name}{sfx})", "",
             "## Done when (#2: v2 set, `ref` cloning, int8, head start 8, the Arabic distilled LocDiT; paired "
             "against the dense model)", "", "| item | result | met |", "|---|---|---|"]
    lines += [f"| {i} | {r} | {'yes' if ok else '**no**'} |" for i, r, ok in items]
    lines += ["", "## Quality (1,320 clips; candidate (difference [95 % paired bootstrap]); bold: the interval excludes 0)",
              "", "| model | " + " | ".join(n for _, n, _ in METRICS) + " |", "|---|" + "---:|" * len(METRICS),
              "| dense | " + " | ".join(f"{base[k]['base']:.{dg}f}" for k, _, dg in METRICS) + " |",
              "| pruned, no recovery | " + " | ".join(cell(c_pr, k, dg) for k, _, dg in METRICS) + " |"]
    if c_first is not None:
        lines.append("| recovered, first run | " + " | ".join(cell(c_first, k, dg) for k, _, dg in METRICS) + " |")
    lines += [f"| **recovered{f' ({a.tag})' if a.tag else ''}** | " + " | ".join(cell(c_rec, k, dg) for k, _, dg in METRICS) + " |"]
    if levels:
        lines += ["", "## Time to first audio (examples/benchmark.py, dense and sparse alternating, mean of 2 runs)", "",
                  "| requests | dense avg / p95 | sparse avg / p95 | difference (avg) |", "|---:|---:|---:|---:|"]
        for lv in levels:
            lines.append(f"| {lv} | {mean('dense', lv, 0):.1f} / {mean('dense', lv, 2):.1f} ms | "
                         f"{mean('sparse', lv, 0):.1f} / {mean('sparse', lv, 2):.1f} ms | "
                         f"{mean('sparse', lv, 0) - mean('dense', lv, 0):+.1f} ms |")
    lines += ["", "## Training", "", "| | |", "|---|---|",
              f"| targets | {man['batches']} batches, {man['config']['patches']:,} target patches |",
              f"| validation loss | {vals[0]['val']['loss']:.4f} before recovery -> {best['val']['loss']:.4f} "
              f"(best, step {best['step']}) |"]
    summary = "\n".join(lines) + "\n"
    open(os.path.join(a.sd, f"summary{sfx}.md"), "w").write(summary)
    json.dump({"items": [{"item": i, "result": r, "met": ok} for i, r, ok in items], "recovered": c_rec,
               "pruned": c_pr}, open(os.path.join(a.sd, f"summary{sfx}.json"), "w"), indent=1)
    print(summary)

    q = lambda k, dg: f"{c_rec[k]['cand']:.{dg}f}"  # noqa: E731
    qb = lambda k, dg: f"{base[k]['base']:.{dg}f}"  # noqa: E731
    ttfa_rows = "\n".join(f"| {lv} | {mean('dense', lv, 0):.0f} / {mean('dense', lv, 2):.0f} ms | "
                          f"{mean('sparse', lv, 0):.0f} / {mean('sparse', lv, 2):.0f} ms |" for lv in levels)
    card = f"""---
license: cc-by-nc-sa-4.0
base_model: VoxCPM-Kernel/VoxCPM2-Arabic-Dialects
library_name: voxcpm
pipeline_tag: text-to-speech
language:
- ar
tags:
- text-to-speech
- voxcpm2
- arabic-dialects
- sparsity
- 2:4
---

# VoxCPM2 Arabic Dialects, 2:4 sparse

[VoxCPM2-Arabic-Dialects](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects) with its main language model's
feed-forward layers pruned to the 2:4 pattern (2 of every 4 weights are zero) and then trained back to the dense
model's outputs. The [Turbo engine](https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo) runs these layers on
the GPU's sparse tensor cores, so the first audio comes sooner. The 11 dialect adapters are the dense model's.

## At a glance

| | |
|---|---|
| Base | VoxCPM2-Arabic-Dialects (VoxCPM2, 2B parameters, 48 kHz) with its 11 adapters |
| Changed weights | base_lm's FFN (gate, up, down of its 28 layers): 2:4 sparse along the input; everything else unchanged |
| Pruning | SparseGPT 2:4, calibrated on VoxCPM2-Arabic-Train-v2 contexts with every adapter |
| Recovery | {best['step']} steps of distillation from the dense model with the same adapter |
| Engine | detects the 2:4 weights itself and uses its sparse int8 kernels (no option needed) |
| License | CC BY-NC-SA 4.0 (trained on non-commercial data); VoxCPM2 itself is Apache-2.0 |

## Results

Held-out evaluation set: 17 dialects × 40 sentences × 2 seeds (Iraqi and Moroccan: 30), 1,320 clips, voice cloning
from another clip of the same speaker, int8 engine, the default distilled LocDiT.

| model | CER % | WER % | UTMOS | speaker sim | late stop % | runaway % |
|---|---:|---:|---:|---:|---:|---:|
| dense | {qb('cer_min', 2)} | {qb('wer_min', 2)} | {qb('utmos', 2)} | {qb('voice_sim', 3)} | {qb('late_stop', 2)} | {qb('runaway', 2)} |
| **2:4 sparse (this model)** | {q('cer_min', 2)} | {q('wer_min', 2)} | {q('utmos', 2)} | {q('voice_sim', 3)} | {q('late_stop', 2)} | {q('runaway', 2)} |

Time to first audio, one A10, all requests sent at once (average / p95 of 2 runs each, dense and sparse
alternating):

| requests at once | dense | 2:4 sparse |
|---:|---:|---:|
{ttfa_rows}

- CER / WER: the better of Whisper large-v3 and Cohere Transcribe Arabic; lower is better. UTMOS: predicted
  naturalness, 1-5.

## Usage

```python
from nanovllm_voxcpm import DialectTTS

tts = await DialectTTS.load("VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Sparse24")
await tts.add_voice("ahmed", "ahmed.wav")
audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")
```

The engine logs `28 with a 2:4 sparse FFN` when it uses the sparse kernels (an Ampere or newer GPU).

## Training

| | |
|---|---|
| Data | contexts of VoxCPM2-Arabic-Train-v2 (135 h); evaluation speakers and texts left out; all 11 adapters equally likely |
| Calibration | SparseGPT, layer by layer, gate / up then down |
| Targets | the dense model's LocDiT conditioning, stop logits and FSQ latents on {man['config']['patches']:,} target patches |
| Recovery | only the kept FFN weights train (the zeros stay), AdamW, {best['step']} steps |
| Hardware | one NVIDIA A10 |
| Code | [`Speech-Next/voxcpm2-arabic-training`](https://github.com/Speech-Next/voxcpm2-arabic-training): `arabic_voxcpm/distill/sparse_ffn.py`, `scripts/run_sparse24.sh` |
"""
    open(os.path.join(a.release, "README.md"), "w").write(card)


if __name__ == "__main__":
    main()
