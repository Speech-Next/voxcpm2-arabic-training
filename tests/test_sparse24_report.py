"""scripts/sparse24_report.py on synthetic runs: the Done-when table, the quality and TTFA tables, the card."""

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["cer_min", "wer_min", "utmos", "voice_sim", "dnsmos_ovrl", "adi_ok", "late_stop", "runaway"]


def _report(path, base, diffs):
    comp = {k: {"base": 1.0, "cand": 1.0 + d, "diff": d, "ci": [d - 0.1, d + 0.1]} for k, d in diffs.items()}
    os.makedirs(path, exist_ok=True)
    json.dump({"runs": [{"comparisons": [{"baseline": {"name": base}, "compare": {"ALL": {"n": 1320, **comp}}}]}]},
              open(os.path.join(path, "report.json"), "w"))


def test_report_and_card(tmp_path):
    runs, sd, rel = tmp_path / "runs", tmp_path / "sd", tmp_path / "release"
    _report(runs / "x-recovered", "x-dense", {k: 0.1 for k in KEYS} | {"voice_sim": -0.002, "late_stop": 0.0,
                                                                         "runaway": 0.0})
    _report(runs / "x-pruned", "x-dense", {k: 1.0 for k in KEYS})
    (sd / "ttfa").mkdir(parents=True)
    for kind, base in (("dense", 92.0), ("sparse", 85.0)):
        for i in (1, 2):
            (sd / "ttfa" / f"{kind}-{i}.txt").write_text(
                "| requests at once | TTFB avg | TTFB p50 | TTFB p95 | whole sentence, avg |\n|---:|---:|---:|---:|---:|\n"
                f"| 1 | 23.0 ms | 23.0 ms | 24.0 ms | 0.40 s |\n| 200 | {base} ms | {base} ms | {base + 1} ms | 1.90 s |\n")
    (sd / "train" / "best").mkdir(parents=True)
    (sd / "train" / "log.jsonl").write_text(json.dumps({"step": 0, "val": {"loss": 0.5}}) + "\n"
                                            + json.dumps({"step": 250, "val": {"loss": 0.1}}) + "\n")
    (sd / "train" / "best" / "state.json").write_text(json.dumps({"step": 250, "val": {"loss": 0.1}}))
    (sd / "targets").mkdir()
    (sd / "targets" / "manifest.json").write_text(json.dumps({"batches": 550, "config": {"patches": 1500000}}))
    rel.mkdir()
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sparse24_report.py"), "--name", "x", "--sd",
                          str(sd), "--release", str(rel)], env={**os.environ, "EVAL_RUNS": str(runs)},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    summary = (sd / "summary.md").read_text()
    assert "| base_lm at least 5 ms shorter in the 200-request head step | TTFA at 200 requests +7.0 ms" in summary
    assert "| CER / WER within 0.3 points | CER +0.10, WER +0.10 | yes |" in summary
    assert "| speaker similarity within 0.005 | -0.0020 | yes |" in summary
    items = json.load(open(sd / "summary.json"))["items"]
    assert [i["met"] for i in items] == [True, True, True, True]
    card = (rel / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-nc-sa-4.0") and "| 200 | 92 / 93 ms | 85 / 86 ms |" in card


def test_report_of_a_retry(tmp_path):
    """--tag: the retry's run and train directory, the first recovery as an extra row, a summary of its own."""
    runs, sd, rel = tmp_path / "runs", tmp_path / "sd", tmp_path / "release"
    for run, d in (("x-recovered", 0.5), ("x-recovered-r2", 0.2), ("x-pruned", 1.0)):
        _report(runs / run, "x-dense", {k: d for k in KEYS} | {"voice_sim": 0.0, "late_stop": 0.0, "runaway": 0.0})
    (sd / "ttfa").mkdir(parents=True)
    for kind in ("dense", "sparse"):
        (sd / "ttfa" / f"{kind}-1.txt").write_text(f"| 200 | {92.0 if kind == 'dense' else 85.0} ms | 1 ms | 1 ms | 1 s |\n")
    for t in ("train", "train-r2"):
        (sd / t / "best").mkdir(parents=True)
        (sd / t / "log.jsonl").write_text(json.dumps({"step": 0, "val": {"loss": 0.5}}) + "\n")
        (sd / t / "best" / "state.json").write_text(json.dumps({"step": 3000 if t == "train-r2" else 2000,
                                                               "val": {"loss": 0.1}}))
    (sd / "targets").mkdir()
    (sd / "targets" / "manifest.json").write_text(json.dumps({"batches": 550, "config": {"patches": 1500000}}))
    rel.mkdir()
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sparse24_report.py"), "--name", "x", "--sd",
                          str(sd), "--release", str(rel), "--tag", "r2"], env={**os.environ, "EVAL_RUNS": str(runs)},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    summary = (sd / "summary-r2.md").read_text()
    assert "CER +0.20, WER +0.20 | yes" in summary, summary
    assert "| recovered, first run | **1.50 (+0.50" in summary, summary  # its intervals exclude 0: bold
    assert "| **recovered (r2)** | **1.20 (+0.20" in summary and "(best, step 3000)" in summary, summary
    assert not (sd / "summary.md").exists()


def test_report_with_dense_layers(tmp_path):
    """--pruned with dense layers (run_sparse24.sh DENSE_LAYERS) and --ttfa: the card says how many layers are sparse."""
    runs, sd, rel = tmp_path / "runs", tmp_path / "sd", tmp_path / "release"
    for run in ("x-recovered", "x-recovered-r3", "x-pruned"):
        _report(runs / run, "x-dense", {k: 0.1 for k in KEYS} | {"voice_sim": 0.0, "late_stop": 0.0, "runaway": 0.0})
    (sd / "ttfa-r3").mkdir(parents=True)
    for kind, ms in (("dense", 92.0), ("sparse", 86.0)):
        (sd / "ttfa-r3" / f"{kind}-1.txt").write_text(f"| 200 | {ms} ms | 1 ms | {ms + 1} ms | 1 s |\n")
    (sd / "pruned-r3").mkdir()
    (sd / "pruned-r3" / "source.json").write_text(json.dumps({"source": "s", "dense_layers": [24, 25, 26, 27]}))
    (sd / "train-r3" / "best").mkdir(parents=True)
    (sd / "train-r3" / "log.jsonl").write_text(json.dumps({"step": 0, "val": {"loss": 0.5}}) + "\n")
    (sd / "train-r3" / "best" / "state.json").write_text(json.dumps({"step": 2500, "val": {"loss": 0.1}}))
    (sd / "targets").mkdir()
    (sd / "targets" / "manifest.json").write_text(json.dumps({"batches": 550, "config": {"patches": 1500000}}))
    rel.mkdir()
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sparse24_report.py"), "--name", "x", "--sd",
                          str(sd), "--release", str(rel), "--tag", "r3", "--ttfa", str(sd / "ttfa-r3"), "--pruned",
                          str(sd / "pruned-r3")], env={**os.environ, "EVAL_RUNS": str(runs)}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "TTFA at 200 requests +6.0 ms" in (sd / "summary-r3.md").read_text()
    card = (rel / "README.md").read_text()
    assert "in 24 of its 28 layers" in card and "layers 24, 25, 26, 27 keep their dense FFN" in card
    assert "`24 with a 2:4 sparse FFN`" in card
