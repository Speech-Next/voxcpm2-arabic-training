"""heldout.py: the same comparison on all clips and without the development items."""
import json
import subprocess
import sys


def run(root, name, utmos):
    d = root / name
    d.mkdir()
    rows = [{"id": f"x{i}", "seed": s, "group": "bh" if i < 2 else "eg", "utmos": utmos + 0.01 * i, "cer": 0.05,
             "cer_c": 0.06, "wer": 0.2, "wer_c": 0.3, "dnsmos_ovrl": 3.0, "voice_sim": 0.9, "late_stop": False,
             "runaway": False} for i in range(4) for s in (2000, 2001)]
    (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_heldout(tmp_path):
    run(tmp_path, "base", 2.0)
    run(tmp_path, "cand", 2.5)
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({"id": "x0"}) + "\n")
    out = subprocess.run([sys.executable, "-m", "arabic_voxcpm.eval.heldout", "cand", "base", "--dev", str(dev),
                          "--per-group"], capture_output=True, text=True, check=True,
                         env={"EVAL_RUNS": str(tmp_path), "PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}).stdout
    assert "all clips (comparable to earlier runs): 8 clips" in out
    assert "outside the development subset (unbiased: the decision number): 6 clips" in out
    # without x0 (the development item): x1..x3, base 2.02 on average, the candidate 0.50 above
    assert "| UTMOS | 2.02 | 2.52 | **+0.50** |" in out.split("unbiased")[1]
    assert "| bh | 2 |" in out and "| eg | 4 |" in out
