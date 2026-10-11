import random

import numpy as np

from arabic_voxcpm.eval import build, report
from arabic_voxcpm.eval.metrics import error_rates, normalize_ar


def test_normalize_and_rates():
    assert normalize_ar("إِزَّيَّك يا صاحبي؟") == "ازيك يا صاحبي"
    assert normalize_ar("- يا هلا <hesitation> كيف") == "يا هلا كيف"
    r = error_rates("كيف حالك اليوم", "كيف حالك")
    assert r["wer"] == 1 / 3 and not r["babble"]


def test_pick_round_robin():
    pool = [{"speaker": s, "uid": f"{s}{i}"} for s in "abc" for i in range(5)]
    got = build.pick(pool, 6, random.Random(0))
    assert len(got) == 6 and {r["speaker"] for r in got} == set("abc")
    assert all(sum(r["speaker"] == s for r in got) == 2 for s in "abc")


def test_compare_paired():
    rng = np.random.default_rng(0)
    base, cand = {}, {}
    for i in range(200):
        c = float(rng.uniform(0, 0.2))
        base[(f"x{i}", 1)] = {"group": "eg", "cer": c, "utmos": 3.0}
        cand[(f"x{i}", 1)] = {"group": "eg", "cer": c - 0.02, "utmos": 3.0}
    out = report.compare(base, cand)
    assert out["ALL"]["n"] == 200
    assert abs(out["eg"]["cer"]["diff"] + 2.0) < 1e-6 and out["eg"]["cer"]["ci"][1] < 0
