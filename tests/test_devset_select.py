"""devset.py: the first items per group, with their voices, map and ground-truth rows; select.py: the best UTMOS among
the candidates whose CER and runaways stay within the guards of the reference."""

import json
import os

from arabic_voxcpm.eval import devset, select


def test_devset(tmp_path):
    base = tmp_path / "v2"
    (base / "voices").mkdir(parents=True)
    items = [{"id": f"{g}-{i}", "group": g, "voice": f"{g}_{i % 2}", "text": "t"} for g in ("bh", "eg") for i in range(5)]
    (base / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    voices = [{"id": f"{g}_{k}", "group": g} for g in ("bh", "eg") for k in range(3)]
    (base / "voices.json").write_text(json.dumps(voices))
    (base / "dialect_map.json").write_text("{}")
    (base / "gt_scores.jsonl").write_text("".join(json.dumps({"id": x["id"], "cer": 0.1}) + "\n" for x in items))
    out = tmp_path / "dev"
    got = devset.build(str(base), str(out), per_group=2)
    assert [x["id"] for x in got] == ["bh-0", "bh-1", "eg-0", "eg-1"]
    assert {v["id"] for v in json.load(open(out / "voices.json"))} == {"bh_0", "bh_1", "eg_0", "eg_1"}
    assert [json.loads(l)["id"] for l in open(out / "gt_scores.jsonl")] == ["bh-0", "bh-1", "eg-0", "eg-1"]
    assert os.path.islink(out / "voices") and os.path.islink(out / "dialect_map.json")
    devset.build(str(base), str(out), per_group=2)  # again: the links are kept


def run(tmp_path, name, utmos, cer, runaway):
    d = tmp_path / name
    d.mkdir()
    rows = [{"id": f"x{i}", "seed": 2002, "utmos": utmos, "cer": cer, "cer_c": cer + 0.1, "wer": 0.2, "wer_c": 0.3,
             "runaway": runaway, "dur_ratio": 1.0} for i in range(4)]
    (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(d)


def test_select(tmp_path):
    ref = run(tmp_path, "ref", 2.10, 0.050, 0.0)
    best_but_garbled = run(tmp_path, "a", 2.40, 0.060, 0.0)  # CER +1 point: out
    runaway = run(tmp_path, "b", 2.35, 0.050, 0.25)  # runaways +25 points: out
    good = run(tmp_path, "c", 2.30, 0.052, 0.0)
    worse = run(tmp_path, "d", 2.20, 0.049, 0.0)
    best, table = select.choose(ref, [best_but_garbled, runaway, good, worse])
    assert best == good and table["guards_passed"] and not table["candidates"]["a"]["passes"]
    best, table = select.choose(ref, [best_but_garbled, runaway])  # nothing passes: the best UTMOS, flagged
    assert best == best_but_garbled and not table["guards_passed"]
