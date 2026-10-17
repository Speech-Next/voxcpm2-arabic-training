import csv
import json
import os

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from arabic_voxcpm.eval import pairs  # noqa: E402

TEXT = "والعكس أيضا الشركات الأمريكية تستفيد"
WORDS = [("والعكس", 0.1, 0.7), ("أيضا", 0.7, 1.04), ("الشركات", 1.3, 2.2), ("الأمريكية", 2.2, 2.8)]


def test_opening_ok():
    assert pairs.opening_ok(WORDS, TEXT)
    assert pairs.opening_ok([("وال", 0.1, 0.3), ("عكس", 0.3, 0.7)], TEXT)  # a split clitic
    assert not pairs.opening_ok([], TEXT)
    assert not pairs.opening_ok([("اشتركوا", 0.0, 0.5), ("في", 0.5, 0.6), ("القناة", 0.6, 1.0)], TEXT)  # filler
    assert not pairs.opening_ok([("والعكس", 1.3, 1.9)], TEXT)  # starts too late
    assert pairs.opening_ok([("والعكس", 1.3, 1.9)], TEXT, max_onset=1.5)


def test_cut_point():
    # 2 words reaching 1.0 s: cut in the pause after "أيضا", GAP_S at most
    at, k = pairs.cut_point(WORDS, 1.0, 2, 3.0)
    assert k == 2 and at == pytest.approx(1.04 + pairs.GAP_S)
    # the next word starts sooner than GAP_S: cut there
    at, k = pairs.cut_point([("a", 0.0, 0.6), ("b", 0.6, 1.1), ("c", 1.15, 1.6)], 1.0, 2, 3.0)
    assert k == 2 and at == pytest.approx(1.15)
    # 3 words needed
    at, k = pairs.cut_point(WORDS, 1.0, 3, 3.0)
    assert k == 3 and at == pytest.approx(2.2)  # "الأمريكية" starts right after
    # the last word: up to the end of the clip
    at, k = pairs.cut_point(WORDS, 2.5, 2, 2.9)
    assert k == 4 and at == pytest.approx(2.9)
    assert pairs.cut_point(WORDS[:1], 1.0, 2, 3.0) is None


def test_cut_fades_out():
    a = np.ones(48000, dtype=np.float32)
    out = pairs.cut(a, 48000, 0.5)
    assert len(out) == 24000 and out[0] == 1.0 and out[-1] == 0.0 and a[23999] == 1.0


class FakeWhisper:
    """Words by file: a cut clip (in data/) keeps the words that end before its length."""

    def __init__(self, by_stem):
        self.by_stem = by_stem

    def words(self, path):
        stem = os.path.basename(path).split("__")[0].removesuffix(".wav")
        ws = self.by_stem[stem]
        if "__" in path:
            dur = sf.info(path).duration
            ws = [w for w in ws if w[2] <= dur + 1e-6]
        return ws


def _runs(tmp_path):
    s = tmp_path / "set"
    s.mkdir()
    items = [{"id": "eg-0", "group": "eg", "text": TEXT}, {"id": "eg-1", "group": "eg", "text": TEXT},
             {"id": "ma-0", "group": "ma", "text": TEXT}]
    (s / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    for name in ("a", "b"):
        r = tmp_path / name
        (r / "wav").mkdir(parents=True)
        (r / "run.json").write_text(json.dumps({"set": str(s)}))
        for it in items:
            for seed in (2000, 2001):
                sf.write(r / "wav" / f"{it['id']}_s{seed}.wav", np.full(3 * 16000, 0.1, np.float32), 16000)
    return str(tmp_path / "a"), str(tmp_path / "b")


def test_build_skips_openings_without_words(tmp_path, monkeypatch):
    a, b = _runs(tmp_path)
    fake = FakeWhisper({"eg-0_s2000": [("اشتركوا", 0.0, 0.6), ("في", 0.6, 0.8), ("القناة", 0.8, 1.2)],
                        "eg-1_s2000": WORDS, "ma-0_s2000": WORDS})
    monkeypatch.setattr(pairs, "words_of", lambda model, path: fake.words(path))
    out = tmp_path / "out"
    got, skipped = pairs.build(a, b, ("main", "PR"), str(out), n=1, model=object(), seed=1)
    assert [p["clip"] for p in got] == ["eg-1_s2000", "ma-0_s2000"]  # eg-0 opens with filler, never chosen
    assert {s["clip"] for s in skipped} <= {"eg-0_s2000"}
    for p in got:
        assert p["text"] == "والعكس أيضا" and p["a"]["asr"] == "والعكس أيضا"
        for side in "ab":
            info = sf.info(out / "data" / f"{p['clip']}__{side}.wav")
            assert info.duration == pytest.approx(1.04 + pairs.GAP_S, abs=1e-3)
    rows = list(csv.DictReader(open(out / "data" / "metadata.csv", encoding="utf-8")))
    assert len(rows) == 2 * len(got) and {r["version"] for r in rows} == {"main", "PR"}
    assert not os.path.exists(out / "data" / "eg-0_s2000__a.wav")
    md = pairs.card(got, skipped, ("main", "PR"), "t", "n", gated_from="VoxCPM-Kernel/Arabic-Hindi-Reference-Voices")
    assert "data/ma-0_s2000__b.wav" in md and "Gated" in md and "| dialect | words |" in md
