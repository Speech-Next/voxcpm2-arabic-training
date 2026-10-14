"""The v2 builder's exclusion logic on tiny synthetic manifests: new texts only from clips no training run could use."""

import argparse
import json
import random

from arabic_voxcpm.eval import build
from arabic_voxcpm.eval.metrics import normalize_ar

WORDS = ["القمر", "البحر", "الجبل", "النهر", "الشمس", "المطر", "الريح", "الطريق", "البيت", "السوق", "المدرسة", "الحديقة",
         "الصباح", "المساء", "الشتاء", "الصيف", "الربيع", "الخريف", "المدينة", "القرية"]


def text(k: int) -> str:
    """A distinct Arabic sentence per k (20-160 characters, >= 3 words, no digits)."""
    return f"ذهبنا إلى {WORDS[k % 20]} مع {WORDS[(k // 20) % 20]} في يوم جميل"


def clip(uid, k, speaker, dialect, split=None, dur=5.0):
    r = {"uid": uid, "text": text(k), "speaker": speaker, "dialect": dialect, "duration_s": dur, "frames": int(dur * 25)}
    return r if split is None else r | {"split": split}


def fixture():
    lah, sada, k = [], [], 0
    for i in range(8):  # held-out speaker eg_h: 8 clips; one shares its text with a training speaker's clip
        lah.append(clip(f"egh{i}", k, "eg_h", "eg"))
        k += 1
    lah.append(clip("egt0", 3, "eg_t", "eg"))  # training speaker, the text of egh3: egh3 counts as seen
    for i in range(6):
        lah.append(clip(f"iqt{i}", k, "iq_t", "iq"))
        k += 1
    for i in range(6):
        sada.append(clip(f"nt{i}", k, "najdi_a", "najdi", "test"))
        sada.append(clip(f"nr{i}", k + 100, "najdi_b", "najdi", "train"))
        k += 1
    for i in range(4):
        sada.append(clip(f"st{i}", k, f"sa_{i}", "sa_unknown", "test"))
        k += 1
    base = [{"group": "eg", "text": lah[0]["text"], "gt_uid": "egh0"}, {"group": "iq", "text": lah[9]["text"], "gt_uid": "iqt0"},
            {"group": "najdi", "text": sada[0]["text"], "gt_uid": "nt0"},
            {"group": "sa", "text": text(999), "gt_uid": "sat0"}]
    held = {"speakers": ["eg_h", "najdi_a"], "uids": [b["gt_uid"] for b in base],
            "texts": [normalize_ar(b["text"]) for b in base]}
    return base, held, lah, sada


def has_audio(corpus, uid, path=False):
    ok = uid != "egh7"  # no recording kept for this one
    return (f"/audio/{corpus}/{uid}.flac" if ok else None) if path else ok


def run_extend(per_group=4, sada_saudi=True, synth=()):
    base, held, lah, sada = fixture()
    h = build.heldout_sets(held)
    seen = build.seen_texts(lah + sada, h, synth)
    return build.extend(base, h, lah, sada, seen, per_group, random.Random(0), has_audio, sada_saudi), lah, sada


def test_seen_texts():
    base, held, lah, sada = fixture()
    h = build.heldout_sets(held)
    assert not build.trainable(lah[1], h) and build.trainable(lah[8], h)  # held-out speaker / training speaker
    assert not build.trainable(sada[0], h) and build.trainable(sada[1], h)  # test / train split
    seen = build.seen_texts(lah + sada, h, ["نص من الكلام المصنوع"])
    assert normalize_ar(text(3)) in seen and normalize_ar(lah[1]["text"]) not in seen
    assert normalize_ar("نص من الكلام المصنوع") in seen


def test_extend_only_unseen():
    (new, stats), lah, sada = run_extend()
    by = {}
    for it in new:
        by.setdefault(it["group"], []).append(it)
    uid = {r["uid"]: r for r in lah + sada}
    assert stats["eg"] == {"base": 1, "pool": 5, "new": 3, "why": "held-out speakers"}
    assert {uid[it["gt_uid"]]["speaker"] for it in by["eg"]} == {"eg_h"}
    assert not {"egh0", "egh3", "egh7"} & {it["gt_uid"] for it in by["eg"]}  # v1's, seen in training, no recording
    assert "iq" not in by and stats["iq"]["new"] == 0 and "training speakers" in stats["iq"]["why"]
    assert all(uid[it["gt_uid"]].get("split") == "test" for it in by["najdi"] + by["sa"])
    assert {it["source"] for it in by["sa"]} == {"sada"} and len(by["sa"]) == 3
    assert all(it["gt_audio"].startswith("/audio/") and it["gt_duration_s"] == 5.0 for it in new)
    texts = [normalize_ar(it["text"]) for it in new]
    assert len(set(texts)) == len(texts)


def test_extend_options():
    (new, stats), _, _ = run_extend(sada_saudi=False)
    assert "sa" not in {it["group"] for it in new} and stats["sa"]["new"] == 0
    (new, _), lah, _ = run_extend(synth=[lah_text for lah_text in [text(1), text(2)]])  # synthetic texts count as seen
    assert not {"egh1", "egh2"} & {it["gt_uid"] for it in new}
    (new, stats), _, _ = run_extend(per_group=1)
    assert new == [] and all(s["new"] == 0 for s in stats.values())


def test_main_extend_writes_superset(tmp_path, monkeypatch):
    base, held, lah, sada = fixture()
    v1 = tmp_path / "v1"
    (v1 / "voices").mkdir(parents=True)
    (v1 / "voices" / "eg_0.flac").write_bytes(b"")
    voices = [{"id": "eg_0", "group": "eg", "speaker": "eg_h", "uid": "egh9", "text": "x", "audio": "voices/eg_0.flac"}]
    build.assign(base, voices)
    (v1 / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in base), encoding="utf-8")
    (v1 / "voices.json").write_text(json.dumps(voices))
    (v1 / "heldout.json").write_text(json.dumps(held, ensure_ascii=False), encoding="utf-8")
    (v1 / "dialect_map.json").write_text("{}")
    rows = {"lahgtna": lah, "sada": sada, "synth": []}
    monkeypatch.setattr(build.manifest, "rows", lambda corpus: rows[corpus])
    monkeypatch.setattr(build, "audio_index", lambda: has_audio)
    out = tmp_path / "v2"
    args = argparse.Namespace(extend=str(v1), out=str(out), per_group=4, seed=0, dry_run=True, sada_saudi=True)
    build.main_extend(args)
    assert not out.exists()  # dry run: counts only
    build.main_extend(argparse.Namespace(**vars(args) | {"dry_run": False}))
    items = [json.loads(line) for line in open(out / "items.jsonl", encoding="utf-8")]
    assert [{k: v for k, v in it.items() if k != "gt_duration_s"} for it in items[:4]] == base  # v1 kept as it was
    assert [it["id"] for it in items[4:]][:1] == [f"{items[4]['group']}-0004"] and len({it["id"] for it in items}) == len(items)
    assert all(it["voice"] == "eg_0" for it in items)  # the one voice: own group or the pool
    h2 = json.load(open(out / "heldout.json", encoding="utf-8"))
    for k in ("speakers", "uids", "texts"):
        assert set(held[k]) <= set(h2[k])
    assert {it["gt_uid"] for it in items} <= set(h2["uids"]) and {"sa_0", "sa_1"} & set(h2["speakers"])
    assert (out / "voices" / "eg_0.flac").exists() and (out / "dialect_map.json").exists()
    assert json.load(open(out / "build.json"))["groups"]["iq"]["new"] == 0
