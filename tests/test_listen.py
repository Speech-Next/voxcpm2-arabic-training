import csv
import json
import wave

import pytest

from arabic_voxcpm.eval import listen


def _wav(path, n=160):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * n)


def _runs(tmp_path):
    """A set of 3 groups x 4 items, run a with both seeds of every item, run b without the last item's."""
    s = tmp_path / "set"
    s.mkdir()
    items = [{"id": f"{g}-{k}", "group": g, "text": f"نص تجريبي رقم {k} للهجة {g}", "voice": "v0"}
             for g in ("eg", "najdi", "ma") for k in range(4)]
    (s / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    (s / "voices.json").write_text("[]")
    runs = {}
    for name in ("a", "b"):
        r = tmp_path / name
        (r / "wav").mkdir(parents=True)
        (r / "run.json").write_text(json.dumps({"set": str(s), "mode": "ref"}))
        for it in items:
            for seed in (2000, 2001):
                if name == "b" and it["id"] == "ma-3":
                    continue
                _wav(str(r / "wav" / f"{it['id']}_s{seed}.wav"))
        (r / "wav" / "eg-0_s2000.wav.tmp").write_bytes(b"partial")  # a write in progress: never picked
        runs[name] = str(r)
    return runs, items


def test_export_blind_page(tmp_path):
    runs, items = _runs(tmp_path)
    out = tmp_path / "ab"
    trials = listen.export(runs["a"], runs["b"], str(out), n=3, seed=1)
    assert len(trials) == 9 and {t["group"] for t in trials} == {"eg", "najdi", "ma"}
    assert "ma-3" not in {t["id"] for t in trials}  # only items both runs generated
    key = json.load(open(out / "key.json"))
    assert [t["trial"] for t in key["trials"]] == list(range(1, 10)) and {t["A"] for t in key["trials"]} <= {"a", "b"}
    for t in trials:
        assert (out / "page" / "audio" / f"{t['trial']:04d}A.wav").exists()
        assert (out / "page" / "audio" / f"{t['trial']:04d}B.wav").exists()
    page = (out / "page" / "index.html").read_text(encoding="utf-8")
    assert 'dir="rtl"' in page and page.count("<audio") == 18 and "http://" not in page and "https://" not in page
    assert "<link" not in page and "@import" not in page and "key.json" not in page
    for name in ("a", "b"):  # nothing on the page names a run
        assert runs[name] not in page
    rows = list(csv.DictReader(open(out / "items.csv", encoding="utf-8")))
    assert [r["id"] for r in rows] == [t["id"] for t in trials] and "A" not in rows[0]
    # the same seed draws the same test
    assert listen.export(runs["a"], runs["b"], str(tmp_path / "ab2"), n=3, seed=1) == trials


def test_score_unblinds(tmp_path):
    runs, _ = _runs(tmp_path)
    out = tmp_path / "ab"
    trials = listen.export(runs["a"], runs["b"], str(out), n=4, seed=3)
    ans = tmp_path / "answers.csv"
    with open(ans, "w", encoding="utf-8-sig", newline="") as f:  # the page writes a BOM
        w = csv.writer(f)
        w.writerow(["listener", "trial", "group", "natural", "dialect"])
        for t in trials:
            letter_b = "A" if t["A"] == "b" else "B"  # the listener always hears run b as more natural
            w.writerow(["x", t["trial"], t["group"], letter_b, "same" if t["group"] == "eg" else ""])
    res = listen.score([str(ans)], str(out / "key.json"))
    nat = res["natural"]["ALL"]
    assert nat["b"] == len(trials) and nat["a"] == 0 and nat["win_b"] == 1.0
    lo, hi = nat["ci"]
    assert 0.5 < lo < 1.0 and hi == pytest.approx(1.0)
    dia = res["dialect"]
    assert dia["ALL"]["n"] == dia["eg"]["same"] == 4 and dia["ALL"]["score_b"] == 0.5 and "najdi" not in dia
    text = listen.fmt(res, str(out / "key.json"))
    assert "more natural" in text and "| ALL |" in text
    with open(ans, "a", encoding="utf-8") as f:
        f.write(f"x,{trials[0]['trial']},eg,C,\n")
    with pytest.raises(SystemExit):
        listen.score([str(ans)], str(out / "key.json"))


def test_wilson():
    lo, hi = listen.wilson(5, 10)
    assert lo < 0.5 < hi and lo == pytest.approx(1 - hi)
    assert listen.wilson(0, 0) != listen.wilson(0, 0)  # nan: no decisive answer
