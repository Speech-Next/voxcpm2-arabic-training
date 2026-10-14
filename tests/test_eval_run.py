"""Modes, stop checks, the empty-audio fix and the one-command run (CPU, small fixtures)."""

import json
import os
import random
import wave

import numpy as np
import pytest

from arabic_voxcpm.eval import common, generate, report, run
from arabic_voxcpm.eval import metrics as M
from arabic_voxcpm.eval import score as S


def _wav(path, n):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\1\0" * n)


def test_adapter_of():
    it = {"id": "x-0", "group": "eg", "adapter": "egyptian"}
    assert generate.adapter_of(it, None, ["egyptian"]) == "egyptian"
    assert generate.adapter_of(dict(it, adapter=None), None, ["egyptian"]) is None  # no adapter on purpose
    with pytest.raises(ValueError):  # was a silent base-model run
        generate.adapter_of(dict(it, adapter="qatari"), None, ["egyptian"])
    assert generate.adapter_of(it, {"eg": "none"}, []) is None
    assert generate.adapter_of(it, {"eg": None}, []) is None
    with pytest.raises(ValueError):
        generate.adapter_of(it, {"ps": "palestinian"}, ["palestinian"])  # group missing from the map
    with pytest.raises(ValueError):
        generate.adapter_of(it, {"eg": "egyptian"}, ["gulf"])


def test_adapter_map_and_voices(tmp_path):
    (tmp_path / "dialect_map.json").write_text("{}")
    assert generate.adapter_map("auto", str(tmp_path), tagged=True) == os.path.realpath(tmp_path / "dialect_map.json")
    assert generate.adapter_map("auto", str(tmp_path), tagged=False) is None
    assert generate.adapter_map("items", str(tmp_path), tagged=True) is None
    assert generate.adapter_map("/m.json", str(tmp_path), tagged=False) == "/m.json"
    items = [{"voice": "v0"}, {"voice": "v1"}]
    voices = {"v0": {"text": "نص"}, "v1": {"text": ""}}
    generate.check_voices(items, voices, "ref")
    generate.check_voices(items, {}, "none")
    with pytest.raises(SystemExit):
        generate.check_voices(items, voices, "prompt_ref")  # v1 has no transcript
    with pytest.raises(SystemExit):
        generate.check_voices(items, {"v0": {"text": "x"}}, "ref")


def test_wav_names_and_claim(tmp_path):
    assert common.parse_wav(common.wav_name("bh-0001", 2000)) == ("bh-0001", 2000)
    for name in ("bh-0001_s2000.wav.tmp", "bh-0001_s2000.wav.tmp.wav", "run.json", "x.wav"):
        assert common.parse_wav(name) is None
    out = str(tmp_path / "r")
    cfg = {"set": "/s", "model": "m", "mode": "ref", "dit": "none", "seeds": [1]}
    common.claim(out, cfg)
    common.claim(out, cfg | {"seeds": [1, 2]})  # more seeds: the same run
    with pytest.raises(SystemExit):
        common.claim(out, cfg | {"mode": "none"})  # an interrupted ref run is not resumed as another mode
    common.finish(out, cfg)
    assert not os.path.exists(os.path.join(out, "run.partial.json"))
    with pytest.raises(SystemExit):
        common.claim(out, cfg | {"dit": "student.pt"})
    assert common.run_config(out)["mode"] == "ref"
    json.dump({"set": "/s"}, open(os.path.join(out, "run.json"), "w"))
    assert common.run_config(out)["mode"] == "ref"  # older runs: ref


def test_write_wav_atomic(tmp_path):
    pytest.importorskip("soundfile")
    p = str(tmp_path / "a_s1.wav")
    common.write_wav(p, np.zeros(480, dtype=np.float32), 48000)
    assert os.listdir(tmp_path) == ["a_s1.wav"]


def test_timing():
    t = S.timing(10.0, "ا" * 110, 11.0, cap_s=64.0)  # 110 characters at 11/s: 10 s expected
    assert t["dur_ratio"] == pytest.approx(1.0) and not (t["late_stop"] or t["runaway"] or t["early_stop"])
    t = S.timing(17.0, "ا" * 110, 11.0, cap_s=64.0)
    assert t["late_stop"] and not t["runaway"]
    t = S.timing(25.0, "ا" * 110, 11.0, cap_s=None)
    assert t["runaway"] and not t["hit_cap"]
    t = S.timing(64.0, "ا" * 1000, 11.0, cap_s=400 * common.PATCH_S)  # ran into max_patches: a runaway whatever the ratio
    assert t["hit_cap"] and t["runaway"] and t["late_stop"] and not t["early_stop"]
    assert S.timing(0.0, "ا" * 110, 11.0, None)["early_stop"]


def test_speaking_rates(tmp_path):
    items = [{"id": "a", "group": "eg", "text": "ا" * 100, "gt_duration_s": 10.0},
             {"id": "b", "group": "eg", "text": "ا" * 120, "gt_duration_s": 10.0},
             {"id": "c", "group": "iq", "text": "ا" * 50},
             {"id": "d", "group": "ma", "text": "ا" * 60}]
    (tmp_path / "gt_scores.jsonl").write_text(json.dumps({"id": "d", "duration_s": 5.0}) + "\n")
    r = S.speaking_rates(str(tmp_path), items)
    assert r["eg"] == pytest.approx(11.0) and r["ma"] == pytest.approx(12.0) and "iq" not in r
    assert r["*"] == pytest.approx(12.0)  # the median of 10, 12 and 12
    assert S.speaking_rates(str(tmp_path / "none"), [items[2]])["*"] == S.DEFAULT_CPS


def test_repeat_to_and_empty_audio():
    with pytest.raises(ValueError):
        M.repeat_to(np.zeros(0, dtype=np.float32), 100)  # was an endless loop
    rng = np.random.default_rng(0)
    for n in (1, 7, 100, 999):
        a = rng.standard_normal(n).astype(np.float32)
        old = a.copy()
        while len(old) < 1000:
            old = np.append(old, old)
        assert np.array_equal(M.repeat_to(a, 1000), old)  # the doubling the scored rows were made with


class FakeScorer(S.Scorer):
    """The scoring logic without the models; records the voice each file was compared with."""

    def __init__(self):
        self.calls, self._voice = [], {}
        self.asr = lambda path: "نص"
        self.asr2 = lambda audio: "نص"
        self.utmos = lambda audio: 3.0
        self.dns = lambda audio: {"dnsmos_ovrl": 3.0}
        self.emb = lambda audio: np.ones(2) / np.sqrt(2)
        self.adi = lambda audio, group: {"adi_label": "Gulf"}

    def __call__(self, path, text, voice_path, group, numbers=False):
        self.calls.append((os.path.basename(path), voice_path))
        return super().__call__(path, text, voice_path, group, numbers)


def _set_and_run(tmp_path, mode):
    s = tmp_path / "set"
    (s / "voices").mkdir(parents=True)
    _wav(s / "voices" / "v0.wav", 16000)
    items = [{"id": f"eg-{k}", "group": "eg", "text": "نص " * 10, "voice": "v0", "gt_audio": None,
              "gt_duration_s": 3.0} for k in range(2)]
    (s / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    (s / "voices.json").write_text(json.dumps([{"id": "v0", "audio": "voices/v0.wav", "text": "نص"}]))
    r = tmp_path / f"run-{mode}"
    (r / "wav").mkdir(parents=True)
    json.dump({"set": str(s), "mode": mode, "max_patches": 400}, open(r / "run.json", "w"))
    _wav(r / "wav" / "eg-0_s1.wav", 16000 * 3)
    _wav(r / "wav" / "eg-1_s1.wav", 0)  # an empty generation
    (r / "wav" / "eg-1_s2.wav.tmp").write_bytes(b"")
    return str(s), str(r)


@pytest.mark.parametrize("mode", ["ref", "none"])
def test_score_run_modes(tmp_path, mode):
    pytest.importorskip("librosa")
    pytest.importorskip("jiwer")
    _, r = _set_and_run(tmp_path, mode)
    sc = FakeScorer()
    S.score_run(r, sc)
    assert [c[0] for c in sc.calls] == ["eg-0_s1.wav", "eg-1_s1.wav"]  # the .tmp file skipped
    assert all((v is None) == (mode == "none") for _, v in sc.calls)
    rows = {x["file"]: x for x in map(json.loads, open(os.path.join(r, "scores.jsonl"), encoding="utf-8"))}
    full, empty = rows["eg-0_s1.wav"], rows["eg-1_s1.wav"]
    assert ("voice_sim" in full) == (mode == "ref") and "dur_ratio" in full and "runaway" in full
    assert empty["empty"] and empty["cer"] == 1.0 and empty["early_stop"] and "utmos" not in empty
    md, data = report.build([[r]], [r], absolute=True)
    assert f"mode {mode}" in md
    assert ("| n/a |" in md) == (mode == "none")


def test_fill_missing_timing_only(tmp_path):
    s, r = _set_and_run(tmp_path, "ref")
    rows = [{"file": "eg-0_s1.wav", "id": "eg-0", "seed": 1, "group": "eg", "duration_s": 3.0, "adi_label": "Gulf"}]
    open(os.path.join(r, "scores.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in rows))
    cfg, items, _ = S.run_parts(r)
    out = os.path.join(r, "scores.jsonl")
    assert S.fill_missing(out, None, None, S.run_timing(cfg, items)) == 1
    row = json.loads(open(out).read())
    assert row["dur_ratio"] == pytest.approx(1.0) and row["hit_cap"] is False
    assert S.fill_missing(out, None, None, S.run_timing(cfg, items)) == 0


def _scored(tmp_path, name, mode, set_dir="/s", shift=0.0):
    r = tmp_path / name
    r.mkdir()
    json.dump({"set": set_dir, "mode": mode}, open(r / "run.json", "w"))
    rng = random.Random(0)
    with open(r / "scores.jsonl", "w") as f:
        for k in range(40):
            row = {"id": f"eg-{k}", "seed": 1, "group": "eg", "cer": rng.uniform(0, .2) + shift, "cer_c": .1,
                   "dur_ratio": 1.0, "late_stop": False, "runaway": False}
            if mode != "none":
                row["voice_sim"] = .9
            f.write(json.dumps(row) + "\n")
    return str(r)


def test_report_keeps_modes_apart(tmp_path):
    base_ref = _scored(tmp_path, "base", "ref")
    base_none = _scored(tmp_path, "base-none", "none")
    cand = _scored(tmp_path, "cand-none", "none", shift=0.05)
    path, note = report.pick_base([base_ref, base_none], cand)
    assert path == base_none and note == ""
    path, note = report.pick_base([base_ref], cand)
    assert path == base_ref and "mode ref" in note
    md, data = report.build([[base_ref, base_none]], [cand])
    assert "`base-none` (mode none)" in md and "speaker sim" in md and "| n/a |" in md
    cmp = data["runs"][0]["comparisons"][0]["compare"]
    assert cmp["ALL"]["cer"]["diff"] == pytest.approx(5.0) and cmp["ALL"]["cer"]["ci"][0] > 0
    assert "voice_sim" not in cmp["ALL"]
    line = next(x for x in report.fmt(cmp, "t", "none").splitlines() if x.startswith("| ALL"))
    assert "✗" in line.split("|")[3] and "✓" not in line and "✗" not in line.split("|")[-4]  # no mark on the ratio
    json.dumps(data)


def test_run_commands(tmp_path, monkeypatch):
    a = run.parser().parse_known_args(["--name", "x", "--model", "m", "--dit", "s.pt", "--mode", "prompt_ref",
                                        "--set", "v2", "--head-tokens", "8"])
    args, extra = a
    assert extra == ["--head-tokens", "8"]
    cmd, env = run.generate_cmd(args, "/sets/v2", "/runs/x", extra)
    assert cmd[1:3] == ["-m", "arabic_voxcpm.eval.generate"] and cmd[-2:] == ["--head-tokens", "8"]
    for flag, v in (("--dit", "s.pt"), ("--mode", "prompt_ref"), ("--set", "/sets/v2"), ("--precision", "bf16")):
        assert cmd[cmd.index(flag) + 1] == v
    assert env["PYTHONPATH"].split(os.pathsep)[0] == run.REPO
    args.engine = "dq"
    cmd, _ = run.generate_cmd(args, "/sets/v2", "/runs/x", [])
    assert cmd[2] == "arabic_voxcpm.eval.generate_dq" and "--dit" not in cmd
    monkeypatch.setattr(common, "RUNS_ROOT", str(tmp_path))
    for n in ("base", "base-v2", "base-none"):
        _scored(tmp_path, n, "ref")
    assert [os.path.basename(p) for p in run.baseline_alternatives("base", "/x/v2", "none")] == \
        ["base-none", "base-v2", "base"]


def test_run_refuses_other_config(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    json.dump({"set": "/s", "model": "m", "mode": "ref", "dit": "none", "seeds": [1, 2]}, open(r / "run.json", "w"))
    run.check_existing(str(r), {"set": "/s", "model": "m", "mode": "ref", "dit": "none", "seeds": [1]})
    with pytest.raises(SystemExit):
        run.check_existing(str(r), {"set": "/s", "model": "m", "mode": "none", "seeds": [1]})
    with pytest.raises(SystemExit):
        run.check_existing(str(r), {"set": "/s", "model": "m", "mode": "ref", "seeds": [3]})


class FakeTTS:
    """DialectTTS's surface: records voices and requests, returns 0.1 s of audio (none for the text "فارغ")."""

    log = []

    def __init__(self, tagged):
        self.tags = {} if tagged else None
        self.dialects = ["egyptian", "gulf"]
        self.sample_rate = 16000
        self.voices = {}

    @classmethod
    async def load(cls, model, dit, **kw):
        cls.log.clear()
        return cls(tagged=model == "tagged")

    async def add_voice(self, name, audio, text=None):
        self.voices[name] = text

    async def synthesize(self, text, dialect=None, voice=None, **kw):
        self.log.append((text, dialect, voice, None if voice is None else self.voices[voice], kw["seed"]))
        return np.zeros(0 if text == "فارغ" else 1600, dtype=np.float32)

    async def close(self):
        pass


@pytest.mark.parametrize("mode", ["ref", "prompt_ref", "none"])
def test_generate_modes(tmp_path, monkeypatch, mode):
    pytest.importorskip("soundfile")
    import argparse
    import asyncio
    import sys
    import types

    monkeypatch.setitem(sys.modules, "nanovllm_voxcpm", types.SimpleNamespace(DialectTTS=FakeTTS))
    s = tmp_path / "set"
    s.mkdir()
    items = [{"id": "eg-0", "group": "eg", "text": "نص", "voice": "v0", "adapter": "egyptian"},
             {"id": "ma-1", "group": "ma", "text": "فارغ", "voice": "v0", "adapter": None}]
    (s / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    (s / "voices.json").write_text(json.dumps([{"id": "v0", "audio": "v0.flac", "text": "نص المرجع"}], ensure_ascii=False))
    (s / "dialect_map.json").write_text(json.dumps({"eg": "egyptian", "ma": None}))
    out = tmp_path / "run"
    args = argparse.Namespace(set=str(s), out=str(out), model="tagged", dit="none", dit_cache=None, head_tokens=0,
                              precision="bf16", mode=mode, seeds=[1, 2], adapter_map="auto", text_prefix="",
                              concurrency=4, max_patches=400, normalize=True)
    asyncio.run(generate.run(args))
    got = sorted(FakeTTS.log, key=lambda x: (x[0], x[4]))
    want_voice = None if mode == "none" else "v0"
    want_text = "نص المرجع" if mode == "prompt_ref" else None
    assert got == [(t, d, want_voice, want_text, seed) for t, d in (("فارغ", None), ("نص", "egyptian")) for seed in (1, 2)]
    assert sorted(os.listdir(out / "wav")) == ["eg-0_s1.wav", "eg-0_s2.wav", "ma-1_s1.wav", "ma-1_s2.wav"]
    cfg = common.run_config(str(out))
    assert cfg["mode"] == mode and cfg["engine"] == "library" and cfg["adapter_map"].endswith("dialect_map.json")
    assert not (out / "run.partial.json").exists()
    with pytest.raises(SystemExit):  # the same directory, another mode
        asyncio.run(generate.run(argparse.Namespace(**vars(args) | {"mode": "none" if mode != "none" else "ref"})))
    (s / "dialect_map.json").write_text(json.dumps({"eg": "lebanese", "ma": None}))
    with pytest.raises(ValueError):  # an adapter the model lacks: an error before anything is generated
        asyncio.run(generate.run(argparse.Namespace(**vars(args) | {"out": str(tmp_path / "run2")})))
    assert not (tmp_path / "run2").exists()


FAKE_DQ = '''
import dataclasses
import numpy as np

LOG = []


@dataclasses.dataclass
class Request:
    seq_id: str
    text: str
    prompt_latents: object
    prompt_text: str
    out_q: object
    max_generate_length: int = 250
    cfg_value: float = 2.0
    seed: object = None
    lora_name: object = None
    ref_latents: object = None
    loop: object = None


class DqCoreEngine:
    def __init__(self, model_path, adapters=None, out_sr=16000, **kw):
        self.adapters, self.out_sr = adapters, out_sr

    def start(self):
        pass

    def stop(self):
        pass

    def encode_latents(self, data, fmt):
        return b"lat:" + fmt.encode()

    def submit(self, req):
        LOG.append(req)
        req.loop.call_soon(req.out_q.put_nowait, ("audio", np.zeros(800, dtype=np.float32)))
        req.loop.call_soon(req.out_q.put_nowait, ("end", None))
'''


@pytest.mark.parametrize("mode", ["ref", "prompt_ref", "none"])
def test_generate_dq_modes(tmp_path, monkeypatch, mode):
    pytest.importorskip("soundfile")
    import argparse
    import asyncio
    import sys

    from arabic_voxcpm.eval import generate_dq

    pkg = tmp_path / "dq" / "dq_engine"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "engine.py").write_text(FAKE_DQ)
    monkeypatch.delitem(sys.modules, "dq_engine", raising=False)
    monkeypatch.delitem(sys.modules, "dq_engine.engine", raising=False)
    model = tmp_path / "model"
    (model / "adapters" / "egyptian").mkdir(parents=True)
    s = tmp_path / "set"
    s.mkdir()
    items = [{"id": "eg-0", "group": "eg", "text": "نص", "voice": "v0", "adapter": "egyptian"},
             {"id": "ma-1", "group": "ma", "text": "نص", "voice": "v0", "adapter": None}]
    (s / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    (s / "v0.flac").write_bytes(b"x")
    (s / "voices.json").write_text(json.dumps([{"id": "v0", "audio": "v0.flac", "text": "نص المرجع"}], ensure_ascii=False))
    args = argparse.Namespace(set=str(s), out=str(tmp_path / "run"), model=str(model), lora_dir=None,
                              dq_engine_dir=str(tmp_path / "dq"), mode=mode, seeds=[1], seeded=False, timesteps=6,
                              cfg=2.0, out_sr=48000, concurrency=4, gpu_mem=0.5, max_patches=400)
    asyncio.run(generate_dq.run(args))
    reqs = sorted(sys.modules["dq_engine.engine"].LOG, key=lambda r: r.seq_id)
    assert [r.lora_name for r in reqs] == ["egyptian", None] and all(r.seed is None for r in reqs)
    for r in reqs:
        assert (r.ref_latents is None) == (mode == "none")
        assert (r.prompt_latents is not None, r.prompt_text) == ((True, "نص المرجع") if mode == "prompt_ref" else (False, ""))
    cfg = common.run_config(args.out)
    assert cfg["engine"] == "dq_engine" and cfg["mode"] == mode and cfg["sample_rate"] == 48000
    assert sorted(os.listdir(os.path.join(args.out, "wav"))) == ["eg-0_s1.wav", "ma-1_s1.wav"]
