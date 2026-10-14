"""Score a generated evaluation run (and, once per set, the ground-truth recordings).

    python -m arabic_voxcpm.eval.score /workspace/data/arabic/eval/runs/<run> [...]

Per utterance: Whisper large-v3 and Cohere Transcribe Arabic transcripts, CER / WER of each against the item text
(Arabic-normalized; cer / wer = Whisper, cer_c / wer_c = Cohere), babble, UTMOS, DNSMOS, WavLM-SV similarity to the
cloned voice (not for --mode none runs), dialect region (ADI-5: adi_label, adi_ok, adi_p), duration, and the stop
checks (``timing``): dur_ratio = duration / the text's expected duration at its group's real speaking rate, hit_cap
(the utterance reached max_patches), late_stop (ratio > 1.5 or the cap), runaway (ratio > 2 or the cap), early_stop
(ratio < 0.5). An empty generation (< 0.1 s) counts as a full ASR error ("empty", no audio metrics).

Results are cached in <run>/scores.jsonl, so a rerun only scores new files and fills metrics that older rows lack
(--fill-only: only that; the dialect model loads only if a row lacks it, --device cpu next to training; the stop
checks need no model). --gt scores the set's ground-truth recordings into <set>/gt_scores.jsonl: the ASR's own
error on real speech of each dialect, the floor a model is compared against.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import median

from arabic_voxcpm.eval import common
from arabic_voxcpm.eval import metrics as M

LATE, RUNAWAY, EARLY = 1.5, 2.0, 0.5  # duration ratio thresholds
DEFAULT_CPS = 11.0  # normalized characters per second, the median of v1's ground-truth recordings


class Scorer:
    def __init__(self):
        self.asr = M.WhisperASR()
        self.asr2 = M.CohereASR()
        self.dns = M.DNSMOS()
        self.utmos = M.UTMOS()
        self.emb = M.SpeakerEmbedder()
        self.adi = M.DialectID()
        self._voice = {}

    def voice_emb(self, path):
        if path not in self._voice:
            self._voice[path] = self.emb(M.load16k(path))
        return self._voice[path]

    def __call__(self, path: str, text: str, voice_path: str | None, group: str, numbers: bool = False) -> dict:
        audio = M.load16k(path)
        empty = len(audio) < M.MIN_AUDIO_S * M.SR  # nothing generated: a full error, no audio metrics
        hyp, hyp2 = ("", "") if empty else (self.asr(path), self.asr2(audio))
        ref, h1, h2 = text, hyp, hyp2
        if numbers:  # numbers compared as words on both sides (numbers.py)
            ref, h1, h2 = (M.spell_numbers(t, group) for t in (text, hyp, hyp2))
        second = {f"{k}_c": v for k, v in M.error_rates(ref, h2).items()}
        row = {"hyp": hyp, "hyp_c": hyp2, **M.error_rates(ref, h1), **second, "duration_s": len(audio) / M.SR}
        if empty:
            return row | {"empty": True}
        row.update({"utmos": self.utmos(audio), **self.dns(audio), **self.adi(audio, group)})
        if voice_path:
            row["voice_sim"] = float(self.emb(audio) @ self.voice_emb(voice_path))
        return row


def speaking_rates(set_dir: str, items: list[dict]) -> dict:
    """group -> median normalized characters per second of its real recordings ("*": all groups). Durations: the
    item's gt_duration_s (v2 sets), else the set's scored ground truth; a set without either gets DEFAULT_CPS."""
    gt = {}
    path = os.path.join(set_dir, "gt_scores.jsonl")
    if os.path.exists(path):
        gt = {r["id"]: r["duration_s"] for r in map(json.loads, open(path, encoding="utf-8"))}
    by = {}
    for it in items:
        d = it.get("gt_duration_s") or gt.get(it["id"])
        if d and not it.get("numbers"):
            by.setdefault(it["group"], []).append(len(M.normalize_ar(it["text"])) / d)
    rates = {g: median(v) for g, v in by.items()}
    rates["*"] = median([x for v in by.values() for x in v]) if by else DEFAULT_CPS
    return rates


def timing(duration_s: float, text: str, rate: float, cap_s: float | None) -> dict:
    """Stop checks of one utterance: duration ratio against the text's expected duration at ``rate`` characters/s,
    and whether it ran into the max_patches cap (cap_s: None = no cap known)."""
    exp = max(len(M.normalize_ar(text)), 1) / rate
    r = duration_s / exp
    hit = cap_s is not None and duration_s >= cap_s - common.PATCH_S
    return {"dur_ratio": r, "hit_cap": hit, "late_stop": hit or r > LATE, "runaway": hit or r > RUNAWAY,
            "early_stop": not hit and r < EARLY}


def run_timing(cfg: dict, items: dict):
    """row -> its stop checks, for a run of configuration ``cfg`` (run.json) over ``items`` (id -> item)."""
    rates = speaking_rates(cfg["set"], list(items.values()))
    cap = cfg["max_patches"] * common.PATCH_S if cfg.get("max_patches") else None

    def f(r):
        it = items[r["id"]]
        text = M.spell_numbers(it["text"], it["group"]) if it.get("numbers") else it["text"]
        return timing(r["duration_s"], text, rates.get(it["group"], rates["*"]), cap)

    return f


class Lazy:
    """A model loaded on first use (a fill that finds nothing missing loads nothing)."""

    def __init__(self, make):
        self.make, self.obj = make, None

    def __call__(self, *a):
        if self.obj is None:
            self.obj = self.make()
        return self.obj(*a)


def fill_missing(out: str, audio_path, adi=None, timing_of=None) -> int:
    """Rows of ``out`` scored before a metric existed get it (the file rewritten in place): the dialect region (adi,
    from the audio) and the stop checks (timing_of, from the row alone)."""
    if not os.path.exists(out):
        return 0
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    n = 0
    for r in rows:
        todo = False
        if adi is not None and "adi_label" not in r and not r.get("empty"):
            r.update(adi(M.load16k(audio_path(r)), r["group"]))
            todo = True
        if timing_of is not None and "dur_ratio" not in r:
            r.update(timing_of(r))
            todo = True
        n += todo
    if n:
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        os.replace(out + ".tmp", out)
    return n


def gt_audio(set_dir: str):
    paths = {it["id"]: it["gt_audio"] for it in common.items(set_dir)}
    return lambda r: paths[r["id"]]


def run_parts(run: str):
    """(configuration, items by id, voice clips by id) of a run."""
    cfg = common.run_config(run)
    items = {it["id"]: it for it in common.items(cfg["set"])}
    voices = {v["id"]: os.path.join(cfg["set"], v["audio"]) for v in common.voices(cfg["set"])}
    return cfg, items, voices


def score_run(run: str, scorer: Scorer) -> None:
    cfg, items, voices = run_parts(run)
    timing_of = run_timing(cfg, items)
    out = os.path.join(run, "scores.jsonl")
    done = {r["file"] for r in map(json.loads, open(out, encoding="utf-8"))} if os.path.exists(out) else set()
    with open(out, "a", encoding="utf-8") as f:
        for name in sorted(os.listdir(os.path.join(run, "wav"))):
            key = common.parse_wav(name)
            if key is None or name in done:
                continue
            iid, seed = key
            it = items[iid]
            voice = voices[it["voice"]] if cfg["mode"] != "none" and it.get("voice") else None
            row = {"file": name, "id": iid, "seed": seed, "group": it["group"]}
            row.update(scorer(os.path.join(run, "wav", name), it["text"], voice, it["group"], it.get("numbers", False)))
            row.update(timing_of(row))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fill_missing(out, lambda r: os.path.join(run, "wav", r["file"]), scorer.adi, timing_of)
    print(f"scored {run}")


def score_gt(set_dir: str, scorer: Scorer) -> None:
    out = os.path.join(set_dir, "gt_scores.jsonl")
    done = {r["id"] for r in map(json.loads, open(out, encoding="utf-8"))} if os.path.exists(out) else set()
    with open(out, "a", encoding="utf-8") as f:
        for it in common.items(set_dir):
            if it["gt_audio"] and it["id"] not in done:
                f.write(json.dumps({"id": it["id"], "group": it["group"],
                                    **scorer(it["gt_audio"], it["text"], None, it["group"])}, ensure_ascii=False) + "\n")
    fill_missing(out, gt_audio(set_dir), scorer.adi)
    print(f"scored ground truth of {set_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--gt", help="also score this set's ground-truth recordings")
    ap.add_argument("--fill-only", action="store_true", help="only add the metrics that scored rows lack")
    ap.add_argument("--device", default="cuda", help="with --fill-only")
    args = ap.parse_args()
    if args.fill_only:
        adi = Lazy(lambda: M.DialectID(args.device))
        if args.gt:
            print(args.gt, fill_missing(os.path.join(args.gt, "gt_scores.jsonl"), gt_audio(args.gt), adi), "rows filled")
        for r in args.runs:
            cfg, items, _ = run_parts(r)
            print(r, fill_missing(os.path.join(r, "scores.jsonl"), lambda x, r=r: os.path.join(r, "wav", x["file"]), adi,
                                  run_timing(cfg, items)), "rows filled")
        raise SystemExit
    s = Scorer()
    if args.gt:
        score_gt(args.gt, s)
    for r in args.runs:
        score_run(r, s)
