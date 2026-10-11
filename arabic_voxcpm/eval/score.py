"""Score a generated evaluation run (and, once per set, the ground-truth recordings).

    python -m arabic_voxcpm.eval.score /workspace/data/arabic/eval/runs/<run> [...]

Per utterance: Whisper large-v3 and Cohere Transcribe Arabic transcripts, CER / WER of each against the item text
(Arabic-normalized; cer / wer = Whisper, cer_c / wer_c = Cohere), babble, UTMOS, DNSMOS, WavLM-SV similarity to the
cloned voice, dialect region (ADI-5: adi_label, adi_ok, adi_p), duration. Results are cached in <run>/scores.jsonl,
so a rerun only scores new files and fills metrics that older rows lack (--device for those: cpu next to training). --gt scores the set's ground-truth recordings into <set>/gt_scores.jsonl: the ASR's own error on
real speech of each dialect, the floor a model is compared against.
"""

from __future__ import annotations

import argparse
import json
import os

from arabic_voxcpm.eval import metrics as M


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
        hyp = self.asr(path)
        hyp2 = self.asr2(audio)
        ref, h1, h2 = text, hyp, hyp2
        if numbers:  # numbers compared as words on both sides (numbers.py)
            ref, h1, h2 = (M.spell_numbers(t, group) for t in (text, hyp, hyp2))
        second = {f"{k}_c": v for k, v in M.error_rates(ref, h2).items()}
        row = {"hyp": hyp, "hyp_c": hyp2, **M.error_rates(ref, h1), **second, "utmos": self.utmos(audio),
               **self.dns(audio), **self.adi(audio, group), "duration_s": len(audio) / M.SR}
        if voice_path:
            row["voice_sim"] = float(self.emb(audio) @ self.voice_emb(voice_path))
        return row


def fill_missing(out: str, audio_path, adi) -> int:
    """Rows of ``out`` scored before the dialect-ID metric existed get it (the file rewritten in place)."""
    if not os.path.exists(out):
        return 0
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    todo = [r for r in rows if "adi_label" not in r]
    for r in todo:
        r.update(adi(M.load16k(audio_path(r)), r["group"]))
    if todo:
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        os.replace(out + ".tmp", out)
    return len(todo)


def gt_audio(set_dir: str):
    paths = {it["id"]: it["gt_audio"] for it in map(json.loads, open(os.path.join(set_dir, "items.jsonl"), encoding="utf-8"))}
    return lambda r: paths[r["id"]]


def score_run(run: str, scorer: Scorer) -> None:
    cfg = json.load(open(os.path.join(run, "run.json")))
    items = {it["id"]: it for it in map(json.loads, open(os.path.join(cfg["set"], "items.jsonl"), encoding="utf-8"))}
    voices = {v["id"]: os.path.join(cfg["set"], v["audio"]) for v in json.load(open(os.path.join(cfg["set"], "voices.json")))}
    out = os.path.join(run, "scores.jsonl")
    done = {r["file"] for r in map(json.loads, open(out, encoding="utf-8"))} if os.path.exists(out) else set()
    with open(out, "a", encoding="utf-8") as f:
        for name in sorted(os.listdir(os.path.join(run, "wav"))):
            if name in done:
                continue
            iid, seed = name[:-4].rsplit("_s", 1)
            it = items[iid]
            voice = voices[it["voice"]] if cfg.get("mode", "ref") != "none" else None
            row = {"file": name, "id": iid, "seed": int(seed), "group": it["group"]}
            row.update(scorer(os.path.join(run, "wav", name), it["text"], voice, it["group"], it.get("numbers", False)))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fill_missing(out, lambda r: os.path.join(run, "wav", r["file"]), scorer.adi)
    print(f"scored {run}")


def score_gt(set_dir: str, scorer: Scorer) -> None:
    out = os.path.join(set_dir, "gt_scores.jsonl")
    done = {r["id"] for r in map(json.loads, open(out, encoding="utf-8"))} if os.path.exists(out) else set()
    with open(out, "a", encoding="utf-8") as f:
        for it in map(json.loads, open(os.path.join(set_dir, "items.jsonl"), encoding="utf-8")):
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
        adi = M.DialectID(args.device)
        if args.gt:
            print(args.gt, fill_missing(os.path.join(args.gt, "gt_scores.jsonl"), gt_audio(args.gt), adi), "rows filled")
        for r in args.runs:
            print(r, fill_missing(os.path.join(r, "scores.jsonl"), lambda x, r=r: os.path.join(r, "wav", x["file"]), adi),
                  "rows filled")
        raise SystemExit
    s = Scorer()
    if args.gt:
        score_gt(args.gt, s)
    for r in args.runs:
        score_run(r, s)
