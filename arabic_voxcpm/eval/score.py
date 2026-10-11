"""Score a generated evaluation run (and, once per set, the ground-truth recordings).

    python -m arabic_voxcpm.eval.score /workspace/data/arabic/eval/runs/<run> [...]

Per utterance: Whisper large-v3 and Cohere Transcribe Arabic transcripts, CER / WER of each against the item text
(Arabic-normalized; cer / wer = Whisper, cer_c / wer_c = Cohere), babble, UTMOS, DNSMOS, WavLM-SV similarity to the
cloned voice, duration. Results are cached in <run>/scores.jsonl, so a rerun only
scores new files. --gt scores the set's ground-truth recordings into <set>/gt_scores.jsonl: the ASR's own error on
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
        self._voice = {}

    def voice_emb(self, path):
        if path not in self._voice:
            self._voice[path] = self.emb(M.load16k(path))
        return self._voice[path]

    def __call__(self, path: str, text: str, voice_path: str | None) -> dict:
        audio = M.load16k(path)
        hyp = self.asr(path)
        hyp2 = self.asr2(audio)
        second = {f"{k}_c": v for k, v in M.error_rates(text, hyp2).items()}
        row = {"hyp": hyp, "hyp_c": hyp2, **M.error_rates(text, hyp), **second, "utmos": self.utmos(audio),
               **self.dns(audio), "duration_s": len(audio) / M.SR}
        if voice_path:
            row["voice_sim"] = float(self.emb(audio) @ self.voice_emb(voice_path))
        return row


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
            row.update(scorer(os.path.join(run, "wav", name), it["text"], voice))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"scored {run}")


def score_gt(set_dir: str, scorer: Scorer) -> None:
    out = os.path.join(set_dir, "gt_scores.jsonl")
    done = {r["id"] for r in map(json.loads, open(out, encoding="utf-8"))} if os.path.exists(out) else set()
    with open(out, "a", encoding="utf-8") as f:
        for it in map(json.loads, open(os.path.join(set_dir, "items.jsonl"), encoding="utf-8")):
            if it["gt_audio"] and it["id"] not in done:
                f.write(json.dumps({"id": it["id"], "group": it["group"], **scorer(it["gt_audio"], it["text"], None)},
                                   ensure_ascii=False) + "\n")
    print(f"scored ground truth of {set_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--gt", help="also score this set's ground-truth recordings")
    args = ap.parse_args()
    s = Scorer()
    if args.gt:
        score_gt(args.gt, s)
    for r in args.runs:
        score_run(r, s)
