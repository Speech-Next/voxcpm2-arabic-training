"""Batched Habibi-TTS generation of a request set (requests.py): F5-TTS samples many requests at once, padded, so the
GPU is busy (one at a time it took ~3 s per sentence).

Runs in Habibi's environment:

    PYTHONPATH=. /workspace/envs/habibi/bin/python -m arabic_voxcpm.synth.habibi_gen --set <set> --out <out>

Items: {"id", "text", "voice", "habibi": "Unified:<dialect id>" | "<specialized dialect id>"}. Output: wav/<id>_s<seed>.wav
at 24 kHz. The duration rule, loudness handling and text layout are Habibi's infer_batch_process, per request.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from arabic_voxcpm.eval.generate_habibi import Habibi

SR = 24000
HOP = 256
TARGET_RMS = 0.1


def prepare(model, voice: dict, set_dir: str, cache: dict):
    """The voice's mel (24 kHz, loudness raised to Habibi's target) and its ref text, once per voice."""
    import torch
    import torchaudio

    if voice["id"] not in cache:
        import soundfile as sf

        a, sr = sf.read(os.path.join(set_dir, voice["audio"]), dtype="float32")
        a = torch.from_numpy(a if a.ndim == 1 else a.mean(1))[None]
        rms = float(a.square().mean().sqrt())
        if rms < TARGET_RMS:
            a = a * TARGET_RMS / rms
        a = torchaudio.functional.resample(a, sr, SR).to("cuda")
        mel = model.mel_spec(a).permute(0, 2, 1)[0]  # [frames, 100]
        text = voice["text"].strip()
        if not text.endswith((".", "؟", "!", "?", "،")):
            text += "."
        cache[voice["id"]] = (mel, text + " ", rms)
    return cache[voice["id"]]


def main(args) -> None:
    import soundfile as sf
    import torch
    from habibi_tts.model.utils import dialect_id_map

    items = [json.loads(l) for l in open(os.path.join(args.set, "items.jsonl"), encoding="utf-8")]
    voices = {v["id"]: v for v in json.load(open(os.path.join(args.set, "voices.json"), encoding="utf-8"))}
    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    todo = [it for it in items if not os.path.exists(os.path.join(args.out, "wav", f"{it['id']}_s{args.seed}.wav"))]
    by = defaultdict(list)
    for it in todo:
        by[it["habibi"].split(":")[0]].append(it)
    tts = Habibi()
    t0, n, sec, batches = time.time(), 0, 0.0, 0
    for name, group in sorted(by.items()):
        model = tts.model(name)
        cache = {}
        jobs = []
        for it in group:
            mel, ref_text, rms = prepare(model, voices[it["voice"]], args.set, cache)
            ref_len = mel.shape[0]
            gen = it["text"].strip()
            frames_s = SR / HOP
            gen_len = ref_len / len(ref_text.encode()) * len(gen.encode())  # Habibi's rule: the reference's rate
            # ... bounded to 7-20 characters per second (a reference transcript missing words ran one to 60 s)
            gen_len = min(max(gen_len, len(gen) / 20 * frames_s), len(gen) / 7 * frames_s)
            dur = ref_len + int(gen_len)
            text = ref_text + gen
            if name == "Unified":
                text = f"{dialect_id_map[it['habibi'].split(':')[1]]}〈{text}〉"
            jobs.append((dur, it, mel, text, ref_len, rms))
        jobs.sort(key=lambda j: j[0])
        b = 0
        while b < len(jobs):
            e = b + 1
            while e < len(jobs) and (e - b + 1) * jobs[e][0] <= args.frames:
                e += 1
            batch = jobs[b:e]
            b = e
            lens = torch.tensor([j[4] for j in batch], device="cuda")
            cond = torch.nn.utils.rnn.pad_sequence([j[2] for j in batch], batch_first=True)
            dur = torch.tensor([j[0] for j in batch], device="cuda")
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                out, _ = model.sample(cond=cond, text=[j[3] for j in batch], duration=dur, lens=lens, steps=args.nfe,
                                      cfg_strength=2.0, sway_sampling_coef=-1.0)
                for (d, it, _, _, ref_len, rms), o in zip(batch, out):
                    mel = o[ref_len:d].float().T[None]  # [1, 100, frames]
                    wave = tts.vocoder.decode(mel)[0].float().cpu().numpy()
                    if rms < TARGET_RMS:
                        wave = wave * rms / TARGET_RMS
                    path = os.path.join(args.out, "wav", f"{it['id']}_s{args.seed}.wav")
                    sf.write(path + ".tmp.wav", wave.astype(np.float32), SR)
                    os.replace(path + ".tmp.wav", path)
                    n += 1
                    sec += len(wave) / SR
            batches += 1
            if batches % 20 == 0:
                el = time.time() - t0
                print(f"{name}: {n}/{len(todo)} done, {sec / 3600:.2f} h audio in {el / 60:.1f} min "
                      f"({sec / el:.1f}x real time)", flush=True)
    print(f"{n} utterances, {sec / 3600:.2f} h in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--frames", type=int, default=24000, help="batch budget: padded mel frames (x2 for CFG)")
    main(ap.parse_args())
