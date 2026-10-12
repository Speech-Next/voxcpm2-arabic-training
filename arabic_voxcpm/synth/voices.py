"""Reference voices for synthetic speech, with a transcript (F5-TTS / Habibi needs one) and a dialect region.

Two kinds:
- VoxTube clips (data/voxtube.py): other speakers than the corpora's; DNSMOS >= --min-dnsmos.
- The cleanest real clips of each dialect's training speakers (quality.py scores: DNSMOS >= --min-dnsmos-real,
  CER <= --max-cer, 4-10 s), decoded from their latents; at most --per-speaker per speaker.

Every voice gets a Cohere transcript (VoxTube's) or keeps its corpus transcript (real), and an ADI-5 region label.

    python -m arabic_voxcpm.synth.voices --out /workspace/data/arabic/synth
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf

from arabic_voxcpm.data.quality import decode16k
from arabic_voxcpm.eval import metrics as M

VOXTUBE = "/workspace/data/arabic/voxtube"
QUALITY = "/workspace/data/arabic/quality/scores.jsonl"
MODEL = "/workspace/models/VoxCPM2-Arabic-Dialects"


def real_candidates(args) -> list[dict]:
    from arabic_voxcpm.train.dataset import select_rows

    text = {r["uid"]: r["text"] for r in select_rows(heldout=args.heldout)}
    by = defaultdict(lambda: defaultdict(list))
    for line in open(QUALITY, encoding="utf-8"):
        q = json.loads(line)
        if (q["uid"] in text and 4.0 <= q["sec"] <= 10.0 and q["cer"] <= args.max_cer
                and q["dnsmos_ovrl"] >= args.min_dnsmos_real):
            by[q["dialect"]][q["speaker"]].append(q)
    rng = random.Random(args.seed)
    out = []
    for d, spk in sorted(by.items()):
        speakers = sorted(spk)
        rng.shuffle(speakers)
        for s in speakers[: args.speakers_per_dialect]:
            best = sorted(spk[s], key=lambda q: -q["dnsmos_ovrl"])[: args.per_speaker]
            out += [{**q, "text": text[q["uid"]]} for q in best]
    return out


def main(args) -> None:
    import torch

    from arabic_voxcpm import vae as avae
    from arabic_voxcpm.data.quality import BatchedCohere
    from arabic_voxcpm.train.dataset import LatentStore

    os.makedirs(os.path.join(args.out, "voices"), exist_ok=True)
    cohere, adi = BatchedCohere(), M.DialectID()
    rows = []

    vt = [json.loads(l) for l in open(os.path.join(VOXTUBE, "voices.jsonl"))]
    vt = [v for v in vt if v["dnsmos_ovrl"] >= args.min_dnsmos]
    for k in range(0, len(vt), 16):
        part = vt[k : k + 16]
        waves = [M.load16k(os.path.join(VOXTUBE, v["audio"])) for v in part]
        for v, a, t in zip(part, waves, cohere.batch(waves)):
            path = os.path.join(args.out, "voices", v["id"] + ".flac")
            sf.write(path, a, M.SR)
            rows.append({"id": v["id"], "kind": "voxtube", "speaker": v["speaker"], "dialect": None,
                         "audio": os.path.relpath(path, args.out), "text": t, "sec": v["sec"],
                         "dnsmos_ovrl": v["dnsmos_ovrl"], **adi(a, "")})
    print(f"{len(rows)} VoxTube voices", flush=True)

    real = real_candidates(args)
    store, vae = LatentStore(), avae.load_vae(MODEL)
    for k in range(0, len(real), 16):
        part = real[k : k + 16]
        for q, a in zip(part, decode16k(vae, [store[q["uid"]] for q in part])):
            vid = "r_" + q["uid"]
            path = os.path.join(args.out, "voices", vid + ".flac")
            sf.write(path, a, M.SR)
            rows.append({"id": vid, "kind": "real", "speaker": q["speaker"], "dialect": q["dialect"],
                         "audio": os.path.relpath(path, args.out), "text": q["text"], "sec": q["sec"],
                         "dnsmos_ovrl": q["dnsmos_ovrl"], **adi(a, "")})
    torch.cuda.empty_cache()
    with open(os.path.join(args.out, "voices.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for r in rows:
        by[(r["kind"], r["dialect"] or r["adi_label"])] += 1
    print(dict(sorted(by.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/data/arabic/synth")
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v1/heldout.json")
    ap.add_argument("--min-dnsmos", type=float, default=3.0)
    ap.add_argument("--min-dnsmos-real", type=float, default=3.0)
    ap.add_argument("--max-cer", type=float, default=0.05)
    ap.add_argument("--speakers-per-dialect", type=int, default=60)
    ap.add_argument("--per-speaker", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
