"""Shared by the corpus pipelines: encode a source shard's clips and write its latent shard and manifest."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import soundfile as sf

from arabic_voxcpm import vae as avae
from arabic_voxcpm.data.store import ShardWriter

BATCH_SAMPLES = 1_500_000  # 16 kHz samples per encoder batch (~94 s of audio, ~3 GB of activations)


def out_dirs(out: str) -> None:
    for d in ("latents", "manifest", "eval_audio", "raw"):
        os.makedirs(os.path.join(out, d), exist_ok=True)


def done(out: str, name: str) -> bool:
    return os.path.exists(os.path.join(out, "manifest", name + ".jsonl"))


def write_shard(out: str, name: str, vae, rows: list[dict], waves: list[np.ndarray], keep: set[str]) -> dict:
    """rows[i] (manifest fields, "uid" required) and waves[i] (16 kHz mono) -> latents/<name>.pt and
    manifest/<name>.jsonl (rows + "frames"); clips whose uid is in ``keep`` are also saved to eval_audio/."""
    writer = ShardWriter(os.path.join(out, "latents", name + ".pt"))
    order = sorted(range(len(rows)), key=lambda i: len(waves[i]))  # similar lengths per batch: less padding
    batch: list[int] = []

    def flush():
        for i, z in zip(batch, avae.encode(vae, [waves[i] for i in batch])):
            writer.add(rows[i]["uid"], z)
            rows[i]["frames"] = len(z)
        batch.clear()

    for i in order:
        if batch and (len(batch) + 1) * len(waves[i]) > BATCH_SAMPLES:
            flush()
        batch.append(i)
    if batch:
        flush()
    for r, w in zip(rows, waves):
        if r["uid"] in keep:
            sf.write(os.path.join(out, "eval_audio", r["uid"] + ".flac"), w, avae.SAMPLE_RATE)
    tmp = os.path.join(out, "manifest", name + ".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    writer.close()
    os.replace(tmp, os.path.join(out, "manifest", name + ".jsonl"))  # the manifest marks the shard done
    return {"clips": len(rows), "hours": sum(len(w) for w in waves) / avae.SAMPLE_RATE / 3600}


def download(repo: str, name: str, raw_dir: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    for attempt in range(5):
        try:
            return hf_hub_download(repo, name, repo_type="dataset", local_dir=raw_dir, revision=revision)
        except Exception as e:  # connection resets: retry
            if attempt == 4:
                raise
            print(f"download {name}: {e!r}, retrying", flush=True)
            time.sleep(10 * (attempt + 1))
