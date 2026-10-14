"""Shared by the corpus pipelines: encode a source shard's clips and write its latent shard and manifest; or, when the
manifests exist already (restored from a backup without the latents), re-encode only chosen clips' latents
(restore_shard, the pipelines' --restore)."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import soundfile as sf

from arabic_voxcpm import vae as avae
from arabic_voxcpm.data.store import Shard, ShardWriter

BATCH_SAMPLES = 1_500_000  # 16 kHz samples per encoder batch (~94 s of audio, ~3 GB of activations)


def out_dirs(out: str) -> None:
    for d in ("latents", "manifest", "eval_audio", "raw"):
        os.makedirs(os.path.join(out, d), exist_ok=True)


def done(out: str, name: str) -> bool:
    return os.path.exists(os.path.join(out, "manifest", name + ".jsonl"))


def _encode_batch(vae, waves: list[np.ndarray], tries: int = 60) -> list:
    """avae.encode, waiting out CUDA out-of-memory errors (another process on the GPU) for up to ``tries`` x 30 s."""
    import torch

    for k in range(tries):
        try:
            return avae.encode(vae, waves)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"CUDA out of memory ({len(waves)} clips; another process on the GPU?), retrying in 30 s", flush=True)
            time.sleep(30)
    return avae.encode(vae, waves)


def _encode(vae, waves: list[np.ndarray], add) -> None:
    """add(i, latents of waves[i]) for every clip, encoded in batches of similar lengths (less padding), shortest
    first."""
    order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
    batch: list[int] = []

    def flush():
        for i, z in zip(batch, _encode_batch(vae, [waves[i] for i in batch])):
            add(i, z)
        batch.clear()

    for i in order:
        if batch and (len(batch) + 1) * len(waves[i]) > BATCH_SAMPLES:
            flush()
        batch.append(i)
    if batch:
        flush()


def write_shard(out: str, name: str, vae, rows: list[dict], waves: list[np.ndarray], keep: set[str]) -> dict:
    """rows[i] (manifest fields, "uid" required) and waves[i] (16 kHz mono) -> latents/<name>.pt and
    manifest/<name>.jsonl (rows + "frames"); clips whose uid is in ``keep`` are also saved to eval_audio/."""
    writer = ShardWriter(os.path.join(out, "latents", name + ".pt"))

    def add(i, z):
        writer.add(rows[i]["uid"], z)
        rows[i]["frames"] = len(z)

    _encode(vae, waves, add)
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


def read_uids(path: str) -> set[str]:
    """The clips to restore: one uid per line, or JSON lines with a "uid" field (a dataset manifest:
    data/release.py)."""
    out = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.add(json.loads(line)["uid"] if line.startswith("{") else line)
    return out


def manifest_rows(out: str, name: str) -> list[dict]:
    path = os.path.join(out, "manifest", name + ".jsonl")
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def latent_uids(out: str, name: str) -> set[str]:
    path = os.path.join(out, "latents", name + ".pt")
    return set(Shard(path).index) if os.path.exists(path) else set()


def to_restore(out: str, name: str, want: set[str]) -> list[dict]:
    """The manifest rows of shard ``name`` that are wanted and have no latents yet."""
    have = latent_uids(out, name)
    return [r for r in manifest_rows(out, name) if r["uid"] in want and r["uid"] not in have]


def restore_shard(out: str, name: str, vae, rows: list[dict], waves: list[np.ndarray]) -> dict:
    """Latents only, for a shard whose manifest exists: rows (its manifest rows, with "frames") and waves[i] (16 kHz
    mono) -> latents/<name>.pt, added to the clips the file already holds (a shard has the latents of the clips
    restored, not necessarily all its manifest's). Every clip must get its manifest's frame count. The manifest and
    eval_audio/ are left as they are."""
    path = os.path.join(out, "latents", name + ".pt")
    writer = ShardWriter(path)
    if os.path.exists(path):
        old = Shard(path)
        assert not {r["uid"] for r in rows} & set(old.index), f"{name}: clips restored twice"
        for u in old.index:
            writer.add(u, old[u])

    def add(i, z):
        assert len(z) == rows[i]["frames"], \
            f"{name} {rows[i]['uid']}: {len(z)} frames, the manifest has {rows[i]['frames']}"
        writer.add(rows[i]["uid"], z)

    _encode(vae, waves, add)
    writer.close()
    return {"clips": len(rows), "hours": sum(len(w) for w in waves) / avae.SAMPLE_RATE / 3600}


def ahead(pool, fn, items: list, depth: int):
    """(item, fn(item)) for each item in order, fn running up to ``depth`` items ahead in ``pool`` (downloads while
    the current shard is encoded)."""
    futs = {}
    for k, x in enumerate(items):
        for j in range(k, min(k + depth + 1, len(items))):
            if j not in futs:
                futs[j] = pool.submit(fn, items[j])
        yield x, futs.pop(k).result()


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
