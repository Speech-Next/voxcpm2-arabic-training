"""Rabe3/lahgtna-arabic-tts-24khz -> AudioVAE latent shards + a manifest, one parquet shard at a time.

    python -m arabic_voxcpm.data.lahgtna --model /workspace/models/VoxCPM2-Arabic-Dialects --out /workspace/data/arabic/lahgtna

Each source shard (~0.8 GB, ~1500 clips) is downloaded, decoded, resampled 24 -> 16 kHz (the corpus was 16 kHz before
it was upsampled), encoded and deleted; the next one downloads meanwhile. Output per shard: latents/<shard>.pt
(store.ShardWriter) and manifest/<shard>.jsonl. The clips of the corpus's held-out evaluation (and their reference
clips) are also kept as 16 kHz FLAC under eval_audio/. Finished shards are skipped, so the job can be restarted.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

from arabic_voxcpm import text as artext
from arabic_voxcpm import vae as avae
from arabic_voxcpm.data.store import ShardWriter

REPO = "Rabe3/lahgtna-arabic-tts-24khz"
SCORES = ["duration_s", "squim_pesq", "squim_stoi", "squim_sisdr", "snr_db", "bandwidth_hz", "asr_cer",
          "spk_halves_cosine", "vad_speech_ratio"]
BATCH_SAMPLES = 1_500_000  # 16 kHz samples per encoder batch (~94 s of audio, ~3 GB of activations)


def list_shards() -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))


def download(name: str, raw_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    for attempt in range(5):
        try:
            return hf_hub_download(REPO, name, repo_type="dataset", local_dir=raw_dir)
        except Exception as e:  # connection resets: retry
            if attempt == 4:
                raise
            print(f"download {name}: {e!r}, retrying", flush=True)
            time.sleep(10 * (attempt + 1))


def eval_uids() -> set[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "manifests/stage_a_eval.jsonl", repo_type="dataset")
    out = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.update((r["uid"], r["ref_uid"]))
    return out


def decode(audio: dict) -> np.ndarray:
    w, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != avae.SAMPLE_RATE:
        w = soxr.resample(w, sr, avae.SAMPLE_RATE, quality="VHQ")
    return w


def process(path: str, name: str, vae, out: str, keep: set[str], pool) -> dict:
    table = pq.read_table(path)
    rows = table.drop(["audio"]).to_pylist()
    waves = list(pool.map(decode, table.column("audio").to_pylist()))
    del table
    writer = ShardWriter(os.path.join(out, "latents", name + ".pt"))
    order = sorted(range(len(rows)), key=lambda i: len(waves[i]))  # similar lengths per batch: less padding
    batch: list[int] = []
    frames = 0

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
    with open(os.path.join(out, "manifest", name + ".jsonl.tmp"), "w", encoding="utf-8") as f:
        for r in rows:
            m = {"uid": r["uid"], "shard": name, "frames": r["frames"], "dialect": r["dialect"],
                 "speaker": r["speaker_id"], "text": artext.clean(r["text"]), "text_raw": r["text"],
                 "transcript_raw": r["transcript_raw"], "asr_text": r["asr_text"]}
            m.update({k: r[k] for k in SCORES})
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    writer.close()
    os.replace(os.path.join(out, "manifest", name + ".jsonl.tmp"), os.path.join(out, "manifest", name + ".jsonl"))
    return {"clips": len(rows), "hours": sum(len(w) for w in waves) / avae.SAMPLE_RATE / 3600}


def main(args) -> None:
    for d in ("latents", "manifest", "eval_audio", "raw"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    shards = list_shards()
    if args.limit:
        shards = shards[: args.limit]
    todo = [s for s in shards if not os.path.exists(os.path.join(args.out, "manifest", os.path.basename(s)[:-8] + ".jsonl"))]
    print(f"{len(shards)} shards, {len(todo)} to do", flush=True)
    keep = eval_uids()
    vae = avae.load_vae(args.model)
    raw = os.path.join(args.out, "raw")
    # threads: libsndfile and soxr release the GIL (forked processes after CUDA init hung in torch)
    with cf.ThreadPoolExecutor(1) as fetch, cf.ThreadPoolExecutor(args.workers) as pool:
        nxt = fetch.submit(download, todo[0], raw) if todo else None
        for k, s in enumerate(todo):
            path = nxt.result()
            nxt = fetch.submit(download, todo[k + 1], raw) if k + 1 < len(todo) else None
            t0 = time.time()
            stats = process(path, os.path.basename(s)[:-8], vae, args.out, keep, pool)
            os.remove(path)
            print(f"[{k + 1}/{len(todo)}] {s}: {stats['clips']} clips, {stats['hours']:.2f} h in {time.time() - t0:.0f} s",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects", help="for the AudioVAE")
    ap.add_argument("--out", default="/workspace/data/arabic/lahgtna")
    ap.add_argument("--workers", type=int, default=32, help="audio decoding threads")
    ap.add_argument("--limit", type=int, default=0, help="only the first N shards (testing)")
    main(ap.parse_args())
