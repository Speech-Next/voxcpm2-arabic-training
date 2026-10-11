"""Rabe3/lahgtna-arabic-tts-24khz -> AudioVAE latent shards + a manifest, one parquet shard at a time.

    python -m arabic_voxcpm.data.lahgtna --out /workspace/data/arabic/lahgtna

591 h, 13 dialects (bh dz eg iq lb ly ma ps sa sd sy tn ye), 688 speakers clustered by the corpus. Each source shard
(~0.8 GB, ~1500 clips) is downloaded, decoded, resampled 24 -> 16 kHz (the corpus was 16 kHz before it was
upsampled), encoded and deleted; the next one downloads meanwhile. The clips of the corpus's held-out evaluation
(and their reference clips) are also kept as 16 kHz FLAC under eval_audio/. Finished shards are skipped.
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
from arabic_voxcpm.data import common

REPO = "Rabe3/lahgtna-arabic-tts-24khz"
SCORES = ["duration_s", "squim_pesq", "squim_stoi", "squim_sisdr", "snr_db", "bandwidth_hz", "asr_cer",
          "spk_halves_cosine", "vad_speech_ratio"]


def list_shards() -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))


def eval_uids() -> set[str]:
    from huggingface_hub import hf_hub_download

    out = set()
    for line in open(hf_hub_download(REPO, "manifests/stage_a_eval.jsonl", repo_type="dataset"), encoding="utf-8"):
        r = json.loads(line)
        out.update((r["uid"], r["ref_uid"]))
    return out


def decode(audio: dict) -> np.ndarray:
    w, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    w = w.mean(1)
    return soxr.resample(w, sr, avae.SAMPLE_RATE, quality="VHQ") if sr != avae.SAMPLE_RATE else w


def rows_of(table, name: str) -> list[dict]:
    out = []
    for r in table.drop(["audio"]).to_pylist():
        m = {"uid": r["uid"], "source": "lahgtna", "shard": name, "dialect": r["dialect"], "speaker": r["speaker_id"],
             "text": artext.clean(r["text"]), "text_raw": r["text"], "transcript_raw": r["transcript_raw"],
             "asr_text": r["asr_text"]}
        m.update({k: r[k] for k in SCORES})
        out.append(m)
    return out


def main(args) -> None:
    common.out_dirs(args.out)
    shards = list_shards()[: args.limit or None]
    todo = [s for s in shards if not common.done(args.out, os.path.basename(s)[:-8])]
    print(f"{len(shards)} shards, {len(todo)} to do", flush=True)
    keep = eval_uids()
    vae = avae.load_vae(args.model)
    raw = os.path.join(args.out, "raw")
    # threads: libsndfile and soxr release the GIL (forked processes after CUDA init hung in torch)
    with cf.ThreadPoolExecutor(1) as fetch, cf.ThreadPoolExecutor(args.workers) as pool:
        nxt = fetch.submit(common.download, REPO, todo[0], raw) if todo else None
        for k, s in enumerate(todo):
            path = nxt.result()
            nxt = fetch.submit(common.download, REPO, todo[k + 1], raw) if k + 1 < len(todo) else None
            t0 = time.time()
            name = os.path.basename(s)[:-8]
            table = pq.read_table(path)
            waves = list(pool.map(decode, table.column("audio").to_pylist()))
            rows = rows_of(table, name)
            del table
            stats = common.write_shard(args.out, name, vae, rows, waves, keep)
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
