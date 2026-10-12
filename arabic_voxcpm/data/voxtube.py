"""Reference voices from VoxTube's Arabic channels (voice-is-cool/voxtube, CC BY-NC-SA 4.0): 228 speakers, no
transcripts, 4 s segments. A voice bank for synthetic speech, not training data.

Per speaker, up to --videos videos; in each, the first two consecutive segments joined (~8 s). Only the needed row
groups' audio is read. Kept with DNSMOS; dialect region (ADI-5) and a transcript are added later by voices.py.

    python -m arabic_voxcpm.data.voxtube --scan /workspace/data/arabic/voxtube/scan.json   (scripts/voxtube_scan.py)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SR = 16000


def decode(b: bytes) -> np.ndarray:
    import librosa
    import soundfile as sf

    a, sr = sf.read(io.BytesIO(b), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    return a if sr == SR else librosa.resample(a, orig_sr=sr, target_sr=SR)


def plan(scan: list[dict], videos: int, seed: int) -> dict:
    """speaker -> [(file, row group, [row index, row index])]: a consecutive segment pair per chosen video."""
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(dict))  # spk -> video -> seg -> (file, rg, idx, gender)
    for x in scan:
        for i, s, v, g, gen in zip(x["ar_idx"], x["spk"], x["video"], x["seg"], x["gender"]):
            by[s][v][g] = (x["file"], x["rg"], i, gen)
    out = {}
    for s, vids in by.items():
        pairs = []
        for v in sorted(vids):
            segs = vids[v]
            for g in sorted(segs):
                a, b = segs[g], segs.get(g + 1)
                if b and a[:2] == b[:2]:  # same row group
                    pairs.append((a[0], a[1], [a[2], b[2]], v, g, a[3]))
                    break
        rng.shuffle(pairs)
        out[s] = pairs[:videos]
    return out


def fetch(fs, file: str, rg: int, rows: list[int]) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    for k in range(6):
        try:
            with fs.open(file, block_size=1 << 22) as f:
                col = pq.ParquetFile(f).read_row_group(rg, columns=["audio"]).column("audio").to_pylist()
            return [decode(col[i]["bytes"]) for i in rows]
        except Exception as e:  # the Hub resets connections often from this host
            err = e
            time.sleep(2 + 3 * k)
    raise err


def main(args) -> None:
    from huggingface_hub import HfFileSystem

    from arabic_voxcpm.eval.metrics import DNSMOS

    fs = HfFileSystem()
    scan = json.load(open(args.scan))
    todo = plan(scan, args.videos, args.seed)
    os.makedirs(os.path.join(args.out, "voices"), exist_ok=True)
    meta_path = os.path.join(args.out, "voices.jsonl")
    done = {json.loads(l)["id"] for l in open(meta_path)} if os.path.exists(meta_path) else set()
    jobs = [(s, k, p) for s, ps in sorted(todo.items()) for k, p in enumerate(ps) if f"vt{s}_{k}" not in done]
    print(f"{len(todo)} speakers, {len(jobs)} clips to fetch", flush=True)
    dns = DNSMOS()

    def one(job):
        s, k, (file, rg, rows, video, seg, gender) = job
        a = np.concatenate(fetch(fs, file, rg, rows))
        return s, k, a, video, seg, gender

    import soundfile as sf

    with open(meta_path, "a") as f, ThreadPoolExecutor(args.threads) as ex:
        for n, (s, k, a, video, seg, gender) in enumerate(ex.map(one, jobs)):
            vid = f"vt{s}_{k}"
            sf.write(os.path.join(args.out, "voices", vid + ".flac"), a, SR)
            row = {"id": vid, "speaker": f"voxtube:{s}", "video": video, "segments": [seg, seg + 1], "gender": gender,
                   "sec": round(len(a) / SR, 2), "audio": f"voices/{vid}.flac", **dns(a)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 100 == 0:
                print(n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="/workspace/data/arabic/voxtube/scan.json")
    ap.add_argument("--out", default="/workspace/data/arabic/voxtube")
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
