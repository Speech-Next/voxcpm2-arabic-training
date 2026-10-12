"""Per-clip quality of the training corpora: DNSMOS of the audio and ASR agreement with the transcript (#3).

Only the AudioVAE latents are kept, so the audio scored is their reconstruction (decoded at 48 kHz, resampled to
16 kHz). Cohere Transcribe Arabic transcribes every candidate in batches (Whisper large-v3 was tried first: on 435 of
512 Algerian clips it disagreed with the transcript, and it beat Cohere on 14, never across a 10 % CER line). DNSMOS
runs on CPU workers alongside.

Candidates: the training rows (select_rows: the evaluation set excluded), per speaker at most --speaker-hours and per
dialect at most --dialect-hours, drawn round-robin over the speakers so the dialect cap spreads over them.

    python -m arabic_voxcpm.data.quality --out /workspace/data/arabic/quality/scores.jsonl

Resumable: uids already in --out are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from arabic_voxcpm import vae as avae
from arabic_voxcpm.eval import metrics as M
from arabic_voxcpm.train.dataset import FPS, LatentStore, select_rows

MODEL = "/workspace/models/VoxCPM2-Arabic-Dialects"
HELDOUT = "/workspace/data/arabic/eval/v1/heldout.json"


def candidates(rows: list[dict], speaker_hours: float, dialect_hours: float, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["dialect"]][r["speaker"]].append(r)
    out = []
    for d, spk in sorted(by.items()):
        queues = []
        for s in sorted(spk):
            q = spk[s][:]
            rng.shuffle(q)
            queues.append(q)
        rng.shuffle(queues)
        used = defaultdict(float)
        total = 0.0
        while queues and total < dialect_hours * 3600:
            nxt = []
            for q in queues:
                r = q.pop()
                sec = r["frames"] / FPS
                if used[r["speaker"]] + sec <= speaker_hours * 3600:
                    out.append(r)
                    used[r["speaker"]] += sec
                    total += sec
                    if q:
                        nxt.append(q)
                if total >= dialect_hours * 3600:
                    break
            queues = nxt
    return out


class BatchedCohere(M.CohereASR):
    def batch(self, waves: list[np.ndarray]) -> list[str]:
        """One transcript per waveform. The processor splits a long input (~30 s and more) into chunks, one output
        each; decode() joins them back with audio_chunk_index."""
        inputs = self.processor(waves, sampling_rate=M.SR, return_tensors="pt", language="ar")
        chunks = inputs.pop("audio_chunk_index", None)
        inputs.to(self.model.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=256)
        texts = self.processor.decode(out, skip_special_tokens=True, audio_chunk_index=chunks, language="ar")
        texts = texts if isinstance(texts, list) else [texts]
        assert len(texts) == len(waves), (len(texts), len(waves))
        return [t.strip() for t in texts]


_DNS = None


def _worker_init() -> None:
    """One thread per DNSMOS worker: forked from the main process, numpy's BLAS pool would otherwise spin a thread per
    core in every worker (load 75 on this host, the GPU waiting at 17 %)."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def _dnsmos(audio: np.ndarray) -> dict:
    global _DNS
    if _DNS is None:
        _DNS = M.DNSMOS(threads=1)
    return _DNS(audio)


@torch.inference_mode()
def decode16k(vae, lat: list[torch.Tensor], chunk: int = 8) -> list[np.ndarray]:
    """16 kHz audio of each clip's latents. The decoder's 48 kHz activations are large: a few clips at a time, and the
    cache is released for the ASRs (CTranslate2 allocates outside torch)."""
    import torchaudio.functional as AF

    out = []
    for k in range(0, len(lat), chunk):
        part = lat[k : k + chunk]
        z = torch.zeros(len(part), 64, max(len(x) for x in part), device="cuda")
        for i, x in enumerate(part):
            z[i, :, : len(x)] = x.T.to("cuda", torch.float32)
        y = AF.resample(vae.decode(z)[:, 0], vae.out_sample_rate, M.SR)  # [B, samples]
        out += [y[i, : len(x) * avae.HOP].float().cpu().numpy() for i, x in enumerate(part)]
        del z, y
    torch.cuda.empty_cache()
    return out


def main(args) -> None:
    rows = select_rows(heldout=args.heldout)
    cand = candidates(rows, args.speaker_hours, args.dialect_hours, args.seed)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["uid"] for l in open(args.out, encoding="utf-8")}
    todo = [r for r in cand if r["uid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    hours = sum(r["frames"] for r in todo) / FPS / 3600
    print(f"{len(cand)} candidates ({sum(r['frames'] for r in cand) / FPS / 3600:.1f} h), {len(todo)} to score "
          f"({hours:.1f} h)", flush=True)
    todo.sort(key=lambda r: r["frames"])  # similar lengths per batch
    store = LatentStore()
    vae = avae.load_vae(MODEL)
    asr = BatchedCohere()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pool = ProcessPoolExecutor(args.dnsmos_workers, initializer=_worker_init)
    t0, n_done, sec_done = time.time(), 0, 0.0
    with open(args.out, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch):
            batch = todo[b : b + args.batch]
            waves = decode16k(vae, [store[r["uid"]] for r in batch])
            dns = pool.map(_dnsmos, waves, chunksize=4)
            hyps = [h for k in range(0, len(waves), args.asr_batch) for h in asr.batch(waves[k : k + args.asr_batch])]
            out = []
            for r, h, d in zip(batch, hyps, dns):
                e = M.error_rates(r["text"], h)
                o = {"uid": r["uid"], "source": r["source"], "dialect": r["dialect"], "speaker": r["speaker"],
                     "sec": r["frames"] / FPS, "cohere": h, "cer": e["cer"], "wer": e["wer"]}
                o.update({k: round(v, 4) for k, v in d.items()})
                out.append(o)
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch)
            sec_done += sum(o["sec"] for o in out)
            if (b // args.batch) % 20 == 0:
                el = time.time() - t0
                print(f"{n_done}/{len(todo)} clips, {sec_done / 3600:.1f} h audio in {el / 60:.1f} min "
                      f"({sec_done / max(el, 1e-9):.0f}x real time)", flush=True)
    pool.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/data/arabic/quality/scores.jsonl")
    ap.add_argument("--heldout", default=HELDOUT)
    ap.add_argument("--speaker-hours", type=float, default=3.0)
    ap.add_argument("--dialect-hours", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--asr-batch", type=int, default=16)
    ap.add_argument("--dnsmos-workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
