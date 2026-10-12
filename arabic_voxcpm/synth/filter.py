"""Synthetic speech into a training corpus: keep a generated clip only if Cohere hears its text (CER <= --max-cer),
DNSMOS >= --min-dnsmos and its speaking rate is plausible; encode the kept clips to AudioVAE latents.

    python -m arabic_voxcpm.synth.filter --set <set> --gen <generator output> --name <shard prefix> [--delete]

Writes /workspace/data/arabic/synth/{latents,manifest}/<name>-<k>.{pt,jsonl} (corpus "synth", read like lahgtna and
SADA) and scores-<name>.jsonl (every clip, kept or not). --delete removes each wav once handled (disk).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from arabic_voxcpm import vae as avae
from arabic_voxcpm.data.quality import BatchedCohere, _dnsmos, _worker_init
from arabic_voxcpm.data.store import ShardWriter
from arabic_voxcpm.eval import metrics as M

ROOT = "/workspace/data/arabic/synth"
MODEL = "/workspace/models/VoxCPM2-Arabic-Dialects"


def main(args) -> None:
    items = {it["id"]: it for it in map(json.loads, open(os.path.join(args.set, "items.jsonl"), encoding="utf-8"))}
    wav_dir = os.path.join(args.gen, "wav")
    names = sorted(n for n in os.listdir(wav_dir) if n.endswith(".wav") and ".tmp" not in n)
    for sub in ("latents", "manifest"):
        os.makedirs(os.path.join(ROOT, sub), exist_ok=True)
    k = len([n for n in os.listdir(os.path.join(ROOT, "latents")) if n.startswith(args.name + "-")])
    asr, vae = BatchedCohere(), avae.load_vae(MODEL)
    pool = ProcessPoolExecutor(16, initializer=_worker_init)
    kept = total = 0
    with open(os.path.join(ROOT, f"scores-{args.name}.jsonl"), "a", encoding="utf-8") as sf_out:
        for b in range(0, len(names), args.shard):
            part = names[b : b + args.shard]
            waves = [M.load16k(os.path.join(wav_dir, n)) for n in part]
            dns = list(pool.map(_dnsmos, waves, chunksize=4))
            hyps = [h for i in range(0, len(waves), 16) for h in asr.batch(waves[i : i + 16])]
            writer = ShardWriter(os.path.join(ROOT, "latents", f"{args.name}-{k:04d}.pt"))
            rows, keep_waves = [], []
            for n, a, h, d in zip(part, waves, hyps, dns):
                it = items[n[:-4].rsplit("_s", 1)[0]]
                sec = len(a) / M.SR
                e = M.error_rates(it["text"], h)
                rate = len(it["text"]) / max(sec, 1e-3)
                ok = e["cer"] <= args.max_cer and d["dnsmos_ovrl"] >= args.min_dnsmos and 5.0 <= rate <= 25.0
                sf_out.write(json.dumps({"id": it["id"], "dialect": it["dialect"], "cohere": h, "cer": e["cer"],
                                         "sec": round(sec, 2), "rate": round(rate, 1), "kept": ok, **d},
                                        ensure_ascii=False) + "\n")
                if ok:
                    rows.append({"uid": f"{args.name}-{it['id']}", "source": "synth", "dialect": it["dialect"],
                                 "speaker": f"synth:{it['speaker']}", "voice": it["voice"], "text": it["text"],
                                 "gen": args.name, "cer": round(e["cer"], 4), "dnsmos_ovrl": d["dnsmos_ovrl"]})
                    keep_waves.append(a)
            for i in range(0, len(keep_waves), 16):
                for r, z in zip(rows[i : i + 16], avae.encode(vae, keep_waves[i : i + 16])):
                    r["frames"] = len(z)
                    writer.add(r["uid"], z)
            writer.close()
            with open(os.path.join(ROOT, "manifest", f"{args.name}-{k:04d}.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            k += 1
            kept += len(rows)
            total += len(part)
            if args.delete:
                for n in part:
                    os.remove(os.path.join(wav_dir, n))
            print(f"{total}/{len(names)} clips, {kept} kept", flush=True)
    pool.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-cer", type=float, default=0.05)
    ap.add_argument("--min-dnsmos", type=float, default=3.0)
    ap.add_argument("--shard", type=int, default=1024)
    ap.add_argument("--delete", action="store_true")
    main(ap.parse_args())
