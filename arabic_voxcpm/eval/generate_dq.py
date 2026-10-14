"""The evaluation set through the base inference (dq_engine, VoxCPM-Kernel/voxcpm-inference): the baseline the library
is measured against, on the same texts, voices and adapters as generate.py.

Runs in dq_engine's environment (stock nano-vllm-voxcpm 2.0.4 + ttfa-bench/dq_engine/compat-nanovllm-2.0.4.patch):

    PYTHONPATH=.:<dir holding dq_engine/> <dq python> -m arabic_voxcpm.eval.generate_dq --out runs/base-dq [--mode ref]

Cloning as the library's modes: ref = the clip as timbre reference, prompt_ref = + the clip and its transcript as
prompt, none = zero-shot. 2.0.4 has no seeded noise (runner._derive_noise_seed): requests are unseeded unless
--seeded, and each "seed" is a replicate (run.json "seeded"). Output: wav/<id>_s<seed>.wav, run.json.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from arabic_voxcpm.eval import common
from arabic_voxcpm.eval.generate import adapter_of, check_voices


async def run(args) -> None:
    import numpy as np

    sys.path.insert(0, args.dq_engine_dir)
    from dq_engine.engine import DqCoreEngine, Request

    items = common.items(args.set)
    voices = {v["id"]: v for v in common.voices(args.set)}
    check_voices(items, voices, args.mode)
    if not os.path.isdir(args.model):  # a Hugging Face repo id
        from huggingface_hub import snapshot_download

        args.model = snapshot_download(args.model)
    lora_dir = args.lora_dir or os.path.join(args.model, "adapters")
    names = sorted(d for d in os.listdir(lora_dir) if os.path.isdir(os.path.join(lora_dir, d))) if os.path.isdir(lora_dir) else []
    adapters = {it["id"]: adapter_of(it, None, names) for it in items}
    cfg = vars(args) | {"engine": "dq_engine", "adapter_map": None, "dit": "none", "lora_dir": lora_dir}
    common.claim(args.out, cfg)
    engine = DqCoreEngine(args.model, inference_timesteps=args.timesteps, max_num_seqs=args.concurrency,
                          max_model_len=1024, max_num_batched_tokens=16384, gpu_memory_utilization=args.gpu_mem,
                          out_sr=args.out_sr, adapters={n: os.path.join(lora_dir, n) for n in names})
    engine.start()
    try:
        lat = {}
        if args.mode != "none":
            for vid in sorted({it["voice"] for it in items}):
                path = os.path.join(args.set, voices[vid]["audio"])
                lat[vid] = engine.encode_latents(open(path, "rb").read(), os.path.splitext(path)[1].lstrip("."))
        sem = asyncio.Semaphore(args.concurrency)
        loop = asyncio.get_running_loop()

        async def one(it, seed):
            path = os.path.join(args.out, "wav", common.wav_name(it["id"], seed))
            if os.path.exists(path):
                return
            v = lat.get(it["voice"])
            clone = {} if args.mode == "none" else {"ref_latents": v}
            if args.mode == "prompt_ref":
                clone |= {"prompt_latents": v, "prompt_text": voices[it["voice"]]["text"]}
            q, chunks = asyncio.Queue(), []
            async with sem:
                engine.submit(Request(seq_id=f"{it['id']}_s{seed}_{time.time_ns()}", text=it["text"], out_q=q, loop=loop,
                                      prompt_latents=clone.get("prompt_latents"), prompt_text=clone.get("prompt_text", ""),
                                      ref_latents=clone.get("ref_latents"), max_generate_length=args.max_patches,
                                      cfg_value=args.cfg, lora_name=adapters[it["id"]],
                                      seed=seed if args.seeded else None))
                while True:
                    kind, payload = await q.get()
                    if kind == "end":
                        break
                    if kind == "error":
                        raise RuntimeError(f"{it['id']}: {payload}")
                    chunks.append(np.asarray(payload, dtype=np.float32))
            common.write_wav(path, np.concatenate(chunks) if chunks else np.zeros(0, np.float32), engine.out_sr)

        t0 = time.time()
        await asyncio.gather(*[one(it, s) for s in args.seeds for it in items])
        common.finish(args.out, cfg | {"seconds": round(time.time() - t0, 1), "dialects": names,
                                       "sample_rate": engine.out_sr})
        print(f"{len(items) * len(args.seeds)} utterances in {time.time() - t0:.0f} s -> {args.out}")
    finally:
        engine.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects")
    ap.add_argument("--lora-dir", default=None, help="default: <model>/adapters")
    ap.add_argument("--dq-engine-dir", default=os.environ.get("DQ_ENGINE_DIR", "/workspace/voxcpm-dialects/base_inference"),
                    help="the directory holding the dq_engine package")
    ap.add_argument("--mode", choices=common.MODES, default="ref")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001])
    ap.add_argument("--seeded", action="store_true", help="pass the seeds (needs runner._derive_noise_seed)")
    ap.add_argument("--timesteps", type=int, default=6, help="dq_engine's default")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--out-sr", type=int, default=48000, help="the model's rate, as the library's runs")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-patches", type=int, default=400)
    a = ap.parse_args()
    a.set = os.path.abspath(common.set_dir(a.set))
    asyncio.run(run(a))
