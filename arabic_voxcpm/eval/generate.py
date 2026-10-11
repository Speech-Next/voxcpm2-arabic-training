"""Generate the evaluation set with one model configuration (runs in the inference library's environment).

    PYTHONPATH=/workspace/voxcpm2-arabic-dialects-turbo python -m arabic_voxcpm.eval.generate \\
        --set /workspace/data/arabic/eval/v1 --out /workspace/data/arabic/eval/runs/current-teacher --dit none

Every item is spoken in its group's adapter (items.jsonl "adapter", or --adapter-map), cloning its voice (--mode ref:
timbre only, prompt_ref: with the reference transcript, none: no cloning), once per seed. Output: wav/<id>_s<seed>.wav
at the model's sample rate and run.json (the configuration).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time


async def run(args) -> None:
    import soundfile as sf
    from nanovllm_voxcpm import DialectTTS

    items = [json.loads(l) for l in open(os.path.join(args.set, "items.jsonl"), encoding="utf-8")]
    voices = json.load(open(os.path.join(args.set, "voices.json"), encoding="utf-8"))
    amap = json.load(open(args.adapter_map, encoding="utf-8")) if args.adapter_map else None
    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    dit = None if args.dit == "none" else args.dit
    tts = await DialectTTS.load(args.model, dit, dit_cache=args.dit_cache, max_concurrency=args.concurrency,
                                head_tokens=args.head_tokens, w8a8=args.precision == "int8")
    try:
        if args.mode != "none":
            for v in voices:
                await tts.add_voice(v["id"], os.path.join(args.set, v["audio"]),
                                    text=v["text"] if args.mode == "prompt_ref" else None)
        sem = asyncio.Semaphore(args.concurrency)

        async def one(it, seed):
            path = os.path.join(args.out, "wav", f"{it['id']}_s{seed}.wav")
            if os.path.exists(path):
                return
            adapter = amap.get(it["group"]) if amap is not None else it["adapter"]
            if adapter is not None and adapter not in tts.dialects:
                adapter = None
            text = (args.text_prefix.format(group=it["group"]) if args.text_prefix else "") + it["text"]
            async with sem:
                audio = await tts.synthesize(text, dialect=adapter, voice=None if args.mode == "none" else it["voice"],
                                             seed=seed, max_patches=args.max_patches)
            sf.write(path, audio, tts.sample_rate)

        t0 = time.time()
        await asyncio.gather(*[one(it, s) for s in args.seeds for it in items])
        json.dump({k: v for k, v in vars(args).items()} | {"seconds": round(time.time() - t0, 1),
                                                            "dialects": tts.dialects, "sample_rate": tts.sample_rate},
                  open(os.path.join(args.out, "run.json"), "w"), indent=1)
        print(f"{len(items) * len(args.seeds)} utterances in {time.time() - t0:.0f} s -> {args.out}")
    finally:
        await tts.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="/workspace/data/arabic/eval/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects")
    ap.add_argument("--dit", default="none", help="distilled LocDiT (path or repo:file); none = the original LocDiT")
    ap.add_argument("--dit-cache", default=None)
    ap.add_argument("--head-tokens", type=int, default=0, help="0 = no head start (quality runs)")
    ap.add_argument("--precision", choices=["bf16", "int8"], default="bf16",
                    help="bf16: the model itself (a new model's SmoothQuant statistics are not calibrated yet)")
    ap.add_argument("--mode", choices=["ref", "prompt_ref", "none"], default="ref")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001])
    ap.add_argument("--adapter-map", help="JSON {group: adapter or null} instead of the items' current adapters")
    ap.add_argument("--text-prefix", default="", help="e.g. '[{group}] ' for a dialect-tagged model")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-patches", type=int, default=400)
    asyncio.run(run(ap.parse_args()))
