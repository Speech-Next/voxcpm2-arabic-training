"""Generate the evaluation set with one model configuration (runs in the inference library's environment).

    PYTHONPATH=/workspace/voxcpm2-arabic-dialects-turbo python -m arabic_voxcpm.eval.generate \\
        --set /workspace/data/arabic/eval/v1 --out /workspace/data/arabic/eval/runs/current-teacher --dit none

Every item is spoken in its group's adapter (items.jsonl "adapter"; a tagged model: the set's dialect_map.json, or
--adapter-map), cloning its voice (--mode ref: timbre only, prompt_ref: with the reference transcript from
voices.json, none: no cloning), once per seed. A group without an adapter (null / "none") is spoken by the base model;
an adapter the model lacks is an error. Output: wav/<id>_s<seed>.wav at the model's sample rate and run.json (the
configuration). An interrupted run resumes; a directory holding another configuration is refused (common.claim).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from arabic_voxcpm.eval import common


def adapter_map(spec: str | None, set_dir: str, tagged: bool) -> str | None:
    """The map file to use: "auto" = the set's dialect_map.json for a tagged model (one model, dialect tags), the
    items' own adapters otherwise; None / "items" = the items' adapters; else a path."""
    if spec in (None, "items"):
        return None
    if spec == "auto":
        path = os.path.join(set_dir, "dialect_map.json")
        return os.path.realpath(path) if tagged and os.path.exists(path) else None
    return spec


def adapter_of(it: dict, amap: dict | None, dialects) -> str | None:
    """The item's adapter (or dialect tag); None = the base model, only where the group is mapped to none on purpose
    (null or "none" in the map / the items' CURRENT_ADAPTER). A group missing from the map or an adapter the model
    does not have raises: no silent base-model run."""
    if amap is None:
        a = it["adapter"]
    elif it["group"] in amap:
        a = amap[it["group"]]
    else:
        raise ValueError(f"{it['id']}: group {it['group']!r} is not in the adapter map")
    if a is None or a == "none":
        return None
    if a not in dialects:
        raise ValueError(f"{it['id']}: adapter {a!r} is not one of the model's ({', '.join(sorted(dialects))})")
    return a


def check_voices(items: list[dict], voices: dict, mode: str) -> None:
    """The items' voices exist, and with prompt_ref each has its transcript."""
    if mode == "none":
        return
    used = {it["voice"] for it in items}
    if used - set(voices):
        raise SystemExit(f"voices missing from voices.json: {sorted(used - set(voices))}")
    bare = sorted(v for v in used if not voices[v].get("text"))
    if mode == "prompt_ref" and bare:
        raise SystemExit(f"prompt_ref needs each voice's transcript; none for {bare} (rebuild with eval.build)")


async def run(args) -> None:
    from nanovllm_voxcpm import DialectTTS

    items = common.items(args.set)
    voices = {v["id"]: v for v in common.voices(args.set)}
    check_voices(items, voices, args.mode)
    dit = None if args.dit == "none" else args.dit
    tts = await DialectTTS.load(args.model, dit, dit_cache=args.dit_cache, max_concurrency=args.concurrency,
                                head_tokens=args.head_tokens, w8a8=args.precision == "int8",
                                gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.85),
                                kv_blocks=getattr(args, "kv_blocks", 600))
    try:
        amap_path = adapter_map(args.adapter_map, args.set, tts.tags is not None)
        amap = json.load(open(amap_path, encoding="utf-8")) if amap_path else None
        adapters = {it["id"]: adapter_of(it, amap, tts.dialects) for it in items}  # all checked before generating
        cfg = vars(args) | {"engine": "library", "adapter_map": amap_path}
        common.claim(args.out, cfg)
        if args.mode != "none":
            for vid in sorted({it["voice"] for it in items}):
                await tts.add_voice(vid, os.path.join(args.set, voices[vid]["audio"]),
                                    text=voices[vid]["text"] if args.mode == "prompt_ref" else None)
        sem = asyncio.Semaphore(args.concurrency)

        async def one(it, seed):
            path = os.path.join(args.out, "wav", common.wav_name(it["id"], seed))
            if os.path.exists(path):
                return
            text = (args.text_prefix.format(group=it["group"]) if args.text_prefix else "") + it["text"]
            async with sem:
                audio = await tts.synthesize(text, dialect=adapters[it["id"]],
                                             voice=None if args.mode == "none" else it["voice"],
                                             seed=seed, max_patches=args.max_patches, normalize=args.normalize)
            common.write_wav(path, audio, tts.sample_rate)

        t0 = time.time()
        await asyncio.gather(*[one(it, s) for s in args.seeds for it in items])
        common.finish(args.out, cfg | {"seconds": round(time.time() - t0, 1), "dialects": tts.dialects,
                                       "sample_rate": tts.sample_rate})
        print(f"{len(items) * len(args.seeds)} utterances in {time.time() - t0:.0f} s -> {args.out}")
    finally:
        await tts.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="v1", help="v1, v2 (under /workspace/data/arabic/eval) or a set directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects")
    ap.add_argument("--dit", default="none", help="distilled LocDiT (path or repo:file); none = the original LocDiT")
    ap.add_argument("--dit-cache", default=None)
    ap.add_argument("--head-tokens", type=int, default=0, help="0 = no head start (quality runs)")
    ap.add_argument("--precision", choices=["bf16", "int8"], default="bf16",
                    help="bf16: the model itself (a new model's SmoothQuant statistics are not calibrated yet)")
    ap.add_argument("--mode", choices=common.MODES, default="ref")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001])
    ap.add_argument("--adapter-map", default="auto",
                    help="JSON {group: adapter or null}; auto: the set's dialect_map.json for a tagged model, the "
                         "items' adapters otherwise; items: always the items' adapters")
    ap.add_argument("--text-prefix", default="", help="e.g. '[{group}] ' for a dialect-tagged model")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-patches", type=int, default=400)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="lower it to share the GPU")
    ap.add_argument("--kv-blocks", type=int, default=600)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false",
                    help="the text as it is (digits included), without the library's text normalization")
    a = ap.parse_args()
    a.set = os.path.abspath(common.set_dir(a.set))
    asyncio.run(run(a))
