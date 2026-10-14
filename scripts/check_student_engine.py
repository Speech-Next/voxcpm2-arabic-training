"""A LocDiT student checkpoint in the inference engine: its format, then one sentence per adapter (GPU).

    PYTHONPATH=<this repo>:$ENGINE $ENGINE/.venv/bin/python scripts/check_student_engine.py \\
        --dit /workspace/runs/dit-adapters/mse-best.pt [--model /workspace/models/VoxCPM2-Arabic-Dialects] \\
        [--out /workspace/runs/dit-adapters/engine-check] [--precision int8] [--head-tokens 8]

1. The checkpoint is what TurboConfig.dit_checkpoint takes (models/voxcpm2/runner.py): "state_dict" with every
   tensor of the model's LocDiT (feat_decoder.estimator.*, the shapes of model.safetensors; the first dit_layers
   layers for a smaller student) and nothing else, "schedule" (t, dt) pairs from 1 to 0, "mean_mode".
2. DialectTTS starts on the model with every adapter registered and the student as its LocDiT (small engine:
   --gpu-memory-utilization, --kv-blocks), clones one voice (the v2 set's first, or --voice), and speaks one sentence
   in each adapter and in the base model. Each output must be finite, not silent and of a plausible duration; written
   to --out/<adapter>.wav with a summary (summary.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

import numpy as np

TEXT = "مرحبا، كيف حالك اليوم؟ الجو جميل والحمد لله، وإن شاء الله نلتقي قريبا."
TEXT_EN = "Hello, aaj ka mausam bahut accha hai, and I hope we meet again very soon."
PREFIX = "feat_decoder.estimator."


def check_format(ckpt_path: str, model_dir: str) -> dict:
    """The checkpoint against the model's LocDiT tensors (names and shapes); returns its metadata."""
    import torch
    from safetensors import safe_open

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt", "cpu") as f:
        want = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys() if k.startswith(PREFIX)}
    layers = ck.get("dit_layers")
    if layers:
        layer = re.compile(re.escape(PREFIX) + r"decoder\.layers\.(\d+)\.")
        want = {k: s for k, s in want.items() if not layer.match(k) or int(layer.match(k).group(1)) < int(layers)}
    got = {k: tuple(v.shape) for k, v in sd.items()}
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    shapes = sorted(k for k in set(want) & set(got) if want[k] != got[k])
    assert not missing and not extra and not shapes, \
        f"not the model's LocDiT: missing {missing[:3]}, extra {extra[:3]}, other shapes {shapes[:3]}"
    sched = ck["schedule"]
    t_span = [float(sched[0][0])] + [float(t - dt) for t, dt in sched]
    assert abs(t_span[0] - 1) < 1e-6 and abs(t_span[-1]) < 1e-6 and all(a > b for a, b in zip(t_span, t_span[1:])), \
        f"schedule {sched} is not a grid from 1 to 0"
    info = {"tensors": len(sd), "dtype": str(next(iter(sd.values())).dtype), "t_span": [round(t, 5) for t in t_span],
            "mean_mode": bool(ck.get("mean_mode")), "dit_layers": layers,
            **{k: ck[k] for k in ("stage", "step", "init", "val_mse_teacher", "adapters") if k in ck}}
    print("checkpoint: " + json.dumps(info, ensure_ascii=False), flush=True)
    return info


async def speak(args, info: dict) -> dict:
    from nanovllm_voxcpm import DialectTTS

    from arabic_voxcpm.eval import common

    tts = await DialectTTS.load(args.model, args.dit, max_concurrency=16, head_tokens=args.head_tokens,
                                gpu_memory_utilization=args.gpu_memory_utilization, kv_blocks=args.kv_blocks,
                                w8a8=args.precision == "int8")
    out = {"checkpoint": os.path.abspath(args.dit), "format": info, "precision": args.precision,
           "head_tokens": args.head_tokens, "adapters": {}}
    try:
        assert tts.tags is None, "a tagged model: this check is for the adapters model"
        voice = args.voice or os.path.join(args.set, common.voices(args.set)[0]["audio"])
        await tts.add_voice("check", voice)
        os.makedirs(args.out, exist_ok=True)
        for name in [*tts.dialects, None]:
            text = TEXT_EN if name == "hindi_english" else TEXT
            t0 = time.time()
            audio = await tts.synthesize(text, dialect=name, voice="check", seed=args.seed, max_patches=300)
            sec = len(audio) / tts.sample_rate
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            ok = bool(np.isfinite(audio).all()) and rms > 1e-3 and 1.0 <= sec <= 20.0
            label = name or "base"
            common.write_wav(os.path.join(args.out, f"{label}.wav"), audio, tts.sample_rate)
            out["adapters"][label] = {"seconds": round(sec, 2), "rms": round(rms, 4), "ok": ok,
                                      "wall_s": round(time.time() - t0, 2)}
            print(f"{label:14} {sec:5.2f} s  rms {rms:.4f}  {'ok' if ok else 'FAILED'}", flush=True)
    finally:
        await tts.close()
    json.dump(out, open(os.path.join(args.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dit", required=True, help="the student checkpoint")
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects", help="the adapters model")
    ap.add_argument("--out", help="default: <checkpoint>-engine-check/")
    ap.add_argument("--set", default="/workspace/data/arabic/eval/v2", help="the voice: this set's first")
    ap.add_argument("--voice", help="a reference clip instead")
    ap.add_argument("--precision", choices=["bf16", "int8"], default="int8")
    ap.add_argument("--head-tokens", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    ap.add_argument("--kv-blocks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--format-only", action="store_true", help="check the checkpoint's format, start no engine")
    args = ap.parse_args()
    args.out = args.out or os.path.splitext(args.dit)[0] + "-engine-check"
    info = check_format(args.dit, args.model)
    if args.format_only:
        return
    res = asyncio.run(speak(args, info))
    bad = [k for k, v in res["adapters"].items() if not v["ok"]]
    print(f"{len(res['adapters']) - len(bad)} of {len(res['adapters'])} ok -> {args.out}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
