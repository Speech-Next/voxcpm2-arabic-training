"""The evaluation set through Habibi-TTS (F5-TTS dialect models, SWivid/Habibi-TTS): a candidate teacher of synthetic
training speech, compared with the current adapters on the same texts, voices and seeds (score.py / report.py).

Runs in Habibi's own environment (/workspace/envs/habibi):

    PYTHONPATH=. /workspace/envs/habibi/bin/python -m arabic_voxcpm.eval.generate_habibi --out runs/habibi-unified
    ... --model specialized   # a group's specialized model where one exists, else the unified one

Cloning: the voice's clip and its transcript (F5-TTS needs both). Output: wav/<id>_s<seed>.wav at 24 kHz, run.json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time

# group -> Habibi dialect id (UNK: inferred from the prompt; Habibi has no Yemeni or Bahraini id)
DIALECT = {"msa": "MSA", "najdi": "SAU", "hejazi": "SAU", "sa": "SAU", "khaliji": "UAE", "bh": "UAE", "iq": "IRQ",
           "eg": "EGY", "dz": "ALG", "ma": "MAR", "tn": "TUN", "ly": "LBY", "sd": "SDN", "lb": "LEV", "sy": "LEV",
           "ps": "LEV", "ye": "UNK"}
SPECIALIZED = {"MSA": 200000, "SAU": 200000, "UAE": 100000, "ALG": 100000, "IRQ": 100000, "EGY": 100000, "MAR": 100000}
REPO = "hf://SWivid/Habibi-TTS"


class Habibi:
    def __init__(self):
        from importlib.resources import files

        from f5_tts.infer.utils_infer import load_vocoder
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
        self.vocoder = load_vocoder(vocoder_name=self.cfg.model.mel_spec.mel_spec_type)
        self.models = {}

    def model(self, name: str):
        """name: "Unified" or a specialized dialect id."""
        if name not in self.models:
            from cached_path import cached_path
            from f5_tts.infer.utils_infer import load_model
            from hydra.utils import get_class

            sub = "Unified/model_200000" if name == "Unified" else f"Specialized/{name}/model_{SPECIALIZED[name]}"
            ckpt = str(cached_path(f"{REPO}/{sub}.safetensors"))
            vocab = str(cached_path(f"{REPO}/{os.path.dirname(sub)}/vocab.txt"))
            cls = get_class(f"f5_tts.model.{self.cfg.model.backbone}")
            self.models = {name: load_model(cls, self.cfg.model.arch, ckpt, mel_spec_type="vocos", vocab_file=vocab)}
        return self.models[name]

    def __call__(self, text: str, ref_audio: str, ref_text: str, group: str, specialized: bool, seed: int):
        import torch
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        from habibi_tts.infer.utils_infer import infer_process
        from habibi_tts.model.utils import dialect_id_map

        d = DIALECT[group]
        name = d if specialized and d in SPECIALIZED else "Unified"
        torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):  # it prints every chunk
            ref, ref_text = preprocess_ref_audio_text(ref_audio, ref_text, show_info=lambda *a: None)
            wave, sr, _ = infer_process(ref, ref_text, text, self.model(name), self.vocoder, show_info=lambda *a: None,
                                        progress=None, dialect_id=dialect_id_map[d] if name == "Unified" else None)
        return wave, sr, name


def main(args) -> None:
    import numpy as np
    import soundfile as sf

    items = [json.loads(line) for line in open(os.path.join(args.set, "items.jsonl"), encoding="utf-8")]
    voices = {v["id"]: v for v in json.load(open(os.path.join(args.set, "voices.json"), encoding="utf-8"))}
    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    tts = Habibi()
    specialized = args.model == "specialized"
    # one model at a time: items ordered by the model they use
    order = sorted(items, key=lambda it: (DIALECT[it["group"]] if specialized and DIALECT[it["group"]] in SPECIALIZED
                                          else "", it["id"]))
    t0, used = time.time(), {}
    for n, it in enumerate(order[: args.limit or None]):
        v = voices[it["voice"]]
        for seed in args.seeds:
            path = os.path.join(args.out, "wav", f"{it['id']}_s{seed}.wav")
            if os.path.exists(path):
                continue
            wave, sr, name = tts(it["text"], os.path.join(args.set, v["audio"]), v["text"], it["group"], specialized, seed)
            used[it["group"]] = name
            sf.write(path + ".tmp.wav", np.asarray(wave, dtype=np.float32), sr)
            os.replace(path + ".tmp.wav", path)
        if n % 50 == 0:
            print(f"{n}/{len(order)} items, {time.time() - t0:.0f} s", flush=True)
    json.dump({"set": args.set, "out": args.out, "model": f"habibi-{args.model}", "models_by_group": used,
               "mode": "ref", "seeds": args.seeds, "seconds": round(time.time() - t0, 1), "sample_rate": 24000},
              open(os.path.join(args.out, "run.json"), "w"), indent=1)
    print(f"{len(items) * len(args.seeds)} utterances in {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="/workspace/data/arabic/eval/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["unified", "specialized"], default="unified")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001])
    ap.add_argument("--limit", type=int, default=0, help="first N items only (a smoke test)")
    main(ap.parse_args())
