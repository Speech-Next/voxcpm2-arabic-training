"""Shared by the evaluation steps (standard library only: generation imports it in the inference environment).

Where sets and runs live, a run's configuration (run.json), the wav names <item id>_s<seed>.wav and their atomic
writes, and the guard that keeps one run directory to one configuration.
"""

from __future__ import annotations

import json
import os
import re

EVAL_ROOT = "/workspace/data/arabic/eval"
RUNS_ROOT = os.path.join(EVAL_ROOT, "runs")
MODES = ("ref", "prompt_ref", "none")  # ref: timbre only, prompt_ref: + the reference transcript, none: no cloning
PATCH_S = 0.16  # one generated patch (4 latent frames at 25 Hz): max_patches * PATCH_S caps an utterance
# what makes two generations of one run directory the same run (seeds may be added)
SAME_RUN = ("set", "engine", "model", "dit", "dit_cache", "precision", "mode", "adapter_map", "text_prefix",
            "normalize", "head_tokens", "max_patches", "timesteps")
_WAV = re.compile(r"^(.+)_s(\d+)\.wav$")


def set_dir(name: str) -> str:
    """A set by name (v1, v2: under EVAL_ROOT) or path."""
    return name if os.path.isdir(name) or os.sep in name else os.path.join(EVAL_ROOT, name)


def run_dir(name: str) -> str:
    return name if os.path.isdir(name) or os.sep in name else os.path.join(RUNS_ROOT, name)


def items(set_dir: str) -> list[dict]:
    return [json.loads(line) for line in open(os.path.join(set_dir, "items.jsonl"), encoding="utf-8")]


def voices(set_dir: str) -> list[dict]:
    return json.load(open(os.path.join(set_dir, "voices.json"), encoding="utf-8"))


def run_config(run: str) -> dict:
    """run.json; runs from before the modes were recorded cloned with ``ref``."""
    cfg = json.load(open(os.path.join(run, "run.json"), encoding="utf-8"))
    cfg.setdefault("mode", "ref")
    return cfg


def wav_name(item_id: str, seed: int) -> str:
    return f"{item_id}_s{seed}.wav"


def parse_wav(name: str) -> tuple[str, int] | None:
    """(item id, seed) of a finished wav; None for anything else (a write in progress, other files)."""
    m = _WAV.match(name)
    return (m.group(1), int(m.group(2))) if m and ".tmp" not in name else None


def write_wav(path: str, audio, sample_rate: int) -> None:
    """Through a temporary file: a killed run leaves no truncated wav that a resume would take as done."""
    import soundfile as sf

    tmp = path + ".tmp"
    sf.write(tmp, audio, sample_rate, format="WAV")
    os.replace(tmp, path)


def claim(out: str, cfg: dict) -> None:
    """Refuse to resume into ``out`` with another configuration (its wavs would be mixed into this run): run.json of
    a finished run or run.partial.json of an interrupted one must agree on SAME_RUN. Then marks ``out`` as started."""
    os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    for name in ("run.json", "run.partial.json"):
        path = os.path.join(out, name)
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
            old.setdefault("mode", "ref")
            diff = {k: (old.get(k), cfg.get(k)) for k in SAME_RUN if k in old and k in cfg and old[k] != cfg[k]}
            if diff:
                raise SystemExit(f"{out} holds another configuration ({name}): "
                                 + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in diff.items())
                                 + "; use another --out or delete it")
    json.dump(cfg, open(os.path.join(out, "run.partial.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def finish(out: str, cfg: dict) -> None:
    """run.json (the run is complete: scripts skip generation when it exists)."""
    json.dump(cfg, open(os.path.join(out, "run.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if os.path.exists(os.path.join(out, "run.partial.json")):
        os.remove(os.path.join(out, "run.partial.json"))
