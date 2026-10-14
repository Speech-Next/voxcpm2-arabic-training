"""One command: generate the evaluation set with a configuration, score it, report it against the baselines.

    python -m arabic_voxcpm.eval.run --name <run> --model <path | hf id> [--engine library | dq] [--dit none | <ckpt>]
        [--dit-cache light] [--precision bf16 | int8] [--mode ref | prompt_ref | none] [--set v1 | v2]
        [--baseline base-adapters ...] [--seeds 2000 2001] [any generate.py option, e.g. --head-tokens 8]

Runs in the scoring environment (/venv/main); generation runs in the engine's own (--engine-python, --engine-path:
ENGINE_PY / ENGINE as the scripts; dq_engine: DQ_PY / DQ_ENGINE_DIR). Steps, each skipped when done:
1. generate.py (generate_dq.py) into <runs>/<name>, unless its run.json exists (with another configuration: refused);
2. score.py: the set's ground truth where unscored, then the run (only new files, missing metrics filled);
3. report.py: <run>/report.md and report.json, the run's per-group means with the ground-truth floor, and per
   baseline the paired bootstrap 95 % intervals per group. A baseline name resolves to <name>-<set>-<mode>,
   <name>-<mode>, <name>-<set>, <name>, the first of the run's set and mode (else the first, labeled).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from arabic_voxcpm.eval import common

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_PY = os.environ.get("ENGINE_PY", "/workspace/ttfa-bench/envs/nanovllm-opt/bin/python")
ENGINE = os.environ.get("ENGINE", "/workspace/nanovllm-voxcpm-opt")
DQ_PY = os.environ.get("DQ_PY", "/workspace/envs/nanovllm-2.0.4/bin/python")
DQ_ENGINE_DIR = os.environ.get("DQ_ENGINE_DIR", "/workspace/voxcpm-dialects/base_inference")


def wanted(args, set_dir: str) -> dict:
    """The run.json fields this command asks for (an existing run must agree)."""
    w = {"set": set_dir, "model": args.model, "mode": args.mode, "seeds": args.seeds}
    if args.engine == "library":
        w |= {"dit": args.dit, "dit_cache": args.dit_cache, "precision": args.precision}
    return w


def check_existing(run: str, want: dict) -> None:
    cfg = common.run_config(run)
    diff = {k: (cfg[k], v) for k, v in want.items() if k in cfg and k != "seeds" and cfg[k] != v}
    if diff:
        raise SystemExit(f"{run} is another configuration: " + ", ".join(f"{k} {a!r} (asked {b!r})"
                                                                         for k, (a, b) in diff.items()))
    missing = sorted(set(want["seeds"]) - set(cfg.get("seeds", [])))
    if missing:
        raise SystemExit(f"{run} lacks seeds {missing}: run generate.py into it, or another --name")


def generate_cmd(args, set_dir: str, out: str, extra: list[str]) -> tuple[list[str], dict]:
    """The generation command and its environment (the engine's python, the engine on PYTHONPATH)."""
    seeds = ["--seeds", *map(str, args.seeds)]
    if args.engine == "dq":
        py, path = args.engine_python or DQ_PY, args.engine_path or DQ_ENGINE_DIR
        cmd = [py, "-m", "arabic_voxcpm.eval.generate_dq", "--dq-engine-dir", path]
    else:
        py, path = args.engine_python or ENGINE_PY, args.engine_path or ENGINE
        cmd = [py, "-m", "arabic_voxcpm.eval.generate", "--dit", args.dit, "--precision", args.precision]
        cmd += ["--dit-cache", args.dit_cache] if args.dit_cache else []
    cmd += ["--set", set_dir, "--out", out, "--model", args.model, "--mode", args.mode, *seeds, *extra]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([REPO, path] + ([os.environ["PYTHONPATH"]]
                                                                     if os.environ.get("PYTHONPATH") else [])))
    return cmd, env


def baseline_alternatives(name: str, set_dir: str, mode: str) -> list[str]:
    s = os.path.basename(set_dir.rstrip("/"))
    names = [f"{name}-{s}-{mode}", f"{name}-{mode}", f"{name}-{s}", name]
    return [common.run_dir(n) for n in dict.fromkeys(names)
            if os.path.exists(os.path.join(common.run_dir(n), "scores.jsonl"))]


def pending(run: str, set_dir: str) -> bool:
    """Whether scoring has work: wavs without a score row, rows without a metric, unscored ground truth."""
    out = os.path.join(run, "scores.jsonl")
    rows = [json.loads(line) for line in open(out, encoding="utf-8")] if os.path.exists(out) else []
    wav = os.path.join(run, "wav")
    files = {n for n in os.listdir(wav) if common.parse_wav(n)} if os.path.isdir(wav) else set()
    if files - {r["file"] for r in rows} or any("adi_label" not in r and not r.get("empty") for r in rows):
        return True
    gt = os.path.join(set_dir, "gt_scores.jsonl")
    done = {json.loads(line)["id"] for line in open(gt, encoding="utf-8")} if os.path.exists(gt) else set()
    return any(it["gt_audio"] and it["id"] not in done for it in common.items(set_dir))


def main(args, extra: list[str]) -> None:
    from arabic_voxcpm.eval import report, score

    set_dir = os.path.abspath(common.set_dir(args.set))
    run = os.path.abspath(common.run_dir(args.name))
    if os.path.exists(os.path.join(run, "run.json")):
        check_existing(run, wanted(args, set_dir))
        print(f"generated already: {run}")
    else:
        cmd, env = generate_cmd(args, set_dir, run, extra)
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, cwd=REPO, check=True)
    if pending(run, set_dir):
        s = score.Scorer()
        score.score_gt(set_dir, s)
        score.score_run(run, s)
    else:  # metrics newer than the rows (the stop checks) need no model
        cfg, items, _ = score.run_parts(run)
        score.fill_missing(os.path.join(run, "scores.jsonl"), None, timing_of=score.run_timing(cfg, items))
    if args.no_report:
        return
    bases = [alt for b in args.baseline if (alt := baseline_alternatives(b, set_dir, args.mode))]
    for b in args.baseline:
        if not baseline_alternatives(b, set_dir, args.mode):
            print(f"baseline {b}: no scored run, left out")
    md, data = report.build(bases, [run], absolute=True)
    data["command"] = sys.argv
    open(os.path.join(run, "report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(data, open(os.path.join(run, "report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(md)
    print(f"-> {run}/report.md, report.json")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="the run: a name under the runs directory, or a path")
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects", help="directory or repo id")
    ap.add_argument("--engine", choices=["library", "dq"], default="library",
                    help="library: the inference library (DialectTTS); dq: the base inference (dq_engine)")
    ap.add_argument("--dit", default="none", help="library: the distilled LocDiT (path or repo:file); none = original")
    ap.add_argument("--dit-cache", default=None, help="library, original LocDiT: light / medium")
    ap.add_argument("--precision", choices=["bf16", "int8"], default="bf16")
    ap.add_argument("--mode", choices=common.MODES, default="ref")
    ap.add_argument("--set", default="v1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001])
    ap.add_argument("--baseline", nargs="+", default=["base-adapters"], help="baseline run names (resolved per set "
                                                                            "and mode, see above)")
    ap.add_argument("--engine-python", default=None)
    ap.add_argument("--engine-path", default=None, help="put on PYTHONPATH for generation")
    ap.add_argument("--no-report", action="store_true", help="generate and score only")
    return ap


if __name__ == "__main__":
    main(*parser().parse_known_args())
