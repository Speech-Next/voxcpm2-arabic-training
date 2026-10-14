"""A versioned training set (#3): the clips one training configuration trains on, one row per segment, with where it
comes from, its license, the filters it passed and where its latents are.

    python -m arabic_voxcpm.data.release --name teacher-v2-train --corpora lahgtna,sada,synth \\
        --heldout /workspace/data/arabic/eval/v1/heldout.json --quality /workspace/data/arabic/quality/scores.jsonl

The selection is train.py's own: select_rows, then split_rows (validation: SADA's validation split plus a --val-frac
sample, at most --val-max clips). Writes <root>/datasets/<name>/
  manifest.jsonl  uid, split (train / val), source, source_file, license, speaker (as training groups them), dialect,
                  corpus_dialect, duration_s, frames, text, filters, scores (Cohere CER, DNSMOS), latents (latent shard,
                  relative to the data root)
  stats.json      clips and hours per split, source and dialect; the filter funnel; pass rates of the quality scores
  README.md       the dataset card
The latents themselves stay in <root>/<corpus>/latents/ (lahgtna.py / sada.py --restore <manifest.jsonl> re-make the
real corpora's from their sources). --check: every row's latents are in its shard with the manifest's frame count.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict

from arabic_voxcpm.data import manifest
from arabic_voxcpm.eval.metrics import normalize_ar
from arabic_voxcpm.train import dataset as D
from arabic_voxcpm.train import dialects
from arabic_voxcpm.train.train import split_rows

SOURCES = {
    "lahgtna": {"repo": "Rabe3/lahgtna-arabic-tts-24khz", "revision": "f3582beced75900456905e509721b9d6457403f3",
                "license": "unspecified (no license declared on the Hub)"},
    "sada": {"repo": "MohamedRashad/SADA22", "revision": "094fe2c0fe4b549a4f34349e6e0622e7c7273c2d",
             "metadata": "khaledalganem/sada2022@refs/convert/parquet (e5692a7639f87153ca0001ae1db83aa5bf18b09f)",
             "license": "CC-BY-NC-SA-4.0"},
    "synth": {"repo": "local: arabic_voxcpm.synth (scripts/run_synth.sh)",
              "license": "synthetic: generated from corpus transcripts and reference voices (see the card)"},
}
SYNTH_FILTER = "synth_filter(cohere_cer<=0.08,dnsmos_ovrl>=3.0,rate_5_25_cps)"  # scripts/run_synth.sh, synth/filter.py
GENERATORS = {"adapters": "VoxCPM2-Arabic-Dialects per-dialect adapters (arabic_voxcpm.eval.generate)",
              "habibi": "Habibi-TTS (arabic_voxcpm.synth.habibi_gen)"}


def hours(frames: int) -> float:
    return frames / D.FPS / 3600


def scan(corpus: str, root: str) -> dict[str, str]:
    """uid -> latent shard stem (manifest/<stem>.jsonl <-> latents/<stem>.pt)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, corpus, "manifest", "*.jsonl"))):
        stem = os.path.basename(path)[:-6]
        for line in open(path, encoding="utf-8"):
            out[json.loads(line)["uid"]] = stem
    return out


def funnel(corpus: str, heldout: dict, scores: dict | None, max_cer: float, min_dnsmos: float,
           root: str) -> tuple[list[dict], set[str]]:
    """select_rows' filters one at a time, in its order: clips and hours left after each; the uids left."""
    spk, uids, texts = set(heldout["speakers"]), set(heldout["uids"]), set(heldout["texts"])
    steps = [
        ("known_dialect", lambda r, s: dialects.of_row(r) is not None),
        ("not_test_split", lambda r, s: r.get("split") != "test"),
        ("not_heldout_speaker_or_clip", lambda r, s: r["speaker"] not in spk and r["uid"] not in uids),
        ("duration_1_20s", lambda r, s: 1.0 <= s <= 20.0),
        ("text_nonempty", lambda r, s: bool(r["text"])),
    ]
    if scores is not None and corpus != "synth":
        steps += [
            ("quality_candidate(<=3h/speaker,<=60h/dialect)", lambda r, s: r["uid"] in scores),
            (f"cohere_cer<={max_cer:g}", lambda r, s: scores[r["uid"]]["cer"] <= max_cer),
            (f"dnsmos_ovrl>={min_dnsmos:g}", lambda r, s: scores[r["uid"]]["dnsmos_ovrl"] >= min_dnsmos),
        ]
    steps += [
        ("rate_2_35_cps", lambda r, s: 2.0 <= len(r["text"]) / s <= 35.0),
        ("not_heldout_text", lambda r, s: normalize_ar(r["text"]) not in texts),
        ("no_digits", lambda r, s: not D._DIGIT.search(r["text"])),
    ]
    rows = manifest.rows(corpus, root)
    out = [{"step": "manifest", "clips": len(rows), "hours": round(hours(sum(r["frames"] for r in rows)), 2)}]
    for name, ok in steps:
        rows = [r for r in rows if ok(r, r["frames"] / D.FPS)]
        out.append({"step": name, "clips": len(rows), "hours": round(hours(sum(r["frames"] for r in rows)), 2)})
    return out, {r["uid"] for r in rows}


def quality_stats(path: str, max_cer: float, min_dnsmos: float) -> dict:
    by = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        q = json.loads(line)
        by[(q["source"], q["dialect"])].append(q)
    out = {}
    for (src, d), qs in sorted(by.items()):
        def agg(sel):
            return {"clips": len(sel), "hours": round(sum(q["sec"] for q in sel) / 3600, 2)}
        out[f"{src}/{d}"] = {
            "scored": agg(qs),
            "cer_ok": agg([q for q in qs if q["cer"] <= max_cer]),
            "dnsmos_ok": agg([q for q in qs if q["dnsmos_ovrl"] >= min_dnsmos]),
            "both_ok": agg([q for q in qs if q["cer"] <= max_cer and q["dnsmos_ovrl"] >= min_dnsmos]),
            "median_cer": round(statistics.median(q["cer"] for q in qs), 4),
            "median_dnsmos_ovrl": round(statistics.median(q["dnsmos_ovrl"] for q in qs), 3),
            "speakers": len({q["speaker"] for q in qs}),
        }
    tot = [q for qs in by.values() for q in qs]
    out["ALL"] = {k: {"clips": sum(v[k]["clips"] for v in out.values()),
                      "hours": round(sum(v[k]["hours"] for v in out.values()), 2)}
                  for k in ("scored", "cer_ok", "dnsmos_ok", "both_ok")}
    out["ALL"]["median_cer"] = round(statistics.median(q["cer"] for q in tot), 4)
    out["ALL"]["median_dnsmos_ovrl"] = round(statistics.median(q["dnsmos_ovrl"] for q in tot), 3)
    return out


def source_file(r: dict, stem: str, n_files: dict) -> str:
    if r["source"] == "lahgtna":
        return f"{SOURCES['lahgtna']['repo']}:data/{stem}.parquet"
    if r["source"] == "sada":
        split, k = stem.rsplit("-", 1)
        return f"{SOURCES['sada']['repo']}:data/{split}-{k}-of-{n_files[split]:05d}.parquet"
    return f"synth/gen/{r['gen']}"


def build(args) -> None:
    root, corpora = args.root, tuple(args.corpora.split(","))
    out_dir = os.path.join(root, "datasets", args.name)
    os.makedirs(out_dir, exist_ok=True)
    rows = D.select_rows(corpora, heldout=args.heldout, quality=args.quality, max_cer=args.max_cer,
                         min_dnsmos=args.min_dnsmos)
    train, val = split_rows(rows, args.val_frac, args.seed)
    val = val[: args.val_max]
    split_of = {r["uid"]: "train" for r in train} | {r["uid"]: "val" for r in val}

    held = json.load(open(args.heldout, encoding="utf-8"))
    scores = None
    if args.quality:
        scores = {q["uid"]: q for q in map(json.loads, open(args.quality, encoding="utf-8"))}
    funnels, left = {}, set()
    for c in corpora:
        funnels[c], kept = funnel(c, held, scores, args.max_cer, args.min_dnsmos, root)
        left |= kept
    assert left == {r["uid"] for r in rows}, "the funnel and select_rows disagree"

    stems = {c: scan(c, root) for c in corpora}
    raw = {c: {r["uid"]: r for r in manifest.rows(c, root)} for c in corpora}
    n_files = defaultdict(int)
    for stem in set(stems.get("sada", {}).values()):
        n_files[stem.rsplit("-", 1)[0]] += 1
    voices = {}
    vpath = os.path.join(root, "synth", "voices.jsonl")
    if "synth" in corpora and os.path.exists(vpath):
        voices = {v["id"]: v for v in map(json.loads, open(vpath, encoding="utf-8"))}

    real_filters = [f["step"] for f in funnels.get("lahgtna", funnels.get("sada", []))[1:]]
    synth_filters = [SYNTH_FILTER] + [f["step"] for f in funnels.get("synth", [])[1:]]
    out_rows = []
    for r in train + val:
        m, c = raw[r["source"]][r["uid"]], r["source"]
        stem = stems[c][r["uid"]]
        o = {"uid": r["uid"], "split": split_of[r["uid"]], "source": c,
             "source_file": source_file(m, stem, n_files), "license": SOURCES[c]["license"],
             "speaker": r["speaker"], "dialect": r["dialect"], "corpus_dialect": m["dialect"],
             "duration_s": round(r["frames"] / D.FPS, 2), "frames": r["frames"], "text": r["text"]}
        if c == "synth":
            v = voices.get(m.get("voice"), {})
            kind = v.get("kind", "?")
            o["license"] = ("synthetic; voice: VoxTube (CC-BY-NC-SA-4.0)" if kind == "voxtube" else
                            "synthetic; voice: a corpus speaker's clip") + "; text: a corpus transcript"
            o["generator"] = GENERATORS[m["gen"].rsplit("-", 1)[0]]
            o["voice"] = m.get("voice")
            o["filters"] = synth_filters
            o["scores"] = {"cohere_cer": m.get("cer"), "dnsmos_ovrl": round(m.get("dnsmos_ovrl", 0.0), 4)}
        else:
            if c == "sada":
                o["sada_split"] = m.get("split")
            o["filters"] = real_filters
            q = (scores or {}).get(r["uid"])
            o["scores"] = {"cohere_cer": round(q["cer"], 4), "dnsmos_ovrl": q["dnsmos_ovrl"]} if q else {}
        o["latents"] = f"{c}/latents/{stem}.pt"
        out_rows.append(o)
    tmp = os.path.join(out_dir, "manifest.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for o in out_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(out_dir, "manifest.jsonl"))  # the restore may be reading it

    def table(sel):
        t = defaultdict(lambda: defaultdict(lambda: {"clips": 0, "hours": 0.0}))
        for o in sel:
            for d in (o["dialect"], "ALL"):
                for s in (o["source"], "real" if o["source"] != "synth" else None, "total"):
                    if s:
                        t[d][s]["clips"] += 1
                        t[d][s]["hours"] += o["duration_s"] / 3600
        return {d: {s: {"clips": v["clips"], "hours": round(v["hours"], 2)} for s, v in sorted(ss.items())}
                for d, ss in sorted(t.items())}

    spk_h = defaultdict(float)
    for o in out_rows:
        spk_h[o["speaker"]] += o["duration_s"] / 3600
    real_spk = {s: h for s, h in spk_h.items() if not s.startswith("synth:")}
    try:
        commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    stats = {
        "name": args.name, "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "code_commit": commit,
        "selection": {"corpora": list(corpora), "heldout": args.heldout, "quality": args.quality,
                      "max_cer": args.max_cer, "min_dnsmos": args.min_dnsmos, "val_frac": args.val_frac,
                      "val_max": args.val_max, "seed": args.seed, "min_s": 1.0, "max_s": 20.0,
                      "quality_candidates": "data/quality.py: per speaker <= 3 h, per dialect <= 60 h, round-robin "
                                            "over speakers (seed 0), evaluation v1 held out"},
        "sources": {c: SOURCES[c] for c in corpora},
        "splits": {s: {"clips": sum(1 for o in out_rows if o["split"] == s),
                       "hours": round(sum(o["duration_s"] for o in out_rows if o["split"] == s) / 3600, 2)}
                   for s in ("train", "val")},
        "train_by_dialect": table([o for o in out_rows if o["split"] == "train"]),
        "all_by_dialect": table(out_rows),
        "speakers": {"real": len(real_spk), "synth_voices": len(spk_h) - len(real_spk),
                     "max_real_speaker_hours": round(max(real_spk.values()), 2)},
        "funnel": funnels,
        "quality_scores": quality_stats(args.quality, args.max_cer, args.min_dnsmos) if args.quality else None,
        "latent_bytes": sum(o["frames"] for o in out_rows) * 64 * 2,
    }
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats["splits"]), f"-> {out_dir}", flush=True)


def check(args) -> None:
    """Every manifest row's latents in its own shard, with the manifest's frame count."""
    import torch

    path = os.path.join(args.root, "datasets", args.name, "manifest.jsonl")
    by = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        by[o["latents"]].append(o)
    missing, bad, n = 0, 0, 0
    for shard, rows in sorted(by.items()):
        p = os.path.join(args.root, shard)
        if not os.path.exists(p):
            missing += len(rows)
            continue
        d = torch.load(p, map_location="cpu", weights_only=True)
        assert d["latents"].dtype == torch.float16 and d["latents"].shape[1] == 64, shard
        idx = {u: i for i, u in enumerate(d["uids"])}
        off = d["offsets"].tolist()
        for o in rows:
            i = idx.get(o["uid"])
            if i is None:
                missing += 1
            elif off[i + 1] - off[i] != o["frames"]:
                bad += 1
            else:
                n += 1
    print(f"{n} clips ok, {missing} without latents, {bad} with another frame count, {len(by)} shards", flush=True)
    if missing or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=manifest.ROOT)
    ap.add_argument("--corpora", default="lahgtna,sada,synth")
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v1/heldout.json")
    ap.add_argument("--quality")
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("--min-dnsmos", type=float, default=3.0)
    ap.add_argument("--val-frac", type=float, default=0.004)
    ap.add_argument("--val-max", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="only check the latents of an existing release")
    a = ap.parse_args()
    check(a) if a.check else build(a)
