"""Build the held-out Arabic evaluation set (issue #2).

    python -m arabic_voxcpm.eval.build --out /workspace/data/arabic/eval/v1

Groups (dialects) and where their texts come from:
- lahgtna dialects with held-out speakers (bh dz eg lb ly ps sd sy tn ye): the corpus's held-out evaluation
  (stage_a_eval), texts and reference clips of speakers that never appear in training;
- lahgtna dialects without held-out speakers (iq ma sa): texts held out from training (their speakers stay in it);
- SADA (najdi hejazi khaliji msa): the corpus's test split.

Per group, PER_GROUP texts (20-160 characters, >= 3 words, Arabic script, round-robin over speakers). Voices: up to
VOICES_PER_GROUP held-out speakers per group, each by a 4-10 s clip that is not an evaluation text; an item clones a
voice of its own group when there is one, otherwise one of the whole pool. Also written: heldout.json, the speakers,
clip ids and normalized texts that training must exclude.

v2 (--extend v1 --out .../v2 --per-group 40; --dry-run prints the counts only): v1's items, voices and seeds kept as
they are, each group brought to --per-group with texts no training run can have seen (``extend``). Training excluded
v1's heldout.json (every entry point reads it through train.dataset.select_rows: test split, held-out speakers, clip
ids and texts out), so new texts come from clips outside that set and their text must not equal any trainable clip's
or a synthetic clip's (normalized):
- lahgtna held-out groups: more clips of the held-out speakers (all their clips are out), with a recording kept;
- SADA groups: more test-split clips;
- iq ma sa: their speakers trained, so every other lahgtna clip of theirs counts as seen (all but one also pass
  select_rows' duration and rate filters). sa takes SADA test-split clips of Saudi speakers (sa_unknown, shamali:
  the "saudi" dialect in training; --no-sada-saudi: none); iq and ma have no unseen source and stay at 30.
Written: items.jsonl (gt_duration_s on every item), voices, dialect_map.json, heldout.json (v1's plus the new clips,
speakers and texts: future training must exclude it), gt_scores.jsonl (v1's rows: only new items to score),
build.json (counts and reasons).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict

from arabic_voxcpm.data import manifest
from arabic_voxcpm.eval import common
from arabic_voxcpm.eval.metrics import normalize_ar

PER_GROUP = 30
VOICES_PER_GROUP = 2
# the current model's adapter for each group (the baseline); None = the base model
CURRENT_ADAPTER = {"eg": "egyptian", "ps": "palestinian", "lb": "jordanian", "sy": "jordanian", "iq": "iraqi",
                   "sa": "saudi", "najdi": "najdi", "hejazi": "hejazi", "khaliji": "emirati", "bh": "qatari"}
# the tagged model's dialect for each group (generate.py --adapter-map dialect_map.json)
DIALECT_OF_GROUP = {"eg": "egyptian", "ps": "palestinian", "lb": "lebanese", "sy": "syrian", "iq": "iraqi", "sa": "saudi",
                    "najdi": "najdi", "hejazi": "hejazi", "khaliji": "gulf", "bh": "bahraini", "msa": "msa",
                    "sd": "sudanese", "ye": "yemeni", "dz": "algerian", "ma": "moroccan", "tn": "tunisian", "ly": "libyan"}
LAHGTNA_HELDOUT = ["bh", "dz", "eg", "lb", "ly", "ps", "sd", "sy", "tn", "ye"]
LAHGTNA_TEXT_ONLY = ["iq", "ma", "sa"]
SADA_GROUPS = ["najdi", "hejazi", "khaliji", "msa"]
SADA_SAUDI = ["sa_unknown", "shamali"]  # SADA labels of the training dialect "saudi" (train.dialects)
_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9\u0660-\u0669\u06f0-\u06f9]")


def text_ok(t: str) -> bool:
    """20-160 characters, >= 3 words, Arabic script only (digits are a text normalization question, issue #4)."""
    return 20 <= len(t) <= 160 and len(t.split()) >= 3 and not _LATIN_OR_DIGIT.search(t)


def pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n rows, round-robin over speakers (each speaker's rows shuffled)."""
    by = defaultdict(list)
    for r in pool:
        by[r["speaker"]].append(r)
    spk = sorted(by)
    rng.shuffle(spk)
    for s in spk:
        rng.shuffle(by[s])
    out = []
    while len(out) < n and any(by[s] for s in spk):
        for s in spk:
            if by[s] and len(out) < n:
                out.append(by[s].pop())
    return out


def assign(items: list[dict], voices: list[dict], start: int = 0) -> None:
    """Ids, voices and baseline adapters of items numbered from ``start`` (v1's order: a voice of the item's own
    group when it has one, otherwise one of the whole pool)."""
    by_group = defaultdict(list)
    for v in voices:
        by_group[v["group"]].append(v["id"])
    pool = [v["id"] for v in voices]
    for k, it in enumerate(items, start):
        own = by_group.get(it["group"])
        it["id"] = f"{it['group']}-{k:04d}"
        it["voice"] = own[k % len(own)] if own else pool[k % len(pool)]
        it["adapter"] = CURRENT_ADAPTER.get(it["group"])


def heldout_sets(held: dict) -> dict:
    return {k: set(held[k]) for k in ("speakers", "uids", "texts")}


def trainable(r: dict, held: dict) -> bool:
    """Whether a training run excluding ``held`` (heldout_sets) could have used clip r. select_rows' other filters
    (dialect, duration, rate, digits) are left out: this over-counts, the safe side for "never seen"."""
    return not (r.get("split") == "test" or r["speaker"] in held["speakers"] or r["uid"] in held["uids"]
                or normalize_ar(r["text"] or "") in held["texts"])


def seen_texts(rows: list[dict], held: dict, extra=()) -> set[str]:
    """Normalized texts a training run excluding ``held`` may have seen: every trainable clip's, and ``extra`` (the
    synthetic corpus: spoken training texts)."""
    return {normalize_ar(r["text"]) for r in rows if r["text"] and trainable(r, held)} | {normalize_ar(t) for t in extra}


def extend(base: list[dict], held: dict, lah: list[dict], sada: list[dict], seen: set[str], per_group: int,
           rng: random.Random, has_audio, sada_saudi: bool = True) -> tuple[list[dict], dict]:
    """New items (no ids yet) that bring every group of ``base`` to per_group texts that no training run saw, and per
    group {"base", "pool", "new", "why"} (module docstring). has_audio(corpus, uid): a recording is kept."""
    have = defaultdict(int)
    for it in base:
        have[it["group"]] += 1
    taken = {normalize_ar(it["text"]) for it in base} | set(held["texts"])

    def ok(r, corpus):
        return (r["uid"] not in held["uids"] and text_ok(r["text"]) and has_audio(corpus, r["uid"])
                and normalize_ar(r["text"]) not in seen)

    pools = {}
    for g in LAHGTNA_HELDOUT:
        pools[g] = [("lahgtna", r) for r in lah if r["dialect"] == g and r["speaker"] in held["speakers"] and ok(r, "lahgtna")]
    sada_test = [r for r in sada if r.get("split") == "test" and 2 <= r["duration_s"] <= 15 and ok(r, "sada")]
    for g in SADA_GROUPS:
        pools[g] = [("sada", r) for r in sada_test if r["dialect"] == g]
    why = {g: "training speakers: every other clip of theirs counts as seen" for g in LAHGTNA_TEXT_ONLY}
    if sada_saudi:
        pools["sa"] = [("sada", r) for r in sada_test if r["dialect"] in SADA_SAUDI]
        why["sa"] = "SADA test split, Saudi speakers (sa_unknown, shamali)"
    out, stats = [], {}
    for g in sorted(have):
        cands, seen_here = [], set()
        for corpus, r in sorted(pools.get(g, []), key=lambda x: x[1]["uid"]):
            n = normalize_ar(r["text"])
            if n not in taken and n not in seen_here:
                seen_here.add(n)
                cands.append(dict(r, corpus=corpus))
        need = max(per_group - have[g], 0)
        chosen = pick(cands, need, rng)
        stats[g] = {"base": have[g], "pool": len(cands), "new": len(chosen),
                    "why": why.get(g, "held-out speakers" if g in LAHGTNA_HELDOUT else "SADA test split")}
        if len(chosen) < need:
            stats[g]["why"] += f"; {len(chosen)} of {need} found"
        for r in chosen:
            out.append({"group": g, "source": r["corpus"], "text": r["text"], "gt_uid": r["uid"],
                        "gt_audio": has_audio(r["corpus"], r["uid"], path=True), "speaker": r["speaker"],
                        "gt_duration_s": r["duration_s"]})
    return out, stats


def audio_index(root: str = manifest.ROOT):
    """has_audio(corpus, uid, path=False): whether the clip's recording is kept (eval_audio/), or its path."""
    kept = {c: {f[:-5] for f in os.listdir(os.path.join(root, c, "eval_audio")) if f.endswith(".flac")}
            for c in ("lahgtna", "sada") if os.path.isdir(os.path.join(root, c, "eval_audio"))}

    def has_audio(corpus, uid, path=False):
        if path:
            return os.path.join(root, corpus, "eval_audio", uid + ".flac") if uid in kept.get(corpus, ()) else None
        return uid in kept.get(corpus, ())

    return has_audio


def main_extend(args) -> None:
    base_dir = common.set_dir(args.extend)
    if os.path.realpath(args.out) == os.path.realpath(base_dir):
        raise SystemExit(f"--out {args.out} is the set being extended: it stays as it is")
    base = common.items(base_dir)
    held_raw = json.load(open(os.path.join(base_dir, "heldout.json"), encoding="utf-8"))
    held = heldout_sets(held_raw)
    lah, sada = manifest.rows("lahgtna"), manifest.rows("sada")
    if not lah or not sada:
        raise SystemExit("the lahgtna and SADA manifests are needed (their transcripts)")
    seen = seen_texts(lah + sada, held, [r["text"] for r in manifest.rows("synth")])
    new, stats = extend(base, held, lah, sada, seen, args.per_group, random.Random(args.seed), audio_index(),
                        args.sada_saudi)
    for g, st in stats.items():
        print(f"{g:8} {st['base']:3} + {st['new']:2} = {st['base'] + st['new']:3}   pool {st['pool']:5}   {st['why']}")
    print(f"{len(base)} + {len(new)} = {len(base) + len(new)} items")
    if args.dry_run:
        return
    voices = common.voices(base_dir)
    assign(new, voices, start=len(base))
    dur = {r["uid"]: r["duration_s"] for r in lah + sada}
    for it in base:
        it.setdefault("gt_duration_s", dur.get(it["gt_uid"]))
    out = args.out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "items.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in base + new)
    json.dump(voices, open(os.path.join(out, "voices.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    shutil.copytree(os.path.join(base_dir, "voices"), os.path.join(out, "voices"), dirs_exist_ok=True)
    for name in ("dialect_map.json", "gt_scores.jsonl"):
        if os.path.exists(os.path.join(base_dir, name)):
            shutil.copy(os.path.join(base_dir, name), os.path.join(out, name))
    json.dump({"speakers": sorted(held["speakers"] | {it["speaker"] for it in new if it["source"] == "sada"}),
               "uids": sorted(held["uids"] | {it["gt_uid"] for it in new}),
               "texts": sorted(held["texts"] | {normalize_ar(it["text"]) for it in new})},
              open(os.path.join(out, "heldout.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump({"base": os.path.abspath(base_dir), "per_group": args.per_group, "seed": args.seed,
               "sada_saudi": args.sada_saudi, "groups": stats}, open(os.path.join(out, "build.json"), "w"), indent=1)
    print(f"-> {out}")


def main(args) -> None:
    from huggingface_hub import hf_hub_download

    rng = random.Random(args.seed)
    lah = {r["uid"]: r for r in manifest.rows("lahgtna")}
    sada = manifest.rows("sada")
    stage = [json.loads(l) for l in open(hf_hub_download("Rabe3/lahgtna-arabic-tts-24khz", "manifests/stage_a_eval.jsonl",
                                                          repo_type="dataset"), encoding="utf-8")]
    lah_audio = os.path.join(manifest.ROOT, "lahgtna", "eval_audio")
    sada_audio = os.path.join(manifest.ROOT, "sada", "eval_audio")
    os.makedirs(os.path.join(args.out, "voices"), exist_ok=True)

    items, voices = [], []
    heldout_speakers = {r["speaker_id"] for r in stage}
    heldout_uids: set[str] = set()

    def add_voice(group, row, audio_dir):
        vid = f"{group}_{len([v for v in voices if v['group'] == group])}"
        dst = os.path.join(args.out, "voices", vid + ".flac")
        shutil.copy(os.path.join(audio_dir, row["uid"] + ".flac"), dst)
        voices.append({"id": vid, "group": group, "speaker": row["speaker"], "uid": row["uid"], "text": row["text"],
                       "audio": os.path.relpath(dst, args.out)})
        heldout_uids.add(row["uid"])

    for g in LAHGTNA_HELDOUT:
        rows = [dict(lah[r["uid"]], ref_uid=r["ref_uid"]) for r in stage if r["dialect"] == g and r["uid"] in lah]
        chosen = pick([r for r in rows if text_ok(r["text"]) and os.path.exists(os.path.join(lah_audio, r["uid"] + ".flac"))],
                      PER_GROUP, rng)
        taken = {r["uid"] for r in chosen}
        refs = sorted({r["ref_uid"] for r in rows} - taken)
        cands = [lah[u] for u in refs if u in lah and 4 <= lah[u]["duration_s"] <= 10
                 and os.path.exists(os.path.join(lah_audio, u + ".flac"))]
        seen = set()
        for r in sorted(cands, key=lambda r: -r.get("squim_pesq", 0)):  # best clip of the best speakers
            if r["speaker"] not in seen and len(seen) < VOICES_PER_GROUP:
                seen.add(r["speaker"])
                add_voice(g, r, lah_audio)
        for r in chosen:
            items.append({"group": g, "source": "lahgtna", "text": r["text"], "gt_uid": r["uid"],
                          "gt_audio": os.path.join(lah_audio, r["uid"] + ".flac"), "speaker": r["speaker"],
                          "gt_duration_s": r["duration_s"]})

    for g in LAHGTNA_TEXT_ONLY:
        rows = [r for r in lah.values() if r["dialect"] == g and text_ok(r["text"]) and r.get("asr_cer", 1) <= 0.1]
        for r in pick(rows, PER_GROUP, rng):
            items.append({"group": g, "source": "lahgtna", "text": r["text"], "gt_uid": r["uid"], "gt_audio": None,
                          "speaker": r["speaker"], "gt_duration_s": r["duration_s"]})

    for g in SADA_GROUPS:
        rows = [r for r in sada if r.get("split") == "test" and r["dialect"] == g]
        chosen = pick([r for r in rows if text_ok(r["text"]) and 2 <= r["duration_s"] <= 15], PER_GROUP, rng)
        taken = {r["uid"] for r in chosen}
        seen = set()
        for r in sorted((r for r in rows if r["uid"] not in taken and 4 <= r["duration_s"] <= 10),
                        key=lambda r: (r["speaker"], r["uid"])):
            if r["speaker"] not in seen and len(seen) < VOICES_PER_GROUP:
                seen.add(r["speaker"])
                add_voice(g, r, sada_audio)
                heldout_speakers.add(r["speaker"])
        for r in chosen:
            items.append({"group": g, "source": "sada", "text": r["text"], "gt_uid": r["uid"],
                          "gt_audio": os.path.join(sada_audio, r["uid"] + ".flac"), "speaker": r["speaker"],
                          "gt_duration_s": r["duration_s"]})
            heldout_speakers.add(r["speaker"])

    assign(items, voices)
    heldout_uids.update(it["gt_uid"] for it in items)
    by_group = defaultdict(list)
    for v in voices:
        by_group[v["group"]].append(v["id"])
    with open(os.path.join(args.out, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump(voices, open(os.path.join(args.out, "voices.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(DIALECT_OF_GROUP, open(os.path.join(args.out, "dialect_map.json"), "w"), indent=1)
    json.dump({"speakers": sorted(heldout_speakers), "uids": sorted(heldout_uids),
               "texts": sorted({normalize_ar(it["text"]) for it in items})},
              open(os.path.join(args.out, "heldout.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    counts = defaultdict(int)
    for it in items:
        counts[it["group"]] += 1
    print(f"{len(items)} items, {len(voices)} voices -> {args.out}")
    print(" ".join(f"{g}:{n}/{len(by_group.get(g, []))}v" for g, n in sorted(counts.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/workspace/data/arabic/eval/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extend", help="a built set (v1) to extend into --out (v2)")
    ap.add_argument("--per-group", type=int, default=40, help="with --extend")
    ap.add_argument("--dry-run", action="store_true", help="with --extend: the counts per group only")
    ap.add_argument("--no-sada-saudi", dest="sada_saudi", action="store_false",
                    help="with --extend: no SADA texts for the lahgtna group sa")
    a = ap.parse_args()
    main_extend(a) if a.extend else main(a)
