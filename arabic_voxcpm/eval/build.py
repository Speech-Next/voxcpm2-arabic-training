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
_LATIN = re.compile(r"[A-Za-z]")


def text_ok(t: str) -> bool:
    return 20 <= len(t) <= 160 and len(t.split()) >= 3 and not _LATIN.search(t)


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
                          "gt_audio": os.path.join(lah_audio, r["uid"] + ".flac"), "speaker": r["speaker"]})

    for g in LAHGTNA_TEXT_ONLY:
        rows = [r for r in lah.values() if r["dialect"] == g and text_ok(r["text"]) and r.get("asr_cer", 1) <= 0.1]
        for r in pick(rows, PER_GROUP, rng):
            items.append({"group": g, "source": "lahgtna", "text": r["text"], "gt_uid": r["uid"], "gt_audio": None,
                          "speaker": r["speaker"]})

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
                          "gt_audio": os.path.join(sada_audio, r["uid"] + ".flac"), "speaker": r["speaker"]})
            heldout_speakers.add(r["speaker"])

    by_group = defaultdict(list)
    for v in voices:
        by_group[v["group"]].append(v["id"])
    pool = [v["id"] for v in voices]
    for k, it in enumerate(items):
        own = by_group.get(it["group"])
        it["id"] = f"{it['group']}-{k:04d}"
        it["voice"] = own[k % len(own)] if own else pool[k % len(pool)]
        it["adapter"] = CURRENT_ADAPTER.get(it["group"])
        heldout_uids.add(it["gt_uid"])
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
    main(ap.parse_args())
