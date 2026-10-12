"""Generation requests for synthetic training speech: per dialect, real transcripts of that dialect read by
reference voices of its region, by the generator that won the bake-off on the evaluation set (issue #3 / #5).

Texts: training transcripts (select_rows: the evaluation set excluded), 20-140 characters, no fillers or markup; a
scored clip's transcript only if it agrees with Cohere (CER <= 0.15). A text is read at most --max-repeat times (by
different voices), which caps the small dialects. Voices (voices.py): the dialect's own real speakers plus VoxTube voices heard
as its region (ADI-5). Each request: one text, one voice, drawn at random; enough requests for --scale x the target
hours, since the filter (filter.py) drops some.

Output: one request set per generator, in the evaluation set's layout (items.jsonl, voices.json):
  <out>/sets/adapters  -> arabic_voxcpm.eval.generate (the current VoxCPM2 adapters)
  <out>/sets/habibi    -> arabic_voxcpm.synth.habibi_gen
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

# dialect -> (generator, model, target hours). Bake-off on the evaluation set, 2026-10-12: Habibi was more
# intelligible for most groups; the adapters for iraqi / saudi (dialect ID), hejazi (Habibi leaked the reference
# transcript) and najdi (UTMOS 3.0 vs 1.9).
PLAN = {
    "gulf": ("habibi", "UAE", 6), "msa": ("habibi", "Unified:MSA", 3), "libyan": ("habibi", "Unified:LBY", 3),
    "tunisian": ("habibi", "Unified:TUN", 4), "moroccan": ("habibi", "Unified:MAR", 3), "algerian": ("habibi", "ALG", 2),
    "lebanese": ("habibi", "Unified:LEV", 2),
    "hejazi": ("adapters", "hejazi", 8), "najdi": ("adapters", "najdi", 6), "saudi": ("adapters", "saudi", 6),
    "iraqi": ("adapters", "iraqi", 6),
}  # target hours after filtering, sized to what the quality filter leaves of the real data (2026-10-12: gulf 0.6 h,
# msa 0.5, hejazi 0.8, iraqi 1.6, libyan 1.9, saudi 2.0, tunisian 2.4, moroccan 3.2, najdi 3.2, lebanese 4.9,
# algerian 5.2; the others 9.5-23 h)
REGION = {"msa": "MSA", "egyptian": "Egyptian", "sudanese": "Egyptian", "palestinian": "Levantine",
          "lebanese": "Levantine", "syrian": "Levantine", "iraqi": "Gulf", "saudi": "Gulf", "najdi": "Gulf",
          "hejazi": "Gulf", "gulf": "Gulf", "bahraini": "Gulf", "yemeni": "Gulf", "libyan": "Maghrebi",
          "tunisian": "Maghrebi", "algerian": "Maghrebi", "moroccan": "Maghrebi"}
CHARS_PER_S = 10.0  # speaking rate of generated speech (Habibi test batch: 10.0 characters per second)
_BAD = re.compile(r"[#\[\]()\"a-zA-Z{}<>/\\|*_=+~]")
_FILLER = re.compile(r"(^|\s)(ا{2,}|آ+|ء+|ام+|اه+|إم+)(?=\s|$)")


def clean(text: str) -> str | None:
    t = " ".join(text.split())
    if _BAD.search(t) or not 20 <= len(t) <= 140:
        return None
    t = _FILLER.sub(" ", t)
    return " ".join(t.split())


def main(args) -> None:
    from arabic_voxcpm.train.dataset import select_rows

    rng = random.Random(args.seed)
    q = {}
    for line in open(args.quality, encoding="utf-8"):
        r = json.loads(line)
        q[r["uid"]] = r["cer"]
    texts = defaultdict(set)
    for r in select_rows(heldout=args.heldout):
        if q.get(r["uid"], 0.0) <= 0.15 and (t := clean(r["text"])):
            texts[r["dialect"]].add(t)
    voices = [json.loads(l) for l in open(os.path.join(args.out, "voices.jsonl"), encoding="utf-8")]
    for gen in ("adapters", "habibi"):
        set_dir = os.path.join(args.out, "sets", gen)
        os.makedirs(set_dir, exist_ok=True)
        items, used = [], {}
        for d, (g, model, hours) in sorted(PLAN.items()):
            if g != gen:
                continue
            pool = [v for v in voices if v["dialect"] == d or (v["kind"] == "voxtube" and v["adi_label"] == REGION[d])]
            tx = sorted(texts[d])
            rng.shuffle(tx)
            scale = args.scale * (args.habibi_share if gen == "habibi" else 1.0)
            n = min(int(hours * scale * 3600 / (sum(map(len, tx)) / len(tx) / CHARS_PER_S)),
                    args.max_repeat * len(tx))
            print(f"{gen:9}{d:12}{model:14}{len(tx):6} texts {len(pool):4} voices "
                  f"({sum(v['kind'] == 'real' for v in pool)} real) -> {n} requests")
            for k in range(n):
                v = rng.choice(pool)
                used[v["id"]] = {"id": v["id"], "audio": os.path.join(args.out, v["audio"]), "text": v["text"]}
                it = {"id": f"{d}-{k:05d}", "group": d, "dialect": d, "text": tx[k % len(tx)], "voice": v["id"],
                      "speaker": v["speaker"]}
                it["adapter" if gen == "adapters" else "habibi"] = model
                items.append(it)
        with open(os.path.join(set_dir, "items.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        json.dump(list(used.values()), open(os.path.join(set_dir, "voices.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=0)
        print(f"{gen}: {len(items)} requests, {len(used)} voices -> {set_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/data/arabic/synth")
    ap.add_argument("--quality", default="/workspace/data/arabic/quality/scores.jsonl")
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v1/heldout.json")
    ap.add_argument("--scale", type=float, default=1.6, help="requests for this many times the target hours")
    ap.add_argument("--habibi-share", type=float, default=0.5,
                    help="Habibi runs ~7x real time on the A10: half its requests keep it near 3.5 h")
    ap.add_argument("--max-repeat", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
