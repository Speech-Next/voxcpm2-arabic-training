"""A development subset of a built evaluation set, for choosing between checkpoints of one run before the evaluation.

    python -m arabic_voxcpm.eval.devset --base /workspace/data/arabic/eval/v2 --out /workspace/data/arabic/eval/v2dev

The first --per-group items of every group (in the set's order), their voices (the base set's files, linked), the
dialect map and the base set's ground-truth scores of those items. Generate it with a seed the evaluation does not use
(run.py --seeds 2002), so the choice and the final #2 numbers come from different samples of the same texts.
"""

from __future__ import annotations

import argparse
import json
import os


def build(base: str, out: str, per_group: int = 10) -> list[dict]:
    items, count = [], {}
    for it in map(json.loads, open(os.path.join(base, "items.jsonl"), encoding="utf-8")):
        if count.get(it["group"], 0) < per_group:
            count[it["group"]] = count.get(it["group"], 0) + 1
            items.append(it)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "items.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    used, ids = {it["voice"] for it in items}, {it["id"] for it in items}
    voices = json.load(open(os.path.join(base, "voices.json"), encoding="utf-8"))
    json.dump([v for v in voices if v["id"] in used], open(os.path.join(out, "voices.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    gt = os.path.join(base, "gt_scores.jsonl")
    if os.path.exists(gt):
        with open(os.path.join(out, "gt_scores.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(line for line in open(gt, encoding="utf-8") if json.loads(line)["id"] in ids)
    for name in ("voices", "dialect_map.json"):
        src, dst = os.path.join(os.path.abspath(base), name), os.path.join(out, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="/workspace/data/arabic/eval/v2")
    ap.add_argument("--out", default="/workspace/data/arabic/eval/v2dev")
    ap.add_argument("--per-group", type=int, default=10)
    a = ap.parse_args()
    print(len(build(a.base, a.out, a.per_group)), "items ->", a.out)
