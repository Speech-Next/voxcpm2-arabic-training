"""The number-heavy evaluation set (issue #4): sets/numbers_v1.jsonl spoken with the main set's voices.

    python -m arabic_voxcpm.eval.numbers --base /workspace/data/arabic/eval/v1 --out /workspace/data/arabic/eval/numbers_v1

The items keep their digits (generate.py --no-normalize passes them to the model as they are). The scorer spells the
numbers out on both sides, in the item text and in the ASR transcript, with the inference library's normalizer
(``spell_numbers``). So an ASR that writes digits is not counted as an error, and a model that misreads "3499" is.
A correct reading in other words than the normalizer's ("عشرة ونص" for 10:30, where it writes "عشرة وثلاثين دقيقة")
still counts as errors: without normalization the model is held to the normalizer's wording.
"""

from __future__ import annotations

import argparse
import json
import os

from arabic_voxcpm.eval.build import CURRENT_ADAPTER

SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sets", "numbers_v1.jsonl")


def build(base: str, out: str, src: str = SET) -> list[dict]:
    voices = json.load(open(os.path.join(base, "voices.json"), encoding="utf-8"))
    first = {}  # group -> the voice of its first item in the main set (text-only groups borrow a voice too)
    for it in map(json.loads, open(os.path.join(base, "items.jsonl"), encoding="utf-8")):
        first.setdefault(it["group"], it["voice"])
    items = []
    for it in map(json.loads, open(src, encoding="utf-8")):
        items.append({**it, "voice": first[it["group"]], "adapter": CURRENT_ADAPTER.get(it["group"]), "gt_audio": None,
                      "numbers": True})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "items.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    used = {it["voice"] for it in items}
    json.dump([v for v in voices if v["id"] in used], open(os.path.join(out, "voices.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    link = os.path.join(out, "voices")
    if not os.path.exists(link):
        os.symlink(os.path.join(os.path.abspath(base), "voices"), link)
    for name in ("dialect_map.json",):
        if os.path.exists(os.path.join(base, name)) and not os.path.exists(os.path.join(out, name)):
            os.symlink(os.path.join(os.path.abspath(base), name), os.path.join(out, name))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="/workspace/data/arabic/eval/v1")
    ap.add_argument("--out", default="/workspace/data/arabic/eval/numbers_v1")
    a = ap.parse_args()
    print(len(build(a.base, a.out)), "items ->", a.out)
