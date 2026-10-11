"""Read the corpus manifests (manifest/<shard>.jsonl, one row per clip) written by the pipelines."""

from __future__ import annotations

import glob
import json
import os

ROOT = "/workspace/data/arabic"
CORPORA = ("lahgtna", "sada")


def rows(corpus: str, root: str = ROOT) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, corpus, "manifest", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            r.setdefault("source", corpus)
            out.append(r)
    return out
