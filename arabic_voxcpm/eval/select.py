"""Choose between checkpoints of one run by what #2 measures, not by a training loss: the candidate with the highest
mean UTMOS on the development runs (devset.py), among those whose best-ASR CER and runaway rate are not worse than the
reference run's (the model being replaced) by more than --max-cer / --max-runaway points.

    python -m arabic_voxcpm.eval.select --reference <dev run> <candidate dev run> ...     prints the chosen run's name
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from arabic_voxcpm.eval import report


def summary(run: str) -> dict:
    rows = list(report.load(run).values())
    mean = lambda k: sum(float(r[k]) for r in rows) / len(rows)  # noqa: E731
    return {"n": len(rows), "utmos": mean("utmos"), "cer_min": 100 * mean("cer_min"), "runaway": 100 * mean("runaway")}


def choose(reference: str, candidates: list[str], max_cer: float = 0.3, max_runaway: float = 1.0) -> tuple[str, dict]:
    ref = summary(reference)
    table = {c: summary(c) for c in candidates}
    ok = {c: s for c, s in table.items()
          if s["cer_min"] <= ref["cer_min"] + max_cer and s["runaway"] <= ref["runaway"] + max_runaway}
    pool = ok or table  # none passes the guards: the best UTMOS anyway, flagged in the table
    best = max(pool, key=lambda c: pool[c]["utmos"])
    return best, {"reference": {os.path.basename(reference): ref},
                  "candidates": {os.path.basename(c): {**s, "passes": c in ok} for c, s in table.items()},
                  "chosen": os.path.basename(best), "guards_passed": bool(ok)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--max-cer", type=float, default=0.3)
    ap.add_argument("--max-runaway", type=float, default=1.0)
    ap.add_argument("--json", help="write the table here")
    a = ap.parse_args()
    best, table = choose(a.reference, a.candidates, a.max_cer, a.max_runaway)
    print(json.dumps(table, indent=1), file=sys.stderr)
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1)
    print(os.path.basename(best))  # the run's name (run.py --name)
