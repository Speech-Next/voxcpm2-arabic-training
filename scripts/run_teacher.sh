#!/bin/bash
# The teacher pipeline, unattended: eval set -> baseline (current adapters) -> train -> export -> eval -> report.
#   scripts/run_teacher.sh <run name> <steps> [extra train.py args]
set -euo pipefail
NAME=$1; STEPS=$2; shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/nanovllm-voxcpm-opt}      # inference library (DialectTTS)
ENGINE_PY=${ENGINE_PY:-/workspace/ttfa-bench/envs/nanovllm-opt/bin/python}
PY=${PY:-/venv/main/bin/python}
SET=/workspace/data/arabic/eval/v1
EVAL=/workspace/data/arabic/eval/runs
RUN=/workspace/runs/$NAME
MODEL=/workspace/models/VoxCPM2-Arabic-Dialects
LOG=/workspace/data/arabic/logs/$NAME.log
mkdir -p "$EVAL" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
step() { echo "=== $(date +%F\ %T) $*"; }
evalrun() {  # generate (skipped when done) -> score, eval/run.py; [run.py / generate.py args]
  (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.run --engine-python "$ENGINE_PY" --engine-path "$ENGINE" \
     --set "$SET" --dit none --mode ref --no-report "$@")
}

step "wait for the corpora"
until [ "$(ls /workspace/data/arabic/lahgtna/manifest/*.jsonl | wc -l)" -ge 112 ]; do sleep 60; done
while ps -eo cmd | grep -q "[a]rabic_voxcpm.data"; do sleep 30; done

step "evaluation set"
[ -f "$SET/items.jsonl" ] || (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.build --out "$SET")

step "baseline: current adapters, original LocDiT"
evalrun --name "$EVAL/base-adapters" --model "$MODEL"  # and the set's ground truth

step "train $NAME ($STEPS steps)"
(cd "$REPO" && PYTHONPATH="$REPO" PYTORCH_ALLOC_CONF=expandable_segments:True $PY -m arabic_voxcpm.train.train \
   --out "$RUN" --steps "$STEPS" --heldout "$SET/heldout.json" $( [ -f "$RUN/last/optim.pt" ] && echo --resume "$RUN/last" ) "$@")

for ck in best last; do
  step "export and evaluate $ck"
  OUT=/workspace/models/arabic-$NAME-$ck
  [ -f "$OUT/dialects.json" ] || (cd "$REPO" && PYTHONPATH="$REPO" $PY -c "
from arabic_voxcpm.train.model import export_merged
export_merged('$MODEL', '$RUN/$ck', '$OUT')")
  evalrun --name "$EVAL/$NAME-$ck" --model "$OUT"  # tagged: the set's dialect_map.json
done
(cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.report --base "$EVAL/base-adapters" \
   "$EVAL/$NAME-best" "$EVAL/$NAME-last" --md "$RUN/report.md")
step "done"
