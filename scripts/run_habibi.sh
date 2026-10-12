#!/bin/bash
# Habibi-TTS (unified and specialized) on the evaluation set, scored and compared with the current adapters.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
EVAL=/workspace/data/arabic/eval/runs
LOG=/workspace/data/arabic/logs/habibi.log
exec >> "$LOG" 2>&1
set -a; source /workspace/.env; set +a
step() { echo "=== $(date +%F\ %T) $*"; }
for m in unified specialized; do
  step "generate habibi-$m"
  [ -f "$EVAL/habibi-$m/run.json" ] || (cd "$REPO" && PYTHONPATH="$REPO" /workspace/envs/habibi/bin/python \
     -m arabic_voxcpm.eval.generate_habibi --out "$EVAL/habibi-$m" --model $m)
done
step "score"
(cd "$REPO" && PYTHONPATH="$REPO" /venv/main/bin/python -m arabic_voxcpm.eval.score "$EVAL/habibi-unified" "$EVAL/habibi-specialized")
(cd "$REPO" && PYTHONPATH="$REPO" /venv/main/bin/python -m arabic_voxcpm.eval.report --base "$EVAL/base-adapters" \
   "$EVAL/habibi-unified" "$EVAL/habibi-specialized" "$EVAL/teacher-v1-last-baselm" --md /workspace/runs/habibi-vs-base.md)
step "done"
