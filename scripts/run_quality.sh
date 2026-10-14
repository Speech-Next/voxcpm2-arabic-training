#!/bin/bash
# Score the training candidates (arabic_voxcpm.data.quality); resumes after a failure with smaller ASR batches.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-/workspace/data/arabic/quality/scores.jsonl}
LOG=/workspace/data/arabic/logs/quality.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
for bs in 32 16 8; do
  echo "=== $(date +%F\ %T) asr batch $bs"
  (cd "$REPO" && PYTHONPATH="$REPO" /venv/main/bin/python -m arabic_voxcpm.data.quality --out "$OUT" --asr-batch $bs "$@") && { echo "=== $(date +%F\ %T) done"; exit 0; }
done
echo "=== $(date +%F\ %T) failed"; exit 1
