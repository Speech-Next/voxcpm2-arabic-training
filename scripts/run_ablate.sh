#!/bin/bash
# Partial merges of a trained teacher, evaluated without retraining: which LoRAs help and which hurt.
#   scripts/run_ablate.sh <run name> <ckpt> <variant>...     variant: <label>:<modules|all>:<scale>
#   e.g. scripts/run_ablate.sh teacher-v1 last lm:base_lm,residual_lm:1 half:all:0.5
set -euo pipefail
NAME=$1; CK=$2; shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/nanovllm-voxcpm-opt}
ENGINE_PY=${ENGINE_PY:-/workspace/ttfa-bench/envs/nanovllm-opt/bin/python}
PY=/venv/main/bin/python
SET=/workspace/data/arabic/eval/v1
EVAL=/workspace/data/arabic/eval/runs
RUN=/workspace/runs/$NAME
MODEL=/workspace/models/VoxCPM2-Arabic-Dialects
LOG=/workspace/data/arabic/logs/$NAME-ablate.log
exec >> "$LOG" 2>&1
set -a; source /workspace/.env; set +a
step() { echo "=== $(date +%F\ %T) $*"; }
RUNS=()
for v in "$@"; do
  IFS=: read -r label mods scale <<< "$v"
  step "variant $label: modules $mods, scale $scale"
  OUT=/workspace/models/arabic-$NAME-$CK-$label
  ONLY=None; [ "$mods" = all ] || ONLY="tuple('$mods'.split(','))"
  [ -f "$OUT/dialects.json" ] || (cd "$REPO" && PYTHONPATH="$REPO" $PY -c "
from arabic_voxcpm.train.model import export_merged
export_merged('$MODEL', '$RUN/$CK', '$OUT', only=$ONLY, lora_scale=$scale)")
  E=$EVAL/$NAME-$CK-$label
  [ -f "$E/run.json" ] || (cd "$REPO" && PYTHONPATH="$REPO:$ENGINE" "$ENGINE_PY" -m arabic_voxcpm.eval.generate --set "$SET" \
     --out "$E" --model "$OUT" --dit none --mode ref --adapter-map "$SET/dialect_map.json")
  (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.score "$E")
  RUNS+=("$E")
done
(cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.report --base "$EVAL/base-adapters" "$EVAL/$NAME-$CK" "${RUNS[@]}" \
   --md "$RUN/ablate-vs-base.md")
(cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.report --base "$EVAL/$NAME-$CK" "${RUNS[@]}" --md "$RUN/ablate-vs-$CK.md")
step "done"
