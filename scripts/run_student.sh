#!/bin/bash
# After the teacher pipeline, unattended: pick the teacher checkpoint -> numbers check (#4) -> teacher data (#6) ->
# LocDiT students, mse then dmd (#7) -> evaluation -> reports.
#   scripts/run_student.sh <teacher run name> <student name> [mse iters] [dmd iters]
set -euo pipefail
TEACHER=$1; NAME=$2; MSE_ITERS=${3:-4000}; DMD_ITERS=${4:-1500}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/nanovllm-voxcpm-opt}
ENGINE_PY=${ENGINE_PY:-/workspace/ttfa-bench/envs/nanovllm-opt/bin/python}
PY=/venv/main/bin/python
SET=/workspace/data/arabic/eval/v1
NUMSET=/workspace/data/arabic/eval/numbers_v1
EVAL=/workspace/data/arabic/eval/runs
TD=/workspace/data/arabic/teacher/$TEACHER
SD=/workspace/runs/$NAME
LOG=/workspace/data/arabic/logs/$NAME.log
mkdir -p "$SD" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
set -a; source /workspace/.env; set +a
step() { echo "=== $(date +%F\ %T) $*"; }
gen() {  # out model set [generate args]
  local out=$1 model=$2 set=$3; shift 3
  [ -f "$out/run.json" ] && return 0
  (cd "$REPO" && PYTHONPATH="$REPO:$ENGINE" "$ENGINE_PY" -m arabic_voxcpm.eval.generate --set "$set" --out "$out" \
     --model "$model" --mode ref --adapter-map "$SET/dialect_map.json" "$@")
}
score() { (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.score "$@"); }
report() { (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.report "$@"); }

step "wait for the teacher pipeline ($TEACHER)"
until grep -qE "^=== .* done$" "/workspace/data/arabic/logs/$TEACHER.log" 2>/dev/null; do
  if ! ps -eo args | grep -q "[r]un_teacher.sh $TEACHER"; then echo "the teacher pipeline stopped without finishing"; exit 1; fi
  sleep 60
done

step "pick the teacher checkpoint (lower CER over both ASRs, best of the two per clip)"
CK=$(cd "$REPO" && PYTHONPATH="$REPO" $PY -c "
from arabic_voxcpm.eval.report import load
m = {ck: sum(r['cer_min'] for r in load('$EVAL/$TEACHER-' + ck).values()) for ck in ('best', 'last')}
print(min(m, key=m.get))")
TM=/workspace/models/arabic-$TEACHER-$CK
echo "teacher: $TM"

step "numbers (#4): the teacher with and without text normalization"
[ -f "$NUMSET/items.jsonl" ] || (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.eval.numbers --base "$SET" --out "$NUMSET")
gen "$EVAL/numbers-$TEACHER-raw" "$TM" "$NUMSET" --dit none --no-normalize
gen "$EVAL/numbers-$TEACHER-norm" "$TM" "$NUMSET" --dit none
score "$EVAL/numbers-$TEACHER-raw" "$EVAL/numbers-$TEACHER-norm"
report --base "$EVAL/numbers-$TEACHER-raw" "$EVAL/numbers-$TEACHER-norm" --md "$SD/numbers.md"

step "teacher data (#6)"
if [ ! -f "$TD/done" ]; then
  rm -rf "$TD"  # a partial run is not resumable (its batches would repeat)
  (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.distill.teacher_data --teacher "$TM" --out "$TD" \
     --heldout "$SET/heldout.json")
  touch "$TD/done"
fi

step "student: mse ($MSE_ITERS iterations)"
[ -f "$SD/mse-done" ] || { (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.distill.dit --teacher "$TM" --data "$TD" \
   --out "$SD" --stage mse --iters "$MSE_ITERS") && touch "$SD/mse-done"; }

step "student: dmd ($DMD_ITERS iterations)"
[ -f "$SD/dmd-done" ] || { (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.distill.dit --teacher "$TM" --data "$TD" \
   --out "$SD" --stage dmd --init "$SD/mse-best.pt" --iters "$DMD_ITERS") && touch "$SD/dmd-done"; }

step "evaluate the students (and the English-distilled LocDiT) on the teacher"
gen "$EVAL/$NAME-mse" "$TM" "$SET" --dit "$SD/mse-best.pt"
gen "$EVAL/$NAME-dmd" "$TM" "$SET" --dit "$SD/dmd-last.pt"
gen "$EVAL/$TEACHER-$CK-olddit" "$TM" "$SET" --dit "kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt"
score "$EVAL/$NAME-mse" "$EVAL/$NAME-dmd" "$EVAL/$TEACHER-$CK-olddit"
report --base "$EVAL/$TEACHER-$CK" "$EVAL/$NAME-mse" "$EVAL/$NAME-dmd" "$EVAL/$TEACHER-$CK-olddit" --md "$SD/vs-teacher.md"
report --base "$EVAL/base-adapters" "$EVAL/$NAME-mse" "$EVAL/$NAME-dmd" "$EVAL/$TEACHER-$CK-olddit" --md "$SD/vs-base.md"
step "done"
