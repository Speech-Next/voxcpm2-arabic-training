#!/bin/bash
# Issue #12, unattended: an Arabic distilled LocDiT for the current dialect adapters.
#   teacher data (the adapters teacher, 10 steps, CFG 2.0) -> replay check -> students (2 steps x 12 layers, mean mode):
#   mse from the English student and from the teacher's LocDiT (the better on #2 goes on) -> spectral -> dmd;
#   each stage's best checkpoint evaluated on #2 (v2 set, ref cloning, int8, head start 8) against the adapters with
#   the original LocDiT (base-adapters) and with the English student (base-distilled).
#     scripts/run_dit_adapters.sh [name]          log: /workspace/data/arabic/logs/<name>.log
# Environment (defaults): ENGINE, ENGINE_PY (the engine's venv), TRAIN_PY (/workspace/envs/train, else ENGINE_PY),
# SCORE_PY (/venv/main: scoring), MODEL, TD (teacher data dir), SD (student runs dir), PATCHES, MSE_ITERS,
# SPEC_ITERS, DMD_ITERS, VERIFY (rows replayed), and extra options per step: TD_ARGS, MSE_ARGS, SPEC_ARGS, DMD_ARGS.
# Every step is skipped when done (teacher data and training resume or restart where they stopped).
set -euo pipefail
NAME=${1:-dit-adapters}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/voxcpm2-arabic-dialects-turbo}
ENGINE_PY=${ENGINE_PY:-$ENGINE/.venv/bin/python}
TRAIN_PY=${TRAIN_PY:-/workspace/envs/train/bin/python}
[ -x "$TRAIN_PY" ] || TRAIN_PY=$ENGINE_PY
SCORE_PY=${SCORE_PY:-/venv/main/bin/python}
MODEL=${MODEL:-/workspace/models/VoxCPM2-Arabic-Dialects}
SET=${SET:-v2}
SET_DIR=/workspace/data/arabic/eval/$SET
TD=${TD:-/workspace/data/arabic/teacher/adapters}
SD=${SD:-/workspace/runs/$NAME}
PATCHES=${PATCHES:-1500000}
VERIFY=${VERIFY:-2000}
MSE_ITERS=${MSE_ITERS:-4000}; SPEC_ITERS=${SPEC_ITERS:-1500}; DMD_ITERS=${DMD_ITERS:-1500}
CURRENT=kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt
EVAL=/workspace/data/arabic/eval/runs
LOG=${LOG:-/workspace/data/arabic/logs/$NAME.log}
mkdir -p "$SD" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True
step() { echo "=== $(date +%F\ %T) $*"; }
train() { (cd "$REPO" && PYTHONPATH="$REPO" "$TRAIN_PY" "$@"); }
evalrun() {  # run name, checkpoint ("none": the original LocDiT), [run.py args]: generate (once) -> score -> report
  local name=$1 dit=$2; shift 2
  (cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" -m arabic_voxcpm.eval.run --engine-python "$ENGINE_PY" \
     --engine-path "$ENGINE" --name "$name" --model "$MODEL" --set "$SET" --mode ref --dit "$dit" --precision int8 \
     --head-tokens 8 "$@")
}
cer() {  # a run's mean CER over both ASRs (best of the two per clip), the #2 choice of run_student.sh
  (cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" -c "
from arabic_voxcpm.eval import report
rows = report.load('$EVAL/$1').values()
print(sum(r.get('cer_min', r.get('cer')) for r in rows) / len(rows))")
}
student() {  # out dir, stage, iterations, init, [dit.py args]
  local out=$1 stage=$2 iters=$3 init=$4; shift 4
  [ -f "$out/$stage-done" ] && return 0
  train -m arabic_voxcpm.distill.dit --teacher "$MODEL" --data "$TD" --out "$out" --stage "$stage" --iters "$iters" \
    --init "$init" "$@" && touch "$out/$stage-done"
}

step "teacher data: the adapters teacher -> $TD ($PATCHES patches)"
train -m arabic_voxcpm.distill.teacher_data --teacher adapters --model "$MODEL" --out "$TD" --patches "$PATCHES" \
  --heldout "$SET_DIR/heldout.json" ${TD_ARGS:-}
step "replay check ($VERIFY rows)"
[ -f "$TD/verify.json" ] || train -m arabic_voxcpm.distill.teacher_data --teacher adapters --model "$MODEL" \
  --out "$TD" --patches "$PATCHES" --heldout "$SET_DIR/heldout.json" --verify "$VERIFY" ${TD_ARGS:-}

step "baselines on $SET (ref, int8, head start 8): original LocDiT, English student"
evalrun "base-adapters-$SET-ref" none --no-report || echo "base-adapters-$SET-ref: kept as it is (another configuration?)"
evalrun "base-distilled-$SET-ref" "$CURRENT" --no-report || echo "base-distilled-$SET-ref: kept as it is"

for init in current teacher; do
  step "student mse from $init ($MSE_ITERS iterations)"
  student "$SD/init-$init" mse "$MSE_ITERS" "$init" ${MSE_ARGS:-}
  evalrun "$NAME-mse-$init" "$SD/init-$init/mse-best.pt" --baseline base-adapters base-distilled
done
C_CUR=$(cer "$NAME-mse-current"); C_TEA=$(cer "$NAME-mse-teacher")
BEST=$("$SCORE_PY" -c "print('current' if $C_CUR <= $C_TEA else 'teacher')")
echo "mse students on #2, mean CER: from current $C_CUR, from teacher $C_TEA -> going on from $BEST"
OUT=$SD/init-$BEST

step "student spectral ($SPEC_ITERS iterations, from $OUT/mse-best.pt)"
student "$OUT" spectral "$SPEC_ITERS" "$OUT/mse-best.pt" --lr 1e-5 --batch 256 --micro 32 ${SPEC_ARGS:-}
evalrun "$NAME-spectral" "$OUT/spectral-best.pt" --baseline base-adapters base-distilled

step "student dmd ($DMD_ITERS iterations, from $OUT/spectral-best.pt)"
student "$OUT" dmd "$DMD_ITERS" "$OUT/spectral-best.pt" --batch 512 --micro 128 ${DMD_ARGS:-}
evalrun "$NAME-dmd" "$OUT/dmd-best.pt" --baseline base-adapters base-distilled
evalrun "$NAME-dmd-last" "$OUT/dmd-last.pt" --baseline base-adapters base-distilled

step "report"
(cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" -m arabic_voxcpm.eval.report --base "$EVAL/base-adapters-$SET-ref" \
   "$EVAL/base-distilled-$SET-ref" "$EVAL/$NAME-mse-current" "$EVAL/$NAME-mse-teacher" "$EVAL/$NAME-spectral" \
   "$EVAL/$NAME-dmd" "$EVAL/$NAME-dmd-last" --md "$SD/report.md" --json "$SD/report.json")
step "done"
