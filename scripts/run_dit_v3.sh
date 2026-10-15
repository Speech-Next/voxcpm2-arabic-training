#!/bin/bash
# Issues #6/#7, unattended: a distilled LocDiT for teacher-v3 (VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3 main, step
#   1,500: one merged model, 17 dialect tags; its LoRA is on base_lm only, so its LocDiT is base VoxCPM2's).
#   teacher data (teacher-v3's LMs teacher-forced over the v2 contexts with the dialect tag in front of the text, its
#   LocDiT solving each patch: 10 steps, CFG 2.0) -> replay check -> students (2 steps x 12 layers, mean mode): mse
#   (from INITS, default the English student; with several, the best on #2 goes on) -> spectral -> dmd; then each
#   stage's best checkpoint evaluated on #2 (v2 set, ref cloning, int8, head start 8) against teacher-v3 with its
#   original LocDiT (teacher-v3-int8), with the English student (teacher-v3-english-int8) and with #12's student for
#   the dialect adapters (teacher-v3-arabic12-int8).
#     scripts/run_dit_v3.sh [name]          log: /workspace/data/arabic/logs/<name>.log
# Every GPU step holds GPU_LOCK (flock; default /workspace/.gpu.lock, empty: no lock) while it runs, so other GPU jobs
# that take the same lock (engine benchmarks) never overlap it; they can run between two steps.
# Environment (defaults): ENGINE, ENGINE_PY (the engine's venv), TRAIN_PY (/workspace/envs/train, else ENGINE_PY),
# SCORE_PY (/workspace/envs/eval: scoring), TEACHER (/workspace/models/arabic-teacher-v3-best, else the Hub snapshot),
# TD (teacher data dir), SD (student runs dir), PATCHES, VERIFY (rows replayed), MSE_ITERS, SPEC_ITERS, DMD_ITERS,
# INITS (current, arabic, teacher or checkpoints), EVAL_STAGES, STOP_AFTER (td, mse, spectral or dmd: stop after that
# step), QUIET_MIN (minutes without a corpus job before starting), and extra options per step: TD_ARGS, MSE_ARGS,
# SPEC_ARGS, DMD_ARGS.
# Every step is skipped when done (teacher data and training resume or restart where they stopped).
set -euo pipefail
NAME=${1:-dit-v3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/voxcpm2-arabic-dialects-turbo}
ENGINE_PY=${ENGINE_PY:-$ENGINE/.venv/bin/python}
TRAIN_PY=${TRAIN_PY:-/workspace/envs/train/bin/python}
[ -x "$TRAIN_PY" ] || TRAIN_PY=$ENGINE_PY
SCORE_PY=${SCORE_PY:-/workspace/envs/eval/bin/python}
[ -x "$SCORE_PY" ] || SCORE_PY=/venv/main/bin/python
TEACHER=${TEACHER:-/workspace/models/arabic-teacher-v3-best}
[ -f "$TEACHER/dialects.json" ] || TEACHER=$("$TRAIN_PY" -c "
from huggingface_hub import snapshot_download as s; print(s('VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3'))")
INITS=${INITS:-current}
GPU_LOCK=${GPU_LOCK-/workspace/.gpu.lock}
SET=${SET:-v2}
SET_DIR=/workspace/data/arabic/eval/$SET
TD=${TD:-/workspace/data/arabic/teacher/teacher-v3}
SD=${SD:-/workspace/runs/$NAME}
PATCHES=${PATCHES:-1500000}
VERIFY=${VERIFY:-2000}
MSE_ITERS=${MSE_ITERS:-4000}; SPEC_ITERS=${SPEC_ITERS:-1500}; DMD_ITERS=${DMD_ITERS:-1500}
EVAL_STAGES=${EVAL_STAGES:-mse spectral dmd dmd-last}
STOP_AFTER=${STOP_AFTER:-}
ENGLISH=VoxCPM-Kernel/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt
ARABIC12=VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k2_dmd.pt
EVAL=/workspace/data/arabic/eval/runs
LOG=${LOG:-/workspace/data/arabic/logs/$NAME.log}
mkdir -p "$SD" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True
step() { echo "=== $(date +%F\ %T) $*"; }
stop_after() { [ "$STOP_AFTER" = "$1" ] && { step "stop after $1 (STOP_AFTER)"; exit 0; }; return 0; }
gpu() {  # a GPU step, holding GPU_LOCK while it runs
  if [ -z "$GPU_LOCK" ]; then "$@"; return; fi
  ( flock 9; "$@" ) 9>>"$GPU_LOCK"
}
train() { (cd "$REPO" && PYTHONPATH="$REPO" "$TRAIN_PY" "$@"); }
evalrun() {  # run name, checkpoint ("none": the original LocDiT), [run.py args]: generate (once) -> score -> report
  local name=$1 dit=$2; shift 2
  (cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" -m arabic_voxcpm.eval.run --engine-python "$ENGINE_PY" \
     --engine-path "$ENGINE" --name "$name" --model "$TEACHER" --set "$SET" --mode ref --dit "$dit" --precision int8 \
     --head-tokens 8 --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.45}" --kv-blocks "${EVAL_KV_BLOCKS:-256}" "$@")
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
  gpu train -m arabic_voxcpm.distill.dit --teacher "$TEACHER" --data "$TD" --out "$out" --stage "$stage" \
    --iters "$iters" --init "$init" "$@" && touch "$out/$stage-done"
}
tag() { basename "$1" .pt; }  # an init's directory name: current, arabic, teacher or the checkpoint's file name

step "$NAME: teacher $TEACHER, data $TD, students $SD, train python $TRAIN_PY, engine $ENGINE"
[ -f "$TEACHER/dialects.json" ] || { echo "not a tagged teacher: $TEACHER"; exit 1; }

step "wait for the corpora (latents restored by arabic_voxcpm.data)"
quiet=0
while [ $quiet -lt "${QUIET_MIN:-5}" ]; do  # QUIET_MIN minutes without a data job: not between two corpora
  if pgrep -f "[a]rabic_voxcpm.data" >/dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done

step "teacher data: teacher-v3 (tagged) -> $TD ($PATCHES patches)"
train -m arabic_voxcpm.distill.teacher_data --teacher "$TEACHER" --out "$TD" --patches "$PATCHES" \
  --heldout "$SET_DIR/heldout.json" --estimate-only ${TD_ARGS:-}
gpu train -m arabic_voxcpm.distill.teacher_data --teacher "$TEACHER" --out "$TD" --patches "$PATCHES" \
  --heldout "$SET_DIR/heldout.json" ${TD_ARGS:-}
step "replay check ($VERIFY rows)"
[ -f "$TD/verify.json" ] || gpu train -m arabic_voxcpm.distill.teacher_data --teacher "$TEACHER" --out "$TD" \
  --patches "$PATCHES" --heldout "$SET_DIR/heldout.json" --verify "$VERIFY" ${TD_ARGS:-}
stop_after td

for init in $INITS; do
  step "student mse from $init ($MSE_ITERS iterations)"
  student "$SD/init-$(tag "$init")" mse "$MSE_ITERS" "$init" ${MSE_ARGS:-}
done
BEST=$(tag "${INITS%% *}")
if [ "$(echo $INITS | wc -w)" -gt 1 ]; then
  best_cer=
  for init in $INITS; do
    t=$(tag "$init")
    gpu evalrun "$NAME-mse-$t" "$SD/init-$t/mse-best.pt" --no-report
    c=$(cer "$NAME-mse-$t")
    echo "mse student from $init on #2: mean CER $c"
    if [ -z "$best_cer" ] || "$SCORE_PY" -c "import sys; sys.exit(not $c < $best_cer)"; then best_cer=$c; BEST=$t; fi
  done
  echo "going on from $BEST"
fi
OUT=$SD/init-$BEST
stop_after mse

step "student spectral ($SPEC_ITERS iterations, from $OUT/mse-best.pt)"
student "$OUT" spectral "$SPEC_ITERS" "$OUT/mse-best.pt" --lr 1e-5 --batch 256 --micro 32 ${SPEC_ARGS:-}
stop_after spectral

step "student dmd ($DMD_ITERS iterations, from $OUT/spectral-best.pt)"
student "$OUT" dmd "$DMD_ITERS" "$OUT/spectral-best.pt" --batch 512 --micro 128 ${DMD_ARGS:-}
stop_after dmd

step "evaluation on $SET (ref, int8, head start 8), the tagged model with the set's dialect_map.json"
gpu evalrun "teacher-v3-int8-$SET-ref" none --no-report
gpu evalrun "teacher-v3-english-int8-$SET-ref" "$ENGLISH" --no-report
gpu evalrun "teacher-v3-arabic12-int8-$SET-ref" "$ARABIC12" --no-report
B=(--baseline teacher-v3-int8 teacher-v3-english-int8 teacher-v3-arabic12-int8)
RUNS=()
for s in $EVAL_STAGES; do
  case $s in
    mse) run=$NAME-mse-$BEST; ck=$OUT/mse-best.pt ;;
    dmd-last) run=$NAME-dmd-last; ck=$OUT/dmd-last.pt ;;
    *) run=$NAME-$s; ck=$OUT/$s-best.pt ;;
  esac
  gpu evalrun "$run" "$ck" "${B[@]}"
  RUNS+=("$EVAL/$run")
done

step "report"
(cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" -m arabic_voxcpm.eval.report --base "$EVAL/teacher-v3-int8-$SET-ref" \
   "$EVAL/teacher-v3-english-int8-$SET-ref" "$EVAL/teacher-v3-arabic12-int8-$SET-ref" "${RUNS[@]}" \
   --md "$SD/report.md" --json "$SD/report.json")
step "done"
