#!/bin/bash
# Issues #6/#7, unattended: a distilled LocDiT for teacher-v3 (VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3 main, step
#   1,500: one merged model, 17 dialect tags; its LoRA is on base_lm only, so its LocDiT is base VoxCPM2's).
#   teacher data (teacher-v3's LMs teacher-forced over the v2 contexts with the dialect tag in front of the text, its
#   LocDiT solving each patch: 10 steps, CFG 2.0) -> replay check -> student on SCHEDULE (default the 3-step grid
#   1 -> 0.7071 -> 0.4239 -> 0, mean mode) from INIT (default the engine's 3-step Arabic student): mse (MSE_ITERS) ->
#   spectral (SPEC_ITERS, best by val_spec) [-> dmd (DMD_ITERS; 0 = none: on the 3-step student, DMD after spectral
#   lowered UTMOS)] -> TRAIN_DONE -> the checkpoint chosen on a development subset of #2 (eval/devset.py, seed 2002,
#   which #2 does not use) among INIT itself, mse-best, spectral-best [, dmd-last], by UTMOS within the CER / runaway
#   guards of INIT (eval/select.py) -> #2 (v2 set, ref, int8, head start 8; the tagged model with the set's
#   dialect_map.json) against teacher-v3 with INIT, with the 2-step Arabic student (K2_RUN when this engine
#   regenerates it bit for bit, else a new run) and in quality mode (QUALITY_RUN: the original LocDiT) ->
#   report.md (all clips) and report-heldout.md (all clips and the items outside the development subset: the unbiased
#   number).
#     scripts/run_dit_v3.sh [name]          log: /workspace/data/arabic/logs/<name>.log
# Every GPU step holds GPU_LOCK (flock; default /workspace/.gpu.lock, empty: no lock) while it runs, so other GPU jobs
# that take the same lock (engine benchmarks) never overlap it; they can run between two steps.
# Environment (defaults): ENGINE (give a pinned checkout), ENGINE_PY (the engine's venv), TRAIN_PY
# (/workspace/envs/train, else ENGINE_PY), SCORE_PY (/workspace/envs/eval: scoring), TEACHER
# (/workspace/models/arabic-teacher-v3-best, else the Hub snapshot), TD (teacher data dir), SD (student runs dir),
# PATCHES, VERIFY (rows replayed), INIT, SCHEDULE, MSE_ITERS, SPEC_ITERS, DMD_ITERS, K2_RUN, QUALITY_RUN, STOP_AFTER
# (td, mse, spectral, dmd, train or dev: stop after that step), QUIET_MIN (minutes without a corpus job before
# starting), and extra options per step: TD_ARGS, MSE_ARGS, SPEC_ARGS, DMD_ARGS.
# Every finished step is skipped (teacher data resumes; a training stage that stopped midway starts again).
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
INIT=${INIT:-VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k3.pt}
SCHEDULE=${SCHEDULE:-1,0.7071,0.4239,0}
GPU_LOCK=${GPU_LOCK-/workspace/.gpu.lock}
SET=${SET:-v2}
SET_DIR=/workspace/data/arabic/eval/$SET
DEV_DIR=${DEV_DIR:-/workspace/data/arabic/eval/${SET}dev}
TD=${TD:-/workspace/data/arabic/teacher/teacher-v3}
SD=${SD:-/workspace/runs/$NAME}
PATCHES=${PATCHES:-1500000}
VERIFY=${VERIFY:-2000}
MSE_ITERS=${MSE_ITERS:-1500}; SPEC_ITERS=${SPEC_ITERS:-1000}; DMD_ITERS=${DMD_ITERS:-0}
K2_RUN=${K2_RUN:-routing-v3-int8-fast-ar}
QUALITY_RUN=${QUALITY_RUN:-routing-v3-int8-q}
STOP_AFTER=${STOP_AFTER:-}
ARABIC_K2=VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k2_dmd.pt
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
score() { (cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" "$@"); }
evalrun() {  # run name, checkpoint ("none": the original LocDiT), [run.py args]: generate (once) -> score -> report
  local name=$1 dit=$2; shift 2
  score -m arabic_voxcpm.eval.run --engine-python "$ENGINE_PY" --engine-path "$ENGINE" --name "$name" \
    --model "$TEACHER" --mode ref --dit "$dit" --precision int8 --head-tokens 8 \
    --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.45}" --kv-blocks "${EVAL_KV_BLOCKS:-256}" "$@"
}
student() {  # stage, iterations, init, [dit.py args]
  local stage=$1 iters=$2 init=$3; shift 3
  [ -f "$SD/$stage-done" ] && return 0
  gpu train -m arabic_voxcpm.distill.dit --teacher "$TEACHER" --data "$TD" --out "$SD" --stage "$stage" \
    --iters "$iters" --init "$init" --schedule "$SCHEDULE" "$@" && touch "$SD/$stage-done"
}
same_audio() {  # this engine regenerates $EVAL/$K2_RUN (teacher-v3 with the 2-step Arabic student) bit for bit
  local out=$SD/k2-check
  rm -rf "$out"
  (cd "$REPO" && PYTHONPATH="$REPO:$ENGINE" "$ENGINE_PY" -m arabic_voxcpm.eval.generate --set "$SET" --out "$out" \
     --model "$TEACHER" --dit "$ARABIC_K2_PT" --precision int8 --head-tokens 8 --mode ref --seeds 2000 2001 \
     --concurrency 128 --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.45}" --kv-blocks "${EVAL_KV_BLOCKS:-256}")
  "$SCORE_PY" - "$out/wav" "$EVAL/$K2_RUN/wav" <<'EOF'
import filecmp, os, sys
a, b = sys.argv[1:]
names = sorted(os.listdir(b))
diff = [n for n in names if not os.path.exists(os.path.join(a, n)) or not filecmp.cmp(os.path.join(a, n), os.path.join(b, n), shallow=False)]
print(f"{os.path.basename(os.path.dirname(b))} check: {len(names) - len(diff)} / {len(names)} files equal")
sys.exit(1 if diff or len(os.listdir(a)) != len(names) else 0)
EOF
  local ok=$?
  rm -rf "$out"
  return $ok
}
td_complete() {
  "$SCORE_PY" -c "import json, sys; sys.exit(0 if json.load(open('$TD/manifest.json')).get('complete') else 1)" 2>/dev/null
}

step "$NAME: teacher $TEACHER, data $TD, init $INIT, schedule $SCHEDULE, students $SD, engine $ENGINE"
[ -f "$TEACHER/dialects.json" ] || { echo "not a tagged teacher: $TEACHER"; exit 1; }
INIT_PT=$(train -c "from arabic_voxcpm.distill.dit import resolve; print(resolve('$INIT'))")
ARABIC_K2_PT=$(train -c "from arabic_voxcpm.distill.dit import resolve; print(resolve('$ARABIC_K2'))")

if td_complete; then
  step "teacher data: complete ($TD)"
else
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
fi
step "replay check ($VERIFY rows)"
[ -f "$TD/verify.json" ] || gpu train -m arabic_voxcpm.distill.teacher_data --teacher "$TEACHER" --out "$TD" \
  --patches "$PATCHES" --heldout "$SET_DIR/heldout.json" --verify "$VERIFY" ${TD_ARGS:-}
stop_after td

step "student mse ($MSE_ITERS iterations, from $INIT)"
student mse "$MSE_ITERS" "$INIT" ${MSE_ARGS:-}
stop_after mse
step "student spectral ($SPEC_ITERS iterations, from mse-best)"
student spectral "$SPEC_ITERS" "$SD/mse-best.pt" --lr 1e-5 --batch 256 --micro 32 ${SPEC_ARGS:-}
stop_after spectral
CANDS=(mse-best spectral-best)
if [ "$DMD_ITERS" -gt 0 ]; then
  step "student dmd ($DMD_ITERS iterations, from spectral-best)"
  student dmd "$DMD_ITERS" "$SD/spectral-best.pt" --batch 512 --micro 128 ${DMD_ARGS:-}
  CANDS+=(dmd-last)
fi
stop_after dmd
echo "training done $(date +%F\ %T): ${CANDS[*]} in $SD" > "$SD/TRAIN_DONE"
stop_after train

step "choice on the development subset ($DEV_DIR, seed 2002): init ${CANDS[*]}"
[ -f "$DEV_DIR/items.jsonl" ] || score -m arabic_voxcpm.eval.devset --base "$SET_DIR" --out "$DEV_DIR"
gpu evalrun "$NAME-dev-init" "$INIT_PT" --set "$DEV_DIR" --seeds 2002 --no-report
DEV_RUNS=("$EVAL/$NAME-dev-init")
for c in "${CANDS[@]}"; do
  gpu evalrun "$NAME-dev-$c" "$SD/$c.pt" --set "$DEV_DIR" --seeds 2002 --no-report
  DEV_RUNS+=("$EVAL/$NAME-dev-$c")
done
CHOSEN=$(score -m arabic_voxcpm.eval.select --reference "$EVAL/$NAME-dev-init" "${DEV_RUNS[@]}" --json "$SD/choice.json")
CHOSEN=${CHOSEN#"$NAME-dev-"}
step "chosen: $CHOSEN ($SD/choice.json)"
stop_after dev

step "evaluation on $SET (ref, int8, head start 8), the tagged model with the set's dialect_map.json, engine $ENGINE"
BASE_K2=$NAME-base-k2
if gpu same_audio; then BASE_K2=$K2_RUN; else gpu evalrun "$BASE_K2" "$ARABIC_K2_PT" --set "$SET" --no-report; fi
gpu evalrun "$NAME-base-init" "$INIT_PT" --set "$SET" --no-report
step "baselines: $NAME-base-init (INIT), $BASE_K2 (2-step), $QUALITY_RUN (quality mode)"
RUN=$NAME-base-init
if [ "$CHOSEN" != init ]; then
  RUN=$NAME-$CHOSEN
  gpu evalrun "$RUN" "$SD/$CHOSEN.pt" --set "$SET" --baseline "$NAME-base-init" "$BASE_K2" "$QUALITY_RUN"
fi
step "report: $RUN"
score -m arabic_voxcpm.eval.report --base "$EVAL/$NAME-base-init" "$EVAL/$BASE_K2" "$EVAL/$QUALITY_RUN" "$EVAL/$RUN" \
  --md "$SD/report.md" --json "$SD/report.json"
score -m arabic_voxcpm.eval.heldout "$RUN" "$NAME-base-init" "$BASE_K2" "$QUALITY_RUN" --dev "$DEV_DIR/items.jsonl" \
  --per-group > "$SD/report-heldout.md"
step "done"
