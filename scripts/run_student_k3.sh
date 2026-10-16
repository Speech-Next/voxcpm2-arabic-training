#!/bin/bash
# #12 follow-up, unattended: a 3-step Arabic distilled LocDiT, fine-tuned from the 2-step one (INIT, the engine's
#   default student) on #12's recorded teacher data (TD: the adapters teacher's 10-step CFG patches; the targets do
#   not depend on the student's grid, so no new teacher data). Grid (SCHEDULE): the 2-step student's first step kept
#   (1 -> 0.7071), its second split at 0.4239 (the English 4-step student's last two steps).
#   spectral (SPEC_ITERS; best by val_spec) -> dmd (DMD_ITERS, a checkpoint every DMD_SAVE) -> TRAIN_DONE -> the
#   checkpoint to evaluate chosen on a development subset of #2 (eval/devset.py, seed 2002, which #2 does not use) by
#   UTMOS among those whose CER and runaways stay near the 2-step student's (eval/select.py), not by teacher MSE
#   (DMD raises it by design) -> #2 (v2 set, ref, int8, head start 8) against the 2-step student and the adapters with
#   the original LocDiT (base-adapters-int8).
#     scripts/run_student_k3.sh [name]          log: /workspace/data/arabic/logs/<name>.log
# Training starts when WAIT_FILE exists (empty: at once). Every GPU step holds GPU_LOCK (flock; default
# /workspace/.gpu.lock, empty: no lock). The 2-step baseline: BASE_RUN (student-lora-base) when this engine
# regenerates it bit for bit, else a new run.
# Environment (defaults): ENGINE (a checkout of the engine; give a pinned one), ENGINE_PY, TRAIN_PY, SCORE_PY, MODEL,
# TD, SD, INIT, SCHEDULE, SPEC_ITERS, DMD_ITERS, DMD_SAVE, STOP_AFTER (spectral, dmd, train or dev), SPEC_ARGS,
# DMD_ARGS. Every finished step is skipped (a stage that stopped midway starts again).
set -euo pipefail
NAME=${1:-student-k3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/voxcpm2-arabic-dialects-turbo}
ENGINE_PY=${ENGINE_PY:-/workspace/voxcpm2-arabic-dialects-turbo/.venv/bin/python}
TRAIN_PY=${TRAIN_PY:-/workspace/envs/train/bin/python}
[ -x "$TRAIN_PY" ] || TRAIN_PY=$ENGINE_PY
SCORE_PY=${SCORE_PY:-/workspace/envs/eval/bin/python}
[ -x "$SCORE_PY" ] || SCORE_PY=/venv/main/bin/python
MODEL=${MODEL:-$("$TRAIN_PY" -c "from huggingface_hub import snapshot_download as s; print(s('VoxCPM-Kernel/VoxCPM2-Arabic-Dialects', local_files_only=True))")}
INIT=${INIT:-VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k2_dmd.pt}
SCHEDULE=${SCHEDULE:-1,0.7071,0.4239,0}
WAIT_FILE=${WAIT_FILE-/workspace/runs/sparse-8/PHASE2_DONE}
GPU_LOCK=${GPU_LOCK-/workspace/.gpu.lock}
SET=${SET:-v2}
SET_DIR=/workspace/data/arabic/eval/$SET
DEV_DIR=${DEV_DIR:-/workspace/data/arabic/eval/${SET}dev}
TD=${TD:-/workspace/data/arabic/teacher/adapters}
SD=${SD:-/workspace/runs/$NAME}
SPEC_ITERS=${SPEC_ITERS:-1000}; DMD_ITERS=${DMD_ITERS:-1500}; DMD_SAVE=${DMD_SAVE:-500}
STOP_AFTER=${STOP_AFTER:-}
BASE_RUN=${BASE_RUN:-student-lora-base}
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
student() {  # stage, iterations, init, [dit.py args]
  local stage=$1 iters=$2 init=$3; shift 3
  [ -f "$SD/$stage-done" ] && return 0
  gpu train -m arabic_voxcpm.distill.dit --teacher "$MODEL" --data "$TD" --out "$SD" --stage "$stage" \
    --iters "$iters" --init "$init" --schedule "$SCHEDULE" "$@" && touch "$SD/$stage-done"
}
evalrun() {  # run name, checkpoint, [run.py args]: generate (once) -> score -> report
  local name=$1 dit=$2; shift 2
  score -m arabic_voxcpm.eval.run --engine-python "$ENGINE_PY" --engine-path "$ENGINE" --name "$name" \
    --model "$MODEL" --mode ref --dit "$dit" --precision int8 --head-tokens 8 \
    --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.45}" --kv-blocks "${EVAL_KV_BLOCKS:-256}" "$@"
}
same_audio() {  # this engine regenerates $EVAL/$BASE_RUN (the 2-step student on #2) into $SD/base-check bit for bit
  local out=$SD/base-check
  rm -rf "$out"
  (cd "$REPO" && PYTHONPATH="$REPO:$ENGINE" "$ENGINE_PY" -m arabic_voxcpm.eval.generate --set "$SET" --out "$out" \
     --model "$MODEL" --dit "$INIT_PT" --precision int8 --head-tokens 8 --mode ref --seeds 2000 2001 \
     --concurrency 128 --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.45}" --kv-blocks "${EVAL_KV_BLOCKS:-256}")
  "$SCORE_PY" - "$out/wav" "$EVAL/$BASE_RUN/wav" <<'EOF'
import filecmp, os, sys
a, b = sys.argv[1:]
names = sorted(os.listdir(b))
diff = [n for n in names if not os.path.exists(os.path.join(a, n)) or not filecmp.cmp(os.path.join(a, n), os.path.join(b, n), shallow=False)]
print(f"base check: {len(names) - len(diff)} / {len(names)} files equal")
sys.exit(1 if diff or len(os.listdir(a)) != len(names) else 0)
EOF
}

step "$NAME: model $MODEL, init $INIT, schedule $SCHEDULE, data $TD, runs $SD, engine $ENGINE"
INIT_PT=$(train -c "from arabic_voxcpm.distill.dit import resolve; print(resolve('$INIT'))")
if [ -n "$WAIT_FILE" ] && [ ! -f "$WAIT_FILE" ]; then
  step "wait for $WAIT_FILE"
  while [ ! -f "$WAIT_FILE" ]; do sleep 60; done
fi
step "spectral ($SPEC_ITERS iterations, from $INIT)"
student spectral "$SPEC_ITERS" "$INIT" --lr 1e-5 --batch 256 --micro 32 ${SPEC_ARGS:-}
stop_after spectral
step "dmd ($DMD_ITERS iterations, from spectral-best, a checkpoint every $DMD_SAVE)"
student dmd "$DMD_ITERS" "$SD/spectral-best.pt" --batch 512 --micro 128 --save-every "$DMD_SAVE" ${DMD_ARGS:-}
stop_after dmd
echo "training done $(date +%F\ %T): spectral, dmd in $SD" > "$SD/TRAIN_DONE"
stop_after train

step "choice on the development subset ($DEV_DIR, seed 2002)"
[ -f "$DEV_DIR/items.jsonl" ] || score -m arabic_voxcpm.eval.devset --base "$SET_DIR" --out "$DEV_DIR"
CANDS=("$SD/spectral-best.pt")
for s in $(seq "$DMD_SAVE" "$DMD_SAVE" "$DMD_ITERS"); do CANDS+=("$SD/dmd-step$s.pt"); done
gpu evalrun "$NAME-dev-k2" "$INIT_PT" --set "$DEV_DIR" --seeds 2002 --no-report
DEV_RUNS=()
for c in "${CANDS[@]}"; do
  n="$NAME-dev-$(basename "$c" .pt)"
  gpu evalrun "$n" "$c" --set "$DEV_DIR" --seeds 2002 --no-report
  DEV_RUNS+=("$EVAL/$n")
done
CHOSEN=$(score -m arabic_voxcpm.eval.select --reference "$EVAL/$NAME-dev-k2" "${DEV_RUNS[@]}" --json "$SD/choice.json")
CHOSEN=${CHOSEN#"$NAME-dev-"}
step "chosen: $CHOSEN ($SD/choice.json)"
stop_after dev

step "evaluation on $SET (ref, int8, head start 8), engine $ENGINE"
BASE=$NAME-base
if gpu same_audio; then BASE=$BASE_RUN; else gpu evalrun "$BASE" "$INIT_PT" --set "$SET" --no-report; fi
step "2-step baseline: $BASE"
gpu evalrun "$NAME-$CHOSEN" "$SD/$CHOSEN.pt" --set "$SET" --baseline "$BASE" base-adapters-int8
score -m arabic_voxcpm.eval.report --base "$EVAL/$BASE" "$EVAL/base-adapters-int8-$SET-ref" "$EVAL/$NAME-$CHOSEN" \
  --md "$SD/report.md" --json "$SD/report.json"
step "done"
