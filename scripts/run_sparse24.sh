#!/bin/bash
# #8 phase 2, unattended: base_lm's FFN 2:4 sparse on the agreed data (distill/sparse_ffn.py), the evaluation, the
# release directory (no upload).
#     nohup scripts/run_sparse24.sh [name] &          log: /workspace/data/arabic/logs/<name>.log
# Starts when WAIT_FOR exists (the coordinator's GPU order). Every GPU step holds GPU_LOCK (flock) while it runs; other
# GPU jobs taking the same lock run between steps. Every finished step is skipped on a rerun (and prune, record and
# train resume inside). At the end DONE_FILE holds "ok" and the summary's path, or the failed step and the log.
# Steps:
#   0. the pending alternating benchmark of phase 1 (the randomly pruned model) has run: that model is deleted
#   1. prune   SparseGPT 2:4, calibration on CALIB_BATCHES batches            -> SD/pruned (a model directory)
#   2. record  the dense model's targets on PATCHES target patches            -> SD/targets
#   3. train   recovery, STEPS steps                                          -> SD/train (best, last)
#   4. export  the recovered model                                            -> SD/model
#   5. eval    #2 (v2 set, ref cloning, int8, head start 8, DIT) through the sparse-kernel engine: the dense model, the
#              pruned one without recovery, the recovered one; both paired against the dense one
#   6. ttfa    examples/benchmark.py, dense and sparse alternating, twice     -> SD/ttfa
#   7. report  SD/summary.md (#8's Done-when items), the release directory RELEASE with its card
# Environment (defaults below): ENGINE, ENGINE_PY, TRAIN_PY, SCORE_PY, MODEL, DIT, SD, RELEASE, WAIT_FOR, DONE_FILE,
# GPU_LOCK, CALIB_BATCHES, PATCHES, STEPS, LR, TRAIN_ARGS (extra train options).
set -euo pipefail
NAME=${1:-sparse24}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/wt-sparse}
ENGINE_PY=${ENGINE_PY:-/workspace/voxcpm2-arabic-dialects-turbo/.venv/bin/python}
TRAIN_PY=${TRAIN_PY:-/workspace/envs/train/bin/python}
SCORE_PY=${SCORE_PY:-/workspace/envs/eval/bin/python}
MODEL=${MODEL:-/workspace/models/VoxCPM2-Arabic-Dialects}
DIT=${DIT:-VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k2_dmd.pt}
SD=${SD:-/workspace/runs/sparse-8/phase2}
RELEASE=${RELEASE:-/workspace/runs/sparse-8/release/VoxCPM2-Arabic-Dialects-Sparse24}
WAIT_FOR=${WAIT_FOR:-/workspace/runs/teacher-v3.1/DONE}
DONE_FILE=${DONE_FILE:-/workspace/runs/sparse-8/PHASE2_DONE}
GPU_LOCK=${GPU_LOCK-/workspace/.gpu.lock}
CALIB_BATCHES=${CALIB_BATCHES:-24}
PATCHES=${PATCHES:-1500000}
STEPS=${STEPS:-2000}
LR=${LR:-2e-5}
TRAIN_ARGS=${TRAIN_ARGS:-}
RANDOM_MODEL=${RANDOM_MODEL:-/workspace/runs/sparse-8/model-prune24-random}
E2E_LOG=${E2E_LOG:-/workspace/runs/sparse-8/e2e/log.txt}
EVAL=/workspace/data/arabic/eval/runs
LOG=${LOG:-/workspace/data/arabic/logs/$NAME.log}
mkdir -p "$SD" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
export PYTORCH_ALLOC_CONF=expandable_segments:True
CUR=start
step() { CUR="$*"; echo "=== $(date +%F\ %T) $*"; }
finish() {
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED (exit $rc) at step: $CUR. Log: $LOG" > "$DONE_FILE"
    echo "=== $(date +%F\ %T) FAILED at: $CUR"
  fi
}
trap finish EXIT
gpu() {  # a GPU step, holding GPU_LOCK while it runs
  if [ -z "$GPU_LOCK" ]; then "$@"; return; fi
  ( flock 9; "$@" ) 9>>"$GPU_LOCK"
}
py() { (cd "$REPO" && PYTHONPATH="$REPO" "$TRAIN_PY" -m arabic_voxcpm.distill.sparse_ffn "$@"); }

step "$NAME: model $MODEL, dit $DIT, engine $ENGINE, runs $SD; waiting for $WAIT_FOR"
until [ -e "$WAIT_FOR" ]; do sleep 60; done

step "0. phase 1's alternating benchmark on the randomly pruned model, then that model is deleted"
until grep -q "^done" "$E2E_LOG" 2>/dev/null || ! pgrep -f "sparse-8/e2e[.]sh" > /dev/null; do sleep 60; done
rm -rf "$RANDOM_MODEL"

if [ ! -f "$SD/pruned/source.json" ]; then
  step "1. prune ($CALIB_BATCHES calibration batches)"
  gpu py prune --model "$MODEL" --out "$SD/pruned" --seed 1 --calib-batches "$CALIB_BATCHES"
fi
if ! grep -q '"complete": true' "$SD/targets/manifest.json" 2>/dev/null; then
  step "2. record the dense model's targets ($PATCHES target patches)"
  gpu py record --model "$MODEL" --out "$SD/targets" --patches "$PATCHES"
fi
if [ ! -f "$SD/train/DONE" ]; then
  step "3. train ($STEPS steps, lr $LR)"
  # shellcheck disable=SC2086
  gpu py train --pruned "$SD/pruned" --targets "$SD/targets" --out "$SD/train" --steps "$STEPS" --lr "$LR" $TRAIN_ARGS
fi
if [ ! -f "$SD/model/model.safetensors" ]; then
  step "4. export the recovered model"
  py export --pruned "$SD/pruned" --ckpt "$SD/train/best" --out "$SD/model"
fi

evalrun() {  # run name, model, [run.py args]: generate (once) -> score -> report
  local name=$1 model=$2; shift 2
  (cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" -m arabic_voxcpm.eval.run --engine-python "$ENGINE_PY" \
     --engine-path "$ENGINE" --name "$name" --model "$model" --set v2 --mode ref --dit "$DIT" --precision int8 \
     --head-tokens 8 --gpu-memory-utilization 0.45 --kv-blocks 256 "$@")
}
for spec in "$NAME-dense:$MODEL:dit-adapters-dmd" "$NAME-pruned:$SD/pruned:$NAME-dense" "$NAME-recovered:$SD/model:$NAME-dense"; do
  IFS=: read -r name model base <<< "$spec"
  if [ ! -f "$EVAL/$name/report.json" ]; then
    step "5. eval $name ($model, baseline $base)"
    gpu evalrun "$name" "$model" --baseline "$base"
  fi
done

if [ ! -f "$SD/ttfa/sparse-2.txt" ]; then
  step "6. TTFA: dense and sparse alternating, twice"
  mkdir -p "$SD/ttfa"
  bench() {  # out file, model
    (cd "$ENGINE" && PYTHONPATH="$ENGINE" NANOVLLM_BUSY_POLL_S=30 "$ENGINE_PY" examples/benchmark.py --model "$2" \
       --levels 1,8,64,200 --rounds 5 > "$1.tmp" 2> "$1.err" && mv "$1.tmp" "$1")
  }
  export -f bench
  export ENGINE ENGINE_PY
  gpu bash -c "for i in 1 2; do bench '$SD/ttfa/dense-'\$i.txt '$MODEL'; bench '$SD/ttfa/sparse-'\$i.txt '$SD/model'; done"
fi

step "7. summary and the release directory"
if [ ! -f "$RELEASE/model.safetensors" ]; then
  py export --pruned "$SD/pruned" --ckpt "$SD/train/best" --out "$RELEASE" --copy
fi
(cd "$REPO" && PYTHONPATH="$REPO" "$SCORE_PY" scripts/sparse24_report.py --name "$NAME" --sd "$SD" --release "$RELEASE")
echo "ok: $SD/summary.md, release directory $RELEASE (not uploaded)" > "$DONE_FILE"
step "done"
