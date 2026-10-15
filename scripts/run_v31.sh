#!/bin/bash
# teacher-v3.1 (#5 follow-up): teacher-v3 (best, step 1500) continued 1500 steps with corpus weights for the dialects
# where v3 lost against the current adapters on #2. Unattended and resumable (run it again: finished steps are
# skipped). Log: /workspace/data/arabic/logs/teacher-v3.1.log.
#   scripts/run_v31.sh
#
# Why (#2, v2 set, ref cloning, teacher-v3-best against the adapters):
# - Bahraini: CER +1.9, UTMOS -0.20, DNSMOS -0.25;
# - Egyptian: CER +0.6, WER +2.0, 9 % shorter;
# - Saudi: Cohere CER +2.7, dialect ID 84 -> 65 %, 12 % shorter.
#
# Weights, the owner's decision (2026-10-15); every other weight is v3's:
# - lahgtna:bahraini 2;
# - Egyptian 2: lahgtna 2, SADA's 0.04 h 0.5 -> 1;
# - sada:saudi 0.5 -> 2;
# - synth:saudi 2 -> 1.
# Effective hours per epoch 154.9 -> 184.7. Bahraini share 14.9 -> 24.9 %, Egyptian 6.9 -> 11.6 %. Saudi 7.0 -> 3.7 %,
# its real speech 17 -> 34 % of it. Saudi real speech is mostly lahgtna (1.65 h, weight 1); SADA's is 0.35 h.
#
# Recipe as v3, except the schedule:
# - Unchanged: LoRA r64 on base_lm, reference augmentation p 0.5, budget 6000 x accum 2, validation every 500.
# - --init: v3's best weights with a fresh AdamW.
# - Peak 5e-5 after 100 warm-up steps (the fresh moments need a few steps), cosine to 5e-6 over 1500 steps. That is
#   below v3's own rate at its step 1500 (6.9e-5): the weights are refined, and the new weights move the balance.
#   v3's steps 1500-3000 (6.9e-5 -> 1e-5) made #2 slightly worse, so every 500-step checkpoint is kept (--keep-steps)
#   to be evaluated if needed.
#
# GPU: every GPU step holds /workspace/.gpu.lock (other agents share the A10). Training waits for WAIT_FILE (default:
# the student-LoRA run's marker; the owner's order is student LoRAs -> v3.1 -> #7).
#
# Then:
# - merged best / last -> /workspace/models/arabic-teacher-v3.1-{best,last};
# - #2 evaluation (v2, ref, bf16, as v3) against base-adapters-v2-ref and teacher-v3-best (reused: the engine changes
#   since are lossless, checked by regenerating teacher-v3-best bit-identically);
# - report;
# - /workspace/runs/teacher-v3.1/DONE.
set -euo pipefail
NAME=teacher-v3.1
STEPS=1500
INIT=${INIT:-/workspace/runs/teacher-v3/best}
LR=${LR:-5e-5}
WARMUP=${WARMUP:-100}
WAIT_FILE=${WAIT_FILE:-/workspace/runs/student-lora/TRAIN_DONE}
LOCK=/workspace/.gpu.lock
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
ENGINE=${ENGINE:-/workspace/voxcpm2-arabic-dialects-turbo}
ENGINE_PY=${ENGINE_PY:-$ENGINE/.venv/bin/python}
TRAIN_PY=${TRAIN_PY:-/workspace/envs/train/bin/python}
[ -x "$TRAIN_PY" ] || TRAIN_PY=$ENGINE_PY
EVAL_PY=${EVAL_PY:-/workspace/envs/eval/bin/python}
[ -x "$EVAL_PY" ] || EVAL_PY=$TRAIN_PY
snapshot() { "$TRAIN_PY" -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))" 2>/dev/null; }
MODEL=${MODEL:-$(snapshot VoxCPM-Kernel/VoxCPM2-Arabic-Dialects)}
SET=/workspace/data/arabic/eval/v2
EVAL=/workspace/data/arabic/eval/runs
RUN=/workspace/runs/$NAME
QUALITY=/workspace/data/arabic/quality/scores.jsonl
CORPORA=lahgtna,sada,synth
WEIGHTS=sada=0.5,sada:saudi=2,sada:egyptian=1,lahgtna:bahraini=2,lahgtna:egyptian=2,synth:hejazi=2,synth:najdi=2,synth:gulf=2,synth:saudi=1,synth:msa=2
REF_AUG=0.5
BASES=(base-adapters teacher-v3-best)
LOG=/workspace/data/arabic/logs/$NAME.log
mkdir -p "$EVAL" "$RUN" "$(dirname "$LOG")"
echo "log: $LOG"
exec >> "$LOG" 2>&1
step() { echo "=== $(date +%F\ %T) $*"; }
py() { local p=$1; shift; (cd "$REPO" && PYTHONPATH="$REPO" "$p" "$@"); }
gpu() { step "wait for the GPU lock"; flock "$LOCK" "$@"; }
evalrun() {  # name model [run.py args]: generate (skipped when done) -> score (-> report)
  local name=$1 model=$2; shift 2
  gpu bash -c 'cd "$0" && PYTHONPATH="$0" "$@"' "$REPO" "$EVAL_PY" -m arabic_voxcpm.eval.run \
    --engine-python "$ENGINE_PY" --engine-path "$ENGINE" --name "$name" --model "$model" --set "$SET" --dit none \
    --mode ref --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.45}" --kv-blocks "${EVAL_KV_BLOCKS:-256}" "$@"
}

step "$NAME: model $MODEL, init $INIT, train python $TRAIN_PY, eval python $EVAL_PY, engine $ENGINE"
[ -n "$MODEL" ] && [ -f "$MODEL/model.safetensors" ] || { echo "no base model (MODEL=$MODEL)"; exit 1; }
[ -f "$INIT/trainable.safetensors" ] || { echo "no checkpoint to continue (INIT=$INIT)"; exit 1; }
for b in base-adapters-v2-ref teacher-v3-best; do
  [ -f "$EVAL/$b/scores.jsonl" ] || { echo "baseline $b not scored"; exit 1; }
done

if py "$TRAIN_PY" -c "import json, sys; sys.exit(json.load(open('$RUN/last/train_state.json'))['step'] < $STEPS)" 2>/dev/null; then
  step "trained already ($RUN/last)"
else
  if [ -n "$WAIT_FILE" ] && [ ! -f "$WAIT_FILE" ]; then
    step "wait for $WAIT_FILE (the owner's GPU order)"
    while [ ! -f "$WAIT_FILE" ]; do sleep 60; done
  fi
  RESUME=(--init "$INIT"); [ -f "$RUN/last/optim.pt" ] && RESUME=(--resume "$RUN/last")
  step "train $NAME ($STEPS steps, ${RESUME[*]})"
  gpu env PYTORCH_ALLOC_CONF=expandable_segments:True bash -c 'cd "$0" && PYTHONPATH="$0" "$@"' "$REPO" "$TRAIN_PY" \
    -m arabic_voxcpm.train.train \
    --model "$MODEL" --out "$RUN" --steps "$STEPS" --heldout "$SET/heldout.json" --corpora "$CORPORA" \
    --quality "$QUALITY" --lora '{"train_modules": ["base_lm"], "enable_dit": false, "enable_proj": false}' \
    --val-every 500 --save-every 500 --keep-steps --corpus-weights "$WEIGHTS" --ref-aug "$REF_AUG" \
    --lr "$LR" --warmup "$WARMUP" --budget "${BUDGET:-6000}" --accum "${ACCUM:-2}" --max-mem-gb 14 \
    --min-train-hours 130 "${RESUME[@]}"
fi

for ck in best last; do
  OUT=/workspace/models/arabic-$NAME-$ck
  step "export $ck -> $OUT"
  [ -f "$OUT/dialects.json" ] || py "$TRAIN_PY" -c "
from arabic_voxcpm.train.model import export_merged
export_merged('$MODEL', '$RUN/$ck', '$OUT')"
  mkdir -p "$OUT/lora" && cp "$RUN/$ck/trainable.safetensors" "$RUN/$ck/train_state.json" "$OUT/lora/"  # as on the Hub
done

for ck in best last; do
  step "evaluate $NAME-$ck"
  evalrun "$NAME-$ck" "/workspace/models/arabic-$NAME-$ck" --baseline "${BASES[@]}"  # tagged: v2's dialect_map.json
done
py "$EVAL_PY" -m arabic_voxcpm.eval.report --base "$EVAL/base-adapters-v2-ref" --base "$EVAL/teacher-v3-best" \
  "$EVAL/$NAME-best" "$EVAL/$NAME-last" --md "$RUN/report.md"
touch "$RUN/DONE"
step "done"
