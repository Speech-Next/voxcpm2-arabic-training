#!/bin/bash
# teacher-v3 (#5): teacher-v2's recipe on teacher-v2's data, plus #5's next steps. Unattended and resumable (run it
# again: finished steps are skipped). Log: /workspace/data/arabic/logs/teacher-v3.log.
#   scripts/run_v3.sh
#
# As teacher-v2: lahgtna + SADA quality-filtered (Cohere CER <= 10 %, DNSMOS >= 3.0; 97.3 h + 5.6 h) + 32.4 h filtered
# synthetic speech, LoRA r64 on base_lm only (67M trainable), 3000 steps, AdamW 1e-4 cosine, 500 warm-up, budget 12000.
# The held-out list is v2's (the evaluation set; it removes no training clip that v1's kept: the same 47,574 rows).
#
# New (#5's next steps):
# 1. Corpus weights (expected draws per clip and epoch): synthetic x2 in the SADA dialects (hejazi, najdi, gulf, saudi,
#    msa), SADA x0.5. teacher-v2's gap is in hejazi / najdi / khaliji, whose real speech is SADA broadcast audio; there
#    the clean synthetic speech goes from 9:1 to 35:1 (hejazi), 2:1 to 8:1 (najdi), 5:1 to 21:1 (gulf) against SADA.
#    Synthetic speech of the other dialects (Habibi-TTS: maghreb, lebanese; iraqi) stays x1: lahgtna carries them and
#    teacher-v2 already improved them. 155.6 effective hours per epoch (lahgtna 63 %, synthetic 36 %, SADA 2 %), ~2.6
#    epochs in 3000 steps: an up-weighted synthetic clip is seen ~5 times (teacher-v2: 3, val loss flat after step 1500).
# 2. Reference augmentation p 0.5: half of the cloning samples (35 % of all samples) get a degraded reference (noise at
#    5-25 dB SNR, synthetic reverberation RT60 0.2-0.8 s, low-pass 3-7 kHz; augment.py), the target stays clean: a
#    noisy reference must still give clean speech. The other half keeps clean references, so cloning from a clean voice
#    is trained as before. Degraded in the training step (AudioVAE decode -> degrade -> encode): ~1.4 s per step, about
#    25 % of teacher-v2's 5.3 s step, ~+1.1 h over the run. Precomputing is cheaper only over several runs (84x real
#    time: ~1.6 h per variant of the 135 h); set REF_AUG_DIR (and REF_AUG_K, default 2) to precompute and use them.
#
# Then: merged best / last -> /workspace/models/arabic-teacher-v3-{best,last}, evaluation on v2 (ref mode) against the
# current adapters (base-adapters-v2-ref) and teacher-v2 (when its model is in the local Hugging Face cache).
#
# Environment (defaults): ENGINE (inference library), ENGINE_PY, TRAIN_PY (training: torch + voxcpm), EVAL_PY (scoring:
# faster-whisper, transformers, onnxruntime, jiwer, librosa), MODEL (the base VoxCPM2 Arabic model), REF_AUG_DIR.
# Memory: budget 6000 x accum 2 = teacher-v2's 12000 positions per optimizer step at about half the peak, capped at
# 0.62 of the GPU, so the #12 LocDiT student (~7 GB) trains next to it (12000 x 1 reserved 20 GB and OOM'd beside it).
# Before: training peaks at ~13.3 GB as teacher-v2 (the augmentation runs before the forward, ~2 GB, freed for it);
# --max-mem-gb 14 logs a warning above 14 GB. A LocDiT student may train on the same 24 GB GPU.
set -euo pipefail
NAME=teacher-v3
STEPS=3000
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
TEACHER_V2=${TEACHER_V2:-$(snapshot VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v2 || true)}
SET=/workspace/data/arabic/eval/v2
EVAL=/workspace/data/arabic/eval/runs
RUN=/workspace/runs/$NAME
QUALITY=/workspace/data/arabic/quality/scores.jsonl
CORPORA=lahgtna,sada,synth
WEIGHTS=sada=0.5,synth:hejazi=2,synth:najdi=2,synth:gulf=2,synth:saudi=2,synth:msa=2
REF_AUG=0.5
LOG=/workspace/data/arabic/logs/$NAME.log
mkdir -p "$EVAL" "$RUN" "$(dirname "$LOG")"
echo "log: $LOG"
exec >> "$LOG" 2>&1
step() { echo "=== $(date +%F\ %T) $*"; }
py() { local p=$1; shift; (cd "$REPO" && PYTHONPATH="$REPO" "$p" "$@"); }
evalrun() {  # name model [run.py args]: generate (skipped when done) -> score (-> report)
  local name=$1 model=$2; shift 2
  py "$EVAL_PY" -m arabic_voxcpm.eval.run --engine-python "$ENGINE_PY" --engine-path "$ENGINE" \
    --name "$name" --model "$model" --set "$SET" --dit none --mode ref \
    --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.45}" --kv-blocks "${EVAL_KV_BLOCKS:-256}" "$@"  # shares the GPU
}

step "$NAME: model $MODEL, train python $TRAIN_PY, eval python $EVAL_PY, engine $ENGINE"
[ -n "$MODEL" ] && [ -f "$MODEL/model.safetensors" ] || { echo "no base model (MODEL=$MODEL)"; exit 1; }
py "$TRAIN_PY" -c "import voxcpm, torch" || { echo "TRAIN_PY $TRAIN_PY lacks voxcpm / torch"; exit 1; }

step "wait for the corpora (latents restored by arabic_voxcpm.data; train.py refuses < 130 h)"
quiet=0
while [ $quiet -lt 5 ]; do  # 5 minutes without a data job: the restore is not between two corpora
  if ps -eo args | grep -q "[a]rabic_voxcpm.data"; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done

AUG_ARGS=(--ref-aug "$REF_AUG")
if [ -n "${REF_AUG_DIR:-}" ]; then
  step "precompute ${REF_AUG_K:-2} degraded variants per clip -> $REF_AUG_DIR"
  py "$TRAIN_PY" -m arabic_voxcpm.train.augment --out "$REF_AUG_DIR" --k "${REF_AUG_K:-2}" --corpora "$CORPORA" \
    --model "$MODEL" --quality "$QUALITY" --heldout "$SET/heldout.json"
  AUG_ARGS+=(--ref-aug-dir "$REF_AUG_DIR")
fi

if py "$TRAIN_PY" -c "import json, sys; sys.exit(json.load(open('$RUN/last/train_state.json'))['step'] < $STEPS)" 2>/dev/null; then
  step "trained already ($RUN/last)"
else
  step "train $NAME ($STEPS steps)"
  RESUME=(); [ -f "$RUN/last/optim.pt" ] && RESUME=(--resume "$RUN/last")
  PYTORCH_ALLOC_CONF=expandable_segments:True py "$TRAIN_PY" -m arabic_voxcpm.train.train \
    --model "$MODEL" --out "$RUN" --steps "$STEPS" --heldout "$SET/heldout.json" --corpora "$CORPORA" \
    --quality "$QUALITY" --lora '{"train_modules": ["base_lm"], "enable_dit": false, "enable_proj": false}' \
    --val-every 500 --save-every 500 --corpus-weights "$WEIGHTS" "${AUG_ARGS[@]}" \
    --budget "${BUDGET:-6000}" --accum "${ACCUM:-2}" --max-mem-gb 14 --gpu-fraction "${GPU_FRACTION:-0.62}" \
    --min-train-hours 130 "${RESUME[@]}"
fi

for ck in best last; do
  OUT=/workspace/models/arabic-$NAME-$ck
  step "export $ck -> $OUT"
  [ -f "$OUT/dialects.json" ] || py "$TRAIN_PY" -c "
from arabic_voxcpm.train.model import export_merged
export_merged('$MODEL', '$RUN/$ck', '$OUT')"
done

py "$EVAL_PY" -c "import faster_whisper, onnxruntime, jiwer, librosa, transformers" || {
  echo "EVAL_PY $EVAL_PY lacks the scoring packages; set EVAL_PY and run again (training and export are done)"; exit 1; }
BASES=(base-adapters)
step "baseline: current adapters, v2, ref"
evalrun base-adapters-v2-ref "$MODEL" --no-report
if [ -n "$TEACHER_V2" ]; then
  step "baseline: teacher-v2 ($TEACHER_V2), v2, ref"
  evalrun teacher-v2-v2-ref "$TEACHER_V2" --no-report
  BASES+=(teacher-v2)
fi
for ck in best last; do
  step "evaluate $NAME-$ck"
  evalrun "$NAME-$ck" "/workspace/models/arabic-$NAME-$ck" --baseline "${BASES[@]}"  # tagged: v2's dialect_map.json
done
REPORT_BASES=(--base "$EVAL/base-adapters-v2-ref")
[ -n "$TEACHER_V2" ] && REPORT_BASES+=(--base "$EVAL/teacher-v2-v2-ref")
py "$EVAL_PY" -m arabic_voxcpm.eval.report "${REPORT_BASES[@]}" "$EVAL/$NAME-best" "$EVAL/$NAME-last" --md "$RUN/report.md"
step "done"
