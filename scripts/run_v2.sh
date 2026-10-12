#!/bin/bash
# teacher-v2: synthetic speech (run_synth.sh), then the teacher pipeline on the quality-filtered real data + synth,
# LoRA on base_lm only (issue #5: ablations of teacher-v1).
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/run_synth.sh
scripts/run_teacher.sh teacher-v2 3000 --val-every 500 --save-every 500 --corpora lahgtna,sada,synth \
  --quality /workspace/data/arabic/quality/scores.jsonl \
  --lora '{"train_modules": ["base_lm"], "enable_dit": false, "enable_proj": false}'
