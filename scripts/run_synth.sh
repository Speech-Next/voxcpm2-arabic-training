#!/bin/bash
# Synthetic speech (issue #3 / #5), unattended: request sets (synth/requests.py) -> generation in chunks (the current
# adapters through the inference library, Habibi-TTS batched) -> filter + latents (synth/filter.py, wavs deleted).
#   scripts/run_synth.sh [chunk size]
set -euo pipefail
CHUNK=${1:-5000}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-/workspace/nanovllm-voxcpm-opt}
ENGINE_PY=${ENGINE_PY:-/workspace/ttfa-bench/envs/nanovllm-opt/bin/python}
PY=/venv/main/bin/python
SYN=/workspace/data/arabic/synth
LOG=/workspace/data/arabic/logs/synth.log
exec >> "$LOG" 2>&1
set -a; source /workspace/.env; set +a
step() { echo "=== $(date +%F\ %T) $*"; }

split() {  # set -> chunk sets <set>-<k> (items split, same voices)
  $PY - "$1" "$CHUNK" <<'PY'
import json, os, shutil, sys
src, n = sys.argv[1], int(sys.argv[2])
items = open(os.path.join(src, "items.jsonl"), encoding="utf-8").read().splitlines()
for k in range(0, len(items), n):
    d = f"{src}-{k // n}"
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "items.jsonl"), "w", encoding="utf-8").write("\n".join(items[k : k + n]) + "\n")
    shutil.copy(os.path.join(src, "voices.json"), d)
    print(d)
PY
}

for gen in adapters habibi; do
  for set in $(split "$SYN/sets/$gen"); do
    name=$(basename "$set")
    out=$SYN/gen/$name
    if [ -f "$out/filtered" ]; then continue; fi
    step "generate $name"
    if [ $gen = adapters ]; then
      (cd "$REPO" && PYTHONPATH="$REPO:$ENGINE" "$ENGINE_PY" -m arabic_voxcpm.eval.generate --set "$set" --out "$out" \
         --model /workspace/models/VoxCPM2-Arabic-Dialects --dit none --mode ref --seeds 1234 --concurrency 128)
    else
      (cd "$REPO" && PYTHONPATH="$REPO" /workspace/envs/habibi/bin/python -m arabic_voxcpm.synth.habibi_gen \
         --set "$set" --out "$out" --seed 1234 --frames 48000)
    fi
    step "filter $name"
    (cd "$REPO" && PYTHONPATH="$REPO" $PY -m arabic_voxcpm.synth.filter --set "$set" --gen "$out" --name "$name" \
       --max-cer 0.08 --delete)
    touch "$out/filtered"
  done
done
step "done"
