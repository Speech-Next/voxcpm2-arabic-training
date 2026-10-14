#!/bin/bash
# Re-make the real corpora's latents for a dataset release (arabic_voxcpm.data.release) whose manifests were restored
# from a backup without them: only the release's clips, from the source shards that hold them, then check them all.
#   scripts/run_restore.sh [release name, default teacher-v2-train]
# Resumable: shards whose listed clips have latents are skipped. Retries after a failure (connection resets).
set -uo pipefail
NAME=${1:-teacher-v2-train}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/workspace/envs/train/bin/python}
MODEL=${MODEL:-/workspace/models/VoxCPM2-Arabic-Dialects}
REL=/workspace/data/arabic/datasets/$NAME/manifest.jsonl
LOG=/workspace/data/arabic/logs/restore-$NAME.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
if [ -f /workspace/.env ]; then set -a; . /workspace/.env; set +a; fi
export HF_HOME=${HF_HOME:-/workspace/.hf_home} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO"
step() { echo "=== $(date +%F\ %T) $*"; }
retry() { for k in $(seq 1 10); do "$@" && return 0; step "failed ($k), retrying"; sleep 60; done; return 1; }
cd "$REPO"
step "restore $NAME: sada"
retry $PY -m arabic_voxcpm.data.sada --model "$MODEL" --restore "$REL" --splits validation train --fetchers 2 || exit 1
step "restore $NAME: lahgtna"
retry $PY -m arabic_voxcpm.data.lahgtna --model "$MODEL" --restore "$REL" --fetchers 3 || exit 1
step "check"
$PY -m arabic_voxcpm.data.release --name "$NAME" --check && step "done"
