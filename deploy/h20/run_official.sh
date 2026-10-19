#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
PY="$VENV_PATH/bin/python"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/models/MiniMax-H3}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/models/TaoMate-H3-FL2VA}"
TIME_MODE="${1:-30}"
case "$TIME_MODE" in 5|10|15|30) ;; *) echo 'Time mode must be 5, 10, 15 or 30' >&2; exit 2;; esac
RUN_DIR="${2:-$REPO_ROOT/outputs/seated_001_${TIME_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$RUN_DIR" ]]; then echo "Output already exists: $RUN_DIR" >&2; exit 2; fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
export PATH="$VENV_PATH/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_DIR/exit_code"' EXIT
cd "$REPO_ROOT"
git rev-parse HEAD > "$RUN_DIR/code_commit.txt"
"$PY" deploy/h20/preflight.py --model-root "$MODEL_ROOT" --checkpoint "$CHECKPOINT" \
  > "$RUN_DIR/environment.json"
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 deploy/h20/nccl_smoke.py
"$PY" src/taomate-fl2av/infer.py --time "$TIME_MODE" --dry-run > "$RUN_DIR/plan.json"
"$PY" src/taomate-fl2av/infer.py --time "$TIME_MODE" \
  --model-root "$MODEL_ROOT" --checkpoint "$CHECKPOINT" --output "$RUN_DIR/inference"
"$PY" deploy/h20/validate_output.py "$RUN_DIR/inference/result/video.mp4" --time "$TIME_MODE"
printf 'OFFICIAL_CASE_PASSED: %s\n' "$RUN_DIR/inference/result/video.mp4"
