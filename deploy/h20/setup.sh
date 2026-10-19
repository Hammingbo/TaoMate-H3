#!/usr/bin/env bash
# Isolated environment for the official FL2VA seated_001 case (SDPA/BF16).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
BASE_PYTHON="${BASE_PYTHON:-python3.10}"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3,10),(3,11)), "Python 3.10/3.11 required"'
  "$BASE_PYTHON" -m venv "$VENV_PATH"
fi
PY="$VENV_PATH/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] in ((3,10),(3,11)), "Python 3.10/3.11 required"'
if [[ -n "${WHEELHOUSE_PATH:-}" ]]; then
  "$PY" -m pip install --no-index --find-links "$WHEELHOUSE_PATH" setuptools wheel
  "$PY" -m pip install --no-index --find-links "$WHEELHOUSE_PATH" \
    --no-build-isolation -e "$REPO_ROOT"
else
  "$PY" -m pip install --upgrade pip setuptools wheel
  "$PY" -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  "$PY" -m pip install -e "$REPO_ROOT"
fi
"$PY" -m pip check
"$PY" -m pip freeze > "$VENV_PATH/taomate-requirements.lock.txt"
printf 'Environment ready: %s\n' "$PY"
