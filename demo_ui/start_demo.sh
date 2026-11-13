#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${COMFYUI_PYTHON:-}" ]]; then
  python_bin="$COMFYUI_PYTHON"
elif [[ -x "/home/aiml03/miniconda3/envs/comfyenv/bin/python" ]]; then
  python_bin="/home/aiml03/miniconda3/envs/comfyenv/bin/python"
else
  python_bin="python3"
fi

echo "Using Python: $python_bin"
exec "$python_bin" app.py
