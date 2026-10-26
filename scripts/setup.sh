#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
training_python="${PYTHON:-}"
if [[ -z "$training_python" ]]; then
  for candidate in python3.10 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info[:2] in ((3,10),(3,11))' 2>/dev/null; then
      training_python="$candidate"
      break
    fi
  done
fi
if [[ -z "$training_python" ]]; then
  echo 'Python 3.10 or 3.11 is required. Set PYTHON=/path/to/python if needed.' >&2
  exit 1
fi
"$training_python" -c 'import sys; assert sys.version_info[:2] in ((3,10),(3,11)), "Use Python 3.10 or 3.11"'
if [[ ! -f .venv/bin/python ]]; then
  "$training_python" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip 'setuptools<81'
torch_index='https://download.pytorch.org/whl/cu121'
if [[ "${1:-}" == '--cpu' ]]; then torch_index='https://download.pytorch.org/whl/cpu'; fi
.venv/bin/python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url "$torch_index"
.venv/bin/python -m pip install -r requirements/engine.txt
.venv/bin/python -m pip install -e '.[export]'
echo 'Ready. Run bash start.sh to open the training interface.'
