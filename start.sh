#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .venv/bin/python ]]; then bash scripts/setup.sh; fi
exec .venv/bin/python -m distill_tool.cli ui "$@"
