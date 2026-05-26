#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export GENEVAL_PORT="${GENEVAL_PORT:-18085}"
export GENEVAL_HOST="${GENEVAL_HOST:-127.0.0.1}"

# Activate the geneval conda env (mmcv/mmdet are pinned per readme.txt) before running.
python -m gunicorn -c gunicorn.conf.py "app:create_app()"
