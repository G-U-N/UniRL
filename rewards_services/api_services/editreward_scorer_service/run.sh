#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export EDITREWARD_REPO_DIR="${EDITREWARD_REPO_DIR:-$SCRIPT_DIR/EditReward}"
export EDITREWARD_PORT="${EDITREWARD_PORT:-18088}"
export EDITREWARD_HOST="${EDITREWARD_HOST:-127.0.0.1}"
export EDITREWARD_CUDA_DEVICES="${EDITREWARD_CUDA_DEVICES:-0,1}"
export EDITREWARD_WORKERS="${EDITREWARD_WORKERS:-${EDITREWARD_NUM_DEVICES:-2}}"

python -m gunicorn -c gunicorn.conf.py "app:create_app()"
