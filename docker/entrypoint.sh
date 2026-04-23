#!/usr/bin/env bash
# docker/entrypoint.sh
# Default entrypoint for the Boxer SfM Docker container.
# Executes the command passed to `docker run` / `docker compose` or falls back
# to the SfM pipeline entry point.

set -euo pipefail

# Honour PYTHONPATH so the Boxer source tree is always importable
export PYTHONPATH="${PYTHONPATH:-}:/app"

if [ $# -eq 0 ]; then
    exec python run_boxer_sfm.py \
        --metashape_dir /data/input \
        --prompts "${PROMPTS:-coral}" \
        --sam3_mode "${SAM3_MODE:-image}" \
        --output_dir /data/output \
        --sam3_ckpt /app/ckpts \
        --fuse
else
    exec "$@"
fi
