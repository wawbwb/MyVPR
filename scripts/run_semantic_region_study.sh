#!/usr/bin/env bash
# No "all" mode. Uploading or invoking without arguments never starts training.
set -euo pipefail
cd "$(dirname "$0")/.."
# Set visibility BEFORE the Python process starts, never after CUDA init.
export CUDA_VISIBLE_DEVICES="${SRP_PHYSICAL_GPU:-1}"
ACTION="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in
  preflight|plan|cache|smoke|train|eval|summary) ;;
  *) echo "Choose preflight, plan, cache, smoke, train, eval, summary. No implicit all/train." >&2; exit 2 ;;
esac
exec python -u scripts/semantic_region_study.py "$ACTION" "$@"
