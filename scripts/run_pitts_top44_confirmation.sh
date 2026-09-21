#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
PYTHON="${VPR_PYTHON:-/home/wt/.conda/envs/VPR/bin/python}"
mkdir -p logs/pitts_top44_confirmation_v1
exec 9>logs/pitts_top44_confirmation_v1/pipeline.lock
flock -n 9 || { echo 'Confirmation pipeline already running'; exit 1; }
"$PYTHON" -m pytest -q tests/test_candidate_top44.py tests/test_candidate_top44_pitts.py tests/test_top44_confirmation.py
"$PYTHON" -u scripts/pitts_top44_confirmation.py prepare
"$PYTHON" -u scripts/pitts_top44_confirmation.py run --max-new-queries 16
"$PYTHON" -u scripts/pitts_top44_confirmation.py run
"$PYTHON" -u scripts/pitts_top44_confirmation.py report
