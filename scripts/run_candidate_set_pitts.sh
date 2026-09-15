#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
python -u scripts/candidate_set_pitts.py check
python -u scripts/candidate_set_pitts.py preflight
# Reuse existing GSV training cache; extract only if not yet complete.
if [[ ! -f .cache/candidate_set_v1/train/completed.json ]]; then
  python -u scripts/candidate_set_screen.py cache --split train --output .cache/candidate_set_v1/train
fi
python -u scripts/candidate_set_pitts.py train --resume
python -u scripts/candidate_set_pitts.py report --output doc/candidate_set_pitts_report_v1
