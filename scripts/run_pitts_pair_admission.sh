#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_pitts_pair_admission.py
python -u scripts/pitts_pair_admission.py check --output doc/pitts_pair_index_v1
python -u scripts/pitts_pair_admission.py run --output .cache/pitts_pair_admission_v1
