#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
if [[ ! -f doc/candidate_set_plan_v1/completed.json ]]; then
  python -u scripts/candidate_set_screen.py prepare --output doc/candidate_set_plan_v1
fi
python -u scripts/candidate_set_screen.py preflight \
  --output "doc/candidate_set_preflight_$(date +%Y%m%d_%H%M%S)"
# Check task difficulty before paying for training-cache extraction.
python -u scripts/candidate_set_screen.py cache --split dev --output .cache/candidate_set_v1/dev
decision="doc/candidate_set_report_v1/admission_$(date +%Y%m%d_%H%M%S)"
if python -u scripts/candidate_set_screen.py admission --output "$decision"; then
  echo 'Development admission passed'
else
  status=$?
  if [[ "$status" == 3 ]]; then
    echo 'Task is saturated. Download doc/candidate_set_report_v1; no training started.'
    exit 0
  fi
  exit "$status"
fi
python -u scripts/candidate_set_screen.py cache --split train --output .cache/candidate_set_v1/train
if python -u scripts/candidate_set_screen.py train --resume --output doc/candidate_set_train_v1; then
  echo 'Training completed'
else
  status=$?
  if [[ "$status" == 3 ]]; then
    mkdir -p doc/candidate_set_report_v1/train_admission
    cp doc/candidate_set_train_v1/contract.json doc/candidate_set_train_v1/gate.json \
      doc/candidate_set_train_v1/completed.json doc/candidate_set_report_v1/train_admission/
    echo 'Training admission failed. Download doc/candidate_set_report_v1.'
    exit 0
  fi
  exit "$status"
fi
python -u scripts/candidate_set_screen.py cache --split test --output .cache/candidate_set_v1/test
python -u scripts/candidate_set_screen.py evaluate \
  --output doc/candidate_set_report_v1/evaluation
