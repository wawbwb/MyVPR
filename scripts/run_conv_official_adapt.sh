#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
ROOT=logs/conv_official_adapt_v1
mkdir -p "$ROOT"
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the shared lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_official_adaptation.py tests/test_official_transfer_diagnosis.py tests/test_official_boq.py tests/test_conv_residual_ablation.py tests/test_conv_fusion_controls.py
"$PYTHON" -u scripts/conv_official_adapt.py wait_gpu
for ARM in metric_control rank_preserve rank_and_relation; do
    extra=()
    if test -d "$ROOT/$ARM"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/conv_official_adapt.py train --arm "$ARM" --output "$ROOT/$ARM" "${extra[@]}"
done
extra=()
if test -d "$ROOT/aggregate"; then extra+=(--resume); fi
"$PYTHON" -u scripts/conv_official_adapt.py aggregate --output "$ROOT/aggregate" "${extra[@]}"
echo 'Exactly three matched adaptation runs; no automatic extension.'
