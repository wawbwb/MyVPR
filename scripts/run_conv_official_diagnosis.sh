#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
ROOT=logs/conv_official_diagnosis_v1
mkdir -p "$ROOT"
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the shared lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_official_transfer_diagnosis.py tests/test_official_boq.py tests/test_conv_residual_ablation.py tests/test_conv_fusion_controls.py
"$PYTHON" -u scripts/conv_official_diagnosis.py wait_gpu
for ITEM in e1_implementation e2_attenuate e3_low_rho e4_low_lr e5_native_negatives; do
    extra=()
    if test -d "$ROOT/$ITEM"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/conv_official_diagnosis.py run --experiment "$ITEM" --output "$ROOT/$ITEM" "${extra[@]}"
done
extra=()
if test -d "$ROOT/aggregate"; then extra+=(--resume); fi
"$PYTHON" -u scripts/conv_official_diagnosis.py aggregate --output "$ROOT/aggregate" "${extra[@]}"
echo 'Exactly five checks finished; no automatic sweep or extra run.'
