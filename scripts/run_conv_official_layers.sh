#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
ROOT=logs/conv_official_layers_v1
mkdir -p "$ROOT"
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the shared lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_official_layer_selection.py tests/test_official_adaptation.py tests/test_official_transfer_diagnosis.py tests/test_official_boq.py tests/test_conv_residual_ablation.py tests/test_conv_fusion_controls.py
"$PYTHON" -u scripts/conv_official_layers.py wait_gpu
extra=()
if test -d "$ROOT/screen"; then extra+=(--resume); fi
"$PYTHON" -u scripts/conv_official_layers.py screen --output "$ROOT/screen" "${extra[@]}"
if "$PYTHON" -u scripts/conv_official_layers.py gate; then
    for ARM in selected last_only; do
        extra=()
        if test -d "$ROOT/$ARM"; then extra+=(--resume); fi
        "$PYTHON" -u scripts/conv_official_layers.py train --arm "$ARM" --output "$ROOT/$ARM" "${extra[@]}"
    done
else
    status=$?
    if test "$status" -ne 2; then exit "$status"; fi
    echo 'Calibration found no supported alternative; do not force new training.'
fi
extra=()
if test -d "$ROOT/aggregate"; then extra+=(--resume); fi
"$PYTHON" -u scripts/conv_official_layers.py aggregate --output "$ROOT/aggregate" "${extra[@]}"
echo 'Layer study finished; at most two new training runs, no automatic extra search.'
