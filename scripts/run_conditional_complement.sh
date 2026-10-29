#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
ACTION=${1:?Choose tests, smoke, train, or all explicitly. No training has a default.}
case "$ACTION" in tests|smoke|train|all) ;; *) echo 'Use tests|smoke|train|all'; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
ROOT=logs/conditional_complement_v1
"$PYTHON" -m pytest -q tests/test_conditional_complement.py tests/test_official_layer_selection.py tests/test_conv_residual_ablation.py tests/test_official_boq.py
if test "$ACTION" = tests; then exit 0; fi
mkdir -p "$ROOT"
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the shared lock'; exit 1; }
export CONDITIONAL_SHARED_LOCK=held
"$PYTHON" -u scripts/conditional_complement.py wait_gpu
run_arm() {
    local stage=$1 arm=$2
    local -a extra=()
    if test -d "$ROOT/${stage}_${arm}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/conditional_complement.py train --stage "$stage" --arm "$arm" \
        --output "$ROOT/${stage}_${arm}" "${extra[@]}"
}
aggregate() {
    local stage=$1
    local -a extra=()
    if test -d "$ROOT/${stage}_aggregate"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/conditional_complement.py aggregate --stage "$stage" \
        --output "$ROOT/${stage}_aggregate" "${extra[@]}"
}
if test "$ACTION" = smoke || test "$ACTION" = all; then
    for ARM in a_continue b_last_condition c_cross_condition; do run_arm smoke "$ARM"; done
    aggregate smoke
fi
if test "$ACTION" = train || test "$ACTION" = all; then
    "$PYTHON" -c 'from pathlib import Path; from scripts.train_slgd import verify,read; p=Path("logs/conditional_complement_v1/smoke_aggregate"); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS_DISCARD_UPDATES", "Complete all matched smoke checks first"'
    for ARM in a_continue b_last_condition c_cross_condition; do run_arm train "$ARM"; done
    aggregate train
fi
echo 'Requested stage complete. No layer/weight/seed sweep or automatic extension.'
