#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
STAGE=${1:-all}
case "$STAGE" in all|smoke|screen) ;; *) echo 'Usage: bash scripts/run_conv_fusion_controls.sh all|smoke|screen'; exit 2 ;; esac
ROOT=logs/conv_fusion_controls_v1
mkdir -p "$ROOT"
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the shared experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_conv_fusion_controls.py tests/test_hcen_followup.py tests/test_hcen_module_ablation.py tests/test_hcen_plain_replication.py tests/test_hcen_plain_base.py tests/test_hcen_replication.py tests/test_hcen_aux_ablation.py tests/test_hcen_vpr.py tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py tests/test_slgd_mid_supervision.py
"$PYTHON" -u scripts/conv_fusion_controls.py wait_gpu
train_stage() {
    local seed=$1 stage=$2 root="$ROOT/seed${1}"
    for ARM in conv_last concat_token; do
        local extra=()
        if test -d "$root/${ARM}_${stage}"; then extra+=(--resume); fi
        "$PYTHON" -u scripts/conv_fusion_controls.py train --seed "$seed" --arm "$ARM" --stage "$stage" \
            --output "$root/${ARM}_${stage}" "${extra[@]}"
    done
    local extra=()
    if test -d "$root/compare_${stage}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/conv_fusion_controls.py compare --seed "$seed" --stage "$stage" \
        --output "$root/compare_${stage}" "${extra[@]}"
}
if test "$STAGE" != screen; then
    for SEED in 42 43 44; do train_stage "$SEED" smoke; done
fi
if test "$STAGE" != smoke; then
    for SEED in 42 43 44; do
        "$PYTHON" -c 'import sys; from pathlib import Path; from scripts.train_slgd import verify,read; p=Path(sys.argv[1]); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS"' "$ROOT/seed${SEED}/compare_smoke"
    done
    for SEED in 42 43 44; do train_stage "$SEED" screen; done
    extra=()
    if test -d "$ROOT/aggregate"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/conv_fusion_controls.py aggregate --output "$ROOT/aggregate" "${extra[@]}"
fi
echo 'Two fusion controls finished; no automatic budget/model selection.'
