#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
STAGE=${1:-all}
case "$STAGE" in all|smoke|screen) ;; *) echo 'Usage: bash scripts/run_hcen_multiseed.sh all|smoke|screen'; exit 2 ;; esac
mkdir -p logs/hcen_multiseed_v1
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_hcen_replication.py tests/test_hcen_aux_ablation.py tests/test_hcen_vpr.py tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py tests/test_slgd_mid_supervision.py
train_stage() {
    local seed=$1 stage=$2 root="logs/hcen_multiseed_v1/seed${1}"
    for ARM in last_noaux hierarchy_noaux last_lowaux hierarchy_lowaux; do
        local extra=()
        if test -d "$root/${ARM}_${stage}"; then extra+=(--resume); fi
        "$PYTHON" -u scripts/hcen_multiseed.py train --seed "$seed" --stage "$stage" --arm "$ARM" \
            --checkpoint "$RU_CKPT" --output "$root/${ARM}_${stage}" "${extra[@]}"
    done
    local extra=()
    if test -d "$root/compare_${stage}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/hcen_multiseed.py compare --seed "$seed" --stage "$stage" \
        --checkpoint "$RU_CKPT" --output "$root/compare_${stage}" "${extra[@]}"
}
# Complete both seeds' mechanical checks before any new 512-step arm starts.
if test "$STAGE" != screen; then
    for SEED in 43 44; do train_stage "$SEED" smoke; done
fi
if test "$STAGE" != smoke; then
    for SEED in 43 44; do
        "$PYTHON" -c 'import sys; from pathlib import Path; from scripts.train_slgd import verify,read; p=Path(sys.argv[1]); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS"' "logs/hcen_multiseed_v1/seed${SEED}/compare_smoke"
    done
    for SEED in 43 44; do train_stage "$SEED" screen; done
    extra=()
    if test -d logs/hcen_multiseed_v1/aggregate; then extra+=(--resume); fi
    "$PYTHON" -u scripts/hcen_multiseed.py aggregate --checkpoint "$RU_CKPT" \
        --output logs/hcen_multiseed_v1/aggregate "${extra[@]}"
fi
echo 'Initialization-only HCEN replication complete; not independent final validation.'
