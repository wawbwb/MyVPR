#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
STAGE=${1:-all}
case "$STAGE" in all|smoke|screen) ;; *) echo 'Usage: bash scripts/run_hcen_aux_ablation.sh all|smoke|screen'; exit 2 ;; esac
mkdir -p logs/hcen_aux_ablation_v1
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_hcen_aux_ablation.py tests/test_hcen_vpr.py tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py tests/test_slgd_mid_supervision.py
train_stage() {
    local stage=$1
    for ARM in last_noaux hierarchy_noaux last_lowaux hierarchy_lowaux; do
        local extra=()
        if test -d "logs/hcen_aux_ablation_v1/${ARM}_${stage}"; then extra+=(--resume); fi
        "$PYTHON" -u scripts/train_hcen_aux_ablation.py train --stage "$stage" --arm "$ARM" \
            --checkpoint "$RU_CKPT" --output "logs/hcen_aux_ablation_v1/${ARM}_${stage}" "${extra[@]}"
    done
    local extra=()
    if test -d "logs/hcen_aux_ablation_v1/compare_${stage}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/train_hcen_aux_ablation.py compare --stage "$stage" --checkpoint "$RU_CKPT" \
        --output "logs/hcen_aux_ablation_v1/compare_${stage}" "${extra[@]}"
}
if test "$STAGE" != screen; then train_stage smoke; fi
if test "$STAGE" != smoke; then
    "$PYTHON" -c 'from pathlib import Path; from scripts.train_slgd import verify,read; p=Path("logs/hcen_aux_ablation_v1/compare_smoke"); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS"'
    train_stage screen
fi
echo 'HCEN 2x2 screen complete; not full training or guaranteed improvement.'
