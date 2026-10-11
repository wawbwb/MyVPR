#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
REFERENCE_PAPER=${REFERENCE_PAPER:-/tmp/vpr_reid_m4IkDH/paper.pdf}
STAGE=${1:-all}
case "$STAGE" in all|smoke|pilot) ;; *) echo 'Usage: bash scripts/run_hcen_vpr.sh all|smoke|pilot'; exit 2 ;; esac
test -f "$REFERENCE_PAPER"
mkdir -p logs/hcen_vpr_v2
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_hcen_vpr.py tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py tests/test_slgd_mid_supervision.py
train_stage() {
    local stage=$1 source=audit
    if test "$stage" = smoke; then source=smoke; fi
    for MODE in last_only hierarchy hierarchy_rank; do
        local extra=()
        if test -d "logs/hcen_vpr_v2/${MODE}_${stage}"; then extra+=(--resume); fi
        "$PYTHON" -u scripts/train_hcen_vpr.py train --stage "$stage" --mode "$MODE" \
            --checkpoint "$RU_CKPT" --source "logs/slgd_hard_v2_${source}" \
            --cache ".cache/slgd_mid_v1_${stage}" --reference-paper "$REFERENCE_PAPER" \
            --output "logs/hcen_vpr_v2/${MODE}_${stage}" "${extra[@]}"
    done
    local extra=()
    if test -d "logs/hcen_vpr_v2/compare_${stage}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/train_hcen_vpr.py compare --stage "$stage" --checkpoint "$RU_CKPT" \
        --source "logs/slgd_hard_v2_${source}" --cache ".cache/slgd_mid_v1_${stage}" \
        --reference-paper "$REFERENCE_PAPER" --output "logs/hcen_vpr_v2/compare_${stage}" "${extra[@]}"
}
if test "$STAGE" != pilot; then train_stage smoke; fi
if test "$STAGE" != smoke; then
    "$PYTHON" -c 'from pathlib import Path; from scripts.train_slgd import verify,read; p=Path("logs/hcen_vpr_v2/compare_smoke"); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS"'
    train_stage pilot
fi
echo 'HCEN fixed-last matched screen complete; not full training or a guaranteed gain.'
