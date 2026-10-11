#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
STAGE=${1:-all}
case "$STAGE" in all|smoke|pilot) ;; *) echo 'Usage: bash scripts/run_slgd_mid_screen.sh all|smoke|pilot'; exit 2 ;; esac
mkdir -p logs/slgd_mid_v1
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py tests/test_slgd_mid_supervision.py
prepare() {
    local stage=$1 source=audit
    if test "$stage" = smoke; then source=smoke; fi
    local extra=()
    if test -d ".cache/slgd_mid_v1_${stage}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/slgd_mid_screen.py prepare --stage "$stage" --checkpoint "$RU_CKPT" \
        --source "logs/slgd_hard_v2_${source}" --cache ".cache/slgd_mid_v1_${stage}" "${extra[@]}"
}
train_stage() {
    local stage=$1 source=audit
    if test "$stage" = smoke; then source=smoke; fi
    for MODE in retrieval last12 mid10; do
        local extra=()
        if test -d "logs/slgd_mid_v1/${MODE}_${stage}"; then extra+=(--resume); fi
        "$PYTHON" -u scripts/slgd_mid_screen.py train --stage "$stage" --mode "$MODE" \
            --checkpoint "$RU_CKPT" --source "logs/slgd_hard_v2_${source}" \
            --cache ".cache/slgd_mid_v1_${stage}" --output "logs/slgd_mid_v1/${MODE}_${stage}" "${extra[@]}"
    done
    "$PYTHON" -u scripts/slgd_mid_screen.py compare --stage "$stage" --checkpoint "$RU_CKPT" \
        --source "logs/slgd_hard_v2_${source}" --cache ".cache/slgd_mid_v1_${stage}" \
        --root logs/slgd_mid_v1 --output "logs/slgd_mid_v1/compare_${stage}"
}
if test "$STAGE" != pilot; then prepare smoke; train_stage smoke; fi
if test "$STAGE" != smoke; then
    "$PYTHON" -c 'from pathlib import Path; from scripts.train_slgd import verify,read; p=Path("logs/slgd_mid_v1/compare_smoke"); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS"'
    prepare pilot
    train_stage pilot
fi
echo 'Fixed-last three-arm visual supervision screen complete; no CLIP or extended training.'
