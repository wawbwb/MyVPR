#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
mkdir -p logs/slgd_decomposition_v1
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py tests/test_slgd_mid_supervision.py tests/test_slgd_decomposition.py
for STAGE in smoke pilot; do
    SOURCE=audit
    if test "$STAGE" = smoke; then SOURCE=smoke; fi
    EXTRA=()
    if test -d "logs/slgd_decomposition_v1/$STAGE"; then EXTRA+=(--resume); fi
    "$PYTHON" -u scripts/audit_slgd_decomposition.py --stage "$STAGE" --checkpoint "$RU_CKPT" \
        --source "logs/slgd_hard_v2_$SOURCE" --output "logs/slgd_decomposition_v1/$STAGE" \
        --cache ".cache/slgd_decomposition_v1_$STAGE" "${EXTRA[@]}"
done
echo 'Fixed-checkpoint decomposition complete; no training or weight search.'
