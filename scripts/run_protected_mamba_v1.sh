#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
exec 9>logs/query_relation_screen_v1.lock
flock -n 9
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
PYTHON=/home/wt/.conda/envs/VPR/bin/python
"$PYTHON" -m pytest -q tests/test_scan_mamba.py tests/test_scan_mamba_protocol.py tests/test_protected_mamba.py
for stage in smoke pilot; do
    for mode in conv_preserved mamba_plain mamba_preserved; do
        "$PYTHON" -u scripts/train_protected_mamba.py --mode "$mode" --stage "$stage" \
            --checkpoint "$RU_CKPT" --output "logs/protected_mamba_v1/${mode}_${stage}" --resume
    done
done
"$PYTHON" -u scripts/check_protected_mamba_gate.py --root logs/protected_mamba_v1
for mode in conv_preserved mamba_plain mamba_preserved; do
    "$PYTHON" -u scripts/train_protected_mamba.py --mode "$mode" --stage full \
        --checkpoint "$RU_CKPT" --output "logs/protected_mamba_v1/${mode}_full" --resume
done
