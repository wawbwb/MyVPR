#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
exec 9>logs/query_relation_screen_v1.lock
flock -n 9
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
for stage in smoke pilot; do
    for mode in boq conv mamba mamba_consistent; do
        /home/wt/.conda/envs/VPR/bin/python -u scripts/train_scan_mamba.py \
            --mode "$mode" --stage "$stage" --checkpoint "$RU_CKPT" \
            --output "logs/scan_mamba_v2/${mode}_${stage}" --resume
    done
done
/home/wt/.conda/envs/VPR/bin/python -u scripts/check_scan_mamba_gate.py --root logs/scan_mamba_v2
for mode in boq conv mamba mamba_consistent; do
    /home/wt/.conda/envs/VPR/bin/python -u scripts/train_scan_mamba.py \
        --mode "$mode" --stage full --checkpoint "$RU_CKPT" \
        --output "logs/scan_mamba_v2/${mode}_full" --resume
done
