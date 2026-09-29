#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
exec 9>logs/dsa_screen_v1.lock
flock -n 9
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
for mode in pca shuffled_fisher place_fisher; do
    /home/wt/.conda/envs/VPR/bin/python -u scripts/train_dsa.py --mode "$mode" --checkpoint "$RU_CKPT" --output "logs/dsa_screen_v1/${mode}_smoke" --smoke --resume
done
for mode in pca shuffled_fisher place_fisher; do
    /home/wt/.conda/envs/VPR/bin/python -u scripts/train_dsa.py --mode "$mode" --checkpoint "$RU_CKPT" --output "logs/dsa_screen_v1/${mode}" --resume
done
