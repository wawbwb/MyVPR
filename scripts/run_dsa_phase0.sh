#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
exec /home/wt/.conda/envs/VPR/bin/python -u scripts/dsa_subspace_check.py --checkpoint "$RU_CKPT" --output logs/dsa_phase0_v1 --resume
