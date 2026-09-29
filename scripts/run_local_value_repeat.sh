#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PY=/home/wt/.conda/envs/VPR/bin/python
RU='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
for SEED in 43 44; do
  for MODE in local_contrast shuffled_contrast; do
    "$PY" -u scripts/train_local_value_repeat.py --mode "$MODE" --init-seed "$SEED" --checkpoint "$RU" --output "logs/local_value_repeat/${MODE}_seed${SEED}_smoke_v1" --smoke --resume
  done
done
for SEED in 43 44; do
  for MODE in local_contrast shuffled_contrast; do
    "$PY" -u scripts/train_local_value_repeat.py --mode "$MODE" --init-seed "$SEED" --checkpoint "$RU" --output "logs/local_value_repeat/${MODE}_seed${SEED}_screen_v1" --resume
  done
done
"$PY" -m scripts.summarize_local_value_repeat
