#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PY=/home/wt/.conda/envs/VPR/bin/python
RU='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
for MODE in mean_outer global_cov query_cov; do
  "$PY" -u scripts/train_second_order_query.py --mode "$MODE" --checkpoint "$RU" --output "logs/second_order_query/${MODE}_smoke_v1" --smoke --resume
done
for MODE in mean_outer global_cov query_cov; do
  "$PY" -u scripts/train_second_order_query.py --mode "$MODE" --checkpoint "$RU" --output "logs/second_order_query/${MODE}_screen_v1" --resume
done
"$PY" -m scripts.summarize_second_order_query
