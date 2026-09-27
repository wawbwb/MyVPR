#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PY=/home/wt/.conda/envs/VPR/bin/python
for MODE in baseline pointwise spatial shuffled; do
  "$PY" -u scripts/train_consensus.py --mode "$MODE" --output "logs/consensus/${MODE}_smoke_v1" --smoke --resume
done
for MODE in baseline pointwise spatial shuffled; do
  "$PY" -u scripts/train_consensus.py --mode "$MODE" --output "logs/consensus/${MODE}_screen_v1" --resume
done
"$PY" scripts/summarize_consensus.py
