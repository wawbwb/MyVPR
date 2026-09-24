#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PY=/home/wt/.conda/envs/VPR/bin/python
STAGE=${1:?Use smoke or train}
case "$STAGE" in
  smoke) SUFFIX=smoke_v1; EXTRA=(--smoke) ;;
  train) SUFFIX=screen_v1; EXTRA=() ;;
  *) echo 'Use smoke or train'; exit 2 ;;
esac
for MODE in baseline ordinary forced partial; do
  "$PY" -u scripts/train_pmd.py --mode "$MODE" --output "logs/pmd/${MODE}_${SUFFIX}" --resume "${EXTRA[@]}"
done
