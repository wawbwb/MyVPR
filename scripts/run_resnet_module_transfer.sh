#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
PYTHON=/home/wt/.conda/envs/VPR/bin/python
ACTION=${1:-preflight}
case "$ACTION" in
  tests|prepare|preflight|baseline|smoke|train|aggregate|all) ;;
  *) echo 'Choose tests/prepare/preflight/baseline/smoke/train/aggregate/all'; exit 2 ;;
esac
if [[ "$ACTION" = tests ]]; then
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q tests/test_resnet_module_transfer.py
  exit 0
fi
if [[ "$ACTION" = prepare ]]; then
  "$PYTHON" -u scripts/prepare_resnet_boq.py
  exit 0
fi
ROOT=logs/resnet_module_transfer_v1
resume_run() {
  local command=$1 name=$2
  shift 2
  local -a extra=()
  if [[ -d "$ROOT/$name" ]]; then extra+=(--resume); fi
  "$PYTHON" -u scripts/resnet_module_transfer.py "$command" --root "$ROOT" "$@" "${extra[@]}"
}
if [[ "$ACTION" = preflight ]]; then
  CUDA_VISIBLE_DEVICES='' resume_run preflight preflight
  exit 0
fi
export CUDA_VISIBLE_DEVICES=1
ARMS=(conv_last conv_cross conditional_last conditional_cross)
if [[ "$ACTION" = baseline || "$ACTION" = all ]]; then
  resume_run baseline baseline
fi
if [[ "$ACTION" = smoke || "$ACTION" = all ]]; then
  for arm in "${ARMS[@]}"; do
    resume_run train "smoke_$arm" --stage smoke --arm "$arm" --allow-training
  done
  resume_run aggregate smoke_aggregate --stage smoke
fi
if [[ "$ACTION" = train || "$ACTION" = all ]]; then
  for arm in "${ARMS[@]}"; do
    resume_run train "train_$arm" --stage train --arm "$arm" --allow-training
  done
fi
if [[ "$ACTION" = aggregate || "$ACTION" = train || "$ACTION" = all ]]; then
  resume_run aggregate train_aggregate --stage train
fi
echo 'Requested stage complete. No extra seeds, weight/rho search, or automatic extensions.'
