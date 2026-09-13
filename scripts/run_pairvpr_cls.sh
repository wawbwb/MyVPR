#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p 'test_pairvpr_cls*.py'
python -u scripts/diagnose_pairvpr_cls.py \
  --official-repo /home/wt/workspace/Pair-VPR-official \
  --msls-path datasets/msls-val \
  --cache .cache/clearclip_token_drop_msls_v1 \
  --audit doc/pairvpr_official_paired_audit_v1 \
  --output "${1:-doc/pairvpr_cls_diagnostic_v1}"
