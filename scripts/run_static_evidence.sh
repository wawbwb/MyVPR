#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u scripts/static_evidence.py preflight --device cuda:1 \
  --output "doc/static_evidence_preflight_$(date +%Y%m%d_%H%M%S)"
if [[ ! -f .cache/static_evidence_train_v1/completed.json ]]; then
  python -u scripts/static_evidence.py prepare --split train --device cuda:1 \
    --output .cache/static_evidence_train_v1
fi
for mode in full random static; do
  out="doc/static_evidence_train_v1/$mode"
  if [[ -f "$out/completed.json" ]]; then
    echo "Retaining completed $mode; evaluation verifies hashes"
    continue
  fi
  extra=()
  if [[ -f "$out/last.pt" ]]; then extra=(--resume); fi
  python -u scripts/static_evidence.py train --mode "$mode" --device cuda:1 \
    --output "$out" "${extra[@]}"
done
if [[ ! -f .cache/static_evidence_msls_v1/completed.json ]]; then
  python -u scripts/static_evidence.py prepare --split msls --device cuda:1 \
    --output .cache/static_evidence_msls_v1
fi
python -u scripts/static_evidence.py evaluate --device cuda:1 \
  --output doc/static_evidence_eval_v1
