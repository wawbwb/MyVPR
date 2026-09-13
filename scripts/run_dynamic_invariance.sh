#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

plan=doc/dynamic_invariance_plan_v1
if [[ ! -f "$plan/completed.json" ]]; then
  python -u scripts/train_dynamic_invariance.py prepare --output "$plan"
fi
python -u scripts/train_dynamic_invariance.py preflight --device cuda:1 \
  --output "doc/dynamic_invariance_preflight_$(date +%Y%m%d_%H%M%S)"
for mode in plain random semantic; do
  out="doc/dynamic_invariance_train_v1/$mode"
  if [[ -f "$out/completed.json" ]]; then
    echo "Retaining completed $mode; evaluation will verify contracts and checkpoint hashes"
    continue
  fi
  extra=()
  if [[ -f "$out/last.pt" ]]; then extra=(--resume); fi
  python -u scripts/train_dynamic_invariance.py train --device cuda:1 \
    --mode "$mode" --output "$out" "${extra[@]}"
done
python -u scripts/train_dynamic_invariance.py evaluate --device cuda:1 \
  --output doc/dynamic_invariance_dev_v1
