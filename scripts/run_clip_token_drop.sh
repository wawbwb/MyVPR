#!/usr/bin/env bash
# Run from repository root, with the VPR environment already activated.
set -euo pipefail
cd "$(dirname "$0")/.."

# Always rerun preflight, including on a resumed pipeline; it must not skip tests.
python scripts/clip_token_drop.py preflight --device cuda:1 \
  --output "doc/clip_token_drop_preflight_$(date +%Y%m%d_%H%M%S)"

python scripts/clip_token_drop.py cache --split train --device cuda:1 \
  --output .cache/clearclip_token_drop_train_v1
python scripts/clip_token_drop.py cache --split msls --device cuda:1 \
  --output .cache/clearclip_token_drop_msls_v1

for mode in none aligned shuffled; do
  out="doc/clip_token_drop_train_v1/$mode"
  if [[ -f "$out/completed.json" ]]; then
    echo "Completed training retained: $mode (evaluation will verify its contract)"
    continue
  fi
  resume=()
  if [[ -d "$out" ]]; then resume=(--resume); fi
  python scripts/clip_token_drop.py train --mode "$mode" --device cuda:1 \
    --cache .cache/clearclip_token_drop_train_v1 --output "$out" "${resume[@]}"
done

if [[ -f doc/clip_token_drop_eval_v1/completed.json ]]; then
  echo "Evaluation already completed: doc/clip_token_drop_eval_v1/report"
else
  python scripts/clip_token_drop.py evaluate --device cuda:1 \
    --cache .cache/clearclip_token_drop_msls_v1 \
    --runs doc/clip_token_drop_train_v1 --output doc/clip_token_drop_eval_v1
fi
