#!/usr/bin/env bash
# Run with bash; no activation, installation, or destructive cleanup inside.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${REGION_VLAD_GPU:-1}"
: "${RU_CKPT:?Set RU_CKPT to the original trained RU checkpoint}"
: "${SAM_CKPT:?Set SAM_CKPT to sam_vit_b_01ec64.pth}"
test -f "$RU_CKPT"
test -f "$SAM_CKPT"
TASK="${1:-smoke}"
FIT=.cache/region_vlad/fit_gsv_v1
CACHE=.cache/region_vlad/msls_v1
mkdir -p doc/region_vlad_runs
STAMP="$(date +%Y%m%d_%H%M%S)"
common=(--checkpoint "$RU_CKPT" --sam-checkpoint "$SAM_CKPT" --device cuda:0)
case "$TASK" in
  fit)
    python -u scripts/region_vlad_screen.py fit "${common[@]}" --dataset-root datasets/gsv_cities --output "$FIT" 2>&1 | tee "doc/region_vlad_runs/fit_${STAMP}.txt"
    ;;
  smoke)
    python -u scripts/region_vlad_screen.py cache "${common[@]}" --dataset-root datasets/msls-val --model "$FIT" --output .cache/region_vlad/smoke_v1 --limit-images 4 --resume 2>&1 | tee "doc/region_vlad_runs/smoke_${STAMP}.txt"
    ;;
  cache)
    python -u scripts/region_vlad_screen.py cache "${common[@]}" --dataset-root datasets/msls-val --model "$FIT" --output "$CACHE" --resume 2>&1 | tee "doc/region_vlad_runs/cache_${STAMP}.txt"
    ;;
  eval)
    python -u scripts/region_vlad_screen.py eval --dataset-root datasets/msls-val --cache "$CACHE" --output "doc/region_vlad_msls_${STAMP}" --device cuda:0 2>&1 | tee "doc/region_vlad_runs/eval_${STAMP}.txt"
    ;;
  *) echo 'Usage: bash scripts/run_region_vlad_screen.sh {fit|smoke|cache|eval}' >&2; exit 2 ;;
esac
