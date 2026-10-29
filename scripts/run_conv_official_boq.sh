#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
ROOT=logs/conv_official_boq_v1
mkdir -p "$ROOT"
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the shared lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_official_boq.py tests/test_conv_residual_ablation.py tests/test_conv_fusion_controls.py
"$PYTHON" -u scripts/conv_official_boq.py wait_gpu
run() {
    local command=$1 output=$2
    shift 2
    local extra=()
    if test -d "$output"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/conv_official_boq.py "$command" --output "$output" "$@" "${extra[@]}"
}
run baseline "$ROOT/baseline"
for SEED in 42 43 44; do run train "$ROOT/seed${SEED}/smoke" --seed "$SEED" --stage smoke; done
for SEED in 42 43 44; do
    "$PYTHON" -c 'import sys; from pathlib import Path; from scripts.train_slgd import verify,read; p=Path(sys.argv[1]); verify(p); assert all(read(p/"summary.json")["checks"].values())' "$ROOT/seed${SEED}/smoke"
done
for SEED in 42 43 44; do run train "$ROOT/seed${SEED}/train" --seed "$SEED" --stage train; done
run aggregate "$ROOT/aggregate"
echo 'Official BoQ transfer complete; no selection of best seed, size or epoch.'
