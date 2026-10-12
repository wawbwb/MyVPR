#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
STAGE=${1:-all}
case "$STAGE" in all|baseline|smoke|screen) ;; *) echo 'Usage: bash scripts/run_hcen_plain_base.sh all|baseline|smoke|screen'; exit 2 ;; esac
mkdir -p logs/hcen_plain_base_v1
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_hcen_plain_base.py tests/test_hcen_replication.py tests/test_hcen_aux_ablation.py tests/test_hcen_vpr.py tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py tests/test_slgd_mid_supervision.py
extra=()
if test -d logs/hcen_plain_base_v1/baseline; then extra+=(--resume); fi
"$PYTHON" -u scripts/hcen_plain_base.py baseline --output logs/hcen_plain_base_v1/baseline "${extra[@]}"
if test "$STAGE" = baseline; then exit 0; fi
train_stage() {
    local stage=$1
    for MODE in last_only hierarchy; do
        local extra=()
        if test -d "logs/hcen_plain_base_v1/${MODE}_${stage}"; then extra+=(--resume); fi
        "$PYTHON" -u scripts/hcen_plain_base.py train --mode "$MODE" --stage "$stage" \
            --output "logs/hcen_plain_base_v1/${MODE}_${stage}" "${extra[@]}"
    done
    local extra=()
    if test -d "logs/hcen_plain_base_v1/compare_${stage}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/hcen_plain_base.py compare --stage "$stage" \
        --output "logs/hcen_plain_base_v1/compare_${stage}" "${extra[@]}"
}
if test "$STAGE" != screen; then train_stage smoke; fi
if test "$STAGE" != smoke; then
    "$PYTHON" -c 'from pathlib import Path; from scripts.train_slgd import verify,read; p=Path("logs/hcen_plain_base_v1/compare_smoke"); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS"'
    train_stage screen
fi
echo 'Plain-BoQ base transfer screen complete; not a pure gate ablation or stability claim.'
