#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
STAGE=${1:-all}
case "$STAGE" in all|smoke|audit) ;; *) echo 'Usage: bash scripts/run_slgd_layers.sh all|smoke|audit'; exit 2 ;; esac
test -f "$RU_CKPT"
mkdir -p logs/slgd_layers_queue
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_slgd.py tests/test_slgd_hard_protocol.py tests/test_slgd_layers.py
run_audit() {
    local suffix=$1
    local source_suffix=audit
    local extra=()
    if test "$suffix" = smoke; then source_suffix=smoke; extra+=(--smoke); fi
    if test -d "logs/slgd_layers_v1_${suffix}"; then extra+=(--resume); fi
    "$PYTHON" -u scripts/audit_slgd_layers.py --checkpoint "$RU_CKPT" \
        --source "logs/slgd_hard_v2_${source_suffix}" --source-cache ".cache/slgd_hard_v2_${source_suffix}" \
        --output "logs/slgd_layers_v1_${suffix}" --cache ".cache/slgd_layers_v1_${suffix}" "${extra[@]}"
}
if test "$STAGE" != audit; then run_audit smoke; fi
if test "$STAGE" != smoke; then
    "$PYTHON" -c 'from pathlib import Path; from scripts.train_slgd import verify,read; p=Path("logs/slgd_layers_v1_smoke"); verify(p); assert read(p/"summary.json")["verdict"]=="SMOKE_PASS"'
    run_audit audit
fi
echo 'Frozen layer diagnostic complete; no new teacher/student/CLIP training.'
