#!/usr/bin/env bash
# Existing VPR environment, physical GPU1, fresh immutable outputs.
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
STAGE=${1:-all}
case "$STAGE" in all|smoke|pilot) ;; *) echo 'Usage: bash scripts/run_slgd_screen.sh all|smoke|pilot'; exit 2 ;; esac
test -f "$RU_CKPT"
mkdir -p logs/slgd_v1
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_slgd.py
run_one() {
    local mode=$1 stage=$2
    local extra=()
    local out="logs/slgd_v1/${mode}_${stage}"
    if test "$mode" != teacher; then extra+=(--teacher "logs/slgd_v1/teacher_${stage}"); fi
    if test -d "$out"; then extra+=(--resume); fi
    local log
    log=$(mktemp "logs/slgd_v1/${mode}_${stage}_XXXXXX.log")
    "$PYTHON" -u scripts/train_slgd.py --mode "$mode" --stage "$stage" \
        --checkpoint "$RU_CKPT" --output "$out" "${extra[@]}" 2>&1 | tee "$log"
}
gate() {
    local report
    report=$(mktemp -u "logs/slgd_v1/${1}_gate_XXXXXX.json")
    "$PYTHON" -u scripts/check_slgd_gate.py --root logs/slgd_v1 --stage "$1" --output "$report"
}
if test "$STAGE" != pilot; then
    for MODE in teacher retrieval local_distill; do run_one "$MODE" smoke; done
    gate smoke
fi
if test "$STAGE" != smoke; then
    "$PYTHON" -c 'from pathlib import Path; from scripts.check_slgd_gate import compare; assert compare(Path("logs/slgd_v1"),"smoke")["verdict"]=="PASS", "Run smoke first"'
    run_one teacher pilot
    gate teacher
    run_one retrieval pilot
    run_one local_distill pilot
    gate pilot
fi
echo 'SLGD scheduled stages complete. No MSLS/Pitts sweep or CLIP stage launched.'
