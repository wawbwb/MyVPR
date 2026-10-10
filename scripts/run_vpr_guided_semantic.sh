#!/usr/bin/env bash
# GPU1, existing GSV/ADE20K cache, no new data download.
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/wt/.conda/envs/VPR/bin/python
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
CONFIG=config/boq_dinov2_vpr_guided_semantic.yaml
STAGE=${1:-smoke}
case "$STAGE" in smoke|train) ;; *) echo 'Usage: bash scripts/run_vpr_guided_semantic.sh smoke|train'; exit 2 ;; esac
test -f "$RU_CKPT"
mkdir -p logs/vpr_guided_semantic_v1
exec 9>logs/query_relation_screen_v1.lock
flock -n 9 || { echo 'Another experiment holds the GPU experiment lock'; exit 1; }
"$PYTHON" -m pytest -q tests/test_seg_aux.py tests/test_vpr_guided_semantic.py
for MODE in vpr_only plain guided; do
    OUT="logs/vpr_guided_semantic_v1/${MODE}_${STAGE}"
    if test -f "$OUT/completed.json"; then
        "$PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads((p/"completed.json").read_text()); c=json.loads((p/"contract.json").read_text()); from src.models.query_semantic import _file_sha256; assert c["config"]["seg_aux"]["mode"]==sys.argv[2]; assert c["smoke_test"]==(sys.argv[3]=="smoke"); assert d["status"]==("SMOKE PASS" if sys.argv[3]=="smoke" else "TRAINING COMPLETE (not an efficacy verdict)"); assert all(_file_sha256(Path(f))==h for f,h in c["source_sha256"].items()); print("Already complete:",p)' "$OUT" "$MODE" "$STAGE"
        continue
    fi
    EXTRA=()
    if test "$STAGE" = smoke; then
        EXTRA+=(--smoke-test)
    else
        test -f "logs/vpr_guided_semantic_v1/${MODE}_smoke/completed.json" || { echo 'Run smoke stage first'; exit 1; }
        if test -f "$OUT/checkpoints/last.ckpt"; then
            EXTRA+=(--resume "$OUT/checkpoints/last.ckpt")
        fi
    fi
    LOG=$(mktemp "logs/vpr_guided_semantic_v1/${MODE}_${STAGE}_XXXXXX.log")
    "$PYTHON" -u scripts/train_seg_aux.py --config "$CONFIG" --mode "$MODE" \
        --init-checkpoint "$RU_CKPT" --device 0 --output "$OUT" "${EXTRA[@]}" 2>&1 | tee "$LOG"
done
