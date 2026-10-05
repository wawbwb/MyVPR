#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
exec 9>logs/query_relation_screen_v1.lock
flock -n 9
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
for stage in smoke formal; do
    for sampling in mixed broad_matched; do
        for lr in 1e-4 1e-5; do
            extra=()
            suffix=''
            if [ "$stage" = smoke ]; then extra+=(--smoke); suffix='_smoke'; fi
            /home/wt/.conda/envs/VPR/bin/python -u scripts/train_query_relation.py \
                --mode appearance --sampling "$sampling" --learning-rate "$lr" \
                --checkpoint "$RU_CKPT" --output "logs/query_protocol_screen_v1/${sampling}_${lr}${suffix}" \
                --resume "${extra[@]}"
        done
    done
done
