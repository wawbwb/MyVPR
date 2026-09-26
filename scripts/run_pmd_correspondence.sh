#!/usr/bin/env bash
set -euo pipefail
cd /home/wt/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
PY=/home/wt/.conda/envs/VPR/bin/python
"$PY" -u scripts/pmd_correspondence_gate.py --smoke --resume --output logs/pmd/correspondence_smoke_v1
"$PY" -u scripts/pmd_correspondence_gate.py --resume
