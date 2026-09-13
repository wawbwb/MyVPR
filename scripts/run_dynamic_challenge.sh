#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -p test_dynamic_challenge.py
if [[ ! -f .cache/dynamic_challenge_donors_v1/completed.json ]]; then
  python -u scripts/dynamic_challenge.py bank --device cuda:1 --output .cache/dynamic_challenge_donors_v1
fi
if [[ ! -f doc/dynamic_challenge_images_v1/completed.json ]]; then
  python -u scripts/dynamic_challenge.py build --output doc/dynamic_challenge_images_v1
fi
if [[ ! -f .cache/dynamic_challenge_clip_v1/completed.json ]]; then
  python -u scripts/dynamic_challenge.py clip --device cuda:1 --output .cache/dynamic_challenge_clip_v1
fi
python -u scripts/dynamic_challenge.py evaluate --device cuda:1 --output doc/dynamic_challenge_eval_v1
