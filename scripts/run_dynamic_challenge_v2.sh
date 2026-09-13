#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -p 'test_dynamic_challenge*.py'
if [[ ! -f .cache/dynamic_challenge_donors_v2/completed.json ]]; then
  python -u scripts/dynamic_challenge_v2.py bank --device cuda:1 --output .cache/dynamic_challenge_donors_v2
fi
if [[ ! -f doc/dynamic_challenge_images_v2/completed.json ]]; then
  python -u scripts/dynamic_challenge_v2.py build --output doc/dynamic_challenge_images_v2
fi
if [[ ! -f .cache/dynamic_challenge_clip_v2/completed.json ]]; then
  python -u scripts/dynamic_challenge_v2.py clip --device cuda:1 --output .cache/dynamic_challenge_clip_v2
fi
python -u scripts/dynamic_challenge_v2.py evaluate --device cuda:1 --output doc/dynamic_challenge_eval_v2
