#!/usr/bin/env bash
# Re-runnable stage driver. Run from the activated VPR environment.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
stage="${1:-all}"
case "$stage" in
  targets|smoke|teacher|calibrate|features|audit|all) ;;
  *) echo "Usage: bash scripts/run_cc_lsa_gate_a.sh {targets|smoke|teacher|calibrate|features|audit|all}" >&2; exit 2 ;;
esac

target_cache='.cache/cc_lsa/gsv_crop_cls_v1'
teacher_checkpoint='logs/cc_lsa/teacher_seed42/final.pt'
calibration='.cache/cc_lsa/gsv_calibration_v1'
calibration_scratch='/tmp/cc_lsa_calibration_v1'
ru_descriptors='.cache/cc_lsa/ru_msls.npy'
features='.cache/cc_lsa/msls_top100_v1'
audit_output='doc/cc_lsa_gate_a'
mkdir -p doc/cc_lsa_runs

log_run() {
  local label="$1"
  shift
  local log_file
  log_file="$(mktemp "doc/cc_lsa_runs/${label}_$(date +%Y%m%d_%H%M%S)_XXXXXX.txt")"
  printf 'Running %s; log: %s\n' "$label" "$log_file"
  "$@" 2>&1 | tee "$log_file"
}

require_ru() {
  if [[ -z "${RU_CKPT:-}" || ! -f "$RU_CKPT" ]]; then
    echo 'Set RU_CKPT to the existing trained repeatability+uniqueness checkpoint.' >&2
    exit 2
  fi
}

require_teacher() {
  python -c 'import json, sys; from pathlib import Path; from src.cc_lsa_gate_a import file_sha256; p=Path(sys.argv[1]); r=json.loads((p.parent/"run.json").read_text()); c=r["teacher_contract"]; assert r["complete"] is True and r["verdict"] == c["verdict"] == "PASS", "LSA teacher contract FAIL: stop here"; assert r["checkpoint"]["sha256"] == file_sha256(p), "LSA checkpoint SHA mismatch"; assert c["config_sha256"] == file_sha256("config/cc_lsa_teacher.yaml"), "Teacher config changed"; print("LSA teacher contract PASS; checkpoint and config SHA verified")' "$teacher_checkpoint"
}

run_targets() {
  log_run targets python -u scripts/cache_gsv_cc_lsa_targets.py --dataset-root datasets/gsv_cities --output "$target_cache" --device cuda:1 --batch-size 16 --num-workers 8
}

run_smoke() {
  log_run smoke python -u scripts/train_cc_lsa_teacher.py --config config/cc_lsa_teacher.yaml --smoke-test
}

run_teacher() {
  if [[ -f logs/cc_lsa/teacher_seed42/run.json ]]; then
    echo 'Teacher output exists; verifying its final contract.'
  else
    log_run teacher_5ep python -u scripts/train_cc_lsa_teacher.py --config config/cc_lsa_teacher.yaml
  fi
  require_teacher
}

run_calibrate() {
  require_ru
  require_teacher
  if [[ -f "$calibration/calibration.json" ]]; then
    echo "Calibration exists: $calibration; full hashes are rechecked by the final audit."
  else
    log_run calibrate python -u scripts/calibrate_cc_lsa_gate_a.py --config config/cc_lsa_gate_a.yaml --dataset-root datasets/gsv_cities --ru-checkpoint "$RU_CKPT" --lsa-checkpoint "$teacher_checkpoint" --output "$calibration" --scratch-dir "$calibration_scratch" --device cuda:1 --batch-size 16 --num-workers 8
  fi
}

run_features() {
  require_ru
  require_teacher
  # Old AG-SLRD descriptor sidecars did not include index hashes. Write a
  # separate provenance-complete RU cache; leave the old experiment intact.
  if [[ ! -f "$ru_descriptors" || ! -f "$ru_descriptors.json" ]]; then
    log_run ru_descriptors python -u scripts/extract_ag_slrd_msls_descriptors.py ru --checkpoint "$RU_CKPT" --msls-path datasets/msls-val --output "$ru_descriptors" --device cuda:1 --batch-size 32 --num-workers 8 --image-size 280 280
  fi
  log_run msls_features python -u scripts/cache_cc_lsa_msls_features.py --config config/cc_lsa_gate_a.yaml --ru-checkpoint "$RU_CKPT" --ru-descriptors "$ru_descriptors" --lsa-checkpoint "$teacher_checkpoint" --msls-path datasets/msls-val --output "$features" --device cuda:1 --batch-size 16 --num-workers 8
}

run_audit() {
  require_teacher
  log_run gate_a python -u scripts/audit_cc_lsa_gate_a.py --config config/cc_lsa_gate_a.yaml --feature-cache "$features" --calibration "$calibration" --msls-path datasets/msls-val --output "$audit_output" --device cuda:1
  python -c 'import json; from pathlib import Path; r=json.loads(Path("doc/cc_lsa_gate_a/summary.json").read_text()); print("Gate A:", r["verdict"]["verdict"]); print("Pair classifier has not been implemented. Review Gate-A evidence before proceeding.")'
}

if [[ "$stage" == all ]]; then
  require_ru
  run_targets
  run_smoke
  run_teacher
  run_calibrate
  run_features
  run_audit
else
  "run_${stage}"
fi
