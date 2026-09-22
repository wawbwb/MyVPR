# WPPR actual decoder continuation gate

Frozen extension result: remaining1664 retained1663 full44 winners; tail29/30.
Selected12 and full44 both1628 correct, fixed20=1620. Corrections10/regressions2.
This supports a runtime experiment, not a stable cross-domain or speed claim.

## Locked first stage

Use first32 sorted remaining GSV query IDs; no outcome-based selection. Do not
retrain the epoch1 head. Strict official model identity and cached teacher scores
must match. FP32, physical GPU1, one pair per direction for all methods.

Run44 candidates for2 layers in both directions. Keep both intermediate streams
on GPU. Use frozen head to select12; continue those states through layers3–12.
No re-encoding or prefix recomputation in the timed decoder path.

Check all44 reference scores against old extraction, selected12 scores against
direct inference, and shortlist against old layer2 features. Fail on mismatch.
Three rotated-order measurements per query after untimed warm-up. Summaries use
per-query median times; also report peak allocated GPU memory.

This is **decoder-only** latency with all44 dense inputs resident on GPU. It
excludes image encoding, retrieval, disk I/O; it is not an end-to-end speed claim.
The additional need to encode24 candidate images may offset decoder savings.
Shared GPU workloads can bias timing and should be checked before execution.

Server commands:

```bash
python -m pytest -q tests/test_wppr_runtime.py tests/test_wppr_extension.py tests/test_wppr_pilot.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/wppr_runtime_benchmark.py
```

Output: doc/wppr_runtime_v1. A completed run verifies hashes and exits. An
interrupted run restarts the fixed32 timing set to avoid mixing partial repeats.

## Next gate, not launched by this script

If correctness passes and latency supports continuing, implement full-pipeline
timing including feature extraction/cache transfers and fixed-budget baselines.
Then evaluate the same frozen head and12-survivor policy on existing Pitts30k-val
without tuning there. Do not label the first32 GSV timing set a new accuracy test.
