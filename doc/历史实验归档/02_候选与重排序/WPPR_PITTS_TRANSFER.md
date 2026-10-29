# Frozen WPPR Pitts transfer

No retraining, threshold tuning, or survivor-count sweep. Same GSV epoch1 head,
44 candidates, 12 survivors, FP32 official Pair-VPR. Test ALL7608 Pitts30k-val
queries against existing top20/top44 scores. This dataset has historical exposure;
it is a frozen cross-dataset transfer check, not an untouched final test set.

First stage: fixed first32 original query IDs, three rotated repeats of top20,
top44 and WPPR. Includes query image read/encode, global retrieval, candidate
image read/encode, GPU transfers, decoder and selection. Database GLOBAL vectors
are resident; database DENSE features are NOT persisted. No inter-query dense
reuse during timing. OS filesystem caches are uncontrolled, so not a disk-cold
benchmark. Audits/startup excluded. This deliberately tests whether additional
24 image encodings offset decoder savings. It does not represent a system with
a fully precomputed dense-feature bank. No large feature cache is created.

Second stage: all7608 queries; existing44 scores are the reference, selected
continuation scores must reproduce within1e-4. A CPU LRU64 of dense features
reduces extraction cost; its throughput is not used for speed claims. Small
per-query score shards are checksummed and resumable. No online accuracy stopping.
Report paired corrections/regressions vs20 and44, winner/tail retention, and
panorama-cluster bootstrap intervals. Model/image/plan identities are checked.

```bash
python -m pytest -q tests/test_wppr_pitts_transfer.py tests/test_wppr_runtime.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/wppr_pitts_transfer.py
```

Output doc/wppr_pitts_transfer_v1. Interrupted transfer resumes verified shards;
interrupted first32 timing restarts the timing set. Expected runtime must be
estimated from the live training machine, not from theoretical layer counts.

## Audited near-tie recovery

The first run stopped after1981 queries: q1981 ranks29/30 swapped due to a
2.20e-7 descriptor difference. Same44 candidate set; cached query global vector
reproduced the original ranking. The fix accepts ONLY same-set permutations
with descriptor max absolute drift<=1e-6, BOTH recomputed/cached score gaps<=1e-6,
and exact frozen-order reproduction from cached descriptors. All differences
are written to retrieval_ties/queryID.json. Missing cached query descriptor,
different candidate set, larger drift or non-tied rank changes still abort.
Timing remains strict online retrieval and is not retroactively changed.

Known338105b incomplete runs can migrate with identical head/data/runtime hashes.
The original contract, hashes of retained verified shards and timing report are
recorded in legacy_migration.json. This is an audited protocol amendment, not
an unmodified prospective experiment. All1981 prior shards are retained.

## Frozen-candidate accuracy protocol amendment

The next stop at q3666 involved a tied44/45 boundary changing candidate membership.
Accuracy evaluation now takes verified reference candidate IDs/order for EVERY
query, independently of online retrieval. Images, model identity and selected
pair scores remain checked. Prior3666 rows already used these exact frozen
lists; their teacher/labels/shortlist/scores are revalidated on resume. Only
known original/tie-fix contracts can migrate. frozen_candidate_migration.json
preserves their contract and shard/timing hashes without overwriting the earlier
legacy migration audit. No training, survivor-count change or label-based choice.

Existing first32 online timing is unchanged. Accuracy is explicitly conditional
on fixed retrieved candidates, NOT an online retrieval robustness measurement.
This amendment follows an observed numerical boundary issue and must be disclosed.
