# Frozen top20 versus top44: full Pitts extension

## Locked question

Does the previously selected frozen Pair-VPR top44 reranking retain a positive
paired R@1 difference over top20 on the remaining 6584 Pitts30k-val queries?
Reuse the original 1024 query results and the full 10000-reference database.
No training, weight selection, K search, score calibration, or accuracy-based
early stopping is permitted. Both variants slice the same FP32 directional-sum
scores. Original query indices, not subset positions, identify all outcomes.

The remaining sample is NOT an independent test set: Pitts was used previously
and panorama groups may overlap the original development subset. Report that
overlap explicitly. This experiment tests extension within Pitts, not semantic
benefit, model novelty, or general stable gains across unseen datasets.

## Statistics

Primary: paired R@1 difference on remaining queries. Resample panorama IDs
(filename stem before pitch/yaw, retaining parent path), keeping all their views
and paired outcomes together. Use 10000 cluster bootstrap draws, seed42,
query-weighted R@1 and a percentile 95% interval. Positive net with lower bound
above zero supports this sample; positive net with interval crossing zero is
inconclusive. Nearby panoramas can still be correlated. Query-IID exact McNemar
is supplementary, not the primary test. No correction for past model selection
is implied. Prior1024 and all7608 metrics are descriptive, not new independent
confirmations. Preserve corrections, regressions, coverage and R5/10/20.

## Execution and recovery

Run on training server only:

```bash
bash scripts/run_pitts_top44_confirmation.sh
```

The pipeline tests helpers, locks image/index/model/code hashes, times the first
16 fixed remaining queries, resumes all 6584, and writes the report only after
completion. No interim accuracy is printed. The pilot is saved and reused.
Each query is atomically saved with a checksum; rerunning resumes completed
queries. Separate process/pipeline locks prevent concurrent writers. A corrupt
checksum fails explicitly rather than silently trusting partial output.

Work: `.cache/pitts_top44_confirmation_v1`; report:
`doc/pitts_top44_confirmation_v1`. No large dense feature cache is created;
reference features use a bounded 64-entry CPU LRU. 289696 pairs require 579392
directional forwards. Measure runtime before estimating completion, not before
deciding whether results are favorable. Timing includes reference encoding and
natural LRU reuse; first20/next24 sections are not independent cold/warm latency
benchmarks. Existing1024 scores are checked against their original report.

```bash
python scripts/pitts_top44_confirmation.py status
```

After completion, archive only the compact reports and this protocol, not the
per-query model cache. An independent unseen split remains necessary before a
broad claim of stable generalization.
