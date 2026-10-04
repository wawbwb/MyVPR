# QR-BoQ fixed-last matched screen v1

User approved fixed-last protocol after the expanded-gallery audit. This file
supersedes the proposal's GSV checkpoint-selection plan, before formal training.
Do not modify DSA records or call QR preflight a performance improvement.

Preflight: three20736-parameter branches passed real updates/frozen checks.
RU expanded GSV gallery20480, queries4096:4064 correct (99.21875%),32 errors in27
places. Three errors involve the same pair of places with minimum recorded
location distance6.76m; other29 have minimum distance>100m. This is a place-level
metadata check, not sampled-image GPS or a corrected GT. GSV remains saturated.

## Locked protocol

- Three modes: appearance, aligned, shuffled. Same20736 parameter tensors,
  zero-output initialization seed42, frozen original RU and frozen BoQ.
- Three epochs,320 batches each:256 broad batches cover4096 places once,
  64 historical hard batches interleaved one per four broad batches. Identical
  plan across arms, four views/place,16 places/batch. Broad batches do not have
  geographic exclusion; historical hard batches retain their prior exclusion.
- AdamW1e-4,zero weight decay,FP32,gradient clip1. Original MultiSimilarity
  loss/miner, historical photometric augmentation, deterministic place/epoch
  draws. Microbatch8 forward then concatenate all64 descriptors for mining.
  Skip zero-loss updates, report effective steps (at most960).
- PRIMARY result is epoch3, no best checkpoint or adaptive early stopping.
  GSV expanded-gallery metrics and descriptor drift are diagnostics only.
  Query identities and gallery remain fixed from completed preflight. Same-place
  identity GT, self image excluded; retain all queries including ambiguity flags.
- MSLS/Pitts evaluated only at original RU and fixed epoch3; initial baseline
  must reproduce675/740 and7160/7608. Report paired corrections/regressions,
  no selection or hyperparameter tuning from these exposed benchmarks.
- Same relative-geometry branch/module/code as completed QR preflight. No new
  architecture changes after observing trained-arm results. Correct layout must
  beat both controls and RU without cross-dataset regression to justify more
  seeds. One successful screen would still not demonstrate stable improvement.

All three augmented two-batch trainer smoke runs must finish before ANY formal
arm. Check zero-start, actual VPR update, frozen state and checkpoint roundtrip.
Fixed-last and absence of best selection are covered by a remote-only test.
Immutable source/metadata/code/plan hashes, resume every16 batches and epoch end;
unsaved work replays. Keep all old experiments. GPU1 PyTorch allocation capped40%
to coexist with Ollama; never terminate an unrelated service. Any exception stops
the queue; numerical failures must not be converted into method conclusions.

Launch: `bash scripts/run_query_relation_screen.sh`.
Outputs: `logs/query_relation_screen_v1/{appearance,aligned,shuffled}`;
`progress.json`, `history.json`, `last.pt`, benchmark per-query predictions,
`summary.json`, `completed.json`. No best.pt exists in this protocol.
