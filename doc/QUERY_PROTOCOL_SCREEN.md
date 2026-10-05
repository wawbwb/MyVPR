# QR training-protocol control, registered before running

The QR aligned structure did not improve RU or the appearance control. This is
not evidence that all relational structures fail. Before adding another module,
test two training factors with the same appearance-only 20,736-parameter network.

## Four fixed arms

| Sampling | LR | Purpose |
|---|---|---|
| mixed | 1e-4 | Rerun the historical appearance protocol under this code revision |
| mixed | 1e-5 | Lower update size with the same sampled batches |
| broad_matched | 1e-4 | Replace historical hard batches, same total image exposures |
| broad_matched | 1e-5 | Factorial combination |

All initialize from RU, seed42, train4096 places for3 epochs,320 batches/epoch,
16 places x4 views, FP32, AdamW wd0, clipping1, identical augmentation. The256
broad batches are identical across all arms. The64 additional slots either use
the historical hard plan or1024 uniformly permuted training places without
replacement within those extra slots (seed52031+epoch). Those places also appear
in the base coverage. Effective optimizer steps may differ due to zero loss;
report them rather than silently padding updates. Same-place false negatives
and hard sampling are not independently isolated by this contrast.

All four smoke tests precede formal training. Final epoch3 only; do not pick
epochs or adjust hyperparameters using MSLS/Pitts. GSV is diagnostic only.
Compare paired corrections/regressions, RU-relative counts, GSV margins,
descriptor drift and effective steps. Preserve all old runs and checkpoints.

Lower LR causing less degradation with zero gain is preservation, not innovation.
A broad-vs-mixed difference is evidence about this sampling intervention, not
proof that subset size, augmentation, or mining are the cause. This screen does
not test full-GSV training or establish a new architecture. Any favorable result
remains exploratory on repeatedly exposed benchmarks and needs matched repeats.
No automatic aligned retuning or extension follows this screen.

Run: `bash scripts/run_query_protocol_screen.sh`. Output:
`logs/query_protocol_screen_v1`. The modified trainer uses new immutable
contracts: do not resume historical QR runs with it.
