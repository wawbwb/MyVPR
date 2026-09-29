# LCV-BoQ initialization confirmation

Original seed42: GSV-selected local contrast epoch2 has MSLS673 (RU675),
Pitts7165 (RU7160); shuffled epoch3 has MSLS673, Pitts7161. This is not a
cross-dataset success. Fixed-last results are also preserved, not substituted.

Add only initialization seeds43 and44, each with local_contrast and
shuffled_contrast. Keep original trainer, completed runs and plan immutable.
The separate replication trainer differs only in explicit adapter initialization
and recording its seed. CPU RNG fork restores the surrounding RNG stream.
Dataset seed42, per-place/epoch view and augmentation seeds, original RU weights,
seed42019 shuffled grid, hard batch plan, all hyperparameters and GSV selection
stay fixed. This measures initialization sensitivity, NOT independently resampled
training datasets, shuffle layouts or minibatch schedules. Existing seed42 used
the legacy RNG-stream initialization; it is an exploratory reference, not rerun.

Run four two-batch smoke runs first, then four three-round runs, 768 batches each.
Report GSV-selected and fixed round3 separately. Do not select by MSLS/Pitts or
report best seed. Summarize per-seed corrections/regressions, mean/range net counts
and wins against frozen RU and same-seed shuffled control. Do not pool the same
queries over seeds as independent observations or claim population significance.

Decision: if the added seeds do not reproduce Pitts improvement and real-neighbor
advantage, or MSLS degradation persists, stop this route rather than tuning to
these exposed benchmarks. A positive three-seed pattern would justify further
validation, not automatically establish stable or independent-test benefit.

Training machine: `bash scripts/run_local_value_repeat.sh`.
Runs: `logs/local_value_repeat`; summary: `doc/local_value_repeat_v1`.
