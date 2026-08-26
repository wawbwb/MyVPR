# Dynamic-category negative-prior screen

## Why this screen is different

The completed CLIP semantic-region experiment is negative for seed 42.  This
screen does not tune that target, reuse CLIP propagation, multiply DINO
features, or retrain a model.  It asks a narrower causal question:

> Does explicitly reducing BoQ attention to segmented dynamic objects improve
> the frozen repeatability+uniqueness (RU) checkpoint?

The fixed torchvision DeepLabV3-MobileNetV3 teacher predicts Pascal-VOC pixel
labels on the clean MSLS image.  Hard dynamic pixels (`person`, `bicycle`,
`car`, `motorbike`, `bus`, and `train`) are area-pooled to the DINOv2 `20x20`
patch grid.  For patch coverage `m` and the preregistered `beta=0.5`, both BoQ
cross-attention blocks receive

```text
attention_bias = -0.5 * m
```

A fully dynamic patch therefore multiplies its prior attention odds by
`exp(-0.5) = 0.607`.  The bias is finite, so patches are downweighted rather
than deleted.  The teacher has no `rider` or `truck` category; that limitation
must remain explicit when interpreting a negative result.

## Matched controls

All five branches use one checkpoint and the exact same raw DINO/RU feature
map for every image:

| Branch | Intervention |
| --- | --- |
| `baseline` | `attention_bias=None`; the historical checkpoint path |
| `zero_bias` | an all-zero float mask; controls the CUDA attention-kernel path |
| `aligned` | the correct image's dynamic mask |
| `shuffled` | a wrong image's mask from a seeded global derangement |
| `random` | a seeded spatial permutation of the same image's 400 mask values |

The shuffled donor permutation is generated separately inside the database
and each query night/season-membership stratum, has no fixed points, and is
independent of batch size.  Thus a night or season query always receives a
mask from another query with the same condition memberships.
The random branch exactly preserves every image's mask values and coverage.
Neither control uses a per-image z-score.

The mask is cached first, then the segmentation teacher is removed.  VPR
evaluation computes DINO plus the learned RU gate once per image batch and
only repeats the relatively small BoQ aggregation.  Standard, night, and
season metrics reuse the same database and query descriptors.  The condition
generator intersects `n2d` / `w2s` / `s2w` candidates with the standard MSLS
query manifest and reuses the corresponding standard GT rows.  Thus every
condition query maps back to the standard query index by exact path.

The repository's night/season files are **custom condition subsets searched
against the standard full MSLS database**.  They are not the official MSLS
condition subtasks, which select a condition-specific database as well as
queries.  Report them as robustness slices of the standard full-DB protocol,
not as official `n2d`, `w2s`, or `s2w` scores.

## Commands

Run from the repository root on the training machine:

```bash
conda activate VPR

python -m pytest -q \
  tests/test_dynamic_category_prior.py \
  tests/test_msls_condition_splits.py \
  tests/test_condition_eval_loader.py

# Regenerate even when old manifests exist.  The old generator omitted the
# panorama/standard-query-universe restriction.
python scripts/generate_msls_condition_splits.py \
  --msls-path datasets/msls-val \
  --conditions night season \
  --report doc/msls_condition_split_audit.json \
  --force

RU_DIR=logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints
find "$RU_DIR" -maxdepth 1 -type f -name '*.ckpt' -print
RU_CKPT="$(find "$RU_DIR" -maxdepth 1 -type f -name 'epoch(26)_*.ckpt' -print -quit)"
test -f "$RU_CKPT"

if [ ! -f .cache/dynamic_prior/msls_val_deeplabv3_mbv3_grid20.npz ]; then
  python scripts/cache_dynamic_category_masks.py \
    --msls-path datasets/msls-val \
    --output .cache/dynamic_prior/msls_val_deeplabv3_mbv3_grid20.npz \
    --report-dir doc/dynamic_category_mask_audit \
    --device cuda:1 \
    --batch-size 16 \
    --num-workers 8 \
    --seg-size 520 520 \
    --grid-size 20 20 \
    --seed 42
fi

python scripts/eval_dynamic_category_prior.py \
  --checkpoint "$RU_CKPT" \
  --msls-path datasets/msls-val \
  --mask-cache .cache/dynamic_prior/msls_val_deeplabv3_mbv3_grid20.npz \
  --output doc/dynamic_category_prior_screen_b0.5 \
  --scratch-dir /tmp/dynamic_prior_screen \
  --device cuda:1 \
  --batch-size 32 \
  --num-workers 8 \
  --image-size 280 280 \
  --beta 0.5 \
  --seed 42 \
  --conditions night season
```

If the standard 19,611-image mask cache already completed successfully, keep
and reuse it.  Regenerating the condition manifests changes only which of the
standard 740 queries are reported in night/season slices; it does not change
the cached image index or any mask.  The split audit records how many old
condition candidates were outside the standard query universe and how many of
those were panoramas.

The default teacher weights are downloaded once from the official PyTorch
model host and then reused from the torch hub cache.  If the training machine
cannot reach that host, download
`deeplabv3_mobilenet_v3_large-fc3c493d.pth` elsewhere, copy it to the training
machine, and add this option to the cache command:

```bash
--teacher-weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth
```

The segmentation cache defaults to FP32 so hard argmax labels do not depend on
mixed-precision boundary rounding.  If memory is still tight, reduce its batch
size; do not enable `--amp` for the preregistered first result.

Do not commit the teacher weights, `.cache/`, or temporary descriptor files.
The five float32 descriptor matrices need about 5 GiB of scratch space and are
deleted automatically unless `--keep-descriptors` is supplied.

## Outputs and decision rule

The mask audit contains `run.json` and an unbiased seeded 12-image montage.
The retrieval output contains:

- `summary.csv`: R@1/5/10 plus changes versus historical and zero-bias paths;
- `query_outcomes.csv`: top-1 prediction/correctness for every query/branch;
- `paired_comparisons.csv`: aligned-only versus comparator-only correct counts;
- `run.json`: hashes, exact controls, parameters, results, and an automatic
  pass/fail verdict.  It also records all query/GT manifest hashes, exact
  standard-query overlap, and the custom-condition protocol label.

Before interpreting semantics, baseline must reproduce RU R@1 `91.22%` within
`0.15` percentage points.  The screen passes only when all of these hold:

1. aligned R@1 beats zero-bias, shuffled, and random on full MSLS;
2. aligned loses no more than `0.3` points versus baseline on full MSLS;
3. on night or season, aligned gains at least `1.0` point versus zero-bias and
   beats zero-bias, shuffled, and random.

If it fails, stop this teacher/injection/strength route before training.  A
failure does not disprove semantic VPR in general.  Only after a pass should
`beta=0.25` and `beta=1.0` be run as separately recorded confirmations by
reusing the same mask cache and choosing new output directories.
