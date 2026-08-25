# DINOv2-BoQ local semantic-region experiment

## Hypothesis

Global CLIP similarity is not used as a place label.  The full target is built
from three local signals:

1. same-place cross-view DINO token repeatability;
2. uniqueness against the hardest different place in the current batch;
3. a frozen CLIP semantic-spatial affinity that only propagates the DINO
   reliability inside confident local regions.

The student gate is `1 + 0.2 * tanh(score)`.  Its 1x1 convolution is
zero-initialised, so the initial model is exactly the visual baseline and the
modulation always stays in `[0.8, 1.2]`.  CLIP is used only once to create a
sparse cache and is absent from training checkpoints and inference.

## Required comparisons

All six runs use full GSV-Cities, 280x280 images, P=40, K=4, 40 epochs and
seed 42.

| Config | Purpose |
| --- | --- |
| `boq_dinov2_photometric.yaml` | visual baseline with the same spatially preserving augmentation |
| `boq_dinov2_semantic_region_repeatability_only.yaml` | DINO repeatability control |
| `boq_dinov2_semantic_region_repeatability_uniqueness_only.yaml` | DINO repeatability + uniqueness, without semantics |
| `boq_dinov2_semantic_region_semantic_only.yaml` | local semantic-confidence control |
| `boq_dinov2_semantic_region_shuffled.yaml` | wrong-place semantic-region control |
| `boq_dinov2_semantic_region_full.yaml` | repeatability + uniqueness + aligned semantics |

The original RandAugment BoQ result is not the direct semantic ablation
baseline: RandAugment may rotate, translate or shear an image, whereas the
cached 14x14 semantic grid is computed from the clean photograph.  Semantic
runs therefore use photometric-only augmentation, and the first config above
controls for that change.

Treat the semantic signal as supported only when full beats both the
repeatability+uniqueness no-semantics control and shuffled semantics. Since
MSLS-val has 740 queries,
0.14 percentage points is approximately one query; use at least a 1 point
gain on a difficult condition before spending on three seeds and SALAD
replication.

For MSLS night/season evaluation, use
`scripts/eval_condition_robustness.py --device cuda:1 --image-size 280 280`.
The loader restores `semantic_region_gate.*` from the checkpoint; evaluating a
semantic checkpoint with an older loader that knows only the backbone and
aggregator would silently measure the wrong model.

## Seed-42 outcome

All six 40-epoch runs completed.  The table reports MSLS-val R@1 in percent;
the late mean is the arithmetic mean over epochs 20--39.

| Config | Best | Epoch 20--39 mean | Final |
| --- | ---: | ---: | ---: |
| photometric | 90.27 | 89.925 | 89.86 |
| repeatability only | 90.95 | 90.604 | 90.81 |
| repeatability + uniqueness | 91.22 | 90.562 | 90.54 |
| semantic only | 90.54 | 89.893 | 90.14 |
| shuffled semantics | 91.22 | 90.983 | 90.95 |
| full aligned semantics | 89.86 | 89.709 | 89.73 |

Training succeeded mechanically, but the semantic hypothesis is not supported
for seed 42.  Full aligned semantics is 1.36 points (about 10 of 740 queries)
below both the no-semantics and shuffled controls at their best checkpoints,
and is also below the photometric baseline.  This fails the pre-registered
criterion, so three-seed and SALAD replication should wait for a target-level
diagnosis.  Night/season results have not yet been collected.

## Aligned-versus-shuffled propagation diagnostic

`scripts/visualize_semantic_region_delta.py` isolates the target construction
from learned-backbone differences.  It loads one neutral
repeatability+uniqueness checkpoint, takes one fixed `P=40`, `K=4` batch,
computes the raw DINO feature map once, then compares the aligned cache with a
place-rolled cache.  The plotted propagation quantity is exactly

```text
delta = confidence * (weighted_neighbor_reliability - base_reliability)
```

Run this from the repository root on the training machine:

```bash
git pull --ff-only origin main
conda activate VPR

RU_DIR=logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints
find "$RU_DIR" -maxdepth 1 -type f -name '*.ckpt' -print
RU_CKPT="$(find "$RU_DIR" -maxdepth 1 -type f -name 'epoch(26)_*.ckpt' -print -quit)"
test -f "$RU_CKPT"

python -m pytest -q \
  tests/test_semantic_region_gate.py \
  tests/test_semantic_region_delta_visualization.py

python scripts/visualize_semantic_region_delta.py \
  --feature-ckpt "$RU_CKPT" \
  --device cuda:1 \
  --clean-input \
  --batch-index 0 \
  --num-workers 0 \
  --num-samples 8 \
  --output doc/semantic_region_delta_batch0_clean
```

The output directory contains 4x4 PNG panels, `summary.csv`,
`diagnostic_tensors.pt`, and `run.json`.  Aligned and shuffled deltas use one
shared symmetric colour range: red is a positive reliability change, blue is a
negative change, and near-white is approximately zero.  Inspect both the
ranked examples and the `random_audit` example.  Aligned propagation is suspect
if it repeatedly changes road, sky, vegetation, or generic building regions
more strongly than stable facade details, signs, windows, and other
place-specific structures.

## Offline target counterfactual sweep

The seed-42 retrieval verdict above is already negative.  The offline sweep is
not needed to decide whether the completed experiment succeeded; it is used to
decide whether a redesigned target is worth retraining.  It reads the saved
diagnostic tensor bundle and needs neither the dataset nor a checkpoint:

```bash
python scripts/sweep_semantic_region_counterfactual.py \
  --input doc/semantic_region_delta_batch0_clean/diagnostic_tensors.pt \
  --output doc/semantic_region_counterfactual_batch0
```

The sweep keeps the original confidence values at retained patches and sets
rejected patches to zero.  Absolute thresholds `0.5`, `0.6`, `0.7` and an
exact per-image top-20% mask are crossed with three target transforms and two
propagation compositions:

- `production_roundtrip`: the current `base10 -> base14 -> out10` path;
- `ru_additive`: `base10 + resize(delta14, 10)`, which is exactly the RU target
  when the semantic mask is empty;
- `per_image_zscore`: the current per-image unit-variance transform;
- `shared_base`: one fixed scale estimated from the complete RU batch;
- `center_only`: subtract the per-image mean without dividing by its standard
  deviation.

The dense production row reproduces the saved target before the sweep.  On the
fixed clean batch, an empty semantic mask in `production_roundtrip` still
changes the final target from direct RU by mean absolute `0.10062`, with spatial
correlation `0.9722`.  This exposes a previously unmeasured interpolation
confound: semantic full/shuffled resize the entire base map `10 -> 14 -> 10`,
whereas RU does not.  It cannot explain full versus shuffled because both use
the round trip, but it confounds comparisons against RU.

For the RU-preserving additive composition with `center_only`, the principal
aligned results are:

| Mask | Patch coverage | Aligned delta retained | Target change from RU | Aligned--shuffled target difference | Saturated target fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 100.00% | 100.00% | 0.05866 | 0.05879 | 4.17% |
| confidence >= 0.5 | 29.69% | 40.04% | 0.02855 | 0.04296 | 5.45% |
| confidence >= 0.6 | 12.64% | 19.94% | 0.01583 | 0.02734 | 5.75% |
| confidence >= 0.7 | 4.38% | 7.73% | 0.00714 | 0.01321 | 5.98% |
| top 20% per image | 20.41% | 29.66% | 0.02276 | 0.03566 | 5.57% |

Hard masking therefore works mechanically and removing unit-variance scaling
reduces the dense aligned saturation from roughly 36% in production to roughly
4%.  Threshold `0.7` is almost a semantic-off control and leaves three of 160
images with no active patch.  Threshold `0.6` is the most useful triage point:
it retains about one fifth of the aligned propagation magnitude without
forcing low-confidence patches into every image as top-20% does.  The selected
mask montage nevertheless remains scattered over generic road, vegetation,
facade, and vehicle regions, so this fixed-batch target analysis is not evidence
of a retrieval improvement.  A revised method still needs matched RU, aligned,
and shuffled training before any R@1 claim.

## Next route: explicit dynamic-category negative prior

Do not retrain the sparse CLIP propagation target yet.  The next preregistered
screen is a frozen-checkpoint intervention that applies segmented dynamic-area
coverage as a negative BoQ cross-attention-logit prior, with aligned,
role-preserving shuffled, and exact-value random controls.  Its implementation,
commands, limitations, and stop/go rule are documented in
`doc/DYNAMIC_CATEGORY_PRIOR_SCREEN.md`.  This route is intentionally isolated
from CLIP and does not reuse the interpolation or per-image unit-variance
target construction diagnosed above.
