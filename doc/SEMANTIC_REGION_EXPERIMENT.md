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
