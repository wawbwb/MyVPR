# Dynamic-category negative-prior screen

## Scope and protocol

This is a frozen-checkpoint causal screen. It asks whether reducing BoQ
attention to segmented dynamic objects helps the trained
repeatability+uniqueness (RU) model. It does not retrain the model and it does
not reuse the failed CLIP propagation target.

DeepLabV3-MobileNetV3 predicts Pascal-VOC labels on each clean image. The
dynamic classes are `person`, `bicycle`, `car`, `motorbike`, `bus`, and
`train`. Their hard-argmax pixel area is pooled to the DINOv2 `20x20` grid and
inserted into both BoQ cross-attention blocks as

```text
attention_bias = -beta * dynamic_patch_fraction
```

The preregistered first value is `beta=0.5`. A fully dynamic patch therefore
multiplies its prior attention odds by `exp(-0.5) = 0.607`; it is downweighted,
not deleted. The teacher has no `rider` or `truck` class.

All five variants use the same checkpoint and the same DINO/RU feature map:

| Variant | Intervention |
| --- | --- |
| `baseline` | `attention_bias=None` |
| `zero_bias` | all-zero bias; numerical-path control |
| `aligned` | the image's correct dynamic mask |
| `shuffled` | another image's mask, with a role/condition-preserving derangement |
| `random` | an exact spatial permutation of the same image's mask values |

The standard 18,871-image database is unchanged. The generator reads all
non-panorama `n2d`, `w2s`, and `s2w` query candidates, computes same-city 25 m
positives in the exact global standard-DB index space, and excludes candidates
with no positive. It writes three atomic conditions:

- `night` (`n2d`);
- `winter2summer` (`w2s`);
- `summer2winter` (`s2w`).

It also writes a `season` compatibility aggregate and a deterministic query
union. The union begins with the standard 740 queries in their original order,
then appends condition-only queries in atomic-condition order. One descriptor
and one mask are therefore reused when a query belongs to multiple reports.

This remains a **custom full-database robustness protocol**, not the official
MSLS condition protocol: the official subtasks filter the database as well as
the queries. Do not label these rows as official `n2d`, `w2s`, or `s2w`
scores.

The metadata contains 55 night candidates and 996 combined season candidates.
Image existence does not guarantee a valid retrieval query: candidates with
no reference within 25 m are excluded. The previous files indicate that the
season aggregate may fall to about 988 queries, but the newly generated audit
is authoritative; do not hard-code 988 in analysis.

## Remove the invalid intersection run

Run these commands only from the repository root on the training machine. The
targets are the four old manifests without `_full_db_`, the old 19,611-image
cache/audit, and retrieval outputs produced from the incorrect 5-query night /
84-query season intersection.

```bash
test -f scripts/generate_msls_condition_splits.py
test -d datasets/msls-val

rm -f -- \
  datasets/msls-val/msls_val_night_qImages.npy \
  datasets/msls-val/msls_val_night_gt_25m.npy \
  datasets/msls-val/msls_val_season_qImages.npy \
  datasets/msls-val/msls_val_season_gt_25m.npy \
  doc/msls_condition_split_audit.json \
  .cache/dynamic_prior/msls_val_deeplabv3_mbv3_grid20.npz \
  msls_condition_fix.bundle

rm -rf -- \
  doc/dynamic_category_mask_audit \
  doc/dynamic_category_prior_screen_b0.5 \
  doc/dynamic_category_prior_screen_b0.5_fixed_splits
```

The old mask values were valid for the standard 740 queries, but that cache is
incomplete for condition-only queries. Removing it avoids accidentally
combining two different image indices. The evaluator also checks the complete
ordered path manifest and will reject the old cache.

## Generate and audit the corrected manifests

```bash
conda activate VPR

python -m pytest -q \
  tests/test_dynamic_category_prior.py \
  tests/test_dynamic_category_eval_index.py \
  tests/test_msls_condition_splits.py \
  tests/test_msls_condition_loader.py \
  tests/test_condition_eval_loader.py

python scripts/generate_msls_condition_splits.py \
  --msls-path datasets/msls-val \
  --cities cph sf \
  --conditions night winter2summer summer2winter \
  --distance-threshold 25 \
  --expected-standard-queries 740 \
  --report doc/msls_condition_split_audit.json \
  --force
```

Inspect the actual retained counts before starting GPU work:

```bash
python - <<'PY'
import json
from pathlib import Path

audit = json.loads(
    Path("doc/msls_condition_split_audit.json").read_text()
)
for name, row in audit["conditions"].items():
    print(
        name,
        "candidate=", row["candidate_queries_before_panorama_exclusion"],
        "panorama=", row["excluded_panorama_queries"],
        "no_positive=", row["excluded_no_positive_queries"],
        "retained=", row["retained_queries"],
        "standard_overlap=", row["standard_query_overlap"],
        "condition_only=", row["condition_only_queries"],
    )
print("season aggregate=", audit["aggregates"]["season"]["retained_queries"])
print("query union=", audit["query_union"]["num_queries"])
print(
    "membership counts=",
    audit["query_union"]["condition_membership_counts"],
)
print(
    "singleton memberships=",
    audit["query_union"]["singleton_condition_membership_paths"],
)
PY
```

Do not continue if generation reports a shared standard query whose recomputed
GT disagrees with `msls_val_gt_25m.npy`, a missing image, invalid metadata, or
an unexpected empty condition. Those checks fail closed by design.
Also stop if `singleton memberships` is non-empty: the shuffled-mask control
cannot form a no-fixed-point donor permutation for a one-image membership
stratum without weakening condition matching.

Generated dataset files are:

```text
msls_val_night_full_db_qImages.npy
msls_val_night_full_db_gt_25m.npy
msls_val_winter2summer_full_db_qImages.npy
msls_val_winter2summer_full_db_gt_25m.npy
msls_val_summer2winter_full_db_qImages.npy
msls_val_summer2winter_full_db_gt_25m.npy
msls_val_season_full_db_qImages.npy
msls_val_season_full_db_gt_25m.npy
msls_val_condition_union_qImages.npy
```

## Cache the complete query union

The old cache cannot be reused. Build a new cache whose name explicitly says
`full_db_condition_union`:

```bash
python scripts/cache_dynamic_category_masks.py \
  --msls-path datasets/msls-val \
  --output .cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz \
  --report-dir doc/dynamic_category_mask_audit_full_db_condition_union \
  --device cuda:1 \
  --batch-size 16 \
  --num-workers 8 \
  --seg-size 520 520 \
  --grid-size 20 20 \
  --seed 42
```

The default teacher pass is FP32 so hard-argmax boundaries do not depend on
mixed-precision rounding. Reduce `--batch-size` if needed; do not enable
`--amp` for the preregistered result. If the official weight host is
unavailable, copy
`deeplabv3_mobilenet_v3_large-fc3c493d.pth` to the machine and add:

```bash
--teacher-weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth
```

## Run the corrected screen

```bash
RU_DIR=logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints
find "$RU_DIR" -maxdepth 1 -type f -name '*.ckpt' -print
RU_CKPT="$(find "$RU_DIR" -maxdepth 1 -type f -name 'epoch(26)_*.ckpt' -print -quit)"
test -f "$RU_CKPT"

python scripts/eval_dynamic_category_prior.py \
  --checkpoint "$RU_CKPT" \
  --msls-path datasets/msls-val \
  --mask-cache .cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz \
  --output doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries \
  --scratch-dir /tmp/dynamic_prior_screen_full_db_condition_queries \
  --device cuda:1 \
  --batch-size 32 \
  --num-workers 8 \
  --image-size 280 280 \
  --beta 0.5 \
  --seed 42 \
  --conditions night winter2summer summer2winter
```

The five float32 descriptor matrices require roughly 5 GiB and are deleted
unless `--keep-descriptors` is supplied. Do not commit teacher weights,
`.cache/`, or scratch descriptors. Keep the split audit, mask audit, and final
retrieval result after checking their hashes and query counts.

## Decision rule

The retrieval directory contains `summary.csv`, `query_outcomes.csv`,
`paired_comparisons.csv`, and `run.json`. Before interpreting the intervention,
the baseline must reproduce RU overall R@1 `91.22%` within `0.15` percentage
points. The screen passes only when:

1. aligned beats zero-bias, shuffled, and random on standard MSLS-val;
2. aligned loses no more than `0.3` points versus baseline overall;
3. aligned gains at least `1.0` point versus zero-bias and beats all controls
   on at least one complete atomic condition.

If it fails, stop this teacher/injection/strength route before training. A
failure rejects this implementation, not semantic VPR in general. Only after
a pass should `beta=0.25` and `beta=1.0` be run as separate confirmations.

## Diagnose baseline BoQ attention routing

After a failed screen, the following audit determines whether the frozen RU
BoQ baseline already avoids dynamic regions. It uses the same checkpoint,
complete query-union cache, transforms, and feature path as the retrieval
screen, but always calls BoQ with `attention_bias=None`:

```bash
python scripts/audit_boq_dynamic_attention.py \
  --checkpoint "$RU_CKPT" \
  --msls-path datasets/msls-val \
  --mask-cache .cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz \
  --output doc/boq_baseline_dynamic_attention_audit \
  --device cuda:1 \
  --batch-size 32 \
  --num-workers 8 \
  --image-size 280 280 \
  --seed 42 \
  --conditions night winter2summer summer2winter
```

The audited baseline is
`DINO -> trained RU semantic_region_gate -> BoQ(attention_bias=None)`. It is
the baseline variant of this screen, not the photometric-only checkpoint.
The script temporarily requests per-head weights from each BoQ
`MultiheadAttention` and fails unless the descriptor and legacy head-mean
attention are reproduced.

The primary continuous-mask statistic is:

```text
attention_mass = sum_patch(attention_probability * dynamic_area_fraction)
uniform_expected_mass = mean_patch(dynamic_area_fraction)
micro_enrichment = sum_image(attention_mass) / sum_image(dynamic_area_fraction)
```

`micro_enrichment < 1` means relative avoidance and `> 1` means enrichment.
Do not interpret enrichment alone as semantic alignment because vehicles and
BoQ attention may both prefer the lower image. `aligned_minus_random_mass`
preserves every mask value within the same image and changes only its spatial
arrangement, so it controls coverage but may still expose a common lower-image
bias. The shuffled comparison preserves reference/query role, the exact
condition-membership stratum, and the population's typical mask positions.
Evidence for image-specific dynamic-object attention therefore requires
aligned to beat both random and shuffled at group level.

Important outputs are:

```text
summary.csv                         group/layer primary results
head_summary.csv                    per-layer, per-head results
query_slot_summary.csv              per learned BoQ query results
per_image.csv                       image-level metrics and selection audit
mean_position_attention.jpg         dataset-level spatial routing maps
attention_balanced_random.jpg       deterministic qualitative sample
attention_high_aligned_minus_random.jpg
attention_low_aligned_minus_random.jpg
head_details/                       selected per-head overlays
run.json                            hashes, dimensions, invariants, definitions
```

Heatmaps display `num_patches * attention`, so uniform attention is exactly
`1`. Every image uses one dataset-wide p99 colour scale; there is no per-image
min-max or z-score. These maps describe where learned BoQ queries read spatial
tokens. They are not causal pixel saliency because values, output projections,
LayerNorm, token mixing, and the final signed FC projection also affect the
descriptor.
