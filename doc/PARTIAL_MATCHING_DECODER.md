# Partial Matching Decoder — implementation gate

Goal: improve pair recognition on identical fixed top20 candidates, not candidate
expansion or inference acceleration. This is a research hypothesis, not a
confirmed novelty or accuracy claim.

## Implemented structure

`src/models/partial_matching.py`: frozen official Pair-VPR blocks1–11 produce
patch states. A shared projection computes cross-image patch affinities; a
log-space Sinkhorn layer with a learnable dustbin produces partial transport.
The same transport updates both patch streams. Unmatched mass is NOT normalized
away. CLS is excluded from transport. Original block12 and classifier consume
the modified features, permitting recognition gradients to reach correspondence
parameters. Zero-initialized output projection reproduces the base at insertion.
Prefix is detached; block12 is frozen but differentiable with respect to inputs.

Modes: ordinary independent row-softmax; forced balanced transport; partial
transport with dustbin. Projection widths match; the partial arm has an active
dustbin scalar, which is unused in other arms. This is not claimed to be exactly
equal active parameter count. A continue-training original baseline is still
required for formal experiments.

Partial transport/dustbins are established machinery (SuperGlue CVPR2020);
local-interaction VPR reranking also exists (R2Former CVPR2023). Novelty must be
established separately; inserting optimal transport alone is not a new theory.

## Preflight only

`scripts/pmd_preflight.py`: two first eligible GSV train queries, each with a
positive and highest-ranked global negative. Reuse index/hashes, encode only
these images; verify model identity, global descriptors, zero-start pair scores.
Five CPU/GPU FP32 optimization steps on the adapter with place pairwise softplus
loss (fixed score temperature10 to avoid saturation), no teacher-score fitting.
Check finite key/dustbin gradients, score change and frozen base parameters.
No checkpoint is saved, no held-out data is inspected. Cached prefix states
are only a small in-memory implementation check, not a large feature cache.

```bash
python -m pytest -q tests/test_partial_matching.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/pmd_preflight.py
```

## Four-arm GSV screening

`scripts/train_pmd.py` implements a matched first screen, not a final efficacy
experiment. All arms unfreeze the original final decoder block; the remaining
official weights stay frozen. Baseline bypasses the adapter. Ordinary, forced,
and partial modes share projection dimensions and initialization. Partial adds
one active dustbin scalar. Dropout stays disabled in every arm.

Policy is locked in the run contract: 1024 hash-selected reachable GSV train
queries, 256 place-disjoint GSV dev queries, fixed original top20, three epochs,
seed42, final-block LR1e-5, adapter LR1e-4. Each query uses one positive (rotating
by epoch) and its highest-global-ranked negative. No candidate expansion and no
teacher-score regression. Dev chooses the earliest best epoch, including epoch0;
both best and last epoch are reported, so selecting the unchanged initial model
cannot be mistaken for training improvement. This previously explored GSV dev
pool is a screening set, NOT untouched confirmation data.

Atomic checkpoints include trainable weights, AdamW moments, RNG state, epoch and
query cursor; save every32 queries and at epoch boundaries. Resume repeats only
work since the last saved update. Original frozen weights are reconstructed from
the strictly verified official checkpoint. Small CPU feature LRU only; no new
dense on-disk feature cache. Checkpoints are retained.

Run on the training machine (physical GPU1 selected by the launcher):

```bash
python -m pytest -q tests/test_partial_matching.py tests/test_train_pmd.py
bash scripts/run_pmd_screen.sh smoke
bash scripts/run_pmd_screen.sh train
python scripts/summarize_pmd_screen.py
```

The launcher resumes each arm automatically. Smoke checks initial score identity,
finite training and checkpoint round-trips. Unit tests additionally compare the
next optimizer update after resume. The paired report counts corrections and
regressions against the frozen teacher AND the continued-training baseline.
Single-seed development results cannot establish stable gains or novelty.

## Still not implemented

## Expanded development evaluation (locked after first screen)

`scripts/eval_pmd_expanded.py` evaluates the remaining1792 GSV dev queries,
excluding the256 used for checkpoint selection. It freezes each arm's existing
best checkpoint (baseline epoch0, ordinary epoch3, forced/partial epoch1), keeps
the original top20 candidates, and performs no further optimization or epoch
selection. This pool has been explored by prior experiments: it is expanded
development screening, NOT an independent test. Original frozen pair outcomes
are reported alongside all four arms. Partial is compared pairwise against each
control; reachable and unreachable original errors are counted separately.

Frozen encoder/prefix computation is shared; all four trained final blocks are
applied separately. Scores are first checked against two original selection-set
queries. Only small per-query score shards are written, not dense feature caches.
`--resume` validates source/checkpoint/code hashes and reuses verified shards.

```bash
python -m pytest -q tests/test_eval_pmd_expanded.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_pmd_expanded.py --resume
```

## Remaining research work

Synthetic correspondence supervision and independent confirmation evaluation
remain subsequent work. A smoke PASS only establishes functioning implementation.
Never infer patch correspondences from place labels.
Monitor dustbin collapse and use known-transform correspondence examples when
adding correspondence supervision; easy smoke loss is not efficacy evidence.
