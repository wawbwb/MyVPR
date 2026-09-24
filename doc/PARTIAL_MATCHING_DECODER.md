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

## Not implemented/authorized by this gate

Formal training, checkpoint/resume, synthetic correspondence supervision,
held-out retrieval evaluation, controlled unfreezing and full four-arm ablation
remain subsequent work. A PASS only establishes functioning implementation.
Before training, lock place-disjoint GSV train/dev, matched sampling/budget and
fixed top20 evaluation. Never infer patch correspondences from place labels.
Monitor dustbin collapse and use known-transform correspondence examples when
adding correspondence supervision; easy smoke loss is not efficacy evidence.
