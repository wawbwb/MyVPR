# Spatial consensus inside the Pair-VPR decoder

Research hypothesis, not established novelty or efficacy. No semantic masks,
dustbin, unmatched rejection, candidate expansion or latency claim.

## Why change direction

PMD expanded GSV screen lost5 queries with0 corrections. Synthetic correspondence
training exposed a location/rejection tradeoff; detached rejection preserved
location learning but its rejection ROC was worse than joint partial matching.
These findings do not establish repeated-pattern ambiguity as the error cause.
The new hypothesis must be validated on actual image pairs, not inferred as fact.

## Literature basis

- NCNet, NeurIPS2018: learn spatially consistent match sets in a4D correlation
  volume; weak image-pair supervision is possible. This motivates neighborhood
  support rather than independently judging a match.
  https://proceedings.neurips.cc/paper/2018/hash/8f7d807e1f53eff5f9efbe5cb81090fb-Abstract.html
- CHMNet, CVPR2021: trainable Hough voting in transformation space and geometric
  consistency. Our first implementation does NOT reproduce its6D scale-space
  architecture or claim its scale invariance.
  https://openaccess.thecvf.com/content/CVPR2021/html/Min_Convolutional_Hough_Matching_Networks_CVPR_2021_paper.html
- R2Former, CVPR2023: VPR pair reranking combines correlations, attention and
  coordinates. Shows local relations can inform pair classification, not proof
  that our extra module improves an already strong Pair-VPR.
  https://arxiv.org/abs/2304.03410
- Pair-VPR: existing strong paired-image decoder and classifier reused here.
  https://arxiv.org/abs/2410.06614

## Implemented structure

Frozen decoder blocks1..11 -> projected patch correlation C(qy,qx,dy,dx) ->
factorized source-grid and target-grid3x3 convolutions (8 hidden channels) ->
symmetrized contextual bias -> bidirectional softmax messages -> zero-start
residual token update -> original block12 / CLS classifier.

Unlike a scalar verifier on pooled evidence, this modifies token interaction
before final pair scoring. Unlike PMD, no Sinkhorn, dustbin, rejection head or
threshold. The factorization provides a structured neighborhood prior but is
not a full4D convolution implementation. Arbitrary perspective/large rotation
robustness is NOT guaranteed. Repeated texture can itself form a consistent false
pattern, so consensus is not a universal disambiguation solution.

This is an NCNet-inspired integration, not automatically a publishable invention.
Novelty would require a defensible difference and comparison beyond this module.

## Controls and next gate

Spatial consensus vs fixed-coordinate-shuffled consensus (identical parameters)
vs pointwise-only correlation processing vs original continue-training baseline.
Pointwise control is smaller; report parameter counts, do not call it matched.
Both matching directions share parameters. Fixed top20 throughout.

Current delivery: structure, unit tests and two-train-query real-image gradient
preflight only. Preflight checks zero-start scores and gradients, not efficacy.
No long training is started by this delivery. Next screen must include enough
reachable real errors, same initialization/budget and paired correction/regression
counts. Prefer real place-pair ranking loss; geometric supervision cannot be
inferred from same-place labels. If spatial does not beat coordinate scrambling,
do not attribute any benefit to geometric consensus.

No dense feature cache: one529x529 correlation map is about1.07MiB FP32 per pair;
8-channel intermediates are larger and autograd adds memory. This is NOT an
estimate of total GPU usage. Existing encoder/checkpoints/candidate caches reused.

## Authorized first training screen

Four arms baseline/pointwise/spatial/shuffled; same1024 hash-selected reachable
training queries and fixed20 candidates,3 epochs,seed42. All2048 GSV dev queries
evaluated every epoch, including epoch0 reproduction. Entire dev has previously
been examined; this remains development screening, not new test evidence.
Select earliest best dev epoch including0; also report final epoch to expose
selection effects. Last-block LR1e-5, adapter1e-4, pairwise softplus temperature10,
weight decay.001. Other original parameters frozen, dropout disabled. Shared
sampling/order, positive rotates by epoch, top-ranked negative fixed.
Checkpoint weights, optimizer and RNG every32 queries. No new dense disk cache.

`scripts/run_consensus_screen.sh` first runs all four smoke tests, then all four
three-epoch runs, then paired summary including spatial vs shuffled/pointwise and
reachable original errors. Any failure stops the queue; re-run resumes checkpoints.
No automatic claim of efficacy based on loss or a nonzero gradient.

```bash
python -m pytest -q tests/test_consensus_matching.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/consensus_preflight.py
```
