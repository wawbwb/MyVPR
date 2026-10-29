# DSQ-BoQ: matched depth-selection development screen

## Motivation and scope

The consensus decoder screen did not beat its controls. This experiment changes
single-image global aggregation instead of reranking a fixed candidate set.
It is a proposed structural hypothesis, not a confirmed novelty or improvement.

References:

- BoQ (CVPR 2024): learned query pooling, https://arxiv.org/abs/2405.07364
- EffoVPR (ICLR 2025): internal ViT features for VPR pooling,
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/6a1b224b153e55c40a6359f9c9fb9d8c-Abstract-Conference.html
- SelaVPR++: frozen-backbone intermediate-feature adaptation,
  https://arxiv.org/abs/2502.16601
- QAA is related adaptive query aggregation, not the proposed depth router:
  https://arxiv.org/abs/2507.03831 . Multi-layer fusion alone is not novel.

## Implementation

Extract the pre-final-norm block 6/9/12 patch maps in one frozen RU-DINO forward.
Keep the historical frozen RU gate on the final layer in every arm; do not apply
that gate to intermediate layers. No segmentation, CLIP, or semantic supervision.
Share the existing BoQ projections, query vectors and BoQ blocks among depth paths.
Per-depth LayerNorm affine alignment precedes shared pooling. For each query slot,
a two-layer router selects among three depth-pooled vectors. Fixed-equal fusion is
the router ablation. Add the fused vector through a zero-initialized projection to
the historical final-layer slot, before the unchanged output projection and L2 norm.

This residual warm start makes all four arms reproduce the RU descriptor before
training. It is an implementation choice beyond the initial weighted-sum sketch.
Do not initialize both residual projection and its upstream features to zero.
Same-layer branches use the same affine initialization as the depth arms, allowing
an extra-capacity control. All active arms execute four shared BoQ paths; baseline
executes one. Identical updates do NOT imply identical FLOPs; report this distinction.

## Locked first screen

- Modes: baseline, last_repeat (12/12/12 with adaptive router), equal_depth,
  adaptive_depth (6/9/12 with per-image/per-query router).
- Original RU SHA256: `38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e`.
- Freeze DINO and RU gate. Train the BoQ aggregator in every arm. Added parameters
  train only where active. Equal fusion freezes its unused router parameters.
- Seed42, hash-selected 4096 GSV places, 4 views each, 16 places per batch.
- Three passes over this subset = 768 updates, NOT three full GSV epochs.
- MultiSimilarity loss/miner, identical photometric augmentation and place order.
- LR: original aggregator 1e-5, new parameters 1e-4; AdamW wd .001, FP32, clip1.
- Full MSLS-val and Pitts30k-val, unchanged 280px resolution and descriptor width.
- Record epoch0 and every pass. Preserve fixed-last outcomes and earliest best
  MSLS epoch; evaluate that same selected epoch on Pitts, not its best Pitts epoch.
- These datasets are historically exposed development sets, not untouched tests.
- Save optimizer/RNG/cursor every256 places; stateless per-place/epoch sampling
  reproduces image/augmentation draws when resuming. No dense feature disk cache.

## Safety and decision

Real-image descriptor equality must pass; MSLS epoch0 must reproduce675/740.
All four two-update smoke runs must pass before formal runs start. Check branch
descriptor-probe gradients on update2 because upstream gradients are initially blocked
by zero output. This probe is never optimized: a zero-loss mined batch must not be
misclassified as a disconnected network. Report actual zero-loss batch counts.
Stop queue on errors. Keep existing experiments/checkpoints untouched.

Only recommend expansion if adaptive depth outperforms both same-layer capacity and
equal-depth controls without exchanging the gain for an unacceptable second-dataset
regression. Then repeat seeds and evaluate generalization; do not call one seed a
stable gain. If only equal fusion helps, attribute the result to multi-depth inputs,
not adaptive routing. No threshold or layer search on the final evaluation results.

Training-machine command: `bash scripts/run_depth_query_screen.sh`.
Summary: `doc/depth_query_screen_v1/summary.json`.
