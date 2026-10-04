# DSA matched short training v1

This is an explicitly exploratory follow-up, NOT a retroactive phase0 pass.
Phase0 variance-retention gate failed. The follow-up cross-view diagnostic found
true Fisher consistently above PCA/shuffled Fisher under full/two half fits at
both layers. Layer11 full-fit centered R1: original57.707%, PCA27.783%, shuffled
13.656%, Fisher71.615%; layer12: 83.285%,66.585%,42.310%,76.790%.
These are intermediate-feature GSV recalls, not final VPR improvements.

## Locked budget and architecture

Three arms: pca / shuffled_fisher / place_fisher. Reuse verified phase0 full-fit
rank16 Euclidean orthonormal bases, no new basis fitting or development tuning.
Only zero-initialized Q/V output matrices in DINO blocks11/12 are trainable:
49,152 parameters. Original RU backbone, BoQ, existing RU gate and basis buffers
remain fixed. Existing RU gate is inherited, not a newly introduced semantic path.
No semantics, candidate expansion or pairwise reranking is added.

Three epochs, 320 batches/epoch, 16 places x4 views/batch. Every epoch covers all
4096 training places exactly once in 256 randomly permuted broad batches; insert
64 historical geographically screened hard batches, one after every four broad
batches. Thus hard batches comprise20%, not the entire training task. Broad
batches use ordinary GSV place labels and have NO new geographic exclusion;
adjacent distinct-place false negatives remain a limitation. Historical hard
batches retain their100m conservative exclusion, not a guarantee of true negatives.
Seed42 data/views, broad order42031+epoch, identical schedules across arms.
AdamW lr1e-4, weight_decay0, grad clip1, FP32, no scheduler or AMP. Original
MultiSimilarity loss/miner. Skip zero-loss optimizer steps and report their count.
At most960 updates/arm; effective updates may differ by arm because mining differs.

Forward chunks8 images, concatenate all64 descriptors BEFORE the common loss;
this does not split the mining batch. Early ten blocks use no_grad, last two do
not. All modules remain eval mode (frozen dropout/stochastic layers); autograd
is enabled on adapters. No BatchNorm batch-stat mismatch is introduced.

## Selection and reporting

GSV development1024 disjoint places, deterministic four clean views, full
4096-image self-excluded retrieval. Select max correct; tie minimum softplus
hardest-negative minus best-positive margin/.05; exact ties earliest. Include
epoch0 so an unchanged RU can win. Record descriptor L2 drift from epoch0.
MSLS/Pitts are evaluated at RU baseline, fixed last and GSV-selected only, not
used for selection. Require baseline675/740 and7160/7608. Both are historically
exposed development benchmarks; not independent final tests.

Report paired corrections/regressions for all arms after completion. Do not call
the method successful unless true-place adaptation beats both controls and RU
without cross-dataset regression. One seed is a screen, not stable evidence.
No automatic larger-budget training or threshold/lr sweep.

## Safety and execution

Run three two-batch smoke tests first; each must show real nonzero VPR updates,
zero-start descriptor agreement, finite gradient connectivity, frozen-weight
identity and checkpoint roundtrip. Any error stops the sequential runner.
Resume saves parameters/optimizer/RNG/cursor every16 batches and at epoch ends;
work after the last save is replayed. Per-place/epoch image RNG remains matched.
No local tests; test and execute only on the training machine's physical GPU1.

```bash
bash scripts/run_dsa_screen.sh
```

Outputs: `logs/dsa_screen_v1/{pca,shuffled_fisher,place_fisher}`.
Each has contract (including exact schedule), progress, history, last/best
checkpoints, predictions, summary and completed hashes. Do not alter old runs.

## CLOSED — user decision, 2026-09-30

All three runs completed and their completion hashes were verified. Identical
batch schedules; frozen original tensors unchanged. No further DSA training,
rank/lr sweeps or additional seeds are planned. Preserve checkpoints and evidence.

| Arm | GSV-selected epoch | MSLS correct /740 | Pitts correct /7608 | Fixed-last MSLS / Pitts |
|---|---:|---:|---:|---:|
| RU | 0 | 675 | 7160 | 675 /7160 |
| PCA | 1 | 673 | 7152 | 669 /7137 |
| Shuffled Fisher | 2 | 675 | 7158 | 675 /7157 |
| True Fisher | 3 | 673 | 7158 | 673 /7158 |

True Fisher paired against RU: MSLS1 correction/3 regressions; Pitts11/13.
Effective updates: PCA718, shuffled716, true716. True Fisher loss fell from
0.10796 to0.08274; mean GSV descriptor L2 drift reached0.09349. This was not
an inactive branch. All three epochs of true/shuffled Fisher retained exactly
the same GSV correctness outcomes as RU (4081/4096); checkpoint selection was
driven only by the auxiliary margin tiebreaker on a saturated development set.

Conclusion: current fixed-subspace Q/V adaptation failed to demonstrate final
VPR gains. Positive intermediate-vector diagnostics do not establish beneficial
token-level parameter-update directions. This does not disprove all supervised
subspace learning. Do not relabel phase0 as passed or claim stable gains.
