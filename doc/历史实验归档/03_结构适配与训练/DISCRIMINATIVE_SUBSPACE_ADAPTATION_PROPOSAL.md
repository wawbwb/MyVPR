# Discriminative-subspace backbone adaptation — proposal only

Status: not implemented, not trained, not an established novelty. LCV is closed.

## Hypothesis

Recent local experiments modified aggregation while preserving the RU backbone.
This does not establish that the backbone is the limiting factor. Test a different
location: Q/V projections in DINO blocks11 and12, with the original BoQ/RU gate
frozen. Constrain the low-rank update input to directions with high between-place
relative to within-place variation, estimated using GSV training views only.

## References

- SelaVPR, ICLR2024: frozen-foundation lightweight VPR adaptation,
  https://arxiv.org/abs/2402.14505
- LoRA: low-rank updates to frozen Transformer weights,
  https://arxiv.org/abs/2106.09685
- EigenPlaces, ICCV2023: training across viewpoints for place discrimination,
  https://arxiv.org/abs/2308.10832

These support adaptation and multi-view supervision, not the efficacy or novelty
of the proposed Fisher-subspace constraint. Low-rank adaptation and discriminant
analysis are established; combining them is not automatically publishable.

## Candidate architecture

For each selected block, average the frozen attention-input normalized patch
tokens per training image (no CLS), obtaining h_pv. Compute within-place scatter
Sw and between-place scatter Sb, equal-weighting places. Solve the regularized
generalized eigenproblem Sb*u=lambda*(Sw+eps*I)*u. Choose rank16, orthonormalize
the selected span in Euclidean space, and freeze the resulting U. Shrinkage/eps,
layer choice and rank must be locked using training data, not MSLS/Pitts.

Change each Q/V projection from W*x to W*x+B*(U.T*x), B initially zero.
Train B only, separately for Q and V in each block; original W, U, BoQ and RU gate
stay frozen. Four 768x16 matrices = 49,152 trainable parameters for ViT-B.
The zero-start reproduces RU; the trainable update alters backbone token content
and attention rather than adding pooled statistics after feature extraction.
Upstream inputs at block12 drift as block11 adapts: fixed-U validity must be
measured, not assumed. Image-average statistics may not transfer to token-level
use; this is a principal risk to test early.

## Controls and experimental discipline

Three matched fixed-basis arms: top PCA span; Fisher span from place-label-shuffled
groups of four views; true-place Fisher span. Equal rank, orthonormal bases, same
B initialization and optimizer/batches. Shuffled groups preserve group sizes.
Frozen RU is the reference. An ordinary trainable-A-and-B LoRA baseline is useful
for follow-up but has more parameters and must not be described as matched.

Before training, compare spans and held-out GSV between/within scatter ratios;
the training Fisher objective being high is tautological, not evidence. Check
eigenspectrum/conditioning, basis stability on train-only resamples, and whether
true and shuffled spans are meaningfully distinct. Do not whiten descriptor outputs.

Reuse existing datasets, not a new benchmark download. Avoid treating the prior
88 mined pairs as the entire task: use broadly covered GSV training places plus
a fixed minority of geographically screened difficult batches, identical across
arms. Exact sample count, update budget and mixture require a locked implementation
contract before launch. Selection stays on disjoint-by-place GSV development;
MSLS/Pitts remain exposed development benchmarks, not independent final tests.

Single-image inference, no semantics/segmentation/CLIP/reranking or candidate
expansion. Backbone gradients are needed through the last two blocks, unlike the
old detached feature helper: do not reuse its no_grad feature path unchanged.
No persistent dense patch cache; accumulate scatters online or retain only small
image-level vectors. No superiority or wall-clock estimate before measurement.

Stop if the pre-training subspace check is uninformative or the true-place arm
does not beat both fixed-basis controls and RU without cross-dataset regression.
Only expand seeds after a useful first screen; preserve fixed-last and GSV-selected
outcomes. Do not convert a few selected-query gains into a stable-benefit claim.
