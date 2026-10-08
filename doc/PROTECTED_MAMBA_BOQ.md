# Frozen-RU Protected Mamba-BoQ — 2026-10-08

## Why this experiment

The completed SC-Mamba v2 screen jointly fine-tuned BoQ. All four arms fell
below original RU: RU 675/740 MSLS and 7160/7608 Pitts; BoQ 663/7115,
Conv 665/7110, Mamba 666/7115, scan-consistent Mamba 665/7108. All completed
11529 Adam calls, selective-state gradients were observed, frozen backbone and
checkpoint roundtrip checks passed. Mamba's small advantage over the degraded
BoQ control does not establish an improvement. This motivates isolating mixer
learning from drift of the original aggregator, NOT rejecting all Mamba methods.

References: [Mamba](https://arxiv.org/abs/2312.00752),
[VMamba](https://arxiv.org/abs/2401.10166),
[MT-FusionNet](https://doi.org/10.1016/j.neucom.2026.133927),
[MambaAdapt](https://www.mdpi.com/1424-8220/26/18/5799).
Mamba/VMamba motivate selective state modeling and directional scans (prior art).
MT-FusionNet and MambaAdapt provide VPR motivation for complementary fusion and
representation preservation. This implementation is NOT a reproduction of
their architecture, temporal adaptation or experience replay. Their reported
results use different protocols; they do not establish success on our datasets.

## Network and hypothesis

Freeze the complete RU backbone, gate, projection, learned BoQ queries and
readout. Reuse those weights on an unchanged RU path and a differentiable
correction path; only a 76992-parameter Mamba spatial mixer is trainable.
No semantic labels, image sequences, pairwise decoder or candidate expansion.
The 4 directional scans reuse the prior verified real selective recurrence.
No new dependency, environment or pretrained-Mamba download is required.

For unit RU descriptor r, frozen BoQ maps mixed tokens to proposal p. Project
its difference into r's tangent space:

    v = (p-r) - dot(p-r,r)*r
    delta = 0.1*v / (0.1 + ||v||)
    d = normalize(r + delta)

Zero-initialize the mixer output; initial d reproduces RU (tolerance 2e-6).
Bypass returns the EXACT original frozen RU path, not a retrained aggregator.
The correction is smooth, orthogonal to r and has norm <0.1. Therefore cosine
to RU is at least 1/sqrt(1.01), about 0.9950, up to numerical error.
**This does not guarantee rank retention or better recall.** The bound limits
descriptor displacement, not the ordering of similar database images.
Two frozen-BoQ readouts for Mamba plus the reference path add inference cost;
there is no acceleration claim. Pure PyTorch prefix scan has O(L log L) work,
not the official linear-work fused CUDA implementation.

The hypothesis is that directional token context can provide a useful bounded
correction while preserving the strong learned RU descriptor geometry. Adding
Mamba or cosine distillation alone is NOT a novelty claim; meaningful gains,
ablations and a further prior-art review would still be required.

## Three predeclared arms

| Arm | Mixer | Loss |
|---|---|---|
| conv_preserved | 79296-parameter local Conv | VPR + preservation |
| mamba_plain | four shared SSM scans | VPR only |
| mamba_preserved | identical Mamba initialization | VPR + preservation |

All share frozen RU and the same 0.1 tangent bound. "Plain" means without the
auxiliary preservation loss, NOT an unbounded network. Original RU is a fixed
evaluation baseline, not a zero-parameter training arm.
Preservation = T^2 KL(P_RU || P_student) + 0.1*(1-cos(d,r)), T=0.07,
where P are softmax batch image-image similarities with self-diagonal excluded.
The teacher uses the SAME augmented pixels, is detached and never changes.
Relations include positives and negatives without benchmark ground truth.
Primary MultiSimilarity VPR loss uses all64 images from16 places x4 views.
Weights/cap are design choices, not established optimal values; locked before
benchmark evaluation and no post-hoc sweep is launched automatically.

## Protocol and interpretation

Use the same GSV hash split, all-training-place epoch permutation, stateless
per-place views/augmentation, seed42, FP32, microbatch4 and identical Adam
clocks as SC-Mamba v2. Only mixer parameters learn, AdamW lr1e-4, weight decay0,
gradient clip1. The1024-place holdout is held out from this adaptation, not
necessarily from the historical RU pretraining. No new datasets are needed.

Run remote tests, all3 smoke4-batch arms, then all3 pilot128-batch arms. Check
initial RU equality, frozen full-RU tensors, exact bypass, nonzero upstream
SSM gradients, correction norm/cosine bounds and checkpoint roundtrip. Initial
and final probes reuse the same8 augmented batches, and read-only gradient
probes report preservation versus VPR gradient norm/cosine on the same2 batches.
The512-image GSV gallery is previously saturated: not an accuracy gate.

Mechanical checks only gate full runs; if passed, all3 full arms start FRESH RU
for3 complete epochs (not pilot weights), fixed final checkpoint, no benchmark
model selection. Initial benchmarks reproduce675 MSLS/7160 Pitts. Report final
R@1/5/20 and paired corrections/regressions. A benefit claim requires beating
RU and matched controls, followed by independent seeds; these historically
exposed benchmarks remain exploratory, not a clean final test. If preservation
only returns to RU without net corrections, it is stability, not improvement.

Runs: `logs/protected_mamba_v1/{mode}_{smoke,pilot,full}`.
Launcher: `bash scripts/run_protected_mamba_v1.sh`. Tests run ONLY remotely.
Any failed test, nonfinite update or mechanical check stops the queue. Shared
flock prevents overlap with prior experiments. Atomic last.pt every16 batches
and epoch end supports --resume under immutable code/data/model contracts.
Existing scan_mamba_v1/v2 artifacts and all checkpoints are retained.
