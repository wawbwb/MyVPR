# Scan-consistent Mamba-BoQ (SC-Mamba-BoQ), exploratory v1

## Status and claim boundary

QR and its protocol/loss diagnostics are closed; preserve their checkpoints.
This is a new nonsemantic, single-image aggregation hypothesis. Adding Mamba
or four-direction scanning is NOT itself new. We test whether explicit
row/column descriptor consistency improves transferable place representation.
No claim of first use, demonstrated novelty, view invariance or speedup.

References: Mamba https://arxiv.org/abs/2312.00752 ; VMamba
https://arxiv.org/abs/2401.10166 ; Spatial-Mamba https://arxiv.org/abs/2410.15091 .
Related direct VPR work exists, including MT-FusionNet
https://doi.org/10.1016/j.neucom.2026.133927 . Full novelty review remains open.

## Model

Retain pretrained RU DINOv2 and historical RU gate frozen. Train the BoQ jointly
with a new module after its input projection/normalization, before query
aggregation. Feature grid20x20, projected384 channels. Bottleneck64, real state8,
delta rank4. Shared Mamba-1 style block: input projection, causal depthwise conv4,
SiLU, input-dependent delta/B/C, stable diagonal A=-exp(A_log), D skip, SiLU gate,
zero-initialized output projection. No semantic supervision or candidate changes.

Scan row-major and its reverse, column-major and its reverse. Undo permutations
and average forward/reverse in each axis. Two residual feature maps enter the
same BoQ. Inference descriptor=normalize(d_row+d_col). This uses TWO BoQ reads;
cost must be measured, not presented as free acceleration. The ordinary Mamba
and consistency arms use exactly the same architecture/inference computation.

L = MultiSimilarityLoss(fused descriptor) + lambda * .5*||d_row-d_col||^2.
Lambda=0 for ordinary Mamba, .1 for consistency. All descriptors normalized.
Both paths receive gradients, no stop-gradient. Retrieval loss opposes trivial
collapse, but zero residual also satisfies consistency: measure activity/drift
and ordinary-Mamba-relative retrieval, never declare success from consistency
alone. Scan invariance is NOT physical viewpoint invariance.

Pure PyTorch independently implemented real selective recurrence; prefix affine
composition checked against a sequential oracle in forward and backward. No
Mamba package or CUDA environment upgrade. This reference implementation has
O(L log L) work/memory, unlike the official linear-work fused selective scan.
It validates a modeling hypothesis, NOT official Mamba efficiency. Bottleneck
compression is a deliberate variation, not an official pretrained Mamba model.

## Matched arms

- boq: original BoQ fine-tuning control.
- conv: gated bottleneck depthwise2D-conv mixer, near-matched new parameter count.
- mamba: same architecture as proposal, lambda0.
- mamba_consistent: lambda.1; the proposed method hypothesis.

All original BoQ weights train at1e-5; mixer1e-4; AdamW wd0, FP32, clip1.
All start at RU descriptors, seed42. No historical hard-batch injection.
Hold out1024 places by a new fixed identity hash; training uses all remaining
GSV places, uniformly permuted without replacement per epoch,16 places x4 views.
Full epochs drop at most15 trailing places; subsequent epoch permutes anew.
These labels are place-disjoint within this experiment, not unseen RU-pretraining
data. No location-based deduplication claim. Fixed image augmentation and sampled
views are paired across arms. Microbatch4 checkpoint/recompute; mine the complete
64-image batch. Frozen backbone extracted with no gradients; no feature cache.

## Execution stages

All four smoke runs (4 batches each) must complete before pilot. Check initial
RU equivalence, finite actual updates, upstream selective-state gradient after
zero-start, frozen backbone, checkpoint roundtrip. Tests run ONLY remotely.

Pilot128 batches per arm from the all-GSV schedule (2048 places). It is NOT a
full epoch and NOT a performance verdict. Initial/final loss is measured on the
same8 fixed augmented batches;512 held-out images give only a small within-GSV
mechanism diagnostic. Examine loss, scan disagreement, descriptor drift,
gradient activity, memory and measured batch runtime. Do not choose a best
checkpoint or retune using MSLS/Pitts. Pilot stops automatically after four arms.

Only after pilot review, full stage starts fresh RU,3 complete training epochs,
all eligible train places, fixed-last MSLS and Pitts report. Code supports full
stage but launcher does NOT auto-start it. Independent repeats and matched
ordinary-Mamba/conv/BoQ comparisons required before claiming useful innovation.
An ineffective128-step pilot alone must not rule out the method.

Resume every16 batches and epoch end; contracts include schedule, source hashes,
metadata and software versions. Do not overwrite existing runs. Completed
artifacts have hashes. GPU1 PyTorch allocation cap40%; no unrelated process kill.

Run: `bash scripts/run_scan_mamba_pilot.sh`.
Outputs: `logs/scan_mamba_v1/{mode}_{smoke,pilot}`.

## v1 results and v2 protocol correction — 2026-10-07

All v1 pilots completed. Fixed8-batch average loss starts0.042167;
final BoQ0.029263, conv0.026616, Mamba0.026597, consistent0.025802.
The512-image within-holdout gallery is saturated at512 correct for every arm,
so it is not useful for an accuracy verdict. Ordinary/consistent scan disparity
is1.02e-6/1.46e-6. Consistent made128 Adam updates versus91 in other arms, because
tiny auxiliary losses activated updates on zero-VPR batches. The apparent
advantage is confounded by optimizer clocks; preserve v1 as a diagnostic, not
evidence of consistency effectiveness.

v2 changes NO architecture or loss weight. Every batch calls AdamW in EVERY arm,
including zero VPR batches; absent gradients become explicit zeros so every
trainable parameter has an identical step clock. Record positive VPR batch count
separately. Checkpoint and contract revisions forbid resuming v1 into v2.

Initial/final diagnostics add weighted consistency/VPR gradient norms and cosine
on the SAME first2 augmented batches. This is read-only; no optimizer update.
Also compare descriptors with the mixer bypassed while retaining trained BoQ:
this measures mixer sensitivity, not RU equivalence or a separately trained arm.

User authorized full training after corrected mechanical checks. New launcher
`bash scripts/run_scan_mamba_v2.sh` runs all4 smoke then all4 pilot stages, checks
hashes/matched schedules/128 updates/nonzero gradients/bypass response, then
automatically runs all4 full stages from FRESH RU, not pilot checkpoints.
No performance-based gate or tuning. All3 complete GSV training epochs, same
schedule, fixed-last MSLS/Pitts. Initial full benchmark must reproduce675/7160.
Outputs `logs/scan_mamba_v2`; no v1 data removed. Rough runtime budget is many
hours on GPU1, unlike the128-batch pilot. Any failure stops the queue.
