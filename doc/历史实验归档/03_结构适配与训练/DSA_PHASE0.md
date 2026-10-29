# DSA phase 0: locked implementation contract

Implementation: `scripts/dsa_subspace_check.py`, `src/discriminative_subspace.py`,
`src/models/discriminative_subspace.py`. This is a subspace feasibility check,
not formal VPR training. No optimizer is constructed and no parameter is updated.

Reuse the completed QSO mean-outer run's original 4096 GSV training and 1024
place-disjoint development IDs. Verify source completion hashes, RU checksum and
GSV metadata. Four deterministic clean epoch-0 views per place; average normalized
attention-input patch tokens in blocks 11/12, excluding CLS. Save image-level
vectors only (approximately 120 MiB), not dense patch features. Resume requires
identical code/data/policy contract; atomic per-batch shards are reused.

For each block fit rank-16 PCA, shuffled-place Fisher and true-place Fisher on
training vectors only. Equal-place scatters use biased covariance; regularized
within scatter is `0.9 Sw + (0.1 + 1e-6) trace(Sw)/768 I`. Euclidean-orthonormalize
the selected span. Development labels are used only for the diagnostic, not fitting.
Report eigenvalues, conditioning, projected between/within variances and overlap
between two train-half fitted spans. Shuffling preserves four images per group,
but does not guarantee every shuffled image comes from a different place.

Predeclared heuristic gate, required at BOTH blocks:

- True Fisher development between/within ratio exceeds both controls by 5%.
- Retain at least 25% of PCA's development between-place variance.
- Mean squared principal-angle cosine with shuffled Fisher is below 0.95.

These thresholds are engineering screening rules, not theoretical guarantees or
paper-derived success criteria. Train-half overlap is diagnostic, not an extra
post-hoc gate. Even PASS does not establish token-level transfer or VPR gains.

Install fixed-basis Q/V adapters in blocks 11/12: `W x + B U^T x`, B zero-start,
49,152 trainable parameters, original RU/BoQ weights frozen. Execute all controls
on real GSV images: reproduce RU descriptors within 2e-6, require finite nonzero
gradients for all four B matrices, no gradients on frozen weights, B remains zero.
The probe scalar is only a connectivity test, not a VPR loss or training step.

Run on physical GPU1 in the existing VPR environment:

```bash
export CUDA_VISIBLE_DEVICES=1
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
python -u scripts/dsa_subspace_check.py --checkpoint "${RU_CKPT:?}" --output logs/dsa_phase0_v1 --resume
```

Read `progress.json` and `summary.json` under the output. `completed.json` means
the diagnostic completed, not that it passed. No automatic formal training.
After PASS, first lock the optimizer/broad-GSV plus minority-hard batch protocol
and matched three-arm budget; then test actual recall and backbone drift.
MSLS/Pitts have been repeatedly inspected and remain exposed development data.
