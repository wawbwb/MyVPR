# DSA cross-view diagnostic (exploratory, no training)

Phase0 finished with STOP_SUBSPACE_GATE: Fisher retained 16.36%/10.09% of PCA
between-place variance, below the locked 25% gate, despite better held-out scatter
ratios. This follow-up does not overwrite that verdict or loosen its threshold.

Reuse verified phase0 vectors (4096 training / 1024 development places). No image
extraction, GPU inference, model update, new datasets or developer-label fitting.
Fit rank16 bases on full training and on each even/odd half of training places.
Shuffle within each half AFTER splitting, so source places never cross halves.
This differs intentionally from phase0's shuffled-group-half overlap diagnostic.

For each block (11/12), use all 12 ordered distinct view pairs. Each pair has
1024 query images and 1024 reference images, one per development place. Correct
means identical place ID, not a geographical-distance GT. Different place IDs
can be physically nearby; results are diagnostic, not MSLS/Pitts recall.

Compare original 768D image-average intermediate features, PCA16, shuffled
Fisher16, true Fisher16. Original means NOT final RU descriptors. Primary metric:
cosine after subtracting that fitting split's TRAINING mean and, for the three
controls, projecting into the Euclidean-orthonormal basis. No whitening. Report
uncentered cosine as a fixed secondary check, never choose whichever looks best.

Report R1/5/10, paired corrections/regressions and 2000 place-cluster bootstrap
intervals (seed42030); all 12 pairs of a place move together. Intervals are
exploratory/unadjusted and don't establish cross-city generalization. A descriptive
CONSISTENT_CONTROL_ADVANTAGE requires positive primary R1 deltas against BOTH
rank-matched controls at BOTH layers under full/half0/half1 fitting. No automatic
training launch, even if this check passes. Original-feature comparisons are
reported separately because dimensions differ. These are diagnostic retrievals,
not evidence of useful patch-level attention updates or final VPR improvement.

Run on training machine CPU only:

```bash
python -u scripts/dsa_retrieval_check.py --source logs/dsa_phase0_v1 --output logs/dsa_retrieval_v1
```

No resume/overwrite; interrupted runs retain evidence and require a new output.
`completed.json` hashes the outputs; `summary.json` and `per_place_ranks.npz`
retain all comparisons and the source development-place ordering.
