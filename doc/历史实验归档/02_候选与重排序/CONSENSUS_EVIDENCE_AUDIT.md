# Fixed-weight neighborhood evidence audit

## Question

Does neighborhood evidence distinguish positive candidates from negative candidates,
even when the trained pair classifier fails? This is an exploratory mechanism check,
not a new retrieval method or independent validation.

## Locked setup

- Spatial consensus epoch 3 (`logs/consensus/spatial_screen_v1/last.pt`), no updates.
- All 15 reachable frozen-model errors in the existing GSV dev split.
- 64 frozen-correct mixed-label queries chosen by a fixed SHA256 ordering before
  evidence extraction, plus all spatial epoch-3 regressions. Groups may overlap.
- All original top-20 candidates, no candidate expansion or dense disk cache.
- Verify completed source hashes, image/encoder identity, checkpoint identity, and
  reproduce saved epoch-3 native candidate scores before accepting results.

## Fixed higher-is-better metrics

1. Mean bidirectional maximum normalized patch similarity.
2. Local grid coherence: neighboring source patches should map to distinct target
   patches with displacement within one patch of the source displacement. This
   tests translation-like consistency, not scale/rotation-invariant geometry.
3. Learned consensus bias support: raw match probabilities weight row-centered
   consensus bias, averaged in both matching directions.
4. The same support after coordinate scrambling.
5. Normal support minus scrambled support.

Each metric is averaged over the two decoder directions. Frozen and trained pair
scores are reference metrics. No sign flipping, coefficient fitting, threshold
search, or selecting positive references using diagnostic evidence is allowed.
The reference positive and negative are chosen by frozen pair scores. Reports also
include ranking among all candidates and per-query positive-negative AUC (ties 0.5).

## Interpretation limits

Cases were stratified using ground truth on historically exposed development data.
Results can motivate a hypothesis, not certify generalization. A promising metric
must also preserve correct controls; counting rescued error cases alone is biased.
Several metrics are inspected, so a post-hoc winner needs separate confirmation.
No signal in these particular summaries rules out neither all spatial evidence nor
all possible readouts. It does argue against extending this exact branch blindly.

## Execution

Tests run only on the training machine:

```bash
python -m pytest -q tests/test_consensus_evidence.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/audit_consensus_evidence.py
```

Default report directory: `doc/consensus_evidence_v1`; an existing output is never
overwritten. This short diagnostic does not implement resume.
