# PMD correspondence supervision gate

## Prior result

Expanded fixed-top20 GSV dev:1792 queries, frozen1747 correct; ordinary1743,
forced1743, partial1742. All three corrected0 original errors; regressions4/4/5.
There were13 reachable errors and32 unreachable errors. Existing best checkpoints
were fixed before expansion. These are historically exposed development data,
not independent confirmation. Current PMD configuration is not supported as an
accuracy improvement. Retain checkpoints; do not extend its VPR training.

## New bounded hypothesis

Can direct, known-transform supervision teach local matching and rejection
before attempting VPR transfer? This does NOT yet train an improved VPR model.

From each GSV image resize to406x406, then independently crop two322x322 views
at integer14-pixel patch offsets, optionally horizontal-flip and mask a5x5
patch block. No post-crop resizing. Source patch IDs define exact bidirectional
targets. Outside overlap and masked patches are unmatched. Place labels never
serve as patch correspondences. Contextual encoder receptive fields can cross
patch boundaries; labels describe source geometry, not isolated feature content.

Use frozen official Pair-VPR encoder and blocks1..11, in both input directions.
Train only shared LayerNorm/key projection and, where applicable, dustbin scalar.
Value/output projections and final VPR scoring are not trained or evaluated.
Three modes: independent softmax WITH dustbin (rejection-capable control), forced
balanced transport, partial transport. Same projection initialization; matched
and unmatched NLL groups equally weighted. Forced transport has matched-only
loss because rejection is structurally impossible; its rejection score is not
evidence of partial transport superiority. No automatic success threshold.

256 hash-selected GSV train images,64 place-disjoint dev images,3 epochs,
AdamW1e-4, seed42; train alternates crop-only and occlusion. Initial/final dev
reports both conditions, matched accuracy WITH rejection counted as error,
unmatched rejection recall and matched false-rejection rate. Patch observations
are correlated; do not treat patch count as independent statistical sample size.
Same-source synthetic views are easier than revisit images and are not proof of
cross-view robustness. No epoch selection, no VPR accuracy claim.

Atomic checkpoint every16 examples; deterministic transformations allow cursor
resume. No dropout, no random minibatch sampling; image hashes and official
checkpoint identity locked. No dense feature cache. Checkpoints retained.

```bash
python -m pytest -q tests/test_pmd_correspondence.py
bash scripts/run_pmd_correspondence.sh
```

Only if this diagnostic learns useful correspondences without excessive false
rejection should a separate plan add supervision to VPR training. Include harder
photometric transforms, shortcut controls and real-revisit diagnostics before
claiming a mechanism generalizes. Dustbin/optimal transport are established
matching tools, not novelty by themselves.
