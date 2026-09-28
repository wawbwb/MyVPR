# CD-BoQ: redundancy diagnosis before training

DSQ-BoQ is stopped at the user's request. Its four arms completed; adaptive
depth did not beat RU, equal-depth or last-repeat controls. No DSQ follow-up.

## Hypothesis, not established benefit

Independent row-normalized query attention can reuse the same patches.
Test whether this produces relevant redundancy before paying for training.
Overlap itself is NOT a fault: several queries may use the same landmark well.

References: BoQ https://arxiv.org/abs/2405.07364 ; Slot Attention
https://arxiv.org/abs/2006.15055 ; SALAD https://arxiv.org/abs/2311.15937 ;
VLAD-BuFF https://arxiv.org/abs/2409.19293 . These motivate aggregation and
competition; they do not establish novelty or improvement of this proposal.

## Implemented mechanism

Let A=softmax(S) across patches. c_qn=sum_{j!=q} A_jn/(Q-1).
Competitive attention = softmax(S-lambda_h*log(1+N*c)).
Projected lambda in [0,2], initialized at zero; no ReLU/squared parameter at
zero that would block the initial derivative. Call project_ after each update.
Temperature control uses signed log inverse-temperature in [-2,2].
Shuffled control applies a fixed seed42017 patch permutation to c.
All arms would train one scalar/head; all historical parameters stay frozen.
Use the legacy attention output plus (modified manual output - original manual
output) so zero start is exact without replacing the historical kernel output.
This compatibility implementation computes extra attention and is not an
acceleration claim. No masks, dropout or semantic-conditioned variants supported.

## Current stage: diagnostic only

No optimizer or training queue. Hash-select 1024 GSV places from the first4096
DSQ subset; another1024 from positions4096:5120 are disjoint development places.
They are NOT asserted unseen during original RU training. Four deterministic
views per place; each view retrieves within its4096-image partition, excluding
itself, with three positives. Report all views; correlated views are not
independent statistical evidence. No MSLS/Pitts parameter selection.

Collect per-block, per-image within-head query attention cosine overlap,
normalized entropy and singular-value-entropy effective rank of pre-norm slots.
Report errors/correct distributions and quartile error rates. Interpret jointly:
high overlap alone, easy retrieval or too few errors is insufficient evidence.
Preserve descriptor identity preflight, hashes, immutable split and per-view
outcomes. Atomic resumable per-place shards need about400MiB, no dense token
cache. Original image bytes are not hashed; metadata files and model are hashed.

Training and later MSLS/Pitts evaluations are deliberately NOT launched until
this evidence is reviewed. Planned arms: temperature, competition, shuffled
competition; RU fixed. Any claimed benefit must beat matched controls, repeat
across seeds and distinguish exposed development sets from unseen tests.

Remote test: python -m pytest -q tests/test_competitive_query.py

Remote diagnosis (physical GPU1 only):

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/audit_competitive_query.py --checkpoint "$RU_CKPT" --output doc/competitive_query_audit_v1 --resume
```
