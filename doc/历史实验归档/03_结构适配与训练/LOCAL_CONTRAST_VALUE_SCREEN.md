# LCV-BoQ: local-contrast value encoding

## Scope and references

QSO did not improve retrieval and is stopped. This is a different structural
hypothesis, not a verified novelty: preserve existing query-key routing and
enrich per-head values with explicit local differences. DINO already has spatial
context; the hypothesis is incremental utility, not that it lacks spatial cues.

- Convpass: lightweight convolutional ViT adaptation, https://arxiv.org/abs/2207.07039
- SelaVPR: frozen-foundation VPR adaptation, https://arxiv.org/abs/2402.14505
- CDC: neighborhood-minus-center operators, https://arxiv.org/abs/2003.04092
  (face anti-spoofing evidence, not proof of VPR gains).

## Exact structure

For each existing BoQ block, keep its encoder, learned queries, Q/K/V projections,
per-head attention and final projection frozen. Normalize post-encoder tokens
and project 384 channels to 32, reshape to the registered 20x20 grid. For each
of eight offsets in a 3x3 neighborhood, learn one weight per reduced channel.
Use replicate padding. Apply GELU then a zero-start bias-free projection to 384.
Split into original attention heads, multiply each head's original attention by
its own added values, concatenate and apply the frozen attention output projection
without a second bias. Add to historical output before existing slot LayerNorm.
Do not average heads to form this residual. Output descriptor dimensions unchanged.

Matched modes (identical trainable weights/initialization, 49,664 parameters total
for two 384-channel blocks):

- local_conv: sum of weighted eight neighbor features (no center tap).
- local_contrast: sum of weighted (neighbor - center) features.
- shuffled_contrast: fixed seed42019 permutation, same difference operator, inverse
  permutation before attention. It destroys real neighborhood adjacency, not the
  correspondence between restored positions and original attention.

This eight-neighbor convolution is a parameter-matched control, not a generic
nine-tap 3x3 convolution. The geometry control has the same learned parameters.
No semantic input, reranking, new backbone weights, or extra training loss.

## Shared hard-negative batches

Verify original QSO training contracts and completed training-side audit. Use only
its 4096 training places, excluding 1024 development places. Mine unique place
pairs from frozen-RU errors and margin <=0.02 correct queries using their recorded
best negative. No use of MSLS/Pitts outcomes to build batches.

For geographic filtering, keep a center and maximum coordinate radius per place
using all its recorded coordinates. Require center distance minus both radii >=
100m for **every pair of different places in a batch**. Abort on invalid metadata
or insufficient safe pairs. This reduces false-negative risk, not a guarantee.

Each batch: four mined pairs (eight places) plus eight randomly drawn distinct,
geographically compatible places. Hard places intentionally repeat; this is not
a one-pass epoch. Save all exposure counts and three fixed 256-batch schedules
(seeds42/43/44). Augmentation/views follow original per-place/epoch seeded sampler;
clean epoch0 mined negatives are not guaranteed hard under later augmentation.

All three arms share exactly these batches, 4 views/place, 3 rounds (768 batches),
AdamW lr1e-4 wd0, FP32, gradient clip1 and MultiSimilarity miner/loss. Sampling is
an experimental common condition, not claimed structural novelty. No comparison
of loss magnitudes with old random-batch QSO runs as a controlled result.

## Verification and selection

Training-machine-only unit tests: constant-field zero contrast, boundary behavior,
per-head formula, unchanged attention, exact parameter matching, gradients after
two updates, frozen weights, geographic filters and repeatable batches.
Real-image smoke for every mode before formal queue: RU zero-start error <=2e-6,
finite gradients, checkpoint roundtrip and unchanged frozen weights. Original
MSLS675/Pitts7160 must reproduce. Failures stop the queue.

Keep GSV epoch0 and each round; select highest GSV correct then lowest margin loss,
earliest exact tie. Report fixed round3 and GSV-selected paired MSLS/Pitts results.
These are historically exposed development benchmarks, not independent tests.
Only expand if real-neighbor contrast beats both controls and RU across datasets.
If all improve equally, attribute evidence to adaptation/shared sampling, not
local differences. One seed is not a stable-benefit or publication-novelty claim.

Resume is contract-checked; optimizer/cursor/RNG saved every256 place presentations.
No persistent dense descriptor cache is needed; preserve all historical results.

Run on training machine: `bash scripts/run_local_value_screen.sh`.
Outputs: `doc/local_value_plan_v1`, `logs/local_value`, `doc/local_value_screen_v1`.
