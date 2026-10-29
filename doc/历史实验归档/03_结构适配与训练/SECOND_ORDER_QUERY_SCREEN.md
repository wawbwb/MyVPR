# QSO-BoQ matched structural pilot

CD-BoQ is not expanded: overlap fell but pre-normalization slot effective rank
did not increase; MSLS +1 query, Pitts -3 vs RU. Preserve its artifacts.

## Hypothesis and sources

Explicit query-conditioned second-order local-feature statistics may complement
the weighted mean in BoQ. This is a proposed mechanism, not established novelty
or likely guaranteed improvement. BoQ as a whole is nonlinear, not merely a mean.

- Bilinear CNN, ICCV2015: https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html
- Compact Bilinear Pooling, CVPR2016: https://openaccess.thecvf.com/content_cvpr_2016/html/Gao_Compact_Bilinear_Pooling_CVPR_2016_paper.html
- Learning second-order statistics for place recognition, Neurocomputing2020:
  https://doi.org/10.1016/j.neucom.2020.02.001

Second-order VPR is already prior art; candidate contribution is query-conditioned
covariance residuals in pretrained BoQ, subject to further novelty review.

## Exact implementation

Keep frozen historical cross-attention, including its head-specific value output.
Average its attention weights across heads for ONLY the additional branch.
Project LayerNorm(key tokens) to16D with a trainable, bias-free projection.
Each BoQ block gets independent new parameters, shared across all its query slots.

- mean_outer: weighted mean outer product m_q*m_q^T (first-moment-only control).
- global_cov: uniform population covariance, repeated to all query slots.
- query_cov: A-weighted population covariance E[uu^T]-m_q*m_q^T.

Take the136 upper-triangle entries; multiply off-diagonals by sqrt(2), apply
x/sqrt(abs(x)+1e-4) and L2 normalization eps1e-6. This is elementwise smooth
power normalization, NOT matrix square-root or whitening. It discards overall
scale, so the hypothesis concerns relative covariance structure, not raw variance
amplitude. All controls use the same representation shape and normalization.

Map136->64->BoQ dimension with GELU; final bias-free projection starts at zero.
Add before original norm_out. Original attention is never modified. No explicit
spatial geometry, semantics, external teacher, candidate expansion or dense cache.
All arms have identical trainable parameter counts and initial weights, but not
identical computation costs. Log and do not claim acceleration.

## Frozen protocol

- Original RU SHA256 38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e.
- Freeze all old parameters/buffers. Train projection, hidden layer and zero-output
  projection only. Verify frozen state byte equality after the run.
- Same4096 hash-selected GSV places, four views/place,16 places/batch, three subset
  passes (768 seen batches, not full GSV epochs). No extra image cache.
- AdamW lr1e-4, zero decay, FP32, gradient clip1. Skip exactly zero mined-loss
  optimizer updates for every arm; record effective updates and zero batches.
- Select on the same disjoint1024-place GSV development set: max correct count,
  then minimum smooth margin loss; earliest exact tie, include epoch0.
- MSLS/Pitts only evaluated at epoch0, fixed last and GSV-selected checkpoint.
  Must reproduce675/740 and7160/7608 before any formal update. They are historically
  exposed development sets, not untouched final tests.
- Real-data smoke runs for all arms precede any formal training. Synthetic tests
  explicitly verify covariance formula, same-mean/different-structure example,
  zero start, bypass equality, equal capacity and upstream gradients after two
  updates. No local-machine execution; tests run on the training machine.
- Atomic optimizer/RNG/cursor checkpoint every256 places; immutable code/data
  hashes for resume. Save tiny added-parameter checkpoints, not copies of RU.

## Interpretation

Query covariance must beat frozen RU and both controls before expansion is
considered. Beating only mean_outer supports second-order information, not query
conditioning; matching global_cov does not validate the proposed contribution.
Report paired fixes/regressions on both datasets; no success from branch activity,
rank increase, or training loss alone. No automatic extra seeds/long training.

Launch on GPU1: bash scripts/run_second_order_query_screen.sh
Progress: logs/second_order_query/{mode}_screen_v1/progress.json
Summary: doc/second_order_query_screen_v1/summary.json
