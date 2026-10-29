# QR-BoQ: query-region relative-layout residual

DSA is closed. This proposal is a different single-image aggregator structure,
not a new backbone subspace, semantic model, candidate expansion or reranker.

## Hypothesis and primary sources

BoQ uses learned queries to aggregate image features. Its backbone/attention are
already contextual and may encode position implicitly; it is incorrect to call
BoQ spatially blind. The hypothesis is narrower: explicitly binding pooled query
content to the relative layout of its attention support may help discriminate
similar-looking places. This has NOT been established on our failures.

- BoQ, CVPR2024: https://openaccess.thecvf.com/content/CVPR2024/html/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.html
- R2Former, CVPR2023: https://arxiv.org/abs/2304.03410 — its pairwise reranker
  combines correlations, attention and coordinates. It motivates testing spatial
  evidence, but does not prove single-image relation aggregation will work.
- Relation Networks, NeurIPS2017: https://papers.nips.cc/paper_files/paper/2017/hash/e6acf4b0f69f6f6e60e9a815938aa1ff-Abstract.html
  — shared pairwise relation encoding and aggregation, not a new invention here.

Candidate contribution is attention-support uncertainty-aware query-to-query
geometry inside a compact global descriptor. No claim of first use or established
novelty; requires further related-work review if experiments become promising.

## Proposed structure

Keep original RU and descriptor width. At the final BoQ cross-attention block,
retain original head-specific output unchanged. For the added branch only, average
attention across heads: A[q,n]. With normalized patch coordinates p[n], compute
mu[q]=sum A[q,n]p[n] and spread v[q]=sum A[q,n]||p[n]-mu[q]||^2.
Use the pooled query content z[q] and pairwise displacement mu[r]-mu[q], distance,
two spreads and attention-overlap as edge inputs. A shared narrow edge MLP
processes projected z[q],z[r] plus these geometric features. Aggregate over r!=q,
then add a zero-output-initialized residual to the original slot before norm_out.
Only new branch parameters train in the first screen; unchanged backbone and RU
path prevent confounding a new structure with backbone finetuning.

All slot pairs (no arbitrary top-k neighbor tuning), narrow projected edge width,
chunk edges if needed. Complexity scales with query-count squared, not patch-pair
matching. Actual GPU memory/time must be measured; no acceleration claim.
Spreads/overlap expose uncertain or diffuse attention to the MLP, not a proven
confidence score. They do not force suppression. Preserve exact RU zero start.
First backward reaches zero-output projection; upstream modules must be checked
after the first genuine update, not incorrectly required nonzero at step0.

Coordinate differences remove a common translation of the measured supports but
do not guarantee camera/viewpoint/crop invariance. No absolute GPS or semantic
labels are consumed by the model. A slot is not guaranteed to be a physical object.

## What differs from prior experiments

- CD-BoQ penalized patch reuse across queries; this models relationships without
  forcing queries to attend to different patches.
- QSO modeled feature covariance within a slot; this models relative spatial
  relationships between slots.
- LCV modified neighboring patch values; this forms a graph across pooled query
  regions, including long-range relationships.
- DSA constrained backbone updates; this leaves the backbone unchanged.

## Matched experiment, before any success claim

RU reference plus three equal-width/trainable-capacity arms: appearance-only
relation, aligned relative geometry, and geometry permuted among slots while
keeping appearance slots fixed. The appearance-only arm supplies a fixed common
geometry vector; all arms retain the same MLP tensors, but effective input
information differs. Label this a capacity control, not identical optimization.
For shuffled control permute centroid/spread together with both axes of overlap,
NOT an entire content-plus-geometry slot permutation (which could leave a
permutation-equivariant graph unchanged). Deterministic shared seed, no labels.

First: correctness tests and real-image forward/backward preflight. Report attention
spread/centroid diversity, branch sensitivity and cost as diagnostics; do not use
a newly invented proxy threshold to claim success. Then one locked short matched
VPR training screen, not a parameter sweep.

Replace the saturated tiny-gallery development selection BEFORE training: retain
place-disjoint GSV development queries, add a fixed wider gallery from existing
GSV training places, record possible nearby-place ambiguities and exclude exact
query images. Select identities/distractor budget without comparing trained arms.
Keep the full query set (not only RU errors); report a frozen-RU difficult stratum
secondarily, alongside all queries. No new dataset download. Broad GSV training
coverage plus fixed minority hard batches, identical across arms. Lock exact
gallery/GT/epochs/lr after code and baseline audit, not from test-arm outcomes.

Train with final VPR retrieval loss, no intermediate-feature auxiliary objective.
Report fixed-last and GSV-selected MSLS/Pitts paired outcomes; benchmarks remain
historically exposed. Only aligned superiority over both controls AND RU without
cross-dataset regression justifies more seeds. If aligned equals shuffled or gains
only arise from capacity, do not claim spatial innovation. Stop the configuration
if matched screen fails; no automatic longer runs.

Major risks: attention centroids collapse; viewpoint shifts destroy layout;
existing contextual features already contain all useful geometry; edge branch
fits dataset framing rather than place identity. This is a testable hypothesis,
not a high-probability guarantee or a demonstrated innovation.

## Implemented first stage (2026-10-04)

`src/models/query_relation.py` adds the branch to the final BoQ cross-attention
only. Head-averaged attention defines centroid/spread/cosine-overlap; six edge
geometry inputs,16D appearance projection and32D edge MLP. Zero32->D output.
All query pairs excluding self, chunked by8 source queries. Appearance supplies
zero geometry; shuffled permutes both geometry axes using seed42040. Original
attention output/head values are unchanged. Three unit tests run remotely only.

`scripts/query_relation_preflight.py` runs two real clean-GSV VPR updates per
arm using the first two historical hard batches; this is connectivity/safety,
not the formal augmentation/training protocol. Frozen tensors are compared
exactly. Then original RU alone evaluates4096 development images against20480
images (all four views of4096 train+1024 development places), excluding self.
Gallery identities are fixed before seeing its recall. Same-place identity GT;
nearby distinct-place ambiguity is explicit, not claimed to be removed.
Do not automatically train after this audit; inspect whether expanded gallery
actually reduces saturation and audit false negatives before finalizing selection.

No checkpoint survives the two smoke updates; no old experiment is overwritten.
Outputs: `logs/query_relation_preflight_v1`. No new image/patch cache, no saved
global feature cache. GPU1 PyTorch allocator capped at40% device memory to leave
room for unrelated services (non-PyTorch allocations are outside this cap).
Launch: `bash scripts/run_query_relation_preflight.sh`.
