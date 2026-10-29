"""Fixed three-arm warm-start study. No search or automatic follow-up runs."""
import hashlib
import numpy as np

from src.official_adaptation import exact_paired


ARMS = ('a_continue', 'b_last_condition', 'c_cross_condition')
WARM_PARAMETER_SHA = 'ecffacbefb6c682e9d14e7426360a2f32cf1a78a8d762d8b00c2a618fe23a404'
POLICY = dict(
    revision=1, arms=list(ARMS), seed=42, smoke_steps=8, steps=2048,
    warm_parameter_sha256=WARM_PARAMETER_SHA, warm_updates=2048,
    warm_source='Sealed conv_official_layers_v1/last_only; previously exposed benchmark choice',
    layers=[8, 10, 12], dim=128, rank=32, rho=.025,
    anchor_lr=1e-5, complement_lr=1e-4, weight_decay=0., clip=1.,
    precision='fp32', microbatch=4, train_size=280,
    eval_sizes=[280, 322], primary_eval_size=322,
    schedule='Same sealed2048 rows/fixed epoch99 four-view order in every arm; smoke first8 discarded',
    initialization='Same trained anchor, NEW Adam/RNG seed42, auxiliary frozen; B/C same complement init',
    training='Freeze all author DINO/BoQ. Update anchor and, in B/C, conditional complement only',
    loss='GPS-filtered MultiSimilarity only; no auxiliary/teacher/rank/decorrelation loss',
    routing=dict(a_continue='anchor only, all anchor inputs L12',
                 b_last_condition='anchor L12; complement L12/L12/L12',
                 c_cross_condition='anchor L12; complement L8/L10/L12'),
    selection='Fixed final2048 for all arms. No best-epoch/size/layer/seed search or extension',
    scope='Single-seed2048-additional-update exploratory validation; exposed GSV/MSLS/Pitts-val, not full training, independent test, stable gain or novelty proof',
    inference='One image -> one12288-d descriptor -> dot product; no cross-image inference/reranking',
)


def named_hash(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters.items():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def matched_contract(contract):
    return {key: value for key, value in contract.items()
            if key not in ('arm',)}


def compare_outcomes(outcomes, official, warm):
    if set(outcomes) != set(ARMS):
        raise ValueError('All three completed arms are required')
    arrays = [np.asarray(x) for x in [official, warm, *outcomes.values()]]
    if (any(x.ndim != 1 or x.dtype != np.bool_ for x in arrays)
            or not len(arrays[0]) or any(x.shape != arrays[0].shape for x in arrays)):
        raise ValueError('Matched nonempty Boolean outcome vectors required')
    result = {}
    for arm in ARMS:
        row = outcomes[arm]
        result[arm] = dict(correct=int(np.sum(row)), queries=len(row),
                           r1=float(np.mean(row)), vs_official=exact_paired(official, row),
                           vs_warm_start=exact_paired(warm, row))
    result['b_vs_a'] = exact_paired(outcomes['a_continue'], outcomes['b_last_condition'])
    result['c_vs_a'] = exact_paired(outcomes['a_continue'], outcomes['c_cross_condition'])
    result['c_vs_b'] = exact_paired(outcomes['b_last_condition'], outcomes['c_cross_condition'])
    return result


def candidate_verdict(reports):
    keys = [f'{dataset}_{size}' for size in (280, 322)
            for dataset in ('msls-val', 'pitts30k-val')]
    if set(reports) != set(keys):
        raise ValueError('Both datasets and BOTH preregistered resolutions required')
    primary = [reports[f'{dataset}_322'] for dataset in ('msls-val', 'pitts30k-val')]
    c_beats_controls = all(r['c_vs_a']['net'] > 0 and r['c_vs_b']['net'] > 0
                           for r in primary)
    c_beats_official = all(r['c_cross_condition']['vs_official']['net'] > 0 for r in primary)
    return dict(
        primary_cross_depth_candidate=c_beats_controls and c_beats_official,
        checks=dict(c_beats_a_and_b_on_both_primary=c_beats_controls,
                    c_beats_official_on_both_primary=c_beats_official),
        verdict=('EXPLORATORY_CROSS_DEPTH_CANDIDATE' if c_beats_controls and c_beats_official
                 else 'NO_REGISTERED_CROSS_DEPTH_GAIN'),
        secondary_280={key: reports[key]['c_cross_condition']['vs_official']['net']
                       for key in keys if key.endswith('_280')},
        stable_gain_proven=False,
        note='Candidate does not mean statistical significance, unseen-test or multi-seed stability; report all regressions and280 results')
