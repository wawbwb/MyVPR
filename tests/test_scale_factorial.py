import numpy as np
import pytest
from src.scale_factorial_statistics import CELLS, factorial, interaction
from scripts import scale_factorial as study
from scripts import scale_agreement_repeat as prior


def report(bits):
    return dict(outcomes=list(map(bool, bits)))


def test_six_cells_separate_scale_gate_and_package_effects():
    official = [dict(base_322=report([0, 0, 0, 0]), descriptor_average=report([1, 0, 0, 0]), group_gate=report([1, 1, 0, 0])) for _ in range(3)]
    innovation = [dict(base_322=report([1, 0, 0, 0]), descriptor_average=report([1, 1, 0, 0]), group_gate=report([1, 1, 1, 1])) for _ in range(3)]
    r = factorial(official, innovation, np.arange(4), 200, 43)
    assert r['cell_correct'] == dict(A=[0]*3, B=[1]*3, C=[2]*3, D=[1]*3, E=[2]*3, F=[4]*3)
    assert r['contrasts']['C_minus_A']['seed_net_queries'] == [2]*3
    assert r['contrasts']['C_minus_B']['seed_net_queries'] == [1]*3
    assert r['contrasts']['F_minus_D']['seed_net_queries'] == [3]*3
    assert r['contrasts']['F_minus_E']['seed_net_queries'] == [2]*3
    assert r['package_interaction']['seed_net_queries'] == [1]*3
    assert r['gate_interaction']['seed_net_queries'] == [1]*3
    assert r['package_interaction']['query_paired']['resampling_units'] == 4


def test_no_double_counting_when_all_gain_is_scale_information():
    family = [dict(base_322=report([0, 0, 1]), descriptor_average=report([1, 0, 1]), group_gate=report([1, 0, 1])) for _ in range(3)]
    r = factorial(family, family, np.arange(3), 200, 44)
    assert r['contrasts']['C_minus_A']['mean_net_queries'] == 1
    assert r['contrasts']['C_minus_B']['query_paired']['percentile_95_ci_pp'] == [0, 0]
    assert r['contrasts']['F_minus_E']['mean_net_queries'] == 0
    assert r['package_interaction']['query_paired']['percentile_95_ci_pp'] == [0, 0]


def test_paired_interaction_averages_seeds_before_sampling_queries():
    a = interaction([[1, -1], [-1, 1], [0, 0]], np.arange(2), 200, 45)
    assert a['seed_net_queries'] == [0, 0, 0]
    assert a['query_paired']['percentile_95_ci_pp'] == [0, 0]
    negative = interaction([[-1, -1], [-1, -1]], np.arange(2), 200, 45)
    assert negative['query_paired']['percentile_95_ci_pp'] == [-100, -100]
    with pytest.raises(ValueError):
        interaction([[1, -1]], np.arange(2), 200, 45)


def test_mismatched_family_and_query_populations_rejected():
    a = [dict(base_322=report([0]), descriptor_average=report([1]), group_gate=report([1])) for _ in range(3)]
    with pytest.raises(ValueError):
        factorial(a, a[:2], np.arange(1), 200)
    with pytest.raises(ValueError):
        factorial(a, a, np.arange(2), 200)


def test_private_training_engine_preserves_original_namespace():
    original = (prior.root, prior.stage_contract, prior.verified, prior.finish, prior.configuration())
    engine = study.private_trainer()
    assert engine is not prior
    assert engine.root is study.root and engine.stage_contract is study.identity
    assert (prior.root, prior.stage_contract, prior.verified, prior.finish, prior.configuration()) == original
    assert engine.training_schedule(43) == prior.training_schedule(43)
    assert engine.train_seed.__globals__ is engine.__dict__


def test_fixed_protocol_and_historical_variant_mapping():
    c = study.configuration()
    assert c['seeds'] == [43, 44, 45] and c['pilot_steps'] == 1024 and c['learning_rate'] == .001
    assert c['microbatch'] == 2 and c['shard_size'] == 64
    assert len(CELLS) == 6 and len(study.VARIANTS) == 9
    r = study.remap_historical(dict(innovation1={'correct': 1}, innovation1_518={'correct': 2}, group_gate={'correct': 3}))
    assert set(r) == {'base_322', 'base_518', 'group_gate'}
    assert r['base_322']['correct'] == 1 and r['base_518']['correct'] == 2
