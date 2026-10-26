import numpy as np
import pytest
from src.scale_repeat_statistics import positive_components, paired_interval, contrast
from scripts.scale_agreement_repeat import configuration, training_schedule


def test_positive_components_are_transitive_and_deterministic():
    groups = positive_components([[1, 2], [3], [2, 3], [9], [10, 11], [11]])
    assert groups.tolist() == [0, 0, 0, 1, 2, 2]
    with pytest.raises(ValueError):
        positive_components([[1], []])


def test_constant_effect_and_variable_cluster_size():
    value = paired_interval(np.full(6, .25), np.array([0, 0, 0, 1, 2, 2]), 200, 42)
    assert value['delta_r1_pp'] == 25
    assert value['percentile_95_ci_pp'] == [25, 25]
    assert value['resampling_units'] == 3 and value['largest_unit_queries'] == 3
    assert not value['adequate_unit_count']


def test_seeds_do_not_multiply_query_sample_size():
    candidate = [[True, False], [False, True]]
    control = [[False, False], [False, False]]
    r = contrast(candidate, control, np.arange(2), 200, 43)
    assert r['seed_net_queries'] == [1, 1]
    assert r['query_paired']['resampling_units'] == 2
    assert r['query_paired']['percentile_95_ci_pp'] == [50, 50]


def test_paired_intervals_reproduce_and_preserve_sign():
    x = np.array([1, 0, -1, .5, 0, 1])
    a = paired_interval(x, np.arange(6), 200, 44)
    b = paired_interval(x, np.arange(6), 200, 44)
    assert a == b
    negative = paired_interval(-x, np.arange(6), 200, 44)
    np.testing.assert_allclose(negative['percentile_95_ci_pp'], -np.array(a['percentile_95_ci_pp'])[::-1])
    with pytest.raises(ValueError):
        contrast([[1, 0]], [[0, 0]], np.arange(2), 200)


def test_fixed_seeds_and_full_epoch_orders():
    c = configuration()
    assert c['seeds'] == [43, 44, 45] and 42 not in c['seeds']
    assert c['pilot_steps'] == 1024 and c['bootstrap_repetitions'] == 2000
    orders = []
    for seed in c['seeds']:
        order = training_schedule(seed)
        assert len(order) == 1024 and all(len(batch) == 16 for batch in order)
        assert order == training_schedule(seed)
        for epoch in range(2):
            assert sorted(p for row in order[epoch*512:(epoch+1)*512] for p in row) == list(range(8192))
        orders.extend([order[0], order[512]])
    assert len({tuple(row) for row in orders}) == 6
