import numpy as np
from scripts.visual_pair_hard_screen import hard_rows, training_order, report


def fixture():
    return {'scores': np.array([[.9, .5], [.9, .89], [.9, .8]], dtype=np.float32),
            'labels': np.array([[True, False], [True, False], [False, True]]),
            'baseline_correct': np.array([True, True, False])}


def test_hard_includes_errors_and_small_margin():
    assert hard_rows(fixture()).tolist() == [False, True, True]


def test_equal_steps_and_deterministic_sampling():
    data = fixture()
    a = training_order(data, np.random.default_rng(42), 'hard_mix')
    b = training_order(data, np.random.default_rng(42), 'hard_mix')
    assert np.array_equal(a, b)
    assert len(a) == len(training_order(data, np.random.default_rng(42), 'uniform'))
    assert hard_rows(data)[a].sum() == len(a)//2


def test_metrics_keep_all_queries():
    data = fixture()
    result = report(data, np.zeros_like(data['scores']), 0.)
    assert result['queries'] == 3
    assert result['correct'] == 2
    assert result['net'] == 0
    assert result['hard_queries'] == 2
    assert result['reachable_errors'] == 1
