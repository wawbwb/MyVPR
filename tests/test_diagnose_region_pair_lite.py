import numpy as np
from scripts.diagnose_region_pair_lite import analyze


def run(region, support, gt, ru=(1., .99, .98)):
    return analyze(np.array([[0, 1, 2]]), np.array([ru]), np.array([region]),
                   np.array([support]), [np.array(gt)])


def test_strong_positive_corrects():
    summary, rows = run([.1, .8, .2], [3, 4, 3], [1])
    assert summary['corrections'] == [0]
    assert rows[0]['positive_beats_all_negatives_region']


def test_global_argmax_blocks_eligible_positive():
    summary, rows = run([.1, .8, .9], [3, 4, 4], [1], (1., .99, .8))
    assert summary['corrections'] == []
    assert summary['eligible_positive_blocked_by_global_argmax_query_ids'] == [0]
    assert rows[0]['selected_position'] == 0
    assert rows[0]['best_positive_blockers'] == ['not_global_region_argmax_including_ties']


def test_regression_details():
    summary, rows = run([.1, .8, .2], [3, 4, 3], [0])
    assert summary['regressions'] == [0]
    assert rows[0]['candidates'][1]['passes_fixed_gate']


def test_unreachable_and_zero_ties():
    summary, rows = run([0., 0., 0.], [0, 0, 0], [8])
    assert summary['unreachable_ru_errors'] == 1
    assert summary['reachable_ru_errors'] == 0
    assert rows[0]['best_positive_position'] is None
    summary, rows = run([0., 0., 0.], [0, 0, 0], [1])
    assert not rows[0]['positive_beats_all_negatives_region']
    assert summary['region_argmax_only_diagnostic']['corrections'] == 0


def test_all_positive_no_negative():
    summary, rows = run([.1, .8, .2], [3, 4, 3], [0, 1, 2])
    assert summary['correct'] == 1
    assert rows[0]['best_negative_position'] is None
