import pytest
from scripts.segvlad_paired import compare_predictions, recall_counts


def test_cumulative_recall():
    assert recall_counts([[2, 0], [1, 0]], [[0], [1]], 2) == [1, 2]


def test_paired_corrections_and_regressions():
    result = compare_predictions([[1], [1], [2]], [[0], [0], [2]], [[0], [1], [2]])
    assert result['correction_query_ids'] == [0]
    assert result['regression_query_ids'] == [1]
    assert result['net'] == 0


def test_mismatch_rejected():
    with pytest.raises(ValueError):
        compare_predictions([[0]], [], [[0]])


def test_empty_gt_rejected():
    with pytest.raises(ValueError):
        recall_counts([[0]], [[]])


def test_official_edge_ids_do_not_create_hits():
    assert recall_counts([[405]], [list(range(-15, 16))], 1) == [0]
