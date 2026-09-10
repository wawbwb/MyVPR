import pytest
from scripts.segvlad_gt_protocols import parse_annotations, check_names, validate_predictions
from scripts.segvlad_paired import recall_counts


def test_ids_not_row_order():
    assert parse_annotations([[1,[0]],[0,[1,2]]],2) == [[1,2],[0]]


def test_duplicate_query_rejected():
    with pytest.raises(ValueError):
        parse_annotations([[0,[1]],[0,[1]]],2)


def test_one_based_rejected():
    with pytest.raises(ValueError):
        parse_annotations([[1,[1]],[2,[2]]],2)


def test_image_stems_must_match_indices(tmp_path):
    with pytest.raises(ValueError):
        check_names(['1.jpg'],tmp_path)


@pytest.mark.parametrize('pred', [[0], [0,1], [0,1,2,3,4]])
def test_short_unique_predictions_are_valid(pred):
    validate_predictions(pred,406,'segvlad',118)


@pytest.mark.parametrize('pred', [[], [0,0], [-1], [406], [True], [1.0], list(range(6)), None])
def test_invalid_predictions_rejected(pred):
    with pytest.raises(ValueError, match='method=segvlad, query=118'):
        validate_predictions(pred,406,'segvlad',118)


def test_short_lists_recall_uses_only_available_candidates():
    assert recall_counts([[0],[1,2]], [[0],[2]]) == [1,2,2,2,2]
