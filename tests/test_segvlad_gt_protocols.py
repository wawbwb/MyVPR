import pytest
from scripts.segvlad_gt_protocols import parse_annotations, check_names


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
