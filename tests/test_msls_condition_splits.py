import numpy as np
import pandas as pd
import pytest

from scripts.generate_msls_condition_splits import (
    _boolean_array,
    _load_city_candidates,
    select_standard_condition_subset,
)


def test_condition_subset_preserves_standard_order_and_ground_truth() -> None:
    standard_paths = ["q/two.jpg", "q/zero.jpg", "q/one.jpg"]
    standard_gt = [np.asarray([2]), np.asarray([0, 1]), np.asarray([3])]

    paths, ground_truth, excluded = select_standard_condition_subset(
        standard_paths,
        standard_gt,
        ["q/one.jpg", "q/panorama.jpg", "q/zero.jpg"],
    )

    assert paths.tolist() == ["q/zero.jpg", "q/one.jpg"]
    assert [values.tolist() for values in ground_truth] == [[0, 1], [3]]
    assert excluded == ["q/panorama.jpg"]


def test_condition_subset_rejects_ambiguous_standard_manifest() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        select_standard_condition_subset(
            ["q/same.jpg", "q/same.jpg"],
            [np.asarray([0]), np.asarray([0])],
            ["q/same.jpg"],
        )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([True, False], [True, False]),
        ([1, 0], [True, False]),
        (["True", "false"], [True, False]),
    ],
)
def test_boolean_metadata_parsing(values: list[object], expected: list[bool]) -> None:
    actual = _boolean_array(pd.Series(values), source="test")
    assert actual.tolist() == expected


def test_boolean_metadata_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _boolean_array(pd.Series(["yes"]), source="test")


def test_city_candidates_track_panorama_rows(tmp_path) -> None:
    query_dir = tmp_path / "cph" / "query"
    query_dir.mkdir(parents=True)
    pd.DataFrame({"key": ["regular", "pano"]}).to_csv(
        query_dir / "postprocessed.csv"
    )
    pd.DataFrame(
        {"key": ["regular", "pano"], "pano": [False, True]}
    ).to_csv(query_dir / "raw.csv")
    pd.DataFrame(
        {
            "n2d": [True, True],
            "w2s": [False, True],
            "s2w": [False, False],
        }
    ).to_csv(query_dir / "subtask_index.csv")

    candidates, panoramas = _load_city_candidates(tmp_path, "cph")

    assert candidates["night"] == {
        "cph/query/images/regular.jpg",
        "cph/query/images/pano.jpg",
    }
    assert panoramas["night"] == {"cph/query/images/pano.jpg"}
    assert candidates["season"] == {"cph/query/images/pano.jpg"}


def test_city_candidates_reject_reordered_subtask_rows(tmp_path) -> None:
    query_dir = tmp_path / "cph" / "query"
    query_dir.mkdir(parents=True)
    pd.DataFrame({"key": ["zero", "one"]}).to_csv(
        query_dir / "postprocessed.csv"
    )
    pd.DataFrame(
        {"key": ["zero", "one"], "pano": [False, False]}
    ).to_csv(query_dir / "raw.csv")
    pd.DataFrame(
        {
            "n2d": [True, False],
            "w2s": [False, True],
            "s2w": [False, False],
        },
        index=[1, 0],
    ).to_csv(query_dir / "subtask_index.csv")

    with pytest.raises(ValueError, match="index order differs"):
        _load_city_candidates(tmp_path, "cph")


def test_city_candidates_reject_reset_index_with_reordered_keys(tmp_path) -> None:
    query_dir = tmp_path / "cph" / "query"
    query_dir.mkdir(parents=True)
    pd.DataFrame({"key": ["zero", "one"]}).to_csv(
        query_dir / "postprocessed.csv"
    )
    pd.DataFrame(
        {"key": ["zero", "one"], "pano": [False, False]}
    ).to_csv(query_dir / "raw.csv")
    pd.DataFrame(
        {
            "key": ["one", "zero"],
            "n2d": [True, False],
            "w2s": [False, True],
            "s2w": [False, False],
        }
    ).to_csv(query_dir / "subtask_index.csv")

    with pytest.raises(ValueError, match="subtask query key order differs"):
        _load_city_candidates(tmp_path, "cph")
