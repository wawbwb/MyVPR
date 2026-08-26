import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_msls_condition_splits import (
    _boolean_array,
    _load_city_query_metadata,
    build_standard_database_coordinates,
    compute_full_database_ground_truth,
    main,
)
from src.dataloaders.valid.msls_condition import (
    CONDITION_FILES,
    CONDITION_UNION_QUERY_FILE,
)


def _write_query_metadata(
    root: Path,
    city: str,
    *,
    keys: list[str],
    coordinates: list[tuple[float, float]],
    panorama: list[bool],
    n2d: list[bool],
    w2s: list[bool],
    s2w: list[bool],
) -> None:
    query_dir = root / city / "query"
    query_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "key": keys,
            "easting": [coordinate[0] for coordinate in coordinates],
            "northing": [coordinate[1] for coordinate in coordinates],
        }
    ).to_csv(query_dir / "postprocessed.csv")
    pd.DataFrame({"key": keys, "pano": panorama}).to_csv(
        query_dir / "raw.csv"
    )
    pd.DataFrame(
        {"key": keys, "n2d": n2d, "w2s": w2s, "s2w": s2w}
    ).to_csv(query_dir / "subtask_index.csv")


def _write_database_metadata(
    root: Path,
    city: str,
    rows: list[tuple[str, float, float]],
) -> None:
    database_dir = root / city / "database"
    database_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "key": [row[0] for row in rows],
            "easting": [row[1] for row in rows],
            "northing": [row[2] for row in rows],
        }
    ).to_csv(database_dir / "postprocessed.csv")


def _load_object_ground_truth(path: Path) -> list[list[int]]:
    values = np.load(path, allow_pickle=True)
    return [np.asarray(positives, dtype=np.int64).tolist() for positives in values]


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


def test_city_query_metadata_keeps_atomic_conditions_and_panorama_flag(
    tmp_path: Path,
) -> None:
    _write_query_metadata(
        tmp_path,
        "cph",
        keys=["night", "winter", "summer", "pano"],
        coordinates=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
        panorama=[False, False, False, True],
        n2d=[True, False, False, True],
        w2s=[False, True, False, False],
        s2w=[False, False, True, False],
    )

    metadata = _load_city_query_metadata(tmp_path, "cph")

    assert metadata.paths.tolist() == [
        "cph/query/images/night.jpg",
        "cph/query/images/winter.jpg",
        "cph/query/images/summer.jpg",
        "cph/query/images/pano.jpg",
    ]
    assert metadata.coordinates.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ]
    assert metadata.panorama.tolist() == [False, False, False, True]
    assert metadata.selected["night"].tolist() == [True, False, False, True]
    assert metadata.selected["winter2summer"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert metadata.selected["summer2winter"].tolist() == [
        False,
        False,
        True,
        False,
    ]


def test_city_query_metadata_rejects_reordered_subtask_rows(tmp_path: Path) -> None:
    _write_query_metadata(
        tmp_path,
        "cph",
        keys=["zero", "one"],
        coordinates=[(0.0, 0.0), (1.0, 1.0)],
        panorama=[False, False],
        n2d=[True, False],
        w2s=[False, True],
        s2w=[False, False],
    )
    subtask_path = tmp_path / "cph" / "query" / "subtask_index.csv"
    subtasks = pd.read_csv(subtask_path, index_col=0)
    subtasks.iloc[::-1].to_csv(subtask_path)

    with pytest.raises(ValueError, match="index order differs"):
        _load_city_query_metadata(tmp_path, "cph")


def test_city_query_metadata_rejects_reset_index_with_reordered_keys(
    tmp_path: Path,
) -> None:
    _write_query_metadata(
        tmp_path,
        "cph",
        keys=["zero", "one"],
        coordinates=[(0.0, 0.0), (1.0, 1.0)],
        panorama=[False, False],
        n2d=[True, False],
        w2s=[False, True],
        s2w=[False, False],
    )
    subtask_path = tmp_path / "cph" / "query" / "subtask_index.csv"
    subtasks = pd.read_csv(subtask_path, index_col=0)
    subtasks = subtasks.iloc[::-1].reset_index(drop=True)
    subtasks.to_csv(subtask_path)

    with pytest.raises(ValueError, match="key order differs"):
        _load_city_query_metadata(tmp_path, "cph")


def test_ground_truth_is_city_scoped_and_uses_global_manifest_indices(
    tmp_path: Path,
) -> None:
    _write_database_metadata(
        tmp_path,
        "cph",
        [
            ("cph_near", 0.0, 0.0),
            ("cph_edge", 25.0, 0.0),
            ("cph_far", 25.1, 0.0),
        ],
    )
    _write_database_metadata(
        tmp_path,
        "sf",
        [("sf_near", 0.0, 0.0), ("sf_far", 100.0, 0.0)],
    )
    # Deliberately interleave cities: positives must remain indices into this
    # exact global standard-DB order, not metadata-row or per-city offsets.
    database_paths = [
        "sf/database/images/sf_far.jpg",
        "cph/database/images/cph_near.jpg",
        "sf/database/images/sf_near.jpg",
        "cph/database/images/cph_edge.jpg",
        "cph/database/images/cph_far.jpg",
    ]
    database_by_city = build_standard_database_coordinates(
        tmp_path, database_paths, ("cph", "sf")
    )

    cph_positives = compute_full_database_ground_truth(
        "cph/query/images/query.jpg",
        np.asarray([0.0, 0.0]),
        database_by_city,
        distance_threshold=25.0,
    )
    sf_positives = compute_full_database_ground_truth(
        "sf/query/images/query.jpg",
        np.asarray([0.0, 0.0]),
        database_by_city,
        distance_threshold=25.0,
    )

    assert cph_positives.tolist() == [1, 3]
    assert sf_positives.tolist() == [2]


def test_generator_writes_atomic_conditions_season_alias_and_standard_first_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_paths = np.asarray(
        [
            "cph/database/images/db_zero.jpg",
            "cph/database/images/db_hundred.jpg",
            "cph/database/images/db_two_hundred.jpg",
        ]
    )
    standard_query = "cph/query/images/q_standard.jpg"
    np.save(tmp_path / "msls_val_dbImages.npy", database_paths)
    np.save(tmp_path / "msls_val_qImages.npy", np.asarray([standard_query]))
    standard_gt = np.empty(1, dtype=object)
    standard_gt[0] = np.asarray([0], dtype=np.int64)
    np.save(tmp_path / "msls_val_gt_25m.npy", standard_gt, allow_pickle=True)
    _write_database_metadata(
        tmp_path,
        "cph",
        [
            ("db_zero", 0.0, 0.0),
            ("db_hundred", 100.0, 0.0),
            ("db_two_hundred", 200.0, 0.0),
        ],
    )

    keys = [
        "q_standard",
        "q_night_extra",
        "q_w2s_shared",
        "q_s2w_extra",
        "q_no_positive",
        "q_panorama",
    ]
    _write_query_metadata(
        tmp_path,
        "cph",
        keys=keys,
        coordinates=[
            (0.0, 0.0),
            (100.0, 0.0),
            (200.0, 0.0),
            (0.0, 0.0),
            (1000.0, 0.0),
            (0.0, 0.0),
        ],
        panorama=[False, False, False, False, False, True],
        n2d=[True, True, False, False, False, True],
        w2s=[False, False, True, False, False, False],
        s2w=[False, False, True, True, True, False],
    )
    image_dir = tmp_path / "cph" / "query" / "images"
    image_dir.mkdir()
    for key in keys:
        (image_dir / f"{key}.jpg").touch()

    report_path = tmp_path / "condition_audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_msls_condition_splits.py",
            "--msls-path",
            str(tmp_path),
            "--cities",
            "cph",
            "--conditions",
            "summer2winter",
            "night",
            "winter2summer",
            "--expected-standard-queries",
            "1",
            "--report",
            str(report_path),
        ],
    )

    main()

    night_query_file, night_gt_file = CONDITION_FILES["night"]
    w2s_query_file, w2s_gt_file = CONDITION_FILES["winter2summer"]
    s2w_query_file, s2w_gt_file = CONDITION_FILES["summer2winter"]
    season_query_file, season_gt_file = CONDITION_FILES["season"]
    assert np.load(tmp_path / night_query_file, allow_pickle=False).tolist() == [
        standard_query,
        "cph/query/images/q_night_extra.jpg",
    ]
    assert _load_object_ground_truth(tmp_path / night_gt_file) == [[0], [1]]
    assert np.load(tmp_path / w2s_query_file, allow_pickle=False).tolist() == [
        "cph/query/images/q_w2s_shared.jpg"
    ]
    assert _load_object_ground_truth(tmp_path / w2s_gt_file) == [[2]]
    assert np.load(tmp_path / s2w_query_file, allow_pickle=False).tolist() == [
        "cph/query/images/q_w2s_shared.jpg",
        "cph/query/images/q_s2w_extra.jpg",
    ]
    assert _load_object_ground_truth(tmp_path / s2w_gt_file) == [[2], [0]]
    assert np.load(tmp_path / season_query_file, allow_pickle=False).tolist() == [
        "cph/query/images/q_w2s_shared.jpg",
        "cph/query/images/q_s2w_extra.jpg",
    ]
    assert _load_object_ground_truth(tmp_path / season_gt_file) == [[2], [0]]
    assert np.load(
        tmp_path / CONDITION_UNION_QUERY_FILE, allow_pickle=False
    ).tolist() == [
        standard_query,
        "cph/query/images/q_night_extra.jpg",
        "cph/query/images/q_w2s_shared.jpg",
        "cph/query/images/q_s2w_extra.jpg",
    ]

    audit = json.loads(report_path.read_text(encoding="utf-8"))
    assert audit["conditions"]["night"]["excluded_panorama_queries"] == 1
    assert audit["conditions"]["night"]["standard_query_overlap"] == 1
    assert (
        audit["conditions"]["summer2winter"]["excluded_no_positive_queries"]
        == 1
    )
    assert audit["conditions"]["summer2winter"]["excluded_no_positive_paths"] == [
        "cph/query/images/q_no_positive.jpg"
    ]
    assert audit["aggregates"]["season"]["duplicate_query_memberships"] == 1
    assert audit["query_union"]["num_standard_queries"] == 1
    assert audit["query_union"]["num_unique_condition_only_queries"] == 3
    assert audit["query_union"]["standard_queries_are_exact_prefix"] is True
    assert audit["query_union"]["condition_membership_counts"] == {
        "1": 2,
        "4": 1,
        "6": 1,
    }
    assert audit["query_union"]["singleton_condition_membership_paths"] == [
        "cph/query/images/q_w2s_shared.jpg",
        "cph/query/images/q_s2w_extra.jpg",
    ]
