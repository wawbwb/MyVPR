from pathlib import Path

import numpy as np
import pytest

from src.dataloaders.valid.msls_condition import (
    CONDITION_FILES,
    CONDITION_UNION_QUERY_FILE,
    MSLSConditionDataset,
    MSLSConditionUnionDataset,
)


def _save_ground_truth(path: Path, values: list[list[int]]) -> None:
    ground_truth = np.empty(len(values), dtype=object)
    for index, positives in enumerate(values):
        ground_truth[index] = np.asarray(positives, dtype=np.int64)
    np.save(path, ground_truth, allow_pickle=True)


def _write_base_manifests(root: Path) -> tuple[list[str], list[str]]:
    references = [
        "cph/database/images/db_zero.jpg",
        "cph/database/images/db_one.jpg",
    ]
    standard_queries = [
        "cph/query/images/q_zero.jpg",
        "cph/query/images/q_one.jpg",
    ]
    np.save(root / "msls_val_dbImages.npy", np.asarray(references))
    np.save(root / "msls_val_qImages.npy", np.asarray(standard_queries))
    return references, standard_queries


def test_full_db_condition_and_union_loaders_keep_shared_index_contract(
    tmp_path: Path,
) -> None:
    references, standard_queries = _write_base_manifests(tmp_path)
    condition_only = "cph/query/images/q_night_only.jpg"
    union = [*standard_queries, condition_only]
    np.save(tmp_path / CONDITION_UNION_QUERY_FILE, np.asarray(union))
    night_query_file, night_gt_file = CONDITION_FILES["night"]
    np.save(
        tmp_path / night_query_file,
        np.asarray([standard_queries[1], condition_only]),
    )
    _save_ground_truth(tmp_path / night_gt_file, [[1], [0]])

    condition = MSLSConditionDataset("night", dataset_path=tmp_path)
    shared = MSLSConditionUnionDataset(dataset_path=tmp_path)

    assert condition.dataset_name == "msls-val-night-full-db"
    assert condition.dbImages.tolist() == references
    assert condition.qImages.tolist() == [standard_queries[1], condition_only]
    assert [np.asarray(gt).tolist() for gt in condition.ground_truth] == [[1], [0]]
    assert shared.standardQImages.tolist() == standard_queries
    assert shared.qImages.tolist() == union
    assert shared.num_references == 2
    assert shared.num_standard_queries == 2
    assert shared.num_queries == 3
    assert shared.image_paths.tolist() == [*references, *union]


def test_season_compatibility_alias_uses_full_db_manifests(tmp_path: Path) -> None:
    references, standard_queries = _write_base_manifests(tmp_path)
    season_query_file, season_gt_file = CONDITION_FILES["season"]
    np.save(tmp_path / season_query_file, np.asarray([standard_queries[0]]))
    _save_ground_truth(tmp_path / season_gt_file, [[0]])

    season = MSLSConditionDataset("season", dataset_path=tmp_path)

    assert season.dataset_name == "msls-val-season-full-db"
    assert season.dbImages.tolist() == references
    assert season.qImages.tolist() == [standard_queries[0]]


@pytest.mark.parametrize(
    "union",
    [
        ["cph/query/images/q_zero.jpg"],
        [
            "cph/query/images/q_one.jpg",
            "cph/query/images/q_zero.jpg",
        ],
    ],
)
def test_union_loader_rejects_missing_or_reordered_standard_prefix(
    tmp_path: Path, union: list[str]
) -> None:
    _write_base_manifests(tmp_path)
    np.save(tmp_path / CONDITION_UNION_QUERY_FILE, np.asarray(union))

    with pytest.raises(ValueError, match="must begin with the complete standard"):
        MSLSConditionUnionDataset(dataset_path=tmp_path)
