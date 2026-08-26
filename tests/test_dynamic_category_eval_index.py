from pathlib import Path

import numpy as np

from scripts.eval_dynamic_category_prior import (
    build_evaluation_sets,
    build_query_condition_strata,
)
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.msls_condition import MSLSConditionUnionDataset
from src.dataloaders.valid.msls_condition_protocol import (
    CONDITION_FILES,
    CONDITION_ORDER,
    CONDITION_UNION_QUERY_FILE,
)


def _save_ground_truth(path: Path, values: list[list[int]]) -> None:
    ground_truth = np.empty(len(values), dtype=object)
    for index, positives in enumerate(values):
        ground_truth[index] = np.asarray(positives, dtype=np.int64)
    np.save(path, ground_truth, allow_pickle=True)


def test_complete_condition_queries_map_into_one_standard_first_union(
    tmp_path: Path,
) -> None:
    # The repository's standard loader requires both validation city folders,
    # even though this manifest-only unit test does not open images.
    (tmp_path / "cph").mkdir()
    (tmp_path / "sf").mkdir()
    references = [
        "cph/database/images/db0.jpg",
        "cph/database/images/db1.jpg",
    ]
    standard_queries = [
        "cph/query/images/q0.jpg",
        "cph/query/images/q1.jpg",
    ]
    night_only = "cph/query/images/q_night.jpg"
    winter_only = "cph/query/images/q_winter.jpg"
    summer_only = "cph/query/images/q_summer.jpg"
    condition_queries = {
        "night": [standard_queries[1], night_only],
        "winter2summer": [winter_only],
        "summer2winter": [night_only, summer_only],
    }
    query_union = [
        *standard_queries,
        night_only,
        winter_only,
        summer_only,
    ]

    np.save(tmp_path / "msls_val_dbImages.npy", np.asarray(references))
    np.save(tmp_path / "msls_val_qImages.npy", np.asarray(standard_queries))
    _save_ground_truth(tmp_path / "msls_val_gt_25m.npy", [[0], [1]])
    np.save(tmp_path / CONDITION_UNION_QUERY_FILE, np.asarray(query_union))
    for condition, paths in condition_queries.items():
        query_file, gt_file = CONDITION_FILES[condition]
        np.save(tmp_path / query_file, np.asarray(paths))
        positives = [[1], [0]] if condition == "night" else [[0]] * len(paths)
        _save_ground_truth(tmp_path / gt_file, positives)

    standard = MapillarySLSDataset(dataset_path=tmp_path, input_transform=None)
    union = MSLSConditionUnionDataset(dataset_path=tmp_path)
    evaluations, memberships = build_evaluation_sets(
        standard,
        union,
        CONDITION_ORDER,
        transform=None,
        msls_path=tmp_path,
    )

    assert [evaluation["name"] for evaluation in evaluations] == [
        "msls-val",
        "msls-val-night-full-db",
        "msls-val-winter2summer-full-db",
        "msls-val-summer2winter-full-db",
    ]
    assert evaluations[0]["query_offsets"].tolist() == [0, 1]
    assert evaluations[1]["query_offsets"].tolist() == [1, 2]
    assert evaluations[1]["num_standard_query_overlap"] == 1
    assert evaluations[1]["num_condition_only_queries"] == 1
    assert evaluations[2]["query_offsets"].tolist() == [3]
    assert evaluations[3]["query_offsets"].tolist() == [2, 4]

    strata, counts = build_query_condition_strata(
        memberships, num_queries=len(query_union)
    )
    assert strata.tolist() == [0, 1, 5, 2, 4]
    assert counts == {"0": 1, "1": 1, "2": 1, "4": 1, "5": 1}
