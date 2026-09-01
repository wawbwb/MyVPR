from pathlib import Path

import numpy as np

from scripts.cache_msls_ag_slrd_layouts import (
    build_role_preserving_shuffle,
    load_msls_index,
)


def _object_ground_truth(rows: list[list[int]]) -> np.ndarray:
    result = np.empty(len(rows), dtype=object)
    for index, values in enumerate(rows):
        result[index] = np.asarray(values, dtype=np.int64)
    return result


def test_role_preserving_shuffle_is_deterministic_derangement() -> None:
    first = build_role_preserving_shuffle(11, 7, seed=42)
    second = build_role_preserving_shuffle(11, 7, seed=42)

    assert first.dtype == np.dtype("int32")
    assert np.array_equal(first, second)
    assert np.unique(first).size == 18
    assert np.all(first != np.arange(18))
    assert np.all(first[:11] < 11)
    assert np.all(first[11:] >= 11)


def test_msls_index_preserves_database_then_query_order(tmp_path: Path) -> None:
    database = np.asarray(["cph/db/a.jpg", "sf/db/b.jpg"])
    queries = np.asarray(["cph/query/q.jpg", "sf/query/r.jpg"])
    np.save(tmp_path / "msls_val_dbImages.npy", database)
    np.save(tmp_path / "msls_val_qImages.npy", queries)
    np.save(
        tmp_path / "msls_val_gt_25m.npy",
        _object_ground_truth([[0], [1]]),
        allow_pickle=True,
    )

    paths, num_references, num_queries, ground_truth, records = load_msls_index(
        tmp_path
    )

    assert paths.tolist() == [*database.tolist(), *queries.tolist()]
    assert num_references == 2
    assert num_queries == 2
    assert [values.tolist() for values in ground_truth] == [[0], [1]]
    assert len(records["path_sequence_sha256"]) == 64
    assert set(records) == {
        "database",
        "queries",
        "ground_truth",
        "path_sequence_sha256",
    }


def test_msls_shuffle_avoids_known_positive_place_overlap() -> None:
    ground_truth = (
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([2]),
        np.asarray([3]),
    )
    donors = build_role_preserving_shuffle(
        4, 4, seed=42, ground_truth=ground_truth
    )
    reference_donors = donors[:4]
    query_donors = donors[4:] - 4
    assert reference_donors[0] not in {0, 1}
    assert reference_donors[1] not in {0, 1}
    for receiver, donor in enumerate(query_donors.tolist()):
        assert set(ground_truth[receiver]).isdisjoint(ground_truth[donor])
