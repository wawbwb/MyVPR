from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_residual_clip_paired import (
    VARIANTS,
    DescriptorStore,
    balanced_sample_indices,
    cross_city_donor_indices,
    descriptor_drift_rows,
    search_and_score,
    summarise_drift,
)


def _city(path: str) -> str:
    return path.replace("\\", "/").split("/", maxsplit=1)[0].lower()


def test_cross_city_donors_are_reproducible_and_role_preserving() -> None:
    image_paths = [
        "cph/database/r0.jpg",
        "cph\\database\\r1.jpg",
        "sf/database/r2.jpg",
        "sf/database/r3.jpg",
        "cph/database/r4.jpg",
        "sf/database/r5.jpg",
        "cph/query/q0.jpg",
        "cph/query/q1.jpg",
        "sf/query/q2.jpg",
        "sf/query/q3.jpg",
    ]
    num_references = 6

    donors, pair_counts = cross_city_donor_indices(
        image_paths, num_references=num_references, seed=42
    )
    repeated, repeated_counts = cross_city_donor_indices(
        image_paths, num_references=num_references, seed=42
    )

    np.testing.assert_array_equal(donors, repeated)
    assert pair_counts == repeated_counts
    assert donors.dtype == np.int64
    assert np.all(donors != np.arange(len(image_paths)))
    assert np.all(donors[:num_references] < num_references)
    assert np.all(donors[num_references:] >= num_references)
    assert all(
        _city(image_paths[source]) != _city(image_paths[int(donor)])
        for source, donor in enumerate(donors)
    )
    assert sum(pair_counts.values()) == len(image_paths)
    assert all(
        key.startswith(("reference:", "query:")) for key in pair_counts
    )


@pytest.mark.parametrize(
    "image_paths, num_references, message",
    (
        (
            [
                "cph/database/r0.jpg",
                "cph/database/r1.jpg",
                "cph/query/q0.jpg",
                "sf/query/q1.jpg",
            ],
            2,
            "reference partition",
        ),
        (
            [
                "cph/database/r0.jpg",
                "sf/database/r1.jpg",
                "cph/query/q0.jpg",
                "cph/query/q1.jpg",
            ],
            2,
            "query partition",
        ),
    ),
)
def test_cross_city_donors_fail_when_a_role_has_only_one_city(
    image_paths: list[str], num_references: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cross_city_donor_indices(image_paths, num_references, seed=42)


def test_balanced_sample_indices_are_deterministic_sorted_and_balanced() -> None:
    indices = balanced_sample_indices(
        num_references=8,
        num_queries=4,
        sample_count=7,
        seed=42,
    )
    repeated = balanced_sample_indices(8, 4, 7, seed=42)

    np.testing.assert_array_equal(indices, repeated)
    assert indices.dtype == np.int64
    assert len(indices) == 7
    assert len(np.unique(indices)) == len(indices)
    assert np.all(indices[:-1] < indices[1:])
    assert int(np.sum(indices < 8)) == 4
    assert int(np.sum(indices >= 8)) == 3


def test_balanced_sample_indices_backfill_a_small_partition_and_cap_at_total() -> None:
    scarce_queries = balanced_sample_indices(10, 1, 6, seed=7)
    assert len(scarce_queries) == 6
    assert int(np.sum(scarce_queries < 10)) == 5
    assert int(np.sum(scarce_queries >= 10)) == 1

    all_indices = balanced_sample_indices(3, 2, 99, seed=7)
    np.testing.assert_array_equal(all_indices, np.arange(5, dtype=np.int64))


def test_descriptor_store_rejects_duplicate_indices(tmp_path: Path) -> None:
    store = DescriptorStore(
        tmp_path / "descriptors",
        image_count=4,
        variant_names=("left", "right"),
        dtype="float32",
    )
    try:
        store.mark_indices(np.asarray([0, 2]))
        with pytest.raises(RuntimeError, match="duplicate indices"):
            store.mark_indices(np.asarray([2, 3]))
        with pytest.raises(ValueError, match="one-dimensional and unique"):
            store.mark_indices(np.asarray([1, 1]))
    finally:
        store.close()


def test_descriptor_store_rejects_missing_images(tmp_path: Path) -> None:
    store = DescriptorStore(
        tmp_path / "descriptors",
        image_count=4,
        variant_names=("left", "right"),
        dtype="float32",
    )
    indices = np.asarray([0, 2], dtype=np.int64)
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    try:
        store.mark_indices(indices)
        store.write("left", indices, values)
        store.write("right", indices, values)
        with pytest.raises(RuntimeError, match="missed 2 images"):
            store.finish()
    finally:
        store.close()


def test_descriptor_store_rejects_a_missing_variant(tmp_path: Path) -> None:
    store = DescriptorStore(
        tmp_path / "descriptors",
        image_count=2,
        variant_names=("left", "right"),
        dtype="float32",
    )
    indices = np.arange(2, dtype=np.int64)
    values = np.eye(2, dtype=np.float32)
    try:
        store.mark_indices(indices)
        store.write("left", indices, values)
        with pytest.raises(RuntimeError, match="not every descriptor variant"):
            store.finish()
    finally:
        store.close()


def test_descriptor_drift_rows_and_summary_report_expected_statistics() -> None:
    bypass = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    descriptors = {name: bypass.clone() for name in VARIANTS}
    descriptors["zero_clip"] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    rows = descriptor_drift_rows(descriptors, sample_indices=[11, 29])
    summary = summarise_drift(rows)

    assert len(rows) == 2 * len(VARIANTS) * 2
    assert {int(row["image_index"]) for row in rows} == {11, 29}
    bypass_zero = next(
        row
        for row in summary
        if row["reference"] == "bypass" and row["variant"] == "zero_clip"
    )
    assert bypass_zero["num_images"] == 2
    assert bypass_zero["cosine_distance_mean"] == pytest.approx(0.5)
    assert bypass_zero["cosine_distance_p50"] == pytest.approx(0.5)
    assert bypass_zero["cosine_distance_p95"] == pytest.approx(0.95)
    assert bypass_zero["cosine_distance_max"] == pytest.approx(1.0)
    assert bypass_zero["l2_mean"] == pytest.approx(np.sqrt(2.0) / 2.0)
    assert bypass_zero["rms_mean"] == pytest.approx(0.5)
    assert bypass_zero["max_abs_mean"] == pytest.approx(0.5)

    aligned_self = next(
        row
        for row in summary
        if row["reference"] == "aligned" and row["variant"] == "aligned"
    )
    for metric in ("cosine_distance", "l2", "rms", "max_abs"):
        assert aligned_self[f"{metric}_max"] == pytest.approx(0.0)


def test_search_and_score_reports_recall_rank_and_signed_margin(
    tmp_path: Path,
) -> None:
    references = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    queries = np.asarray(
        [
            [1.0, 0.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.concatenate((references, queries)), axis=1), 1.0
    )
    descriptor_path = tmp_path / "descriptors.npy"
    np.save(descriptor_path, np.concatenate((references, queries), axis=0))

    result = search_and_score(
        descriptor_path,
        num_references=len(references),
        ground_truth=(np.asarray([0]), np.asarray([0])),
        k_values=(1, 2),
        rank_k=3,
    )

    assert result["recalls"] == {1: 0.5, 2: 1.0}
    np.testing.assert_array_equal(result["hits"][1], [True, False])
    np.testing.assert_array_equal(result["hits"][2], [True, True])
    np.testing.assert_array_equal(result["top1"], [0, 1])
    np.testing.assert_array_equal(result["first_positive_rank"], [1, 2])
    np.testing.assert_array_equal(
        result["positive_found_within_rank_k"], [True, True]
    )
    np.testing.assert_allclose(
        result["best_positive_distance"], [0.0, 0.8], atol=1e-6
    )
    np.testing.assert_allclose(
        result["nearest_negative_distance"], [2.0, 0.4], atol=1e-6
    )
    np.testing.assert_allclose(
        result["positive_negative_margin"], [2.0, -0.4], atol=1e-6
    )
