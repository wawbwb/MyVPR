from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.dynamic_category_prior import (
    dynamic_patch_coverage,
    load_and_validate_mask_cache,
    map_condition_query_indices,
    resolve_dynamic_class_ids,
    role_preserving_derangement,
    save_mask_cache,
    spatially_permute_masks,
    validate_ground_truth,
    validate_overlapping_ground_truth,
)
from src.models.aggregators.boq import BoQ, BoQBlock


def _tiny_boq() -> BoQ:
    torch.manual_seed(7)
    return BoQ(
        in_channels=8,
        proj_channels=64,
        num_queries=2,
        num_layers=2,
        row_dim=4,
    ).eval()


def test_boq_none_path_is_exact_and_old_state_dict_loads_strictly() -> None:
    model = _tiny_boq()
    feature_map = torch.randn(2, 8, 2, 3)

    implicit, implicit_attn = model(feature_map)
    explicit, explicit_attn = model(feature_map, attention_bias=None)

    assert torch.equal(implicit, explicit)
    assert all(
        torch.equal(left, right)
        for left, right in zip(implicit_attn, explicit_attn)
    )
    restored = _tiny_boq()
    restored.load_state_dict(model.state_dict(), strict=True)


def test_zero_bias_is_close_to_baseline_for_all_supported_shapes() -> None:
    model = _tiny_boq()
    feature_map = torch.randn(2, 8, 2, 3)
    baseline, _ = model(feature_map)

    for zero_bias in (
        torch.zeros(2, 6),
        torch.zeros(2, 2, 3),
        torch.zeros(2, 1, 2, 3),
    ):
        actual, _ = model(feature_map, attention_bias=zero_bias)
        torch.testing.assert_close(actual, baseline, rtol=1e-5, atol=1e-6)


def test_negative_bias_reduces_attention_to_masked_keys() -> None:
    torch.manual_seed(11)
    block = BoQBlock(in_dim=64, num_queries=3, nheads=1).eval()
    tokens = torch.randn(2, 6, 64)
    _, _, baseline_attention = block(tokens)
    bias = torch.zeros(2, 6)
    bias[:, -2:] = -20.0
    _, _, suppressed_attention = block(tokens, attention_bias=bias)

    baseline_mass = baseline_attention[..., -2:].sum(dim=-1)
    suppressed_mass = suppressed_attention[..., -2:].sum(dim=-1)
    assert torch.all(suppressed_mass < baseline_mass)
    assert float(suppressed_mass.max()) < 1e-6


@pytest.mark.parametrize(
    "bad_bias",
    [
        torch.zeros(2, 5),
        torch.full((2, 6), float("nan")),
        torch.full((2, 6), 0.1),
    ],
)
def test_boq_rejects_bad_attention_bias(bad_bias: torch.Tensor) -> None:
    model = _tiny_boq()
    with pytest.raises(ValueError):
        model(torch.randn(2, 8, 2, 3), attention_bias=bad_bias)


def test_dynamic_patch_coverage_is_exact_area_pooling() -> None:
    logits = torch.zeros(1, 3, 4, 4)
    logits[:, 0] = 1.0
    logits[:, 1, :2] = 4.0

    coverage, labels = dynamic_patch_coverage(logits, [1], (2, 2))

    expected = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    torch.testing.assert_close(coverage, expected)
    assert torch.all(labels[:, :2] == 1)
    assert torch.all(labels[:, 2:] == 0)


def test_dynamic_class_names_are_metadata_resolved_with_aliases() -> None:
    categories = ("__background__", "bicycle", "motorbike", "person")
    ids, names = resolve_dynamic_class_ids(
        categories, ["bike", "motorcycle", "pedestrian", "person"]
    )
    assert ids == [1, 2, 3]
    assert names == ["bicycle", "motorbike", "person"]
    with pytest.raises(ValueError, match="truck"):
        resolve_dynamic_class_ids(categories, ["truck"])


def test_role_derangement_has_no_fixed_points_or_partition_crossing() -> None:
    query_strata = np.asarray([0, 0, 0, 1, 1, 1, 1])
    donors = role_preserving_derangement(
        19, 7, seed=42, query_strata=query_strata
    )

    assert np.array_equal(np.sort(donors), np.arange(26))
    assert np.all(donors != np.arange(26))
    assert np.all(donors[:19] < 19)
    assert np.all(donors[19:] >= 19)
    assert np.array_equal(
        query_strata[donors[19:] - 19],
        query_strata,
    )
    assert np.array_equal(
        donors,
        role_preserving_derangement(
            19, 7, seed=42, query_strata=query_strata
        ),
    )
    with pytest.raises(ValueError, match="only 1 image"):
        role_preserving_derangement(19, 7, seed=42, query_strata=[0, 0, 0, 0, 0, 0, 1])


def test_spatial_random_control_preserves_values_and_is_batch_independent() -> None:
    masks = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    indices = torch.tensor([5, 11, 29])
    together = spatially_permute_masks(masks, indices, seed=42)
    separately = torch.cat(
        [
            spatially_permute_masks(
                masks[row : row + 1], indices[row : row + 1], seed=42
            )
            for row in range(3)
        ]
    )

    torch.testing.assert_close(together, separately)
    torch.testing.assert_close(
        together.flatten(1).sort(dim=1).values,
        masks.flatten(1).sort(dim=1).values,
    )
    torch.testing.assert_close(together.mean(dim=(1, 2)), masks.mean(dim=(1, 2)))


def _write_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    masks = np.linspace(0.0, 1.0, 5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    paths = [f"city/image_{index}.jpg" for index in range(5)]
    save_mask_cache(
        path,
        masks=masks,
        image_paths=paths,
        num_references=3,
        grid_size=(2, 3),
        segmentation_size=(52, 78),
        model_name="teacher",
        weights_name="weights",
        weights_url="https://example.invalid/weights.pth",
        dynamic_class_names=("car",),
        dynamic_class_ids=(7,),
    )
    return masks, paths


def test_mask_cache_requires_exact_paths_order_boundary_and_grid(tmp_path: Path) -> None:
    cache_path = tmp_path / "masks.npz"
    expected_masks, paths = _write_cache(cache_path)

    masks, metadata = load_and_validate_mask_cache(
        cache_path,
        expected_image_paths=paths,
        expected_num_references=3,
        expected_grid_size=(2, 3),
    )
    torch.testing.assert_close(
        torch.from_numpy(masks),
        torch.from_numpy(expected_masks),
        rtol=1e-3,
        atol=5e-4,
    )
    assert metadata["dynamic_class_names"] == ["car"]

    with pytest.raises(ValueError, match="paths/order"):
        load_and_validate_mask_cache(
            cache_path,
            expected_image_paths=list(reversed(paths)),
            expected_num_references=3,
            expected_grid_size=(2, 3),
        )
    with pytest.raises(ValueError, match="boundary"):
        load_and_validate_mask_cache(
            cache_path,
            expected_image_paths=paths,
            expected_num_references=2,
            expected_grid_size=(2, 3),
        )
    with pytest.raises(ValueError, match="grid"):
        load_and_validate_mask_cache(
            cache_path,
            expected_image_paths=paths,
            expected_num_references=3,
            expected_grid_size=(3, 2),
        )


def test_condition_query_mapping_is_unique_and_order_preserving() -> None:
    full = ["q/zero.jpg", "q/one.jpg", "q/two.jpg"]
    actual = map_condition_query_indices(full, ["q/two.jpg", "q/zero.jpg"])
    assert actual.tolist() == [2, 0]
    with pytest.raises(ValueError, match="not a subset"):
        map_condition_query_indices(full, ["q/missing.jpg"])
    with pytest.raises(ValueError, match="duplicate"):
        map_condition_query_indices(["q/same.jpg", "q/same.jpg"], ["q/same.jpg"])
    with pytest.raises(ValueError, match="condition queries contain duplicate"):
        map_condition_query_indices(full, ["q/one.jpg", "q/one.jpg"])


def test_ground_truth_validation_rejects_empty_duplicate_and_bad_indices() -> None:
    valid = validate_ground_truth(
        [np.asarray([2, 0]), np.asarray([1])],
        num_queries=2,
        num_references=3,
        dataset_name="test",
    )
    assert [values.tolist() for values in valid] == [[2, 0], [1]]

    bad_cases = (
        ([np.asarray([], dtype=np.int64)], "empty"),
        ([np.asarray([0, 0])], "duplicates"),
        ([np.asarray([-1])], "outside"),
        ([np.asarray([3])], "outside"),
        ([np.asarray([1.0])], "not integer"),
        ([np.asarray([[1]])], "1-D"),
    )
    for ground_truth, message in bad_cases:
        with pytest.raises(ValueError, match=message):
            validate_ground_truth(
                ground_truth,
                num_queries=1,
                num_references=3,
                dataset_name="test",
            )


def test_overlapping_ground_truth_must_match_standard_exactly() -> None:
    standard_paths = ["q/zero.jpg", "q/one.jpg"]
    standard_gt = [np.asarray([2, 1]), np.asarray([4])]
    condition_paths = ["q/extra.jpg", "q/zero.jpg"]
    condition_gt = [np.asarray([3]), np.asarray([1, 2])]

    assert validate_overlapping_ground_truth(
        standard_paths,
        standard_gt,
        condition_paths,
        condition_gt,
        condition_name="night",
    ) == (1, 1)

    with pytest.raises(ValueError, match="disagrees with standard MSLS"):
        validate_overlapping_ground_truth(
            standard_paths,
            standard_gt,
            condition_paths,
            [np.asarray([3]), np.asarray([2])],
            condition_name="night",
        )
