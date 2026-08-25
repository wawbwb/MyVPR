import torch

from scripts.visualize_semantic_region_delta import (
    build_propagation_diagnostics,
    roll_cache_by_place,
    select_sample_indices,
)
from src.models.semantic_region_gate import SemanticRegionReliabilityTarget


def _non_identity_cache(
    place_count: int = 2,
    views_per_place: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return place-specific graphs for which a place roll is observable."""

    if place_count != 2:
        raise ValueError("this fixture defines two distinct place graphs")
    permutations = torch.tensor(
        [
            [3, 2, 1, 0],
            [1, 2, 3, 0],
        ],
        dtype=torch.uint8,
    )
    indices = permutations[:, None].expand(-1, views_per_place, -1)
    indices = indices.reshape(place_count * views_per_place, 4, 1).clone()
    weights = torch.ones_like(indices, dtype=torch.float32)
    confidence_by_place = torch.tensor(
        [
            [1.0, 0.8, 0.6, 0.4],
            [0.2, 0.4, 0.8, 1.0],
        ]
    )
    confidence = confidence_by_place[:, None].expand(
        -1, views_per_place, -1
    )
    confidence = confidence.reshape(place_count * views_per_place, 4).clone()
    return indices, weights, confidence


def test_roll_cache_by_place_moves_all_fields_without_mutating_inputs() -> None:
    place_count, views_per_place = 3, 2
    batch_size, patch_count, topk = place_count * views_per_place, 4, 2
    indices = torch.arange(
        batch_size * patch_count * topk, dtype=torch.uint8
    ).reshape(batch_size, patch_count, topk)
    weights = torch.arange(
        batch_size * patch_count * topk, dtype=torch.float32
    ).reshape(batch_size, patch_count, topk)
    weights = weights + 0.25
    confidence = torch.arange(
        batch_size * patch_count, dtype=torch.float32
    ).reshape(batch_size, patch_count)
    confidence = confidence / 10.0
    originals = tuple(
        tensor.clone() for tensor in (indices, weights, confidence)
    )

    rolled = roll_cache_by_place(
        indices,
        weights,
        confidence,
        place_count,
        views_per_place,
    )

    for place in range(place_count):
        donor_place = (place - 1) % place_count
        for view in range(views_per_place):
            destination = place * views_per_place + view
            donor = donor_place * views_per_place + view
            for actual, source in zip(rolled, originals):
                torch.testing.assert_close(actual[destination], source[donor])

    for actual, original in zip((indices, weights, confidence), originals):
        torch.testing.assert_close(actual, original)


def test_diagnostics_match_production_builders_and_are_non_degenerate() -> None:
    place_count, views_per_place = 2, 2
    torch.manual_seed(37)
    raw_featmap = torch.randn(
        place_count * views_per_place, 5, 2, 2, dtype=torch.float32
    )
    indices, weights, confidence = _non_identity_cache(
        place_count, views_per_place
    )
    semantic_config = {
        "match_grid": 2,
        "target_scale": 2.0,
        "place_chunk_size": 1,
        "min_spatial_std": 1e-3,
    }

    diagnostics = build_propagation_diagnostics(
        raw_featmap,
        indices,
        weights,
        confidence,
        place_count,
        views_per_place,
        semantic_config,
    )
    builder_kwargs = diagnostics["builder_kwargs"]
    full_builder = SemanticRegionReliabilityTarget(
        mode="full", **builder_kwargs
    )
    shuffled_builder = SemanticRegionReliabilityTarget(
        mode="shuffled", **builder_kwargs
    )

    reference_full_out10, _ = full_builder._sparse_semantic_smooth(
        diagnostics["base10"],
        indices,
        weights,
        confidence,
        place_count,
        views_per_place,
    )
    reference_shuffled_out10, _ = shuffled_builder._sparse_semantic_smooth(
        diagnostics["base10"],
        indices,
        weights,
        confidence,
        place_count,
        views_per_place,
    )
    torch.testing.assert_close(
        diagnostics["aligned"]["out10"], reference_full_out10
    )
    torch.testing.assert_close(
        diagnostics["shuffled"]["out10"], reference_shuffled_out10
    )

    reference_full_target, _ = full_builder(
        raw_featmap,
        place_count,
        views_per_place,
        indices,
        weights,
        confidence,
    )
    reference_shuffled_target, _ = shuffled_builder(
        raw_featmap,
        place_count,
        views_per_place,
        indices,
        weights,
        confidence,
    )
    torch.testing.assert_close(
        diagnostics["aligned"]["target"], reference_full_target
    )
    torch.testing.assert_close(
        diagnostics["shuffled"]["target"], reference_shuffled_target
    )

    out10_difference = (
        diagnostics["aligned"]["out10"]
        - diagnostics["shuffled"]["out10"]
    ).abs()
    target_difference = (
        diagnostics["aligned"]["target"]
        - diagnostics["shuffled"]["target"]
    ).abs()
    assert out10_difference.max() > 1e-3
    assert target_difference.max() > 1e-3


def test_sample_selection_has_one_random_audit_and_no_duplicates() -> None:
    aligned_delta = torch.zeros(8, 1, 2, 2)
    target_difference = torch.zeros_like(aligned_delta)
    for index in range(8):
        aligned_delta[index].fill_(float(8 - index))
        target_difference[index].fill_(float(index + 1))

    selected, reasons = select_sample_indices(
        aligned_delta,
        target_difference,
        num_samples=5,
        seed=123,
    )
    repeated, repeated_reasons = select_sample_indices(
        aligned_delta,
        target_difference,
        num_samples=5,
        seed=123,
    )

    assert len(selected) == 5
    assert len(set(selected)) == len(selected)
    assert selected == repeated
    assert reasons == repeated_reasons
    random_audits = [
        index for index in selected if "random_audit" in reasons[index]
    ]
    assert len(random_audits) == 1


def test_single_sample_selection_does_not_label_unselected_samples() -> None:
    aligned_delta = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    target_difference = aligned_delta.flip(0)

    selected, reasons = select_sample_indices(
        aligned_delta,
        target_difference,
        num_samples=1,
        seed=7,
    )

    assert len(selected) == 1
    assert set(reasons) == set(selected)
