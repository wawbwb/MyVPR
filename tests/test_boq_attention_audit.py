import math

import torch

from scripts.boq_attention_audit import (
    compute_attention_components,
    compute_mask_overlap,
    fc_energy_slot_weights,
    force_per_head_cross_attention,
)
from src.models.aggregators.boq import BoQ


def _tiny_boq() -> BoQ:
    torch.manual_seed(17)
    return BoQ(
        in_channels=8,
        proj_channels=64,
        num_queries=3,
        num_layers=2,
        row_dim=4,
    ).eval()


def test_per_head_hook_preserves_descriptor_and_legacy_attention_mean() -> None:
    model = _tiny_boq()
    feature_map = torch.randn(2, 8, 2, 3)

    legacy_descriptor, legacy_attention = model(feature_map)
    with force_per_head_cross_attention(model):
        per_head_descriptor, per_head_attention = model(feature_map)

    assert torch.equal(per_head_descriptor, legacy_descriptor)
    assert len(per_head_attention) == 2
    for per_head, legacy in zip(per_head_attention, legacy_attention):
        assert per_head.shape == (2, 1, 3, 6)
        torch.testing.assert_close(per_head.mean(dim=1), legacy)
        torch.testing.assert_close(
            per_head.sum(dim=-1), torch.ones_like(per_head[..., 0])
        )

    # The hook must be removed when the context exits.
    _, restored_attention = model(feature_map)
    assert all(attention.ndim == 3 for attention in restored_attention)


def test_uniform_attention_has_unit_enrichment_for_soft_mask() -> None:
    attentions = [
        torch.full((1, 2, 3, 6), 1.0 / 6.0),
        torch.full((1, 2, 3, 6), 1.0 / 6.0),
    ]
    slot_weights = torch.full((2, 3), 1.0 / 6.0)
    diagnostics = compute_attention_components(
        attentions,
        grid_size=(2, 3),
        fc_slot_weights=slot_weights,
    )
    mask = torch.tensor([[[[1.0, 0.5, 0.0], [0.25, 0.0, 0.0]]]])
    overlap = compute_mask_overlap(diagnostics["component_maps"], mask)

    expected_area = torch.tensor([[1.75 / 6.0]])
    torch.testing.assert_close(overlap["area"], expected_area)
    torch.testing.assert_close(
        overlap["mass"], expected_area[:, :, None].expand(-1, -1, 4)
    )
    torch.testing.assert_close(
        overlap["enrichment"], torch.ones_like(overlap["enrichment"])
    )
    torch.testing.assert_close(
        diagnostics["focus"]["top_10pct_attention_mass"],
        torch.full((1, 4), 1.0 / 6.0),
    )
    torch.testing.assert_close(
        diagnostics["focus"]["top_20pct_attention_mass"],
        torch.full((1, 4), 2.0 / 6.0),
    )


def test_soft_attention_mass_is_exact_and_empty_mask_is_ineligible() -> None:
    maps = torch.tensor([[[0.4, 0.3, 0.2, 0.1]]])
    masks = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    overlap = compute_mask_overlap(maps, masks)

    torch.testing.assert_close(overlap["area"][0, 0], torch.tensor(0.375))
    torch.testing.assert_close(overlap["mass"][0, 0, 0], torch.tensor(0.55))
    torch.testing.assert_close(
        overlap["enrichment"][0, 0, 0], torch.tensor(0.55 / 0.375)
    )
    assert not bool(overlap["eligible"][0, 1])
    assert math.isnan(float(overlap["enrichment"][0, 1, 0]))
    torch.testing.assert_close(overlap["support_area"][0, 0], torch.tensor(0.5))
    torch.testing.assert_close(overlap["support_mass"][0, 0, 0], torch.tensor(0.7))


def test_attention_and_mask_use_row_major_non_square_grid() -> None:
    attention = torch.zeros(1, 1, 1, 6)
    attention[..., 4] = 1.0  # row=1, column=1 in a 2x3 grid
    diagnostics = compute_attention_components(
        [attention],
        grid_size=(2, 3),
        fc_slot_weights=torch.ones(1, 1),
    )
    masks = torch.zeros(1, 2, 2, 3)
    masks[0, 0, 1, 1] = 1.0
    masks[0, 1, 0, 2] = 1.0

    overlap = compute_mask_overlap(diagnostics["component_maps"], masks)

    torch.testing.assert_close(overlap["mass"][0, 0], torch.ones(3))
    torch.testing.assert_close(overlap["mass"][0, 1], torch.zeros(3))
    torch.testing.assert_close(
        diagnostics["component_maps"].sum(dim=-1),
        torch.ones(1, 3),
    )


def test_head_query_layer_aggregation_and_fc_energy_weights_are_exact() -> None:
    layer_one = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        ]
    )
    layer_two = torch.tensor(
        [
            [
                [[0.8, 0.2], [0.6, 0.4]],
                [[0.2, 0.8], [0.4, 0.6]],
            ]
        ]
    )
    fc_weight = torch.tensor([[1.0, 0.0, 0.0, 3.0]])
    slot_weights = fc_energy_slot_weights(
        fc_weight, num_layers=2, num_queries=2
    )
    diagnostics = compute_attention_components(
        [layer_one, layer_two],
        grid_size=(1, 2),
        fc_slot_weights=slot_weights,
    )

    expected_layer_one = layer_one.mean(dim=(1, 2))
    expected_layer_two = layer_two.mean(dim=(1, 2))
    expected_consensus = (expected_layer_one + expected_layer_two) / 2.0
    torch.testing.assert_close(
        diagnostics["layer_maps"],
        torch.stack((expected_layer_one, expected_layer_two), dim=1),
    )
    torch.testing.assert_close(diagnostics["consensus_map"], expected_consensus)
    torch.testing.assert_close(slot_weights, torch.tensor([[0.1, 0.0], [0.0, 0.9]]))
    expected_fc = (
        diagnostics["query_maps"] * slot_weights[None, :, :, None]
    ).sum(dim=(1, 2))
    torch.testing.assert_close(diagnostics["fc_proxy_map"], expected_fc)
