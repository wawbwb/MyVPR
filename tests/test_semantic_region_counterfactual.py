import torch
from torch.nn import functional as F

from scripts.sweep_semantic_region_counterfactual import (
    TargetScaler,
    apply_target_transform,
    build_counterfactual_branch,
    make_hard_mask,
    resize_reliability,
)


def test_absolute_threshold_mask_is_boolean_and_includes_boundary() -> None:
    confidence = torch.tensor(
        [[0.0, 0.49, 0.50, 0.91], [0.50, 0.51, 0.20, 1.0]],
        dtype=torch.float32,
    )
    original = confidence.clone()

    mask = make_hard_mask(confidence, threshold=0.5)

    assert mask.dtype == torch.bool
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [[False, False, True, True], [True, True, False, True]]
        ),
    )
    torch.testing.assert_close(confidence, original)


def test_top_twenty_percent_selects_exactly_per_image() -> None:
    confidence = torch.tensor(
        [
            [0.01, 0.90, 0.30, 0.80, 0.20, 0.10, 0.40, 0.50, 0.60, 0.70],
            [0.99, 0.10, 0.20, 0.30, 0.40, 0.98, 0.50, 0.60, 0.70, 0.80],
        ],
        dtype=torch.float32,
    )

    mask = make_hard_mask(confidence, top_fraction=0.2)

    assert mask.dtype == torch.bool
    torch.testing.assert_close(mask.sum(dim=1), torch.tensor([2, 2]))
    assert mask[0].nonzero(as_tuple=True)[0].tolist() == [1, 3]
    assert mask[1].nonzero(as_tuple=True)[0].tolist() == [0, 5]


def test_resize_reliability_matches_production_bilinear_resize() -> None:
    reliability = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0]],
        dtype=torch.float32,
    )
    expected = F.interpolate(
        reliability.view(2, 1, 2, 2),
        size=(3, 5),
        mode="bilinear",
        align_corners=False,
    )

    actual = resize_reliability(reliability, (3, 5))

    torch.testing.assert_close(actual, expected)


def test_per_image_zscore_matches_production_target_formula() -> None:
    feature_map = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 5.0]]],
            [[[10.0, 14.0], [11.0, 20.0]]],
        ],
        dtype=torch.float32,
    )
    target_scale = 2.0
    flat = feature_map.flatten(1)
    expected = torch.tanh(
        target_scale
        * (
            (flat - flat.mean(dim=1, keepdim=True))
            / flat.std(dim=1, keepdim=True, unbiased=False)
        )
    ).view_as(feature_map)

    actual = apply_target_transform(
        feature_map,
        mode="per_image_zscore",
        target_scale=target_scale,
        eps=1e-6,
    )

    torch.testing.assert_close(actual, expected)


def test_shared_base_uses_one_fixed_batch_scale_not_per_image_scale() -> None:
    base_feature = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[10.0, 12.0], [14.0, 16.0]]],
        ],
        dtype=torch.float32,
    )
    counterfactual = base_feature.clone()
    counterfactual[0] = counterfactual[0] + 4.0
    counterfactual[1] = counterfactual[1] * 0.5
    scaler = TargetScaler.from_base(base_feature, eps=1e-6)
    expected_mean = float(base_feature.mean())
    expected_std = float(base_feature.std(unbiased=False))
    expected = torch.tanh(
        (counterfactual - expected_mean) / expected_std
    )

    actual = apply_target_transform(
        counterfactual,
        mode="shared_base",
        target_scale=1.0,
        scaler=scaler,
        eps=1e-6,
    )

    assert scaler.mean == expected_mean
    assert scaler.std == expected_std
    torch.testing.assert_close(actual, expected)

    flat = counterfactual.flatten(1)
    independently_rescaled = torch.tanh(
        (
            (flat - flat.mean(dim=1, keepdim=True))
            / flat.std(dim=1, keepdim=True, unbiased=False)
        ).view_as(counterfactual)
    )
    assert not torch.allclose(actual, independently_rescaled)


def test_zero_mask_ru_additive_returns_exactly_to_ru_base() -> None:
    base_reliability = torch.tensor(
        [[0.0, 1.0, 8.0, 2.0, -3.0, 5.0, 4.0, 7.0, -1.0]],
        dtype=torch.float32,
    )
    base14 = F.interpolate(
        base_reliability.view(1, 1, 3, 3),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    smoothed14 = base14.flip(1) + 7.0
    confidence14 = torch.linspace(0.1, 1.0, 16).view(1, 16)
    mask = torch.zeros_like(confidence14, dtype=torch.bool)

    branch = build_counterfactual_branch(
        base14,
        smoothed14,
        confidence14,
        base_side=3,
        feature_hw=(5, 5),
        mask=mask,
        transform="center_only",
        target_scale=1.0,
        composition="ru_additive",
        base_reliability=base_reliability,
    )
    expected_pre_target = resize_reliability(base_reliability, (5, 5))
    expected_target = apply_target_transform(
        expected_pre_target,
        mode="center_only",
        target_scale=1.0,
    )

    torch.testing.assert_close(
        branch["effective_confidence"], torch.zeros_like(confidence14)
    )
    torch.testing.assert_close(branch["delta14"], torch.zeros_like(base14))
    torch.testing.assert_close(branch["propagated14"], base14)
    torch.testing.assert_close(
        branch["resized_delta"], torch.zeros_like(base_reliability)
    )
    torch.testing.assert_close(branch["out_base"], base_reliability)
    torch.testing.assert_close(branch["pre_target"], expected_pre_target)
    torch.testing.assert_close(branch["target"], expected_target)


def test_zero_mask_production_roundtrip_preserves_interpolation_effect() -> None:
    base_reliability = torch.tensor(
        [[0.0, 1.0, 8.0, 2.0, -3.0, 5.0, 4.0, 7.0, -1.0]],
        dtype=torch.float32,
    )
    base14 = F.interpolate(
        base_reliability.view(1, 1, 3, 3),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    mask = torch.zeros_like(base14, dtype=torch.bool)

    branch = build_counterfactual_branch(
        base14,
        smoothed14=base14.flip(1),
        confidence14=torch.ones_like(base14),
        base_side=3,
        feature_hw=(5, 5),
        mask=mask,
        transform="center_only",
        target_scale=1.0,
        composition="production_roundtrip",
        base_reliability=base_reliability,
    )
    expected_roundtrip = F.interpolate(
        base14.view(1, 1, 4, 4),
        size=(3, 3),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)

    torch.testing.assert_close(branch["out_base"], expected_roundtrip)
    assert not torch.allclose(branch["out_base"], base_reliability)


def test_counterfactual_branch_retains_confidence_and_two_stage_resize() -> None:
    base14 = torch.tensor([[0.0, 1.0, 2.0, 4.0]], dtype=torch.float32)
    smoothed14 = torch.tensor([[4.0, 3.0, 0.0, 2.0]], dtype=torch.float32)
    confidence14 = torch.tensor([[0.25, 0.50, 0.75, 1.00]])
    mask = torch.tensor([[True, False, True, True]])
    expected_effective = confidence14 * mask.float()
    expected_delta = expected_effective * (smoothed14 - base14)
    expected_propagated = base14 + expected_delta
    expected_out_base = F.interpolate(
        expected_propagated.view(1, 1, 2, 2),
        size=(3, 3),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    expected_resized_delta = F.interpolate(
        expected_delta.view(1, 1, 2, 2),
        size=(3, 3),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    expected_pre_target = F.interpolate(
        expected_out_base.view(1, 1, 3, 3),
        size=(4, 5),
        mode="bilinear",
        align_corners=False,
    )
    flat = expected_pre_target.flatten(1)
    expected_target = torch.tanh(
        2.0
        * (
            (flat - flat.mean(dim=1, keepdim=True))
            / flat.std(dim=1, keepdim=True, unbiased=False)
        )
    ).view_as(expected_pre_target)

    branch = build_counterfactual_branch(
        base14,
        smoothed14,
        confidence14,
        base_side=3,
        feature_hw=(4, 5),
        mask=mask,
        transform="per_image_zscore",
        target_scale=2.0,
    )

    torch.testing.assert_close(branch["mask"], mask)
    torch.testing.assert_close(branch["effective_confidence"], expected_effective)
    torch.testing.assert_close(branch["delta14"], expected_delta)
    torch.testing.assert_close(branch["propagated14"], expected_propagated)
    torch.testing.assert_close(branch["resized_delta"], expected_resized_delta)
    torch.testing.assert_close(branch["out_base"], expected_out_base)
    torch.testing.assert_close(branch["pre_target"], expected_pre_target)
    torch.testing.assert_close(branch["target"], expected_target)
