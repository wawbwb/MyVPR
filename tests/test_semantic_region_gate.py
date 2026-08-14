import torch
from torch import nn
from torch.nn import functional as F

from src.core.vpr_framework import VPRFrameworkDistill
from src.models.semantic_region_gate import (
    SemanticRegionGate,
    SemanticRegionReliabilityTarget,
)


def _identity_cache(batch_size: int, side: int):
    patch_count = side * side
    indices = torch.arange(patch_count, dtype=torch.uint8)
    indices = indices.view(1, patch_count, 1).expand(batch_size, -1, -1)
    weights = torch.ones(batch_size, patch_count, 1)
    confidence = torch.ones(batch_size, patch_count)
    return indices, weights, confidence


def test_gate_starts_as_identity_and_is_bounded() -> None:
    gate = SemanticRegionGate(in_channels=3, alpha=0.2)
    features = torch.randn(4, 3, 5, 5)
    output, score, weights = gate(features)

    torch.testing.assert_close(output, features)
    torch.testing.assert_close(score, torch.zeros_like(score))
    torch.testing.assert_close(weights, torch.ones_like(weights))

    with torch.no_grad():
        gate.proj.weight.fill_(100.0)
        gate.proj.bias.fill_(100.0)
    _, _, weights = gate(features)
    assert weights.min() >= 0.8
    assert weights.max() <= 1.2


def test_semantic_only_does_not_depend_on_dino_features() -> None:
    builder = SemanticRegionReliabilityTarget(
        mode="semantic_only", match_grid=2
    )
    confidence = torch.tensor(
        [[0.0, 0.2, 0.8, 1.0]] * 4, dtype=torch.float32
    )
    first, _ = builder(
        torch.randn(4, 2, 2, 2), 2, 2, semantic_confidence=confidence
    )
    second, _ = builder(
        torch.randn(4, 2, 2, 2) * 100.0,
        2,
        2,
        semantic_confidence=confidence,
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_semantic_only_ignores_nearly_constant_quantized_confidence() -> None:
    builder = SemanticRegionReliabilityTarget(
        mode="semantic_only", match_grid=2, min_spatial_std=1e-3
    )
    confidence = torch.full((4, 4), 0.5, dtype=torch.float16)
    confidence[:, 0] += torch.finfo(torch.float16).eps

    target, stats = builder(
        torch.randn(4, 2, 2, 2), 2, 2, semantic_confidence=confidence
    )

    torch.testing.assert_close(target, torch.zeros_like(target))
    torch.testing.assert_close(
        stats["region_semantic_informative_frac"], torch.zeros(())
    )


def test_shuffled_cache_matches_explicit_place_roll() -> None:
    torch.manual_seed(9)
    features = torch.randn(4, 3, 2, 2)
    indices, weights, confidence = _identity_cache(4, 2)
    confidence[2:] *= 0.3

    shuffled = SemanticRegionReliabilityTarget(mode="shuffled", match_grid=2)
    shuffled_target, _ = shuffled(
        features, 2, 2, indices, weights, confidence
    )

    rolled_indices = indices.view(2, 2, 4, 1).roll(1, 0).flatten(0, 1)
    rolled_weights = weights.view(2, 2, 4, 1).roll(1, 0).flatten(0, 1)
    rolled_confidence = confidence.view(2, 2, 4).roll(1, 0).flatten(0, 1)
    full = SemanticRegionReliabilityTarget(mode="full", match_grid=2)
    full_target, _ = full(
        features,
        2,
        2,
        rolled_indices,
        rolled_weights,
        rolled_confidence,
    )
    torch.testing.assert_close(shuffled_target, full_target)


def test_full_requires_all_cache_tensors() -> None:
    builder = SemanticRegionReliabilityTarget(mode="full", match_grid=2)
    try:
        builder(torch.randn(4, 2, 2, 2), 2, 2)
    except ValueError as error:
        assert "semantic cache tensors" in str(error)
    else:  # pragma: no cover
        raise AssertionError("full target must reject a missing cache")


def test_repeatability_uniqueness_control_matches_identity_semantics() -> None:
    torch.manual_seed(13)
    features = torch.randn(6, 3, 2, 2)
    indices, weights, confidence = _identity_cache(6, 2)

    control = SemanticRegionReliabilityTarget(
        mode="repeatability_uniqueness_only", match_grid=2
    )
    control_target, _ = control(features, 3, 2)
    full = SemanticRegionReliabilityTarget(mode="full", match_grid=2)
    full_target, _ = full(
        features, 3, 2, indices, weights, confidence
    )

    torch.testing.assert_close(control_target, full_target)


def test_vectorized_vpr_components_match_reference_loops() -> None:
    torch.manual_seed(21)
    builder = SemanticRegionReliabilityTarget(
        mode="full", match_grid=2, place_chunk_size=1
    )
    features = torch.randn(6, 3, 2, 2)

    repeatability, uniqueness, _ = builder._vpr_components(features, 3, 2)

    tokens = F.normalize(features.flatten(2).transpose(1, 2), dim=-1)
    tokens = tokens.view(3, 2, 4, 3)
    expected_repeatability = torch.empty(3, 2, 4)
    prototypes = F.normalize(tokens.mean(dim=(1, 2)), dim=-1)
    place_similarity = prototypes @ prototypes.T
    place_similarity.fill_diagonal_(-torch.inf)
    hard_places = place_similarity.argmax(dim=1)
    expected_commonness = torch.empty_like(expected_repeatability)
    for place in range(3):
        for view in range(2):
            expected_repeatability[place, view] = (
                tokens[place, view] @ tokens[place, 1 - view].T
            ).amax(dim=-1)
            negatives = tokens[hard_places[place]].flatten(0, 1)
            expected_commonness[place, view] = (
                tokens[place, view] @ negatives.T
            ).amax(dim=-1)

    torch.testing.assert_close(
        repeatability, expected_repeatability.flatten(0, 1)
    )
    torch.testing.assert_close(
        uniqueness, (1.0 - expected_commonness).flatten(0, 1)
    )


def test_salad_tuple_keeps_cls_token_and_inference_never_needs_cache() -> None:
    class TupleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, images):
            local = images * self.scale
            cls = images.mean(dim=(2, 3))
            return local, cls

    class TupleAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_cls = None

        def forward(self, output):
            local, cls = output
            self.seen_cls = cls
            return torch.cat((local.flatten(1), cls), dim=1)

    aggregator = TupleAggregator()
    framework = VPRFrameworkDistill(
        backbone=TupleBackbone(),
        aggregator=aggregator,
        loss_function=nn.Identity(),
        config_dict={},
        distill_module=None,
        lambda_global=0.0,
        lambda_region=0.0,
        semantic_region_gate=SemanticRegionGate(2),
        semantic_region_target=SemanticRegionReliabilityTarget(
            mode="repeatability_only", match_grid=2
        ),
        lambda_semantic_region=0.02,
    ).eval()
    images = torch.randn(2, 2, 2, 2)

    output = framework(images)

    torch.testing.assert_close(aggregator.seen_cls, images.mean(dim=(2, 3)))
    assert output.shape == (2, 10)


def test_semantic_auxiliary_gradient_updates_gate_not_detached_features() -> None:
    gate = SemanticRegionGate(2)
    features = torch.randn(4, 2, 2, 2, requires_grad=True)
    target = torch.randn(4, 1, 2, 2).tanh()

    score = gate.predict(features.detach())
    loss = torch.nn.functional.smooth_l1_loss(score, target)
    feature_grad, gate_grad = torch.autograd.grad(
        loss, (features, gate.proj.weight), allow_unused=True
    )

    assert feature_grad is None
    assert gate_grad is not None
    assert gate_grad.abs().sum() > 0
