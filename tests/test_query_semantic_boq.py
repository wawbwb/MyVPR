import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.core.vpr_framework import VPRFrameworkDistill
from src.models.aggregators.boq import BoQ, BoQBlock
from src.models.query_semantic import (
    QuerySemanticTarget,
    spatially_randomize_targets,
)


def _tiny_boq(*, semantic: bool) -> BoQ:
    torch.manual_seed(17)
    return BoQ(
        in_channels=4,
        proj_channels=64,
        num_queries=2,
        num_layers=2,
        row_dim=4,
        semantic_num_classes=3 if semantic else None,
        semantic_bias_scale=0.25,
        semantic_temperature=1.0,
        semantic_head_hidden=8,
    ).eval()


def _copy_legacy_parameters(legacy: BoQ, conditioned: BoQ) -> None:
    state = conditioned.state_dict()
    legacy_state = legacy.state_dict()
    assert set(legacy_state).issubset(state)
    state.update(legacy_state)
    conditioned.load_state_dict(state, strict=True)


def test_historical_boq_path_and_state_dict_remain_exact() -> None:
    model = _tiny_boq(semantic=False)
    feature_map = torch.randn(2, 4, 2, 3)

    implicit_descriptor, implicit_attention = model(feature_map)
    explicit_descriptor, explicit_attention = model(
        feature_map,
        attention_bias=None,
        semantic_logits=None,
    )

    assert torch.equal(implicit_descriptor, explicit_descriptor)
    assert all(
        torch.equal(left, right)
        for left, right in zip(implicit_attention, explicit_attention)
    )
    assert not any("semantic_" in key for key in model.state_dict())

    restored = _tiny_boq(semantic=False)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_descriptor, restored_attention = restored(feature_map)
    assert torch.equal(restored_descriptor, implicit_descriptor)
    assert all(
        torch.equal(left, right)
        for left, right in zip(restored_attention, implicit_attention)
    )


def test_zero_initialised_semantic_adapter_is_exactly_the_legacy_path() -> None:
    legacy = _tiny_boq(semantic=False)
    conditioned = _tiny_boq(semantic=True)
    _copy_legacy_parameters(legacy, conditioned)
    feature_map = torch.randn(2, 4, 2, 3)
    arbitrary_logits = torch.randn(2, 3, 2, 3)

    legacy_descriptor, legacy_attention = legacy(feature_map)
    supplied_descriptor, supplied_attention = conditioned(
        feature_map, semantic_logits=arbitrary_logits
    )
    internal_descriptor, internal_attention = conditioned(feature_map)

    assert all(
        torch.count_nonzero(block.semantic_query_proj.weight) == 0
        and torch.count_nonzero(block.semantic_query_proj.bias) == 0
        for block in conditioned.boqs
    )
    assert torch.equal(supplied_descriptor, legacy_descriptor)
    assert torch.equal(internal_descriptor, legacy_descriptor)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(supplied_attention, legacy_attention)
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(internal_attention, legacy_attention)
    )


def test_query_semantic_bias_is_batched_bounded_and_query_specific() -> None:
    block = BoQBlock(
        in_dim=64,
        num_queries=2,
        nheads=1,
        semantic_num_classes=3,
        semantic_bias_scale=0.25,
    ).eval()
    with torch.no_grad():
        block.semantic_query_proj.weight.zero_()
        block.semantic_query_proj.bias.zero_()
        block.semantic_query_proj.weight[0, 0] = 2.0
        block.semantic_query_proj.weight[1, 1] = 2.0

    queries = torch.zeros(2, 2, 64)
    queries[:, 0, 0] = 1.0
    queries[:, 1, 1] = 1.0
    probabilities = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )

    bias = block._query_semantic_bias(queries, probabilities)

    assert bias.shape == (2, 2, 3)
    assert torch.isfinite(bias).all()
    assert float(bias.abs().max()) <= 0.25 + 1e-7
    torch.testing.assert_close(
        bias.mean(dim=-1), torch.zeros(2, 2), atol=1e-7, rtol=0.0
    )
    assert not torch.allclose(bias[:, 0], bias[:, 1])


def test_semantic_head_default_and_external_logits_are_finite() -> None:
    model = _tiny_boq(semantic=True)
    feature_map = torch.randn(2, 4, 2, 3)

    logits = model.predict_semantics(feature_map)
    default_descriptor, default_attention = model(feature_map)
    external_descriptor, external_attention = model(
        feature_map, semantic_logits=logits
    )

    assert logits.shape == (2, 3, 2, 3)
    assert torch.isfinite(logits).all()
    assert default_descriptor.shape == (2, 64 * 4)
    assert torch.isfinite(default_descriptor).all()
    assert len(default_attention) == len(model.boqs)
    assert torch.equal(default_descriptor, external_descriptor)
    assert all(
        torch.equal(left, right)
        for left, right in zip(default_attention, external_attention)
    )
    diagnostics = model.semantic_diagnostics()
    assert diagnostics
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert float(diagnostics["query_semantic_bias_abs_max"]) == 0.0


def test_boq_rejects_bad_semantic_logits_shape_and_values() -> None:
    model = _tiny_boq(semantic=True)
    feature_map = torch.randn(2, 4, 2, 3)

    with pytest.raises(ValueError, match="semantic_logits must have shape"):
        model(feature_map, semantic_logits=torch.randn(2, 3, 3, 2))

    nonfinite = torch.randn(2, 3, 2, 3)
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(feature_map, semantic_logits=nonfinite)


def test_vpr_gradient_reaches_query_adapter() -> None:
    model = _tiny_boq(semantic=True).train()
    feature_map = torch.randn(2, 4, 2, 3)
    semantic_logits = torch.randn(2, 3, 2, 3)
    descriptor, _ = model(feature_map, semantic_logits=semantic_logits)
    probe = torch.linspace(-1.0, 1.0, descriptor.numel()).view_as(descriptor)

    (descriptor * probe).sum().backward()

    adapter_grads = [
        parameter.grad
        for block in model.boqs
        for parameter in block.semantic_query_proj.parameters()
    ]
    assert all(gradient is not None for gradient in adapter_grads)
    assert sum(float(gradient.abs().sum()) for gradient in adapter_grads) > 0.0


def test_framework_detaches_semantic_head_input_from_backbone() -> None:
    class ScaleBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images * self.scale

    backbone = ScaleBackbone()
    aggregator = _tiny_boq(semantic=True).train()
    target = QuerySemanticTarget(
        mode="aligned", num_classes=3, min_confidence=0.0
    )
    framework = VPRFrameworkDistill(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=nn.Identity(),
        config_dict={},
        lambda_global=0.0,
        lambda_region=0.0,
        query_semantic_target=target,
        lambda_query_semantic=1.0,
    )
    images = torch.randn(2, 4, 2, 3, requires_grad=True)
    *_, semantic_logits = framework._student_forward(
        images, return_query_semantic=True
    )
    labels = torch.tensor(
        [[[0, 1, 2], [1, 2, 0]], [[2, 0, 1], [0, 1, 2]]]
    )
    confidence = torch.ones_like(labels, dtype=torch.float32)
    loss, _ = target(
        semantic_logits, labels, confidence, torch.tensor([4, 9])
    )

    loss.backward()

    head_grads = [parameter.grad for parameter in aggregator.semantic_head.parameters()]
    assert all(gradient is not None for gradient in head_grads)
    assert sum(float(gradient.abs().sum()) for gradient in head_grads) > 0.0
    assert images.grad is None
    assert backbone.scale.grad is None


def test_aligned_target_matches_confidence_weighted_cross_entropy() -> None:
    logits = torch.tensor(
        [[[[2.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]], [[0.0, 0.0, 2.0]]]]
    )
    labels = torch.tensor([[[0, 1, 2]]])
    confidence = torch.tensor([[[1.0, 0.5, 0.49]]])
    target = QuerySemanticTarget(
        mode="aligned", num_classes=3, min_confidence=0.5
    )

    loss, stats = target(logits, labels, confidence, torch.tensor([7]))
    per_patch = F.cross_entropy(logits, labels, reduction="none")
    expected = (per_patch[0, 0, 0] + 0.5 * per_patch[0, 0, 1]) / 1.5

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        stats["query_semantic_valid_frac"], torch.tensor(2.0 / 3.0)
    )
    torch.testing.assert_close(
        stats["query_semantic_valid_confidence"], torch.tensor(0.75)
    )
    torch.testing.assert_close(
        stats["query_semantic_accuracy"], torch.ones(())
    )


def test_target_all_ignored_is_differentiable_finite_zero() -> None:
    logits = torch.randn(2, 3, 2, 2, requires_grad=True)
    labels = torch.full((2, 2, 2), 255)
    confidence = torch.ones(2, 2, 2)
    target = QuerySemanticTarget(
        mode="aligned",
        num_classes=3,
        min_confidence=0.5,
        ignore_index=255,
    )

    loss, stats = target(logits, labels, confidence, torch.tensor([1, 2]))
    loss.backward()

    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0
    assert float(stats["query_semantic_valid_frac"]) == 0.0
    assert float(stats["query_semantic_valid_confidence"]) == 0.0


def test_random_target_is_stable_by_cache_index_and_preserves_pairs() -> None:
    labels = torch.arange(12).reshape(2, 2, 3)
    confidence = labels.float().div(20.0)
    cache_indices = torch.tensor([7, 19])

    labels_42, confidence_42 = spatially_randomize_targets(
        labels, confidence, cache_indices, seed=42
    )
    repeated_labels, repeated_confidence = spatially_randomize_targets(
        labels, confidence, cache_indices, seed=42
    )
    reordered_labels, reordered_confidence = spatially_randomize_targets(
        labels.flip(0), confidence.flip(0), cache_indices.flip(0), seed=42
    )
    labels_43, confidence_43 = spatially_randomize_targets(
        labels, confidence, cache_indices, seed=43
    )

    assert torch.equal(labels_42, repeated_labels)
    assert torch.equal(confidence_42, repeated_confidence)
    assert torch.equal(labels_42, reordered_labels.flip(0))
    assert torch.equal(confidence_42, reordered_confidence.flip(0))
    assert not torch.equal(labels_42, labels_43)
    assert not torch.equal(confidence_42, confidence_43)
    for row in range(labels.shape[0]):
        original_pairs = torch.stack(
            (labels[row].flatten().float(), confidence[row].flatten()), dim=1
        )
        random_pairs = torch.stack(
            (
                labels_42[row].flatten().float(),
                confidence_42[row].flatten(),
            ),
            dim=1,
        )
        original_order = original_pairs[:, 0].argsort()
        random_order = random_pairs[:, 0].argsort()
        torch.testing.assert_close(
            random_pairs[random_order], original_pairs[original_order]
        )


def test_random_query_target_uses_the_stable_pair_permutation() -> None:
    labels = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    confidence = torch.tensor([[[0.91, 0.92, 0.93], [0.94, 0.95, 0.96]]])
    cache_indices = torch.tensor([31])
    random_labels, random_confidence = spatially_randomize_targets(
        labels, confidence, cache_indices, seed=42
    )
    logits = 8.0 * F.one_hot(random_labels, num_classes=6).permute(0, 3, 1, 2)
    random_target = QuerySemanticTarget(
        mode="random", num_classes=6, min_confidence=0.5, random_seed=42
    )
    aligned_target = QuerySemanticTarget(
        mode="aligned", num_classes=6, min_confidence=0.5
    )

    actual_loss, actual_stats = random_target(
        logits, labels, confidence, cache_indices
    )
    expected_loss, expected_stats = aligned_target(
        logits, random_labels, random_confidence, cache_indices
    )

    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(
        actual_stats["query_semantic_valid_frac"],
        expected_stats["query_semantic_valid_frac"],
    )
    torch.testing.assert_close(
        actual_stats["query_semantic_valid_confidence"],
        expected_stats["query_semantic_valid_confidence"],
    )
