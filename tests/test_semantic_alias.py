import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.core.vpr_framework import VPRFrameworkDistill
from src.models.distillation import DistillationModule
from src.models.semantic_alias import CLIPSemanticAliasLoss


def _controlled_alias_batch(
    *, requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four two-view places with one geographically unsafe CLIP neighbor."""
    labels = torch.tensor([10, 10, 20, 20, 30, 30, 40, 40])

    # At the equator these longitudes are approximately 0, 11, 111 and
    # 222 metres.  Places 10 and 20 must therefore be rejected as a pair by a
    # 50 m threshold even though they are each other's closest CLIP neighbor.
    place_longitudes = torch.tensor([0.0, 0.0001, 0.0010, 0.0020], dtype=torch.float64)
    coordinates = torch.stack(
        (torch.zeros_like(place_longitudes), place_longitudes), dim=1
    ).repeat_interleave(2, dim=0)

    angles = torch.tensor([0.0, 0.01, 0.10, 2.0])
    place_clip = torch.stack((angles.cos(), angles.sin()), dim=1)
    clip_embeddings = place_clip.repeat_interleave(2, dim=0).detach()

    student_descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
            [-0.8, 0.6],
            [0.0, 1.0],
            [0.6, 0.8],
            [0.0, -1.0],
            [-0.6, -0.8],
        ]
    )
    if requires_grad:
        student_descriptors.requires_grad_()
        clip_embeddings.requires_grad_()
    return student_descriptors, clip_embeddings, labels, coordinates


def test_clip_mining_obeys_gps_and_uses_hardest_cross_view_pair() -> None:
    student, clip_embeddings, labels, coordinates = _controlled_alias_batch(
        requires_grad=True
    )
    objective = CLIPSemanticAliasLoss(
        selection="clip",
        negative_topk=1,
        min_geo_distance_m=50.0,
        student_margin=0.2,
        loss_temperature=0.05,
    )

    loss, stats = objective(student, clip_embeddings, labels, coordinates)

    # Expected directed selections by CLIP angle are 10->30, 20->30,
    # 30->20 and 40->30.  In particular, unsafe 10<->20 is never selected.
    selected_image_groups = [
        ([0, 1], [4, 5]),
        ([2, 3], [4, 5]),
        ([4, 5], [2, 3]),
        ([6, 7], [4, 5]),
    ]
    normalized_student = F.normalize(student.detach(), dim=-1)
    student_similarity = normalized_student @ normalized_student.t()
    expected_hard = torch.stack(
        [
            student_similarity[
                torch.tensor(anchor)[:, None], torch.tensor(negative)[None, :]
            ].amax()
            for anchor, negative in selected_image_groups
        ]
    )
    expected_loss = 0.05 * F.softplus((expected_hard - 0.2) / 0.05).mean()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(
        stats["semantic_alias_hard_student_sim"], expected_hard.mean()
    )

    place_clip = F.normalize(clip_embeddings.detach()[::2], dim=-1)
    expected_place_pairs = torch.tensor([[0, 2], [1, 2], [2, 1], [3, 2]])
    expected_clip_similarity = torch.stack(
        [place_clip[a] @ place_clip[b] for a, b in expected_place_pairs.tolist()]
    ).mean()
    torch.testing.assert_close(
        stats["semantic_alias_selected_clip_sim"], expected_clip_similarity
    )
    torch.testing.assert_close(
        stats["semantic_alias_candidate_pair_count"], torch.tensor(10.0)
    )
    torch.testing.assert_close(
        stats["semantic_alias_candidate_pair_frac"], torch.tensor(10.0 / 12.0)
    )
    torch.testing.assert_close(
        stats["semantic_alias_valid_pair_count"], torch.tensor(4.0)
    )
    # The selected distances average roughly 106 m; selecting unsafe aliases
    # would make this much smaller.
    assert 90.0 < stats["semantic_alias_selected_geo_distance_m"] < 120.0
    torch.testing.assert_close(
        stats["semantic_alias_random_control"], torch.zeros(())
    )
    assert "semantic_alias_loss" not in stats
    assert all(key.startswith("semantic_alias_") for key in stats)

    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    # CLIP is a detached selector, never an optimization target.
    assert clip_embeddings.grad is None


def test_clip_mining_is_invariant_to_noncontiguous_batch_layout() -> None:
    student, clip_embeddings, labels, coordinates = _controlled_alias_batch()
    objective = CLIPSemanticAliasLoss()
    expected_loss, expected_stats = objective(
        student, clip_embeddings, labels, coordinates
    )

    permutation = torch.tensor([5, 0, 7, 2, 4, 1, 6, 3])
    actual_loss, actual_stats = objective(
        student[permutation],
        clip_embeddings[permutation],
        labels[permutation],
        coordinates[permutation],
    )

    torch.testing.assert_close(actual_loss, expected_loss)
    assert actual_stats.keys() == expected_stats.keys()
    for key in actual_stats:
        torch.testing.assert_close(actual_stats[key], expected_stats[key])


def test_random_control_is_fixed_and_does_not_consume_global_rng() -> None:
    student, clip_embeddings, labels, coordinates = _controlled_alias_batch()
    objective = CLIPSemanticAliasLoss(selection="random", negative_topk=2)

    torch.manual_seed(12345)
    rng_before = torch.random.get_rng_state().clone()
    first_loss, first_stats = objective(
        student, clip_embeddings, labels, coordinates
    )
    rng_after = torch.random.get_rng_state()
    torch.testing.assert_close(rng_after, rng_before)

    # Changing the global seed cannot change the fixed random-control mining.
    torch.manual_seed(9876)
    second_loss, second_stats = objective(
        student, clip_embeddings, labels, coordinates
    )
    torch.testing.assert_close(first_loss, second_loss)
    for key in first_stats:
        torch.testing.assert_close(first_stats[key], second_stats[key])
    torch.testing.assert_close(
        first_stats["semantic_alias_random_control"], torch.ones(())
    )
    torch.testing.assert_close(
        first_stats["semantic_alias_valid_pair_count"], torch.tensor(8.0)
    )
    priority = objective._random_control_priority(torch.tensor([10, 20, 30, 40]))
    torch.testing.assert_close(priority, priority.t())


def test_shuffled_control_preserves_symmetric_mining_structure() -> None:
    student, clip_embeddings, labels, coordinates = _controlled_alias_batch()
    objective = CLIPSemanticAliasLoss(selection="shuffled", negative_topk=1)

    loss, stats = objective(student, clip_embeddings, labels, coordinates)

    assert torch.isfinite(loss)
    torch.testing.assert_close(
        stats["semantic_alias_random_control"], torch.zeros(())
    )
    torch.testing.assert_close(
        stats["semantic_alias_shuffled_control"], torch.ones(())
    )
    torch.testing.assert_close(
        stats["semantic_alias_valid_pair_count"], torch.tensor(4.0)
    )


def test_nonfinite_gps_and_no_valid_pair_return_graph_connected_zero() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    clip_embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    labels = torch.tensor([0, 1])
    coordinates = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])

    loss, stats = CLIPSemanticAliasLoss()(student, clip_embeddings, labels, coordinates)

    torch.testing.assert_close(loss, torch.zeros(()))
    assert loss.grad_fn is not None
    assert all(value.ndim == 0 and torch.isfinite(value) for value in stats.values())
    torch.testing.assert_close(
        stats["semantic_alias_valid_place_frac"], torch.zeros(())
    )
    torch.testing.assert_close(
        stats["semantic_alias_valid_pair_count"], torch.zeros(())
    )
    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


def test_geo_filter_uses_minimum_view_distance_not_place_centroid() -> None:
    student = F.normalize(torch.randn(4, 3), dim=-1).requires_grad_()
    clip_embeddings = F.normalize(torch.randn(4, 3), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    # Centroids are about 67 m apart, but the closest cross-place views are
    # only about 22 m apart. A centroid-only filter would admit this pair.
    coordinates = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0010], [0.0, 0.0002], [0.0, 0.0020]],
        dtype=torch.float64,
    )

    loss, stats = CLIPSemanticAliasLoss(min_geo_distance_m=50.0)(
        student, clip_embeddings, labels, coordinates
    )

    torch.testing.assert_close(loss, torch.zeros(()))
    torch.testing.assert_close(
        stats["semantic_alias_candidate_pair_count"], torch.zeros(())
    )


def test_nonfinite_clip_place_is_removed_from_candidate_pool() -> None:
    student = F.normalize(torch.randn(4, 3), dim=-1).requires_grad_()
    clip_embeddings = F.normalize(torch.randn(4, 3), dim=-1)
    clip_embeddings[2, 0] = torch.nan
    labels = torch.tensor([0, 0, 1, 1])
    coordinates = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.001], [0.0, 0.001]]
    )

    loss, stats = CLIPSemanticAliasLoss()(student, clip_embeddings, labels, coordinates)

    assert torch.isfinite(loss)
    torch.testing.assert_close(
        stats["semantic_alias_candidate_pair_count"], torch.zeros(())
    )


def test_topk_is_clamped_and_same_place_is_never_a_candidate() -> None:
    student = torch.eye(3)
    clip_embeddings = torch.eye(3)
    labels = torch.tensor([4, 5, 6])
    coordinates = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

    _, stats = CLIPSemanticAliasLoss(
        negative_topk=99, min_geo_distance_m=0.0
    )(student, clip_embeddings, labels, coordinates)

    # Three places have exactly two different-place candidates each.
    torch.testing.assert_close(
        stats["semantic_alias_candidate_pair_count"], torch.tensor(6.0)
    )
    torch.testing.assert_close(
        stats["semantic_alias_valid_pair_count"], torch.tensor(6.0)
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"selection": "student"}, ValueError),
        ({"negative_topk": 0}, ValueError),
        ({"negative_topk": 1.5}, TypeError),
        ({"min_geo_distance_m": math.nan}, ValueError),
        ({"min_geo_distance_m": -1.0}, ValueError),
        ({"student_margin": 1.01}, ValueError),
        ({"loss_temperature": 0.0}, ValueError),
    ],
)
def test_constructor_rejects_invalid_settings(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        CLIPSemanticAliasLoss(**kwargs)


def test_forward_rejects_malformed_inputs() -> None:
    objective = CLIPSemanticAliasLoss()
    with pytest.raises(ValueError, match="coordinates"):
        objective(
            torch.randn(2, 3),
            torch.randn(2, 4),
            torch.tensor([0, 1]),
            torch.randn(2, 3),
        )
    with pytest.raises(TypeError, match="labels"):
        objective(
            torch.randn(2, 3),
            torch.randn(2, 4),
            torch.tensor([0.0, 1.0]),
            torch.randn(2, 2),
        )


class _FakeGlobalTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.register_buffer(
            "global_embeddings",
            F.normalize(
                torch.tensor(
                    [[1.0, 0.0], [1.0, 0.1], [0.9, 0.2], [0.8, 0.3]]
                ),
                dim=-1,
            ),
        )

    def forward(self, images: torch.Tensor, return_attn: bool = False):
        assert not return_attn
        batch_size = images.shape[0]
        globals_ = self.global_embeddings[:batch_size] * self.scale
        tokens = torch.zeros(batch_size, 1, 2, device=images.device)
        return globals_, tokens


def test_distillation_module_integrates_alias_loss_without_teacher_gradient() -> None:
    teacher = _FakeGlobalTeacher()
    module = DistillationModule(
        teacher=teacher,
        teacher_token_dim=2,
        teacher_global_dim=2,
        student_feat_channels=1,
        student_global_dim=2,
        distill_mode="global_only",
        semantic_alias_enabled=True,
        semantic_alias_min_geo_distance_m=50.0,
    )
    module.train()
    assert not teacher.training

    student = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        requires_grad=True,
    )
    output = module(
        images=torch.randn(4, 3, 8, 8),
        images_aug=None,
        student_featmap=torch.randn(4, 1, 2, 2),
        student_global=student,
        labels=torch.tensor([0, 0, 1, 1]),
        coordinates=torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.001], [0.0, 0.001]]
        ),
        compute_global=False,
        compute_region=False,
    )

    assert output["loss_alias"] > 0
    assert "semantic_alias_selected_clip_sim" in output
    output["loss_alias"].backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.scale.grad is None


def test_framework_unpacks_metadata_without_breaking_legacy_batches() -> None:
    images = torch.randn(2, 2, 3, 4, 4)
    images_aug = torch.randn_like(images)
    labels = torch.tensor([[0, 0], [1, 1]])
    metadata = {"coordinates": torch.randn(2, 2, 2)}

    unpacked = VPRFrameworkDistill._unpack_distillation_batch(
        (images, labels, metadata)
    )
    assert unpacked[0] is images
    assert unpacked[1] is None
    assert unpacked[2] is labels
    assert unpacked[3] is metadata

    augmented = VPRFrameworkDistill._unpack_distillation_batch(
        (images, images_aug, labels, metadata)
    )
    assert augmented[0] is images
    assert augmented[1] is images_aug
    assert augmented[2] is labels
    assert augmented[3] is metadata

    legacy = VPRFrameworkDistill._unpack_distillation_batch((images, labels))
    assert legacy[0] is images
    assert legacy[1] is None
    assert legacy[2] is labels
    assert legacy[3] is None


def test_framework_training_step_routes_metadata_and_alias_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=1)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.conv(images)

    class NormalizeAggregator(nn.Module):
        def forward(self, featmap: torch.Tensor) -> torch.Tensor:
            return F.normalize(featmap.flatten(1), dim=-1)

    class GraphConnectedVPRLoss(nn.Module):
        def forward(self, descriptors: torch.Tensor, labels: torch.Tensor):
            return descriptors.sum() * 0.0 + 1.0, 0.0

    teacher = _FakeGlobalTeacher()
    distill = DistillationModule(
        teacher=teacher,
        teacher_token_dim=2,
        teacher_global_dim=2,
        student_feat_channels=1,
        student_global_dim=4,
        distill_mode="global_only",
        semantic_alias_enabled=True,
    )
    distill.student_global_proj.requires_grad_(False)
    backbone = TinyBackbone()
    framework = VPRFrameworkDistill(
        backbone=backbone,
        aggregator=NormalizeAggregator(),
        loss_function=GraphConnectedVPRLoss(),
        config_dict={},
        distill_module=distill,
        lambda_global=0.0,
        lambda_region=0.0,
        lambda_attn=0.0,
        lambda_alias=0.05,
        distill_warmup_steps=0,
    )
    framework._trainer = SimpleNamespace(global_step=1)
    monkeypatch.setattr(framework, "log", lambda *args, **kwargs: None)

    images = torch.randn(2, 2, 1, 2, 2)
    labels = torch.tensor([[0, 0], [1, 1]])
    metadata = {
        "coordinates": torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.001], [0.0, 0.001]],
            ]
        )
    }
    loss = framework.training_step((images, labels, metadata), batch_idx=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert backbone.conv.weight.grad is not None
    assert backbone.conv.weight.grad.abs().sum() > 0
    assert teacher.scale.grad is None
