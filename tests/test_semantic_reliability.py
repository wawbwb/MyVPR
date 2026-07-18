import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.core.vpr_framework import VPRFrameworkDistill
from src.models.distillation import DistillationModule, SpatialAttentionHead
from src.models.semantic_reliability import VPRSemanticReliabilityTarget


def _semantic_batch():
    e0, e1, e2 = torch.eye(3)
    patch_embeddings = torch.stack(
        [
            torch.stack([e0, e1, e2]),
            torch.stack([e0, e1, -e2]),
            torch.stack([e1, -e0, -e2]),
            torch.stack([e1, -e0, -e2]),
        ]
    )
    global_embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.1],
                [0.9, 0.2],
                [0.8, 0.2],
            ]
        ),
        dim=-1,
    )
    labels = torch.tensor([10, 10, 20, 20])
    return patch_embeddings, global_embeddings, labels


def test_reliability_prefers_stable_place_discriminative_patch() -> None:
    patches, globals_, labels = _semantic_batch()
    builder = VPRSemanticReliabilityTarget(
        temperature=0.1, negative_topk=1, pair_chunk_size=1
    )

    target, valid, stats = builder(patches, globals_, labels)

    assert valid.all()
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(4))
    # Anchor patch 0 matches its positive view but not the semantic negative;
    # patch 1 is common to both places and must receive less reliability.
    assert target[0, 0] > target[0, 1]
    assert stats["reliability_margin"] > 0
    assert 0 < stats["reliability_target_entropy_norm"] <= 1


def test_reliability_is_position_independent_and_chunk_invariant() -> None:
    patches, globals_, labels = _semantic_batch()
    dense = VPRSemanticReliabilityTarget(pair_chunk_size=64)
    chunked = VPRSemanticReliabilityTarget(pair_chunk_size=1)

    expected, expected_valid, _ = dense(patches, globals_, labels)
    actual, actual_valid, _ = chunked(patches, globals_, labels)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_valid, expected_valid)

    # Reordering patches in the other positive view cannot change anchor 0:
    # matching uses semantic nearest neighbours, not corresponding grid cells.
    permuted = patches.clone()
    permuted[1] = permuted[1, torch.tensor([2, 0, 1])]
    permuted_target, _, _ = dense(permuted, globals_, labels)
    torch.testing.assert_close(permuted_target[0], expected[0])

    # Neither batch layout nor a top-k larger than the available negative
    # places may change the result.
    batch_permutation = torch.tensor([2, 0, 3, 1])
    reordered_target, reordered_valid, _ = VPRSemanticReliabilityTarget(
        negative_topk=99
    )(
        patches[batch_permutation],
        globals_[batch_permutation],
        labels[batch_permutation],
    )
    torch.testing.assert_close(reordered_target, expected[batch_permutation])
    torch.testing.assert_close(reordered_valid, expected_valid[batch_permutation])


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0]),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0, 0]),
    ],
)
def test_invalid_short_batches_are_uniform_and_excluded(labels: torch.Tensor) -> None:
    batch_size = labels.numel()
    patches = torch.randn(batch_size, 4, 5)
    globals_ = torch.randn(batch_size, 3)
    builder = VPRSemanticReliabilityTarget()

    target, valid, stats = builder(patches, globals_, labels)

    assert not valid.any()
    torch.testing.assert_close(target, torch.full_like(target, 0.25))
    torch.testing.assert_close(
        stats["reliability_valid_anchor_frac"], torch.zeros(())
    )

    logits = torch.randn(batch_size, 1, 4, requires_grad=True)
    student = logits.softmax(dim=-1)
    loss = DistillationModule.attention_kl_loss(
        student, target, student_hw=(2, 2), valid_mask=valid
    )
    assert torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_positive_only_target_handles_a_single_place() -> None:
    labels = torch.tensor([4, 4, 4])
    patches = torch.randn(3, 4, 6)
    globals_ = torch.randn(3, 3)
    builder = VPRSemanticReliabilityTarget(negative_weight=0.0)

    target, valid, _ = builder(patches, globals_, labels)

    assert valid.all()
    assert torch.isfinite(target).all()
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(3))


def test_zero_fp16_embeddings_remain_finite() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    patches = torch.zeros(4, 4, 6, dtype=torch.float16)
    globals_ = torch.zeros(4, 3, dtype=torch.float16)

    target, valid, stats = VPRSemanticReliabilityTarget(
        temperature=1e-3, pair_chunk_size=1
    )(patches, globals_, labels)

    assert valid.all()
    assert target.dtype == torch.float32
    assert torch.isfinite(target).all()
    assert all(torch.isfinite(value) for value in stats.values())
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(4))


class _FakeSemanticTeacher(nn.Module):
    def __init__(self, patch_tokens: torch.Tensor, global_tokens: torch.Tensor) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.register_buffer("patch_tokens", patch_tokens)
        self.register_buffer("global_tokens", global_tokens)

    def forward(self, images: torch.Tensor, return_attn: bool = False):
        if return_attn:
            raise AssertionError("semantic reliability must not request CLS attention")
        return self.global_tokens * self.scale, self.patch_tokens * self.scale

    def project_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(tokens, dim=-1)


def test_semantic_target_is_frozen_but_student_attention_gets_gradient() -> None:
    torch.manual_seed(17)
    labels = torch.tensor([0, 0, 1, 1])
    patch_tokens = torch.randn(4, 4, 4)
    global_tokens = F.normalize(torch.randn(4, 4), dim=-1)
    teacher = _FakeSemanticTeacher(patch_tokens, global_tokens)
    module = DistillationModule(
        teacher=teacher,
        teacher_token_dim=4,
        teacher_global_dim=4,
        student_feat_channels=2,
        student_global_dim=4,
        distill_mode="global_only",
        attention_target="semantic_reliability",
        reliability_pair_chunk_size=2,
    )
    module.train()
    assert not teacher.training

    logits = torch.randn(4, 1, 4, requires_grad=True)
    output = module(
        images=torch.randn(4, 3, 8, 8),
        images_aug=None,
        student_featmap=torch.randn(4, 2, 2, 2),
        student_global=torch.randn(4, 4),
        student_attn=logits.softmax(dim=-1),
        labels=labels,
        compute_global=False,
        compute_region=False,
    )
    output["loss_attn"].backward()

    assert teacher.scale.grad is None
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert "reliability_valid_anchor_frac" in output


def test_detached_reliability_kl_updates_head_not_backbone() -> None:
    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(2, 2, kernel_size=1)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.conv(images)

    class Aggregator(nn.Module):
        def forward(self, featmap: torch.Tensor) -> torch.Tensor:
            return featmap.flatten(1)

    class UnusedLoss(nn.Module):
        def forward(self, descriptors, labels):  # pragma: no cover
            raise AssertionError("not used by this gradient-isolation test")

    backbone = Backbone()
    head = SpatialAttentionHead(in_channels=2)
    framework = VPRFrameworkDistill(
        backbone=backbone,
        aggregator=Aggregator(),
        loss_function=UnusedLoss(),
        config_dict={},
        distill_module=nn.Identity(),
        spatial_attn_head=head,
        lambda_global=0.0,
        lambda_region=0.0,
        lambda_attn=0.01,
        detach_backbone_for_attn=True,
    )

    _, _, student_attn, raw_featmap = framework._student_forward(
        torch.randn(3, 2, 2, 2)
    )
    distill_attn = framework._attention_for_distillation(raw_featmap, student_attn)
    target = torch.tensor(
        [[0.6, 0.2, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.2, 0.6]]
    )
    loss = DistillationModule.attention_kl_loss(
        distill_attn, target, student_hw=(2, 2)
    )
    backbone_grad, head_grad = torch.autograd.grad(
        loss,
        (backbone.conv.weight, head.proj.weight),
        allow_unused=True,
    )

    assert backbone_grad is None
    assert head_grad is not None
    assert torch.isfinite(head_grad).all()
    assert head_grad.abs().sum() > 0


def test_supervised_framework_forward_never_calls_teacher() -> None:
    patches = torch.randn(2, 4, 4)
    globals_ = F.normalize(torch.randn(2, 4), dim=-1)

    class SpyTeacher(_FakeSemanticTeacher):
        def forward(self, images: torch.Tensor, return_attn: bool = False):
            raise AssertionError("student inference must not execute CLIP")

    class IdentityBackbone(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images

    class FlattenAggregator(nn.Module):
        def forward(self, featmap: torch.Tensor) -> torch.Tensor:
            return featmap.flatten(1)

    teacher = SpyTeacher(patches, globals_)
    distill = DistillationModule(
        teacher=teacher,
        teacher_token_dim=4,
        teacher_global_dim=4,
        student_feat_channels=2,
        student_global_dim=8,
        distill_mode="global_only",
        attention_target="semantic_reliability",
    )
    framework = VPRFrameworkDistill(
        backbone=IdentityBackbone(),
        aggregator=FlattenAggregator(),
        loss_function=nn.Identity(),
        config_dict={},
        distill_module=distill,
        spatial_attn_head=SpatialAttentionHead(in_channels=2),
        lambda_global=0.0,
        lambda_region=0.0,
        lambda_attn=0.01,
    ).eval()

    output = framework(torch.randn(2, 2, 2, 2))
    assert output.shape == (2, 8)
