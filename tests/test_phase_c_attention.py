import pytest
import torch
from torch import nn

from src.core.vpr_framework import VPRFrameworkDistill
from src.models.clip_teacher import CLIPTeacherEncoder
from src.models.distillation import DistillationModule, SpatialAttentionHead


def test_spatial_attention_head_starts_as_identity() -> None:
    head = SpatialAttentionHead(in_channels=8, num_heads=2)
    featmap = torch.randn(3, 8, 4, 5)

    weighted, attention = head(featmap)

    torch.testing.assert_close(weighted, featmap, rtol=1e-6, atol=1e-6)
    expected = torch.full_like(attention, 1.0 / 20.0)
    torch.testing.assert_close(attention, expected, rtol=1e-6, atol=1e-6)


def test_spatial_attention_head_normalises_every_head() -> None:
    torch.manual_seed(7)
    head = SpatialAttentionHead(in_channels=4, num_heads=3)
    with torch.no_grad():
        head.proj.weight.normal_()
        head.proj.bias.normal_()

    _, attention = head(torch.randn(2, 4, 3, 5))

    assert attention.shape == (2, 3, 15)
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones(2, 3),
        rtol=1e-6,
        atol=1e-6,
    )


def test_attention_kl_is_zero_for_matching_distributions() -> None:
    teacher = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    student = teacher.unsqueeze(1).repeat(1, 3, 1)

    loss = DistillationModule.attention_kl_loss(
        student_attn=student,
        teacher_cls_attn=teacher,
        student_hw=(2, 2),
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0.0)


def test_attention_kl_rejects_mismatched_shapes() -> None:
    student = torch.full((2, 1, 4), 0.25)
    teacher = torch.full((2, 4), 0.25)

    with pytest.raises(ValueError, match="batch sizes differ"):
        DistillationModule.attention_kl_loss(student, teacher[:1], (2, 2))

    with pytest.raises(ValueError, match="square grid"):
        DistillationModule.attention_kl_loss(student, torch.full((2, 6), 1 / 6), (2, 2))

    with pytest.raises(ValueError, match="does not match student_hw"):
        DistillationModule.attention_kl_loss(student, teacher, (1, 3))


@pytest.mark.parametrize("batch_first", [True, False])
def test_clip_cls_attention_side_computation_matches_pytorch_mha(
    batch_first: bool,
) -> None:
    """C1 extracts the target without replacing the teacher block forward."""

    class FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln_1 = nn.LayerNorm(8)
            self.attn = nn.MultiheadAttention(
                embed_dim=8,
                num_heads=2,
                dropout=0.0,
                batch_first=batch_first,
            )

    torch.manual_seed(11)
    block = FakeBlock().eval()
    batch_layout = torch.randn(2, 5, 8)
    block_input = batch_layout if batch_first else batch_layout.transpose(0, 1)

    actual = CLIPTeacherEncoder._cls_to_patch_attention(
        block, block_input, batch_first=batch_first
    )

    normalised = block.ln_1(block_input)
    _, all_weights = block.attn(
        normalised,
        normalised,
        normalised,
        need_weights=True,
        average_attn_weights=False,
    )
    expected = all_weights[:, :, 0, 1:].mean(dim=1)
    expected = expected / expected.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_framework_forward_keeps_spatial_head_without_teacher() -> None:
    """The architecture-only control and inference path must not need CLIP."""

    class IdentityBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    class FlattenAggregator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten(1) * self.scale

    class UnusedLoss(nn.Module):
        def forward(self, descriptors, labels):  # pragma: no cover
            raise AssertionError("forward-only test must not compute VPR loss")

    head = SpatialAttentionHead(in_channels=2)
    with torch.no_grad():
        head.proj.weight.fill_(0.5)

    backbone = IdentityBackbone()
    aggregator = FlattenAggregator()
    framework = VPRFrameworkDistill(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=UnusedLoss(),
        config_dict={},
        distill_module=None,
        spatial_attn_head=head,
        lambda_global=0.0,
        lambda_region=0.0,
        lambda_attn=0.0,
    )
    images = torch.randn(2, 2, 3, 4)

    expected_featmap, _ = head(backbone(images))
    expected = aggregator(expected_featmap)

    torch.testing.assert_close(framework(images), expected)
    assert "spatial_attn_head.proj.weight" in framework.state_dict()
    grouped_parameter_ids = {
        id(parameter)
        for group in framework._optimizer_param_groups()
        for parameter in group["params"]
    }
    assert all(id(parameter) in grouped_parameter_ids for parameter in head.parameters())
