from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.backbones.dinov2 import DinoV2
from src.models.residual_clip_fusion import ResidualCLIPFeatureProvider


class _AddBlock(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.register_buffer("delta", torch.tensor(float(delta)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.delta.to(dtype=tokens.dtype)


class _FakeDino(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.embed_dim = width
        self.prepare_calls = 0
        self.patch_embed = nn.Conv2d(
            3, width, kernel_size=14, stride=14, bias=False
        )
        nn.init.constant_(self.patch_embed.weight, 0.01)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            [_AddBlock(delta) for delta in (0.1, 0.2, 0.4, 0.8)]
        )

    def prepare_tokens_with_masks(self, images: torch.Tensor) -> torch.Tensor:
        self.prepare_calls += 1
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls_token + self.pos_embed, patches), dim=1)


class _FakeCLIPEncoder(nn.Module):
    def __init__(self, clip_dim: int = 3, patch_count: int = 4) -> None:
        super().__init__()
        # If the provider were accidentally registered, this distinctive
        # parameter name would appear in DinoV2.state_dict().
        self.provider_sentinel = nn.Parameter(torch.tensor(17.0))
        self.clip_dim = int(clip_dim)
        self.patch_count = int(patch_count)
        self.forward_calls = 0

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        image_value = images.float().mean(dim=(1, 2, 3), keepdim=False)
        channels = torch.arange(
            self.clip_dim, device=images.device, dtype=torch.float32
        )
        patches = torch.arange(
            self.patch_count, device=images.device, dtype=torch.float32
        )
        global_features = image_value[:, None] + channels[None]
        raw_patch_tokens = (
            image_value[:, None, None]
            + patches[None, :, None]
            + channels[None, None, :]
        )
        return F.normalize(global_features, dim=-1), raw_patch_tokens

    def project_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(tokens.float(), dim=-1)


def _build_backbone(monkeypatch: pytest.MonkeyPatch) -> tuple[DinoV2, _FakeCLIPEncoder]:
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _FakeDino())
    backbone = DinoV2(
        backbone_name="dinov2_vitb14",
        num_unfrozen_blocks=2,
        return_cls_token=True,
        residual_clip_fusion={
            "enabled": True,
            "clip_dim": 3,
            "clip_grid_size": [2, 2],
            "clip_chunk_size": 1,
            "mode": "aligned",
            "views_per_place": 2,
            "encoder": {"model_name": "fake", "pretrained": "fake"},
        },
    )
    encoder = _FakeCLIPEncoder()
    backbone._residual_clip_provider = ResidualCLIPFeatureProvider(
        encoder=encoder,
        model_name="fake",
        pretrained="fake",
        chunk_size=1,
        expected_clip_dim=3,
        expected_patch_count=4,
    )
    return backbone, encoder


def test_residual_clip_receives_the_final_dino_patch_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone, encoder = _build_backbone(monkeypatch)
    images = torch.linspace(-1.0, 1.0, 2 * 3 * 28 * 28).reshape(
        2, 3, 28, 28
    )

    with torch.no_grad():
        expected_tokens = backbone.dino.prepare_tokens_with_masks(images)
        for block in backbone.dino.blocks:
            expected_tokens = block(expected_tokens)
        expected_cls = expected_tokens[:, 0]
        expected_local = (
            expected_tokens[:, 1:]
            .permute(0, 2, 1)
            .contiguous()
            .view(2, backbone.out_channels, 2, 2)
        )

    captured: dict[str, torch.Tensor] = {}

    def capture_fusion_input(module, args) -> None:
        captured["dino"] = args[0].detach().clone()

    hook = backbone.residual_clip_fusion.register_forward_pre_hook(
        capture_fusion_input
    )
    try:
        actual_local, actual_cls = backbone(images)
    finally:
        hook.remove()

    # The zero-initialised residual makes the complete backbone exactly equal
    # to the final DINO output, while the pre-hook proves that this final map
    # (not an intermediate frozen-block map) is what the fusion consumes.
    torch.testing.assert_close(captured["dino"], expected_local)
    assert torch.equal(actual_local, expected_local)
    torch.testing.assert_close(actual_cls, expected_cls)

    state_keys = tuple(backbone.state_dict())
    assert any(key.startswith("residual_clip_fusion.") for key in state_keys)
    assert not any("provider" in key for key in state_keys)
    assert not any("provider_sentinel" in key for key in state_keys)
    assert encoder.provider_sentinel.requires_grad is False
    assert encoder.training is False


def test_component_extraction_is_reusable_and_matches_aligned_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone, encoder = _build_backbone(monkeypatch)
    backbone.eval()
    images = torch.stack(
        (
            torch.full((3, 28, 28), -0.25),
            torch.full((3, 28, 28), 0.75),
        )
    )
    fusion = backbone.residual_clip_fusion
    with torch.no_grad():
        fusion.clip_projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, -0.25, 0.75],
                ]
            )
        )
        fusion.residual_adapter.weight.copy_(torch.eye(4))
        fusion.residual_adapter.bias.zero_()

    raw_feature_map, x_cls, clip_global, clip_patches = (
        backbone.extract_residual_clip_components(images)
    )

    assert backbone.dino.prepare_calls == 1
    # Provider chunk_size=1, so two images require exactly two CLIP calls.
    assert encoder.forward_calls == 2
    assert raw_feature_map.shape == (2, 4, 2, 2)
    assert x_cls.shape == (2, 4)
    assert clip_global.shape == (2, 3)
    assert clip_patches.shape == (2, 4, 3)
    assert not raw_feature_map.requires_grad
    assert not clip_global.requires_grad
    assert not clip_patches.requires_grad

    fused = {}
    for mode in ("aligned", "global_only", "wrong_region"):
        fused[mode], _ = fusion(
            raw_feature_map,
            clip_patches,
            clip_global,
            intervention_mode=mode,
        )

    # Reusing extracted components for three interventions reruns neither
    # encoder.  The deliberately non-zero adapter makes each intervention
    # observably different.
    assert backbone.dino.prepare_calls == 1
    assert encoder.forward_calls == 2
    assert not torch.equal(fused["aligned"], fused["global_only"])
    assert not torch.equal(fused["aligned"], fused["wrong_region"])

    ordinary_local, ordinary_cls = backbone(images)
    torch.testing.assert_close(fused["aligned"], ordinary_local)
    torch.testing.assert_close(x_cls, ordinary_cls)
    assert backbone.dino.prepare_calls == 2
    assert encoder.forward_calls == 4
