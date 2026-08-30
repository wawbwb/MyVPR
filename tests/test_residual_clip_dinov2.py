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

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
