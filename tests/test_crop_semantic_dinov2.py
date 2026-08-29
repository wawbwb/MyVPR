from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.backbones.dinov2 import DinoV2


class _FakeBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + 0.01 * self.projection(tokens)


class _FakeDino(nn.Module):
    def __init__(self, width: int = 8, block_count: int = 4) -> None:
        super().__init__()
        self.embed_dim = width
        self.patch_embed = nn.Conv2d(
            3, width, kernel_size=14, stride=14, bias=False
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            [_FakeBlock(width) for _ in range(block_count)]
        )

    def prepare_tokens_with_masks(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls_token + self.pos_embed, patches), dim=1)


def _build(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True) -> DinoV2:
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *args, **kwargs: _FakeDino(),
    )
    film = (
        {
            "enabled": True,
            "hidden_dim": 4,
            "semantic_dim": 6,
            "alpha": 0.1,
            "insert_before_last_n_blocks": 2,
        }
        if enabled
        else None
    )
    return DinoV2(
        backbone_name="dinov2_vitb14",
        num_unfrozen_blocks=2,
        crop_semantic_film=film,
    )


def test_dinov2_inserts_film_before_last_two_blocks_with_exact_zero_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = _build(monkeypatch)
    images = torch.randn(2, 3, 28, 28)

    enabled = backbone(images)
    with backbone.crop_semantic_film.bypass():
        bypassed = backbone(images)

    assert torch.equal(enabled, bypassed)
    assert enabled.shape == (2, 8, 2, 2)
    assert all(
        not parameter.requires_grad
        for block in backbone.dino.blocks[:2]
        for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad
        for block in backbone.dino.blocks[-2:]
        for parameter in block.parameters()
    )


def test_dinov2_training_path_returns_semantics_but_inference_skips_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = _build(monkeypatch)
    images = torch.randn(2, 3, 28, 28)
    semantic_head = backbone.crop_semantic_film.semantic_projection
    original_forward = semantic_head.forward

    def fail_if_called(*args, **kwargs):
        raise AssertionError("teacher-only semantic head ran at inference")

    semantic_head.forward = fail_if_called
    inference_features = backbone(images)
    semantic_head.forward = original_forward

    features, semantic_tokens, raw_scale = (
        backbone.forward_with_crop_semantics(
            images, semantic_batch_indices=torch.tensor([1])
        )
    )
    assert inference_features.shape == features.shape == (2, 8, 2, 2)
    assert semantic_tokens.shape == (1, 4, 6)
    assert raw_scale.shape == (2, 4, 8)
    assert any(
        key.startswith("crop_semantic_film.")
        for key in backbone.state_dict()
    )


def test_dinov2_rejects_a_film_insertion_that_skips_unfrozen_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *args, **kwargs: _FakeDino(),
    )
    with pytest.raises(ValueError, match="immediately before"):
        DinoV2(
            backbone_name="dinov2_vitb14",
            num_unfrozen_blocks=2,
            crop_semantic_film={
                "enabled": True,
                "insert_before_last_n_blocks": 1,
            },
        )
