"""Contract tests for the Phase-A residual CLIP screen."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.residual_clip_fusion import (
    RESIDUAL_CLIP_NEW_KEY_PREFIX,
    ResidualCLIPFeatureProvider,
    ResidualCLIPFusion,
    freeze_for_residual_clip_screen,
    warm_start_residual_clip_model,
)


def _fusion(mode: str = "aligned") -> ResidualCLIPFusion:
    return ResidualCLIPFusion(
        in_channels=2,
        clip_dim=2,
        clip_grid_size=(2, 2),
        mode=mode,
        views_per_place=2,
    )


def _inputs(batch_size: int = 4):
    generator = torch.Generator().manual_seed(7)
    dino = torch.randn(batch_size, 2, 2, 2, generator=generator)
    patches = torch.randn(batch_size, 4, 2, generator=generator)
    global_features = torch.randn(batch_size, 2, generator=generator)
    return dino, patches, global_features


def _set_identity_residual(module: ResidualCLIPFusion) -> None:
    with torch.no_grad():
        module.clip_projection.weight.copy_(torch.eye(2))
        module.residual_adapter.weight.copy_(torch.eye(2))
        module.residual_adapter.bias.zero_()


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    return features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, 2)


def test_residual_clip_is_exact_zero_start() -> None:
    module = _fusion()
    dino, patches, global_features = _inputs()

    fused, residual = module(dino, patches, global_features)

    assert torch.equal(fused, dino)
    assert torch.count_nonzero(residual) == 0
    assert torch.count_nonzero(module.residual_adapter.weight) == 0
    assert torch.count_nonzero(module.residual_adapter.bias) == 0
    diagnostics = module.diagnostics()
    assert diagnostics["residual_clip_residual_rms"].item() == 0.0
    assert diagnostics["residual_clip_residual_max_abs"].item() == 0.0


def test_residual_clip_bypass_is_nested_and_exact() -> None:
    module = _fusion()
    _set_identity_residual(module)
    dino, patches, global_features = _inputs()
    changed, raw_residual = module(dino, patches, global_features)
    assert not torch.equal(changed, dino)
    assert torch.count_nonzero(raw_residual) > 0

    with module.bypass():
        with module.bypass():
            bypassed, still_computed = module(dino, patches, global_features)
            assert torch.equal(bypassed, dino)
            assert torch.count_nonzero(still_computed) > 0
            diagnostics = module.diagnostics()
            assert diagnostics["residual_clip_bypassed"].item() == 1.0
            assert diagnostics["residual_clip_residual_rms"].item() == 0.0

    assert not module.bypassed
    restored, _ = module(dino, patches, global_features)
    torch.testing.assert_close(restored, changed)


@pytest.mark.parametrize(
    "mode", ["aligned", "global_only", "wrong_region", "wrong_place"]
)
def test_each_training_control_selects_the_registered_tokens(mode: str) -> None:
    module = _fusion(mode)
    _set_identity_residual(module)
    dino, patches, global_features = _inputs()
    aligned = module._aligned_local_tokens(patches, (2, 2))

    fused, _ = module(
        dino,
        patches,
        global_features,
        apply_training_control=True,
    )

    if mode == "aligned":
        selected = aligned
    elif mode == "global_only":
        selected = F.normalize(global_features.float(), dim=-1)
        selected = selected[:, None].expand(-1, 4, -1)
    elif mode == "wrong_region":
        selected = aligned.index_select(1, torch.tensor([2, 3, 0, 1]))
    else:
        selected = aligned.reshape(2, 2, 4, 2).roll(1, dims=0)
        selected = selected.reshape_as(aligned)

    dino_tokens = _flatten_features(dino)
    expected = dino_tokens + selected - F.normalize(dino_tokens, dim=-1)
    torch.testing.assert_close(_flatten_features(fused), expected)
    expected_applied = 0.0 if mode == "aligned" else 1.0
    assert (
        module.diagnostics()["residual_clip_control_applied"].item()
        == expected_applied
    )


@pytest.mark.parametrize(
    "mode", ["aligned", "global_only", "wrong_region", "wrong_place"]
)
def test_every_checkpoint_uses_aligned_tokens_in_eval(mode: str) -> None:
    aligned = _fusion("aligned").eval()
    candidate = _fusion(mode).eval()
    _set_identity_residual(aligned)
    candidate.load_state_dict(aligned.state_dict())
    dino, patches, global_features = _inputs()

    expected, _ = aligned(
        dino, patches, global_features, apply_training_control=False
    )
    actual, _ = candidate(
        dino, patches, global_features, apply_training_control=candidate.training
    )

    torch.testing.assert_close(actual, expected)
    assert candidate.diagnostics()["residual_clip_control_applied"].item() == 0.0


def test_wrong_place_rolls_whole_places_without_crossing_view_index() -> None:
    module = _fusion("wrong_place")
    place_count, views_per_place, patch_count, channels = 3, 2, 4, 2
    aligned = torch.arange(
        place_count * views_per_place * patch_count * channels,
        dtype=torch.float32,
    ).reshape(place_count * views_per_place, patch_count, channels)
    global_features = torch.ones(place_count * views_per_place, channels)

    selected, applied = module._select_control_tokens(
        aligned, global_features, apply_training_control=True
    )

    selected = selected.reshape(
        place_count, views_per_place, patch_count, channels
    )
    source = aligned.reshape(
        place_count, views_per_place, patch_count, channels
    )
    assert applied
    for destination_place in range(place_count):
        donor_place = (destination_place - 1) % place_count
        for view_index in range(views_per_place):
            assert torch.equal(
                selected[destination_place, view_index],
                source[donor_place, view_index],
            )


def test_zero_start_gradient_reaches_projection_after_adapter_moves() -> None:
    module = _fusion()
    dino, patches, global_features = _inputs()

    fused, _ = module(dino, patches, global_features)
    fused.square().mean().backward()

    assert module.residual_adapter.weight.grad is not None
    assert torch.count_nonzero(module.residual_adapter.weight.grad) > 0
    assert module.residual_adapter.bias.grad is not None
    assert torch.count_nonzero(module.residual_adapter.bias.grad) > 0
    # Exact zero start intentionally blocks P_C on the first backward pass.
    assert module.clip_projection.weight.grad is not None
    assert torch.count_nonzero(module.clip_projection.weight.grad) == 0

    # A direct SGD-equivalent update keeps this contract test independent of
    # optional optimiser/compiler integrations in the host torch build.
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-0.1)
    module.zero_grad(set_to_none=True)
    assert torch.count_nonzero(module.residual_adapter.weight) > 0

    fused, _ = module(dino, patches, global_features)
    fused.square().mean().backward()
    assert module.clip_projection.weight.grad is not None
    assert torch.count_nonzero(module.clip_projection.weight.grad) > 0


class _FakeCLIPEncoder(nn.Module):
    def __init__(self, clip_dim: int = 3, patch_count: int = 2) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.clip_dim = clip_dim
        self.patch_count = patch_count
        self.chunk_sizes: list[int] = []

    def forward(self, images: torch.Tensor):
        self.chunk_sizes.append(images.shape[0])
        image_ids = images[:, 0, 0, 0].float()
        offsets = torch.arange(
            self.clip_dim, device=images.device, dtype=torch.float32
        )
        global_features = image_ids[:, None] + offsets[None]
        raw_patches = global_features[:, None].expand(
            -1, self.patch_count, -1
        )
        return global_features * self.scale, raw_patches * self.scale

    def project_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


def test_provider_chunks_freezes_detaches_and_preserves_order() -> None:
    encoder = _FakeCLIPEncoder()
    provider = ResidualCLIPFeatureProvider(
        encoder=encoder,
        chunk_size=2,
        expected_clip_dim=3,
        expected_patch_count=2,
    )
    images = torch.zeros(5, 3, 2, 2, requires_grad=True)
    images.data[:, 0, 0, 0] = torch.arange(5, dtype=torch.float32)

    global_features, patch_features = provider.encode(images)

    assert encoder.chunk_sizes == [2, 2, 1]
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert not global_features.requires_grad
    assert not patch_features.requires_grad
    torch.testing.assert_close(global_features[:, 0], torch.arange(5.0))
    torch.testing.assert_close(patch_features[:, 0, 0], torch.arange(5.0))
    audit = provider.audit()
    assert audit["prepared"] is True
    assert audit["trainable_parameters"] == 0
    assert audit["training"] is False


def test_plain_provider_encoder_is_excluded_from_student_state() -> None:
    provider = ResidualCLIPFeatureProvider(encoder=_FakeCLIPEncoder())

    class _Owner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fusion = _fusion()
            self.provider = provider

    owner = _Owner()
    state = owner.state_dict()

    assert state
    assert set(state) == {
        "fusion.clip_projection.weight",
        "fusion.residual_adapter.weight",
        "fusion.residual_adapter.bias",
    }
    assert provider._encoder not in tuple(owner.modules())


class _ResidualBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.legacy = nn.Linear(2, 2)
        self.residual_clip_fusion = _fusion()


class _WarmStartModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ResidualBackbone()
        self.aggregator = nn.Linear(2, 2)
        self.semantic_region_gate = nn.Linear(2, 1)


def _ru_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith(RESIDUAL_CLIP_NEW_KEY_PREFIX)
    }


def _ru_hyper_parameters() -> dict:
    return {
        "seed": 42,
        "datamodule": {
            "train_set_name": "gsv-cities",
            "cities": "all",
            "train_image_size": [280, 280],
            "augmentation_mode": "photometric",
            "batch_size": 40,
            "img_per_place": 4,
        },
        "backbone": {"class": "DinoV2"},
        "aggregator": {"class": "BoQ"},
        "trainer": {"max_epochs": 40},
        "distillation": {
            "semantic_region": {
                "enabled": True,
                "mode": "repeatability_uniqueness_only",
                "lambda_target": 0.02,
                "alpha": 0.2,
            }
        },
    }


def _save_ru_checkpoint(path: Path, model: nn.Module) -> str:
    torch.save(
        {
            "state_dict": _ru_state(model),
            "hyper_parameters": _ru_hyper_parameters(),
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_residual_warm_start_allows_only_new_branch_and_checks_sha(
    tmp_path: Path,
) -> None:
    model = _WarmStartModel()
    checkpoint = tmp_path / "ru.ckpt"
    digest = _save_ru_checkpoint(checkpoint, model)

    report = warm_start_residual_clip_model(
        model, checkpoint, expected_sha256=digest
    )

    expected_new = {
        key
        for key in model.state_dict()
        if key.startswith(RESIDUAL_CLIP_NEW_KEY_PREFIX)
    }
    assert set(report["new_keys"]) == expected_new
    assert report["loaded_keys"] == len(_ru_state(model))
    assert report["residual_zero_initialized"] is True

    wrong_digest = ("0" if digest[0] != "0" else "1") + digest[1:]
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        warm_start_residual_clip_model(
            _WarmStartModel(), checkpoint, expected_sha256=wrong_digest
        )


def test_residual_screen_freezes_everything_except_projection_and_adapter() -> None:
    model = _WarmStartModel()

    trainable = freeze_for_residual_clip_screen(model)

    assert model._residual_clip_base_frozen is True
    assert set(trainable) == {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert set(trainable) == {
        "backbone.residual_clip_fusion.clip_projection.weight",
        "backbone.residual_clip_fusion.residual_adapter.weight",
        "backbone.residual_clip_fusion.residual_adapter.bias",
    }
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
