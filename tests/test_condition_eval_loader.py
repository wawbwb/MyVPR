import torch
from torch import nn

import scripts.eval_condition_robustness as condition_eval
from scripts.eval_condition_robustness import InferenceModel
from src.models.crop_semantic_film import CropSemanticFiLM
from src.models.semantic_region_gate import SemanticRegionGate


def test_condition_inference_applies_semantic_gate_to_tuple_local_features() -> None:
    class TupleBackbone(nn.Module):
        def forward(self, images):
            return images, images.mean(dim=(2, 3))

    class CaptureAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.local = None
            self.cls = None

        def forward(self, output):
            self.local, self.cls = output
            return self.local.flatten(1)

    gate = SemanticRegionGate(2, alpha=0.2)
    with torch.no_grad():
        gate.proj.weight.zero_()
        gate.proj.bias.fill_(10.0)
    aggregator = CaptureAggregator()
    model = InferenceModel(
        TupleBackbone(), aggregator, semantic_region_gate=gate
    )
    images = torch.ones(2, 2, 2, 2)

    model(images)

    torch.testing.assert_close(
        aggregator.local, images * (1.0 + 0.2 * torch.tanh(torch.tensor(10.0)))
    )
    torch.testing.assert_close(aggregator.cls, images.mean(dim=(2, 3)))


class _CropFilmBackbone(nn.Module):
    def __init__(self, crop_semantic_film=None):
        super().__init__()
        self.out_channels = 2
        crop_semantic_film = crop_semantic_film or {}
        self.crop_semantic_film = CropSemanticFiLM(
            in_channels=2,
            hidden_dim=int(crop_semantic_film.get("hidden_dim", 2)),
            semantic_dim=int(crop_semantic_film.get("semantic_dim", 3)),
            alpha=float(crop_semantic_film.get("alpha", 0.1)),
        )

    def forward(self, images):
        tokens = images.flatten(2).transpose(1, 2)
        tokens, semantic, _ = self.crop_semantic_film(
            tokens, return_semantic=False
        )
        assert semantic is None
        return tokens.transpose(1, 2).reshape_as(images)


class _CropFilmAggregator(nn.Module):
    def __init__(self, in_channels=None):
        super().__init__()
        self.projection = nn.Linear(int(in_channels), 4)

    def forward(self, features):
        return self.projection(features.mean(dim=(2, 3)))


def test_condition_loader_strictly_restores_crop_film_backbone(
    tmp_path, monkeypatch
) -> None:
    film_config = {
        "enabled": True,
        "hidden_dim": 2,
        "semantic_dim": 3,
        "alpha": 0.1,
        "insert_before_last_n_blocks": 2,
    }
    backbone = _CropFilmBackbone(film_config)
    aggregator = _CropFilmAggregator(backbone.out_channels)
    gate = SemanticRegionGate(backbone.out_channels, alpha=0.2)
    with torch.no_grad():
        backbone.crop_semantic_film.channel_scale.bias.fill_(0.3)
    source = InferenceModel(
        backbone, aggregator, semantic_region_gate=gate
    ).eval()
    checkpoint_path = tmp_path / "crop-film.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "backbone": {
                    "module": "unused",
                    "class": "CropFilmBackbone",
                    "params": {"crop_semantic_film": film_config},
                },
                "aggregator": {
                    "module": "unused",
                    "class": "CropFilmAggregator",
                    "params": {"in_channels": None},
                },
                "distillation": {
                    "semantic_region": {
                        "enabled": True,
                        "apply_pretrained_gate": True,
                        "lambda_target": 0.0,
                        "alpha": 0.2,
                    }
                },
            },
            "state_dict": source.state_dict(),
        },
        checkpoint_path,
    )

    def fake_get_instance(module_name, class_name, params):
        if class_name == "CropFilmBackbone":
            return _CropFilmBackbone(**params)
        if class_name == "CropFilmAggregator":
            return _CropFilmAggregator(**params)
        raise AssertionError(class_name)

    monkeypatch.setattr(condition_eval, "get_instance", fake_get_instance)
    restored = condition_eval.load_inference_model_from_ckpt(
        checkpoint_path, torch.device("cpu")
    ).eval()
    images = torch.randn(2, 2, 3, 3)

    with torch.no_grad():
        expected = source(images)
        actual = restored(images)

    assert restored.backbone.crop_semantic_film is not None
    torch.testing.assert_close(actual, expected)


class _ResidualEvalBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, tokens):
        return tokens + 0.01 * self.projection(tokens)


class _ResidualEvalDino(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.embed_dim = width
        self.patch_embed = nn.Conv2d(
            3, width, kernel_size=14, stride=14, bias=False
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            [_ResidualEvalBlock(width) for _ in range(4)]
        )

    def prepare_tokens_with_masks(self, images):
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls_token + self.pos_embed, patches), dim=1)


class _ResidualEvalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.provider_sentinel = nn.Parameter(torch.tensor(23.0))

    def forward(self, images):
        from torch.nn import functional as F

        value = images.float().mean(dim=(1, 2, 3))
        channels = torch.arange(3, device=images.device, dtype=torch.float32)
        patches = torch.arange(4, device=images.device, dtype=torch.float32)
        global_features = F.normalize(value[:, None] + channels[None], dim=-1)
        raw_patches = (
            value[:, None, None]
            + patches[None, :, None]
            + channels[None, None]
        )
        return global_features, raw_patches

    def project_patch_tokens(self, tokens):
        from torch.nn import functional as F

        return F.normalize(tokens.float(), dim=-1)


class _ResidualEvalAggregator(nn.Module):
    def __init__(self, in_channels=None) -> None:
        super().__init__()
        self.projection = nn.Linear(int(in_channels), 5)

    def forward(self, features):
        return self.projection(features.mean(dim=(2, 3)))


def test_condition_loader_strictly_restores_residual_clip_without_provider_state(
    tmp_path, monkeypatch
) -> None:
    import pytest

    from src.models.backbones.dinov2 import DinoV2
    from src.models.residual_clip_fusion import ResidualCLIPFeatureProvider

    residual_config = {
        "enabled": True,
        "clip_dim": 3,
        "clip_grid_size": [2, 2],
        "clip_chunk_size": 2,
        "mode": "aligned",
        "views_per_place": 2,
        "encoder": {"model_name": "fake", "pretrained": "fake"},
    }

    monkeypatch.setattr(
        torch.hub, "load", lambda *args, **kwargs: _ResidualEvalDino()
    )

    def make_backbone(params):
        backbone = DinoV2(**params)
        backbone._residual_clip_provider = ResidualCLIPFeatureProvider(
            encoder=_ResidualEvalEncoder(),
            model_name="fake",
            pretrained="fake",
            chunk_size=2,
            expected_clip_dim=3,
            expected_patch_count=4,
        )
        return backbone

    backbone_params = {
        "backbone_name": "dinov2_vitb14",
        "num_unfrozen_blocks": 2,
        "residual_clip_fusion": residual_config,
    }
    backbone = make_backbone(backbone_params)
    aggregator = _ResidualEvalAggregator(backbone.out_channels)
    gate = SemanticRegionGate(backbone.out_channels, alpha=0.2)
    with torch.no_grad():
        backbone.residual_clip_fusion.clip_projection.weight.fill_(0.125)
        backbone.residual_clip_fusion.residual_adapter.weight.copy_(
            0.05 * torch.eye(backbone.out_channels)
        )
        backbone.residual_clip_fusion.residual_adapter.bias.fill_(0.01)
    source = InferenceModel(
        backbone, aggregator, semantic_region_gate=gate
    ).eval()
    source_state = source.state_dict()

    assert any(
        key.startswith("backbone.residual_clip_fusion.")
        for key in source_state
    )
    assert not any("provider" in key for key in source_state)
    assert not any("provider_sentinel" in key for key in source_state)

    checkpoint = {
        "hyper_parameters": {
            "backbone": {
                "module": "unused",
                "class": "ResidualDinoBackbone",
                "params": backbone_params,
            },
            "aggregator": {
                "module": "unused",
                "class": "ResidualAggregator",
                "params": {"in_channels": None},
            },
            "distillation": {
                "semantic_region": {
                    "enabled": True,
                    "apply_pretrained_gate": True,
                    "lambda_target": 0.0,
                    "alpha": 0.2,
                },
                "residual_clip": {"enabled": True, "mode": "aligned"},
            },
        },
        "state_dict": source_state,
    }
    checkpoint_path = tmp_path / "residual-clip.ckpt"
    torch.save(checkpoint, checkpoint_path)

    def fake_get_instance(module_name, class_name, params):
        if class_name == "ResidualDinoBackbone":
            return make_backbone(params)
        if class_name == "ResidualAggregator":
            return _ResidualEvalAggregator(**params)
        raise AssertionError(class_name)

    monkeypatch.setattr(condition_eval, "get_instance", fake_get_instance)
    restored = condition_eval.load_inference_model_from_ckpt(
        checkpoint_path, torch.device("cpu")
    ).eval()
    images = torch.randn(3, 3, 28, 28)

    with torch.no_grad():
        expected = source(images)
        actual = restored(images)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        restored.backbone.residual_clip_fusion.residual_adapter.weight,
        source.backbone.residual_clip_fusion.residual_adapter.weight,
    )
    assert not any(
        "provider" in key or "provider_sentinel" in key
        for key in restored.backbone.state_dict()
    )

    # Removing one adapter tensor must be rejected by the loader's strict
    # backbone restore instead of silently evaluating the zero-start branch.
    incomplete = dict(checkpoint)
    incomplete["state_dict"] = type(source_state)(source_state)
    del incomplete["state_dict"][
        "backbone.residual_clip_fusion.residual_adapter.bias"
    ]
    incomplete_path = tmp_path / "residual-clip-incomplete.ckpt"
    torch.save(incomplete, incomplete_path)
    with pytest.raises(RuntimeError, match="residual_adapter.bias"):
        condition_eval.load_inference_model_from_ckpt(
            incomplete_path, torch.device("cpu")
        )
