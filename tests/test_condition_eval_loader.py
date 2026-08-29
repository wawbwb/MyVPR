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
