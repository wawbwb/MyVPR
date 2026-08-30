from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
AUDITED_RU_SHA256 = (
    "38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e"
)
FORMAL_MODES = ("aligned", "global_only", "wrong_region", "wrong_place")
FORMAL_CONFIGS = {
    mode: CONFIG_DIR / f"boq_dinov2_residual_clip_{mode}.yaml"
    for mode in FORMAL_MODES
}
PREFLIGHT_CONFIG = CONFIG_DIR / "boq_dinov2_residual_clip_preflight.yaml"
ALLOWED_FORMAL_DIFFERENCES = {
    ("backbone", "params", "residual_clip_fusion", "mode"),
    ("distillation", "residual_clip", "mode"),
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert isinstance(config, dict), f"{path.name} must contain a mapping"
    return config


def _formal_configs() -> dict[str, dict]:
    return {mode: _load(path) for mode, path in FORMAL_CONFIGS.items()}


def _flatten(
    value: object, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], object]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[tuple[str, ...], object] = {}
    for key, child in value.items():
        flattened.update(_flatten(child, prefix + (str(key),)))
    return flattened


def test_formal_residual_clip_configs_change_only_the_training_control() -> None:
    configs = _formal_configs()
    flattened = {mode: _flatten(config) for mode, config in configs.items()}
    reference_keys = set(flattened["aligned"])

    assert all(set(values) == reference_keys for values in flattened.values())
    for path in reference_keys - ALLOWED_FORMAL_DIFFERENCES:
        values = {mode: flat[path] for mode, flat in flattened.items()}
        assert len({repr(value) for value in values.values()}) == 1, (
            f"unexpected config difference at {'.'.join(path)}: {values}"
        )

    for mode, config in configs.items():
        assert config["backbone"]["params"]["residual_clip_fusion"][
            "mode"
        ] == mode
        assert config["distillation"]["residual_clip"]["mode"] == mode


def test_residual_clip_configs_pin_phase_a_data_model_and_training_scope() -> None:
    for mode, config in _formal_configs().items():
        data = config["datamodule"]
        backbone = config["backbone"]
        fusion = backbone["params"]["residual_clip_fusion"]
        aggregator = config["aggregator"]
        distill = config["distillation"]
        residual = distill["residual_clip"]
        trainer = config["trainer"]

        assert config["seed"] == 42
        assert data == {
            "train_set_name": "gsv-cities",
            "cities": "all",
            "train_image_size": [280, 280],
            "val_image_size": [280, 280],
            "augmentation_mode": "photometric",
            "img_per_place": 4,
            "batch_size": 40,
            "num_workers": 8,
            "val_set_names": ["msls-val"],
        }

        assert backbone["module"] == "src.models.backbones"
        assert backbone["class"] == "DinoV2"
        assert backbone["params"]["backbone_name"] == "dinov2_vitb14"
        assert backbone["params"]["num_unfrozen_blocks"] == 2
        assert "crop_semantic_film" not in backbone["params"]
        assert fusion == {
            "enabled": True,
            "mode": mode,
            "clip_dim": 512,
            "clip_grid_size": [14, 14],
            "views_per_place": 4,
            "clip_chunk_size": 20,
            "encoder": {
                "model_name": "ViT-B-16",
                "pretrained": "openai",
                "hf_mirror": "https://hf-mirror.com",
            },
        }

        assert aggregator["class"] == "BoQ"
        assert aggregator["params"] == {
            "in_channels": None,
            "proj_channels": 384,
            "num_queries": 64,
            "num_layers": 2,
            "row_dim": 32,
        }
        assert config["loss_function"]["class"] == "VPRLossFunction"
        assert config["loss_function"]["params"] == {
            "loss_fn_name": "MultiSimilarityLoss",
            "miner_name": "MultiSimilarityMiner",
        }

        assert distill["enabled"] is True
        assert distill["lambda_global"] == 0.0
        assert distill["lambda_region"] == 0.0
        assert distill["spatial_attn"] == {
            "enabled": False,
            "lambda_kl": 0.0,
        }
        assert distill["semantic_alias"] == {
            "enabled": False,
            "lambda": 0.0,
        }
        assert distill["semantic_positive"] == {
            "enabled": False,
            "lambda": 0.0,
        }
        assert distill["distill_warmup_steps"] == 0
        assert distill["query_semantic"] == {
            "enabled": False,
            "mode": "architecture_only",
            "lambda_target": 0.0,
        }
        assert distill["crop_semantic_film"] == {
            "enabled": False,
            "mode": "architecture_only",
            "lambda_target": 0.0,
        }

        ru_gate = distill["semantic_region"]
        assert ru_gate == {
            "enabled": True,
            "mode": "repeatability_uniqueness_only",
            "apply_pretrained_gate": True,
            "alpha": 0.2,
            "lambda_target": 0.0,
            "match_grid": 10,
            "target_scale": 2.0,
            "place_chunk_size": 8,
        }
        assert residual == {
            "enabled": True,
            "mode": mode,
            "run_tag": None,
            "diagnostic_interval": 100,
        }

        assert trainer == {
            "optimizer": "adamw",
            "lr": 0.0002,
            "wd": 0.001,
            "warmup": 500,
            "max_epochs": 3,
            "max_steps": -1,
            "milestones": [2],
            "lr_mult": 0.1,
            "accelerator": "gpu",
            "devices": [1],
            "precision": "16-mixed",
            "init_checkpoint": None,
            "init_checkpoint_sha256": AUDITED_RU_SHA256,
            "freeze_base": True,
        }


def test_preflight_is_exactly_the_aligned_run_capped_at_500_steps() -> None:
    aligned = _load(FORMAL_CONFIGS["aligned"])
    preflight = _load(PREFLIGHT_CONFIG)
    aligned_flat = _flatten(aligned)
    preflight_flat = _flatten(preflight)

    assert set(aligned_flat) == set(preflight_flat)
    differences = {
        path
        for path in aligned_flat
        if aligned_flat[path] != preflight_flat[path]
    }
    assert differences == {
        ("distillation", "residual_clip", "run_tag"),
        ("trainer", "max_steps"),
    }
    assert preflight["backbone"]["params"]["residual_clip_fusion"][
        "mode"
    ] == "aligned"
    assert preflight["distillation"]["residual_clip"] == {
        "enabled": True,
        "mode": "aligned",
        "run_tag": "preflight",
        "diagnostic_interval": 100,
    }
    assert preflight["trainer"]["max_epochs"] == 3
    assert preflight["trainer"]["max_steps"] == 500
