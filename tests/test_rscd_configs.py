from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
AUDITED_RU_SHA256 = (
    "38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e"
)
FORMAL_MODES = (
    "no_mask",
    "uniform_block",
    "shuffled_semantic",
    "aligned_rscd",
)
FORMAL_CONFIGS = {
    mode: CONFIG_DIR / f"boq_dinov2_rscd_{mode}.yaml"
    for mode in FORMAL_MODES
}
PREFLIGHT_CONFIG = CONFIG_DIR / "boq_dinov2_rscd_preflight.yaml"
ALLOWED_FORMAL_DIFFERENCES = {("distillation", "rscd", "mode")}


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


def test_formal_rscd_configs_change_only_the_mask_policy() -> None:
    configs = _formal_configs()
    flattened = {mode: _flatten(config) for mode, config in configs.items()}
    reference_keys = set(flattened["no_mask"])

    assert all(set(values) == reference_keys for values in flattened.values())
    for path in reference_keys - ALLOWED_FORMAL_DIFFERENCES:
        values = {mode: flat[path] for mode, flat in flattened.items()}
        assert len({repr(value) for value in values.values()}) == 1, (
            f"unexpected config difference at {'.'.join(path)}: {values}"
        )

    assert {
        mode: config["distillation"]["rscd"]["mode"]
        for mode, config in configs.items()
    } == {mode: mode for mode in FORMAL_MODES}


def test_rscd_configs_pin_data_model_cache_and_training_contract() -> None:
    for mode, config in _formal_configs().items():
        data = config["datamodule"]
        backbone = config["backbone"]
        aggregator = config["aggregator"]
        distill = config["distillation"]
        rscd = distill["rscd"]
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

        assert backbone == {
            "module": "src.models.backbones",
            "class": "DinoV2",
            "params": {
                "backbone_name": "dinov2_vitb14",
                "num_unfrozen_blocks": 2,
            },
        }
        assert aggregator == {
            "module": "src.models.aggregators",
            "class": "BoQ",
            "params": {
                "in_channels": None,
                "proj_channels": 384,
                "num_queries": 64,
                "num_layers": 2,
                "row_dim": 32,
            },
        }
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
        assert distill["residual_clip"] == {
            "enabled": False,
            "mode": "aligned",
        }

        assert distill["semantic_region"] == {
            "enabled": True,
            "mode": "repeatability_uniqueness_only",
            "apply_pretrained_gate": True,
            "alpha": 0.2,
            "lambda_target": 0.0,
            "match_grid": 10,
            "target_scale": 2.0,
            "place_chunk_size": 8,
        }
        assert rscd == {
            "enabled": True,
            "mode": mode,
            "cache_dir": (
                ".cache/ade20k_patch_labels/"
                "segformer_b0_ade20k_grid20"
            ),
            "stats_path": (
                ".cache/ade20k_patch_labels/segformer_b0_ade20k_grid20/"
                "rscd_class_stats.json"
            ),
            "min_confidence": 0.5,
            "block_size": 2,
            "max_mask_fraction": 0.15,
            "replacement": "detached_image_mean",
            "lambda_relation": 0.05,
            "random_seed": 42,
            "diagnostic_interval": 100,
            "run_tag": None,
        }

        assert trainer == {
            "optimizer": "adamw",
            "lr": 0.000002,
            "wd": 0.001,
            "warmup": 0,
            "max_epochs": 3,
            "max_steps": -1,
            "milestones": [],
            "lr_mult": 0.1,
            "accelerator": "gpu",
            "devices": [1],
            "precision": "16-mixed",
            "init_checkpoint": None,
            "init_checkpoint_sha256": AUDITED_RU_SHA256,
            "freeze_base": False,
        }


def test_preflight_is_aligned_rscd_with_only_registered_runtime_changes() -> None:
    aligned = _load(FORMAL_CONFIGS["aligned_rscd"])
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
        ("distillation", "rscd", "diagnostic_interval"),
        ("distillation", "rscd", "run_tag"),
        ("trainer", "max_steps"),
    }
    assert preflight["distillation"]["rscd"]["mode"] == "aligned_rscd"
    assert preflight["distillation"]["rscd"]["run_tag"] == "preflight"
    assert preflight["distillation"]["rscd"]["diagnostic_interval"] == 10
    assert preflight["trainer"]["max_steps"] == 500
    assert preflight["trainer"]["max_epochs"] == 3


def test_rscd_uses_low_lr_continuation_not_crop_film_restart_schedule() -> None:
    for config in _formal_configs().values():
        trainer = config["trainer"]
        assert trainer["lr"] == 2e-6
        assert trainer["warmup"] == 0
        assert trainer["milestones"] == []
