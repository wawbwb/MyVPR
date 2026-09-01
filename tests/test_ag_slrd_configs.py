from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
CONFIGS = {
    mode: CONFIG_DIR / f"ag_slrd_semantic_teacher_{mode}.yaml"
    for mode in ("aligned", "shuffled")
}
ALLOWED_DIFFERENCES = {
    ("mode",),
    ("trainer", "output_dir"),
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert isinstance(config, dict), f"{path.name} must contain a mapping"
    return config


def _flatten(
    value: object, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], object]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[tuple[str, ...], object] = {}
    for key, child in value.items():
        flattened.update(_flatten(child, prefix + (str(key),)))
    return flattened


def _configs() -> dict[str, dict]:
    return {mode: _load(path) for mode, path in CONFIGS.items()}


def test_teacher_controls_differ_only_in_registered_fields() -> None:
    configs = _configs()
    flattened = {mode: _flatten(config) for mode, config in configs.items()}
    reference_keys = set(flattened["aligned"])

    assert all(set(values) == reference_keys for values in flattened.values())
    for path in reference_keys - ALLOWED_DIFFERENCES:
        values = {mode: flat[path] for mode, flat in flattened.items()}
        assert len({repr(value) for value in values.values()}) == 1, (
            f"unexpected config difference at {'.'.join(path)}: {values}"
        )

    assert {mode: config["mode"] for mode, config in configs.items()} == {
        "aligned": "aligned",
        "shuffled": "shuffled",
    }
    assert configs["aligned"]["trainer"]["output_dir"].endswith(
        "/aligned"
    )
    assert configs["shuffled"]["trainer"]["output_dir"].endswith(
        "/shuffled"
    )


def test_teacher_configs_pin_cache_and_split_contract() -> None:
    expected_cache = {
        "dir": ".cache/ade20k_semantic_layout/gsv_grid70",
        "expected_schema": "openvpr_ade20k_semantic_layout",
        "expected_version": 1,
        "expected_grid": [70, 70],
        "expected_num_classes": 12,
    }
    expected_data = {
        "dataset_root": "datasets/gsv_cities",
        "cities": "all",
        "views_per_place": 4,
        "places_per_batch": 40,
        "num_workers": 8,
        "split_algorithm": "sha256_place_v1",
        "holdout_modulus": 10,
        "holdout_remainder": 0,
    }
    for config in _configs().values():
        assert config["seed"] == 42
        assert config["cache"] == expected_cache
        assert config["data"] == expected_data


def test_teacher_configs_pin_model_and_training_contract() -> None:
    expected_model = {
        "num_classes": 12,
        "embed_dim": 32,
        "channels": [64, 128, 256],
        "descriptor_dim": 512,
        "ignore_index": 255,
    }
    expected_trainer_common = {
        "epochs": 10,
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "precision": "16-mixed",
        "loss_precision": "32-true",
        "max_grad_norm": 10.0,
        "amp": {
            "init_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "max_retries_per_batch": 8,
            "max_total_retries": 32,
        },
        "device": "cuda:1",
    }
    for mode, config in _configs().items():
        assert config["model"] == expected_model
        trainer = dict(config["trainer"])
        assert trainer.pop("output_dir") == (
            f"logs/ag_slrd_semantic_teacher/{mode}"
        )
        assert trainer == expected_trainer_common


def test_teacher_config_is_phase_zero_only() -> None:
    forbidden_keys = {
        "backbone",
        "aggregator",
        "distillation",
        "init_checkpoint",
        "lambda_relation",
    }
    for config in _configs().values():
        assert forbidden_keys.isdisjoint(config)
        flattened_names = {".".join(path) for path in _flatten(config)}
        assert not any("msls" in name.lower() for name in flattened_names)
