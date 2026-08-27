from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from argparser import update_config_with_args_and_defaults
from src.models.query_semantic import (
    freeze_for_query_semantic_screen,
    warm_start_query_semantic_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
AUDITED_RU_SHA256 = (
    "38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e"
)
QUERY_CONFIGS = {
    mode: CONFIG_DIR / f"boq_dinov2_query_semantic_{mode}.yaml"
    for mode in ("architecture_only", "aligned", "shuffled", "random")
}
ALLOWED_CONFIG_DIFFERENCES = {
    ("distillation", "query_semantic", "mode"),
    ("distillation", "query_semantic", "cache_dir"),
    ("distillation", "query_semantic", "lambda_target"),
}


def _load_query_configs() -> dict[str, dict]:
    configs = {}
    for mode, path in QUERY_CONFIGS.items():
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        assert isinstance(config, dict), f"{path.name} must contain a mapping"
        configs[mode] = config
    return configs


def _flatten_mapping(
    value: object, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], object]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[tuple[str, ...], object] = {}
    for key, child in value.items():
        flattened.update(_flatten_mapping(child, prefix + (str(key),)))
    return flattened


def _namespace(**overrides: object) -> argparse.Namespace:
    values = {
        "config": None,
        "train": False,
        "seed": None,
        "silent": False,
        "compile": False,
        "dev": False,
        "display_theme": None,
        "train_set": None,
        "val_sets": None,
        "train_image_size": None,
        "val_image_size": None,
        "batch_size": None,
        "img_per_place": None,
        "num_workers": None,
        "backbone": None,
        "aggregator": None,
        "loss_function": None,
        "optimizer": None,
        "lr": None,
        "wd": None,
        "warmup": None,
        "milestones": None,
        "lr_mult": None,
        "max_epochs": None,
        "accelerator": None,
        "devices": None,
        "precision": None,
        "init_checkpoint": None,
        "freeze_base": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_query_semantic_configs_differ_only_in_control_fields() -> None:
    configs = _load_query_configs()
    flattened = {mode: _flatten_mapping(config) for mode, config in configs.items()}

    key_sets = {mode: set(values) for mode, values in flattened.items()}
    reference_keys = key_sets["architecture_only"]
    assert all(keys == reference_keys for keys in key_sets.values())

    for path in reference_keys - ALLOWED_CONFIG_DIFFERENCES:
        values = {mode: flat[path] for mode, flat in flattened.items()}
        assert len({repr(value) for value in values.values()}) == 1, (
            f"unexpected config difference at {'.'.join(path)}: {values}"
        )

    query_sections = {
        mode: config["distillation"]["query_semantic"]
        for mode, config in configs.items()
    }
    assert {
        mode: section["mode"] for mode, section in query_sections.items()
    } == {
        "architecture_only": "architecture_only",
        "aligned": "aligned",
        "shuffled": "shuffled",
        "random": "random",
    }
    cache_path = ".cache/ade20k_patch_labels/segformer_b0_ade20k_grid20"
    assert query_sections["architecture_only"]["cache_dir"] is None
    assert all(
        query_sections[mode]["cache_dir"] == cache_path
        for mode in ("aligned", "shuffled", "random")
    )
    assert query_sections["architecture_only"]["lambda_target"] == 0.0
    assert all(
        query_sections[mode]["lambda_target"] == 0.05
        for mode in ("aligned", "shuffled", "random")
    )

    for mode, config in configs.items():
        query = query_sections[mode]
        assert config["seed"] == 42
        assert config["datamodule"]["augmentation_mode"] == "photometric"
        assert config["aggregator"]["params"]["semantic_num_classes"] == 150
        assert query["num_classes"] == 150
        assert config["distillation"]["semantic_region"][
            "apply_pretrained_gate"
        ] is True
        assert config["distillation"]["semantic_region"][
            "lambda_target"
        ] == 0.0
        assert config["trainer"]["freeze_base"] is True
        assert config["trainer"]["init_checkpoint"] is None
        assert (
            config["trainer"]["init_checkpoint_sha256"]
            == AUDITED_RU_SHA256
        )


def test_dinov2_hub_ref_matches_recorded_ru_training_path() -> None:
    source = (
        REPO_ROOT / "src" / "models" / "backbones" / "dinov2.py"
    ).read_text(encoding="utf-8")
    assert "facebookresearch/dinov2:main" in source


def test_query_semantic_cli_warm_start_arguments_override_yaml() -> None:
    config = {
        "trainer": {
            "init_checkpoint": "from-yaml.ckpt",
            "freeze_base": False,
        }
    }
    merged = update_config_with_args_and_defaults(
        config,
        _namespace(init_checkpoint="from-cli.ckpt", freeze_base=True),
    )
    assert merged["trainer"]["init_checkpoint"] == "from-cli.ckpt"
    assert merged["trainer"]["freeze_base"] is True


def test_query_semantic_cli_defaults_preserve_yaml_warm_start_settings() -> None:
    config = {
        "trainer": {
            "init_checkpoint": "from-yaml.ckpt",
            "freeze_base": True,
        }
    }
    merged = update_config_with_args_and_defaults(config, _namespace())
    assert merged["trainer"]["init_checkpoint"] == "from-yaml.ckpt"
    assert merged["trainer"]["freeze_base"] is True


class _SemanticBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.legacy = nn.Linear(2, 2)
        self.semantic_query_proj = nn.Linear(2, 3)


class _SemanticAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.legacy = nn.Linear(2, 2)
        self.semantic_head = nn.Conv2d(2, 3, kernel_size=1)
        self.boqs = nn.ModuleList([_SemanticBlock()])

    def semantic_parameters(self):
        yield from self.semantic_head.parameters()
        yield from self.boqs[0].semantic_query_proj.parameters()


class _QuerySemanticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.aggregator = _SemanticAggregator()
        self.semantic_region_gate = nn.Conv2d(2, 1, kernel_size=1)


def _legacy_ru_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("aggregator.semantic_head.")
        and not (
            key.startswith("aggregator.boqs.")
            and ".semantic_query_proj." in key
        )
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


def test_ru_warm_start_allows_only_new_query_semantic_keys(tmp_path: Path) -> None:
    model = _QuerySemanticModel()
    state = _legacy_ru_state(model)
    checkpoint_path = tmp_path / "ru.ckpt"
    torch.save(
        {
            "state_dict": {
                f"_orig_mod.{key}": value for key, value in state.items()
            },
            "hyper_parameters": _ru_hyper_parameters(),
        },
        checkpoint_path,
    )

    report = warm_start_query_semantic_model(model, checkpoint_path)

    expected_new = {
        key
        for key in model.state_dict()
        if key.startswith("aggregator.semantic_head.")
        or (
            key.startswith("aggregator.boqs.")
            and ".semantic_query_proj." in key
        )
    }
    assert set(report["new_keys"]) == expected_new
    assert report["loaded_keys"] == len(state)


def test_ru_warm_start_rejects_wrong_checkpoint_hash(tmp_path: Path) -> None:
    model = _QuerySemanticModel()
    checkpoint_path = tmp_path / "ru.ckpt"
    torch.save(
        {
            "state_dict": _legacy_ru_state(model),
            "hyper_parameters": _ru_hyper_parameters(),
        },
        checkpoint_path,
    )
    actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    wrong = ("0" if actual[0] != "0" else "1") + actual[1:]

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        warm_start_query_semantic_model(
            model, checkpoint_path, expected_sha256=wrong
        )


@pytest.mark.parametrize("failure", ["missing_legacy", "unexpected"])
def test_ru_warm_start_rejects_non_semantic_key_mismatches(
    tmp_path: Path, failure: str
) -> None:
    model = _QuerySemanticModel()
    state = _legacy_ru_state(model)
    if failure == "missing_legacy":
        state.pop("backbone.bias")
    else:
        state["unexpected.weight"] = torch.zeros(1)
    checkpoint_path = tmp_path / f"{failure}.ckpt"
    torch.save(
        {
            "state_dict": state,
            "hyper_parameters": _ru_hyper_parameters(),
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="unsafe RU warm start"):
        warm_start_query_semantic_model(model, checkpoint_path)


def test_ru_warm_start_rejects_wrong_experiment_provenance(
    tmp_path: Path,
) -> None:
    model = _QuerySemanticModel()
    provenance = _ru_hyper_parameters()
    provenance["distillation"]["semantic_region"]["mode"] = "full"
    checkpoint_path = tmp_path / "full.ckpt"
    torch.save(
        {
            "state_dict": _legacy_ru_state(model),
            "hyper_parameters": provenance,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="not repeatability"):
        warm_start_query_semantic_model(model, checkpoint_path)


def test_ru_warm_start_rejects_partially_existing_semantic_branch(
    tmp_path: Path,
) -> None:
    model = _QuerySemanticModel()
    state = _legacy_ru_state(model)
    state["aggregator.semantic_head.weight"] = (
        model.aggregator.semantic_head.weight.detach().clone()
    )
    checkpoint_path = tmp_path / "partial-query.ckpt"
    torch.save(
        {
            "state_dict": state,
            "hyper_parameters": _ru_hyper_parameters(),
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="must not contain query-semantic"):
        warm_start_query_semantic_model(model, checkpoint_path)


def test_frozen_query_semantic_screen_leaves_only_adapters_trainable() -> None:
    model = _QuerySemanticModel()

    trainable = freeze_for_query_semantic_screen(model)

    assert trainable
    assert model._query_semantic_base_frozen is True
    assert set(trainable) == {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert all(
        name.startswith("aggregator.semantic_head.")
        or (
            name.startswith("aggregator.boqs.")
            and ".semantic_query_proj." in name
        )
        for name in trainable
    )
