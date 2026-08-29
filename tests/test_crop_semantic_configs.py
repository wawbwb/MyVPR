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
    "architecture_only",
    "aligned",
    "wrong_region",
    "wrong_place",
)
FORMAL_CONFIGS = {
    mode: CONFIG_DIR / f"boq_dinov2_crop_semantic_film_{mode}.yaml"
    for mode in FORMAL_MODES
}
PREFLIGHT_CONFIG = (
    CONFIG_DIR / "boq_dinov2_crop_semantic_film_preflight.yaml"
)
ALLOWED_FORMAL_DIFFERENCES = {
    ("distillation", "crop_semantic_film", "mode"),
    ("distillation", "crop_semantic_film", "lambda_target"),
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


def test_formal_crop_semantic_configs_change_only_the_causal_variable() -> None:
    configs = _formal_configs()
    flattened = {mode: _flatten(config) for mode, config in configs.items()}
    reference_keys = set(flattened["architecture_only"])

    assert all(set(values) == reference_keys for values in flattened.values())
    for path in reference_keys - ALLOWED_FORMAL_DIFFERENCES:
        values = {mode: flat[path] for mode, flat in flattened.items()}
        assert len({repr(value) for value in values.values()}) == 1, (
            f"unexpected config difference at {'.'.join(path)}: {values}"
        )

    crop_sections = {
        mode: config["distillation"]["crop_semantic_film"]
        for mode, config in configs.items()
    }
    assert {
        mode: section["mode"] for mode, section in crop_sections.items()
    } == {
        "architecture_only": "architecture_only",
        "aligned": "aligned",
        "wrong_region": "wrong_region",
        "wrong_place": "wrong_place",
    }
    assert crop_sections["architecture_only"]["lambda_target"] == 0.0
    assert all(
        crop_sections[mode]["lambda_target"] == 0.05
        for mode in ("aligned", "wrong_region", "wrong_place")
    )


def test_crop_semantic_configs_pin_preregistered_data_and_model_contract() -> None:
    for mode, config in _formal_configs().items():
        data = config["datamodule"]
        backbone = config["backbone"]
        film = backbone["params"]["crop_semantic_film"]
        aggregator = config["aggregator"]
        distill = config["distillation"]
        crop = distill["crop_semantic_film"]
        teacher = crop["teacher"]
        trainer = config["trainer"]

        assert config["seed"] == 42
        assert data["train_set_name"] == "gsv-cities"
        assert data["cities"] == "all"
        assert data["train_image_size"] == [280, 280]
        assert data["val_image_size"] == [280, 280]
        assert data["augmentation_mode"] == "photometric"
        assert data["batch_size"] == 40
        assert data["img_per_place"] == 4

        assert backbone["class"] == "DinoV2"
        assert backbone["params"]["backbone_name"] == "dinov2_vitb14"
        assert backbone["params"]["num_unfrozen_blocks"] == 2
        assert film == {
            "enabled": True,
            "hidden_dim": 128,
            "semantic_dim": 512,
            "alpha": 0.1,
            "insert_before_last_n_blocks": 2,
        }
        assert aggregator["class"] == "BoQ"
        assert "semantic_num_classes" not in aggregator["params"]

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
        assert distill["query_semantic"]["enabled"] is False
        assert distill["query_semantic"]["lambda_target"] == 0.0
        assert distill["distill_warmup_steps"] == 500

        ru_gate = distill["semantic_region"]
        assert ru_gate["enabled"] is True
        assert ru_gate["mode"] == "repeatability_uniqueness_only"
        assert ru_gate["apply_pretrained_gate"] is True
        assert ru_gate["lambda_target"] == 0.0

        assert crop["enabled"] is True
        assert crop["crop_grid"] == [2, 2]
        assert crop["teacher_chunk_size"] == 20
        assert crop["diagnostic_interval"] == 100
        assert crop["run_tag"] is None
        assert teacher == {
            "model_name": "ViT-B-16",
            "pretrained": "openai",
            "hf_mirror": "https://hf-mirror.com",
        }

        assert trainer["max_epochs"] == 5
        assert trainer["max_steps"] == -1
        assert trainer["warmup"] == 500
        assert trainer["freeze_base"] is False
        assert trainer["init_checkpoint"] is None
        assert trainer["init_checkpoint_sha256"] == AUDITED_RU_SHA256

        # 768 -> 128 -> 512 semantic projection and 128 -> 768 FiLM.
        assert film["hidden_dim"] == 128
        assert film["semantic_dim"] == teacher_output_dim("ViT-B-16")


def teacher_output_dim(model_name: str) -> int:
    """Dimension pinned by the preregistered OpenCLIP teacher."""
    dimensions = {"ViT-B-16": 512}
    return dimensions[model_name]


def test_preflight_is_the_aligned_run_capped_at_exactly_500_steps() -> None:
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
        ("distillation", "crop_semantic_film", "run_tag"),
        ("trainer", "max_steps"),
    }
    assert aligned["trainer"]["max_steps"] == -1
    assert preflight["trainer"]["max_steps"] == 500
    assert preflight["trainer"]["max_epochs"] == 5
    assert preflight["distillation"]["crop_semantic_film"]["mode"] == (
        "aligned"
    )
    assert preflight["distillation"]["crop_semantic_film"]["run_tag"] == (
        "preflight"
    )
