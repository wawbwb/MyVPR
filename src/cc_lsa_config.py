"""Exact schema validation for the registered CC-LSA Gate-A config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.cc_lsa_features import CC_LSA_DINO_FEATURE_STAGE, CC_LSA_LOCAL_GRID


CC_LSA_GATE_CONFIG_SCHEMA = "openvpr_cc_lsa_gate_a_config"
CC_LSA_GATE_CONFIG_VERSION = 1
REGISTERED_CONTROLS = (
    "dino_full",
    "raw_clip",
    "token_permutation",
    "wrong_place",
    "dino_count_weight_matched",
)


def _exact(value: Any, expected: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        found = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - found)}, "
            f"unexpected={sorted(found - expected)}"
        )
    return value


def load_gate_a_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _exact(
        config,
        {
            "schema",
            "version",
            "seed",
            "features",
            "calibration",
            "score",
            "thresholds",
            "controls",
        },
        context="Gate-A config",
    )
    if (
        config["schema"] != CC_LSA_GATE_CONFIG_SCHEMA
        or config["version"] != CC_LSA_GATE_CONFIG_VERSION
    ):
        raise ValueError("unsupported CC-LSA Gate-A config schema/version")
    _exact(
        config["features"],
        {"image_size", "local_grid", "dino_feature_stage", "dtype"},
        context="features",
    )
    _exact(
        config["calibration"],
        {
            "gsv_target_cache",
            "max_holdout_images_per_city",
            "hard_negative_scope",
            "semantic_quantile",
        },
        context="calibration",
    )
    _exact(
        config["score"],
        {
            "top_k",
            "top_edges",
            "token_permutation_seed",
            "matched_seeds",
            "matched_percentile",
        },
        context="score",
    )
    _exact(
        config["score"]["matched_seeds"], {"start", "stop"}, context="matched_seeds"
    )
    _exact(
        config["thresholds"],
        {
            "expected_queries",
            "expected_ru_correct",
            "minimum_reachable_errors",
            "minimum_corrections",
            "control_gap_queries",
            "minimum_rank_improved",
        },
        context="thresholds",
    )
    if config["seed"] != 42:
        raise ValueError("registered Gate-A seed is 42")
    if config["features"]["image_size"] != [280, 280]:
        raise ValueError("registered Gate-A image size is 280x280")
    if tuple(config["features"]["local_grid"]) != CC_LSA_LOCAL_GRID:
        raise ValueError("registered Gate-A local grid is 14x14")
    if config["features"]["dino_feature_stage"] != CC_LSA_DINO_FEATURE_STAGE:
        raise ValueError("registered Gate-A DINO feature stage differs")
    if config["features"]["dtype"] != "float16":
        raise ValueError("registered Gate-A cache dtype is float16")
    calibration = config["calibration"]
    if calibration["max_holdout_images_per_city"] != 128:
        raise ValueError("registered calibration uses 128 holdout images per city")
    if calibration["hard_negative_scope"] != "same_city_different_place_ru_top1":
        raise ValueError("registered calibration hard-negative scope differs")
    if float(calibration["semantic_quantile"]) != 0.95:
        raise ValueError("registered semantic quantile is 0.95")
    score = config["score"]
    if score["top_k"] != 100 or score["top_edges"] != 20:
        raise ValueError("registered Gate-A uses top_k=100 and top_edges=20")
    if score["token_permutation_seed"] != 42:
        raise ValueError("registered token permutation seed is 42")
    if score["matched_seeds"] != {"start": 0, "stop": 100}:
        raise ValueError("registered matched seeds are 0..99")
    if score["matched_percentile"] != 95:
        raise ValueError("registered matched percentile is 95 with higher method")
    if tuple(config["controls"]) != REGISTERED_CONTROLS:
        raise ValueError("Gate-A controls or their registered order differ")
    thresholds = config["thresholds"]
    expected_thresholds = {
        "expected_queries": 740,
        "expected_ru_correct": 675,
        "minimum_reachable_errors": 8,
        "minimum_corrections": 8,
        "control_gap_queries": 4,
        "minimum_rank_improved": 37,
    }
    if thresholds != expected_thresholds:
        raise ValueError("registered Gate-A thresholds differ")
    return config
