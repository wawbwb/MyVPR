from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.cc_lsa_config import load_gate_a_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("section,key,value", [
    ("score", "top_k", 50), ("score", "top_edges", 10),
    ("score", "matched_seeds", {"start": 0, "stop": 10}),
    ("thresholds", "minimum_corrections", 1),
    ("features", "local_grid", [20, 20]),
    ("calibration", "semantic_quantile", 0.5),
])
def test_gate_config_rejects_protocol_drift(tmp_path, section, key, value):
    config = load_gate_a_config(ROOT / "config/cc_lsa_gate_a.yaml")
    changed = deepcopy(config)
    changed[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="registered"):
        load_gate_a_config(path)


def test_gate_config_rejects_missing_placebo(tmp_path):
    config = load_gate_a_config(ROOT / "config/cc_lsa_gate_a.yaml")
    config["controls"].remove("wrong_place")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="controls"):
        load_gate_a_config(path)


def test_gate_config_rejects_unknown_keys(tmp_path):
    config = load_gate_a_config(ROOT / "config/cc_lsa_gate_a.yaml")
    config["score"]["fusion_weight"] = 0.1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_gate_a_config(path)
