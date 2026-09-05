"""Cache integrity and scoring integration tests; no models or downloads."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_cc_lsa_gate_a import FIXED_VARIANTS, _validate_cache, score_all_pairs
from src.cc_lsa_gate_a import (
    CC_LSA_FEATURE_SCHEMA, CC_LSA_VERSION, file_sha256, implementation_sha256,
)


def _arrays():
    # Three references followed by one query, four local tokens per image.
    local = np.tile(np.eye(4, dtype=np.float16), (4, 1, 1))
    return {
        "candidates": np.asarray([[0, 1, 2]], dtype=np.int32),
        "candidate_scores": np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32),
        "feature_indices": np.arange(4, dtype=np.int64),
        "global_to_feature": np.arange(4, dtype=np.int32),
        "wrong_place_indices": np.asarray([1, 2, 3, 0], dtype=np.int32),
        "ru_best_positive_rank": np.asarray([2], dtype=np.int16),
        "dino_local": local.copy(), "lsa_local": local.copy(),
        "raw_clip_local": local.copy(),
    }


def test_feature_audit_always_rejects_modified_array(tmp_path: Path):
    arrays = _arrays()
    for name, value in arrays.items():
        np.save(tmp_path / f"{name}.npy", value, allow_pickle=False)
    manifest = {
        "schema": CC_LSA_FEATURE_SCHEMA, "version": CC_LSA_VERSION,
        "complete": True, "config_sha256": "fixture",
        "implementation_sha256": implementation_sha256(),
        "num_references": 3, "num_queries": 1, "feature_rows": 4,
        "dino_dim": 4, "semantic_dim": 4,
        "array_sha256": {
            f"{name}.npy": file_sha256(tmp_path / f"{name}.npy") for name in arrays
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"score": {"top_k": 3}, "features": {"local_grid": [2, 2]}}
    _, loaded = _validate_cache(tmp_path, config_sha="fixture", config=config)
    assert loaded["candidates"].tolist() == [[0, 1, 2]]
    del loaded
    np.save(tmp_path / "global_to_feature.npy", np.zeros(4, dtype=np.int32))
    with pytest.raises(ValueError, match="SHA mismatch"):
        _validate_cache(tmp_path, config_sha="fixture", config=config)


def test_all_variants_share_visual_candidates_and_zero_padding():
    scores, random_scores = score_all_pairs(
        _arrays(), num_references=3, tau_lsa=0.5, tau_raw=0.5,
        top_edges=20, token_seed=42, matched_seeds=(0, 1, 2),
        device=torch.device("cpu"),
    )
    assert tuple(scores) == FIXED_VARIANTS
    # Four perfect MNN matches, divided by the registered top-20 denominator.
    np.testing.assert_allclose(scores["aligned_lsa"], [[0.2, 0.2, 0.2]])
    np.testing.assert_allclose(scores["dino_full"], scores["aligned_lsa"])
    np.testing.assert_allclose(scores["raw_clip"], scores["aligned_lsa"])
    np.testing.assert_allclose(random_scores, np.full((3, 1, 3), 0.2))
    assert np.all(scores["token_permutation"] <= scores["aligned_lsa"])
