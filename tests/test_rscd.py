from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.models.rscd import (
    RSCDMaskBuilder,
    apply_token_mask,
    load_rscd_stats,
    pairwise_relation_loss,
    warm_start_rscd_model,
)
from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_SCHEMA,
    QUERY_SEMANTIC_CACHE_VERSION,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache_and_stats(tmp_path: Path) -> tuple[Path, Path, dict]:
    cache = tmp_path / "cache"
    cache.mkdir()
    arrays = {
        "labels.npy": np.zeros((2, 4, 4), dtype=np.uint8),
        "confidence.npy": np.full((2, 4, 4), 255, dtype=np.uint8),
        "shuffled_indices.npy": np.asarray([1, 0], dtype=np.int32),
    }
    for filename, array in arrays.items():
        np.save(cache / filename, array)
    array_hashes = {filename: _sha256(cache / filename) for filename in arrays}
    city_digest = "a" * 64
    manifest = {
        "schema": QUERY_SEMANTIC_CACHE_SCHEMA,
        "version": QUERY_SEMANTIC_CACHE_VERSION,
        "complete": True,
        "num_images": 2,
        "num_classes": 2,
        "classes": ["stable", "unsupported"],
        "grid_size": [4, 4],
        "array_sha256": array_hashes,
        "cities": [{"name": "TestCity", "sha256": city_digest}],
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stats = {
        "schema": "openvpr_rscd_class_stats",
        "version": 1,
        "complete": True,
        "created_utc": "2026-08-31T00:00:00Z",
        "cache_schema": QUERY_SEMANTIC_CACHE_SCHEMA,
        "cache_version": QUERY_SEMANTIC_CACHE_VERSION,
        "grid_size": [4, 4],
        "num_classes": 2,
        "min_confidence": 0.5,
        "min_confidence_quantized": 128,
        "source": {
            "cache_dir": str(cache),
            "dataset_root": str(tmp_path / "dataset"),
            "num_images": 2,
            "eligible_min_views": 4,
            "model_name": "teacher",
            "requested_revision": "revision",
            "resolved_commit": "commit",
            "class_names": ["stable", "unsupported"],
        },
        "source_hashes": {
            "manifest.json": _sha256(manifest_path),
            **array_hashes,
            "city_csv": {"TestCity": city_digest},
        },
        "protocol": {
            "presence": "any high-confidence patch",
            "eligible_place": "at least four views",
            "repeatability_numerator": "sum m*(m-1)",
            "repeatability_support": "sum m*(n-1)",
            "frequency": "places_present/eligible_places",
            "nuisance": "1-repeatability*(1-frequency)",
            "min_support": 2,
            "confidence_comparison": "quantized confidence >= threshold",
            "confidence_quantization": "ceil(min_confidence*255)",
        },
        "totals": {
            "cities": 1,
            "eligible_places": 4,
            "eligible_images": 2,
            "cached_images": 2,
            "patches_per_image": 16,
        },
        "classes": [
            {
                "id": 0,
                "name": "stable",
                "views_present": 2,
                "places_present": 1,
                "repeatability_numerator": 1,
                "support": 2,
                "repeatability": 0.5,
                "frequency": 0.25,
                "nuisance": 0.625,
                "valid": True,
                "invalid_reason": None,
            },
            {
                "id": 1,
                "name": "unsupported",
                "views_present": 0,
                "places_present": 0,
                "repeatability_numerator": 0,
                "support": 0,
                "repeatability": None,
                "frequency": 0.0,
                "nuisance": None,
                "valid": False,
                "invalid_reason": "support_below_minimum",
            },
        ],
    }
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return stats_path, manifest_path, stats


def test_load_rscd_stats_verifies_cache_and_maps_invalid_class_to_zero(tmp_path):
    stats_path, manifest_path, _ = _write_cache_and_stats(tmp_path)
    stats = load_rscd_stats(
        stats_path,
        cache_manifest=manifest_path,
        expected_min_confidence=0.5,
    )
    assert stats.grid_size == (4, 4)
    assert stats.class_names == ("stable", "unsupported")
    assert stats.nuisance_scores == pytest.approx((0.625, 0.0))
    assert stats.valid_classes == (True, False)


def test_load_rscd_stats_fails_closed_on_formula_or_cache_hash(tmp_path):
    stats_path, manifest_path, stats = _write_cache_and_stats(tmp_path)
    stats["classes"][0]["nuisance"] = 0.7
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="nuisance is inconsistent"):
        load_rscd_stats(stats_path)

    stats_path, manifest_path, stats = _write_cache_and_stats(tmp_path / "other")
    stats["source_hashes"]["labels.npy"] = "b" * 64
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="labels.npy SHA256 disagrees"):
        load_rscd_stats(
            stats_path, cache_manifest=manifest_path, verify_array_files=False
        )


def _maps_for_builder(batch: int = 4):
    # Every non-overlapping 2x2 cell is internally label-consistent.  Aligned
    # and donor maps expose all 16 candidates, so the quota is controlled only
    # by max_coverage.
    block_labels = torch.tensor(
        [
            [0, 1, 2, 0],
            [1, 2, 0, 1],
            [2, 0, 1, 2],
            [0, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    labels = block_labels.repeat_interleave(2, 0).repeat_interleave(2, 1)
    labels = labels.unsqueeze(0).repeat(batch, 1, 1)
    donor = labels.roll(shifts=2, dims=2)
    confidence = torch.ones_like(labels, dtype=torch.float32)
    indices = torch.arange(100, 100 + batch, dtype=torch.long)
    return labels, confidence, indices, donor, confidence.clone()


@pytest.mark.parametrize("mode", ["aligned", "uniform", "shuffled"])
def test_active_masks_are_deterministic_nonoverlapping_and_share_quota(mode):
    labels, confidence, indices, donor, donor_confidence = _maps_for_builder()
    builder = RSCDMaskBuilder(
        [0.2, 0.6, 1.0],
        valid_classes=[True, True, True],
        mode=mode,
        confidence_threshold=0.5,
        max_coverage=0.125,
        seed=42,
        grid_size=(8, 8),
    )
    assert list(builder.parameters()) == []
    assert builder.state_dict() == {}
    first, first_stats = builder.build(
        labels,
        confidence,
        indices,
        7,
        donor_labels=donor,
        donor_confidence=donor_confidence,
    )
    second, _ = builder.build(
        labels,
        confidence,
        indices,
        7,
        donor_labels=donor,
        donor_confidence=donor_confidence,
    )
    assert torch.equal(first, second)
    # 8/64 tokens = two complete, non-overlapping 2x2 blocks per image.
    assert first.dtype is torch.bool
    assert torch.equal(first.sum((1, 2)), torch.full((4,), 8))
    block_sums = first.view(4, 4, 2, 4, 2).sum((2, 4))
    assert set(block_sums.unique().tolist()) <= {0, 4}
    assert first_stats["rscd_mask_coverage"].item() == pytest.approx(0.125)
    assert first_stats["rscd_quota_blocks"].item() == pytest.approx(2.0)


def test_all_active_modes_have_exact_per_image_common_quota():
    labels, confidence, indices, donor, donor_confidence = _maps_for_builder()
    masks = {}
    for mode in ("aligned", "uniform", "shuffled"):
        builder = RSCDMaskBuilder(
            [0.2, 0.6, 1.0],
            mode=mode,
            confidence_threshold=0.5,
            max_coverage=0.1875,
            seed=9,
            grid_size=(8, 8),
        )
        masks[mode], _ = builder.build(
            labels,
            confidence,
            indices,
            11,
            donor_labels=donor,
            donor_confidence=donor_confidence,
        )
    token_counts = [mask.sum((1, 2)) for mask in masks.values()]
    assert all(torch.equal(token_counts[0], counts) for counts in token_counts[1:])
    assert torch.equal(token_counts[0], torch.full((4,), 12))


def test_no_mask_returns_zero_without_donor_and_active_mode_requires_donor():
    labels, confidence, indices, _, _ = _maps_for_builder(batch=2)
    no_mask = RSCDMaskBuilder(
        [1.0, 1.0, 1.0],
        mode="no_mask",
        confidence_threshold=0.5,
        grid_size=(8, 8),
    )
    mask, stats = no_mask.build(labels, confidence, indices, 0)
    assert not bool(mask.any())
    assert stats["rscd_mask_coverage"].item() == 0.0

    aligned = RSCDMaskBuilder(
        [1.0, 1.0, 1.0],
        mode="aligned",
        confidence_threshold=0.5,
        grid_size=(8, 8),
    )
    with pytest.raises(ValueError, match="require donor maps"):
        aligned.build(labels, confidence, indices, 0)


def test_builder_rejects_undecoded_confidence_and_out_of_range_labels():
    labels, confidence, indices, donor, donor_confidence = _maps_for_builder(1)
    builder = RSCDMaskBuilder(
        [1.0, 1.0, 1.0],
        mode="aligned",
        confidence_threshold=0.5,
        grid_size=(8, 8),
    )
    with pytest.raises(TypeError, match="decoded floating point"):
        builder.build(
            labels,
            (confidence * 255).to(torch.uint8),
            indices,
            0,
            donor_labels=donor,
            donor_confidence=donor_confidence,
        )
    bad_labels = labels.clone()
    bad_labels[0, 0, 0] = 3
    with pytest.raises(ValueError, match="outside the class range"):
        builder.build(
            bad_labels,
            confidence,
            indices,
            0,
            donor_labels=donor,
            donor_confidence=donor_confidence,
        )


def test_apply_token_mask_uses_detached_image_mean():
    featmap = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    mask = torch.tensor([[[True, False], [False, False]]])
    output = apply_token_mask(featmap, mask)
    assert output[0, 0, 0, 0].item() == pytest.approx(2.5)
    output.sum().backward()
    assert torch.equal(
        featmap.grad, torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]])
    )


def test_pairwise_relation_loss_detaches_clean_target():
    masked = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], requires_grad=True
    )
    clean = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )
    loss = pairwise_relation_loss(masked, clean)
    assert loss.ndim == 0 and loss.item() > 0
    loss.backward()
    assert masked.grad is not None and bool((masked.grad != 0).any())
    assert clean.grad is None
    assert pairwise_relation_loss(clean.detach(), clean.detach()).item() == 0.0


class _TinyRU(nn.Module):
    def __init__(self, builder: RSCDMaskBuilder | None = None):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.aggregator = nn.Linear(2, 2)
        self.semantic_region_gate = nn.Linear(2, 1)
        if builder is not None:
            self.rscd_builder = builder


def _ru_hparams():
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
        "distillation": {
            "semantic_region": {
                "enabled": True,
                "mode": "repeatability_uniqueness_only",
                "lambda_target": 0.02,
                "alpha": 0.2,
            }
        },
        "trainer": {"max_epochs": 40},
    }


def test_strict_ru_warm_start_accepts_parameterless_builder(tmp_path):
    source = _TinyRU()
    checkpoint_path = tmp_path / "ru.ckpt"
    torch.save(
        {"hyper_parameters": _ru_hparams(), "state_dict": source.state_dict()},
        checkpoint_path,
    )
    builder = RSCDMaskBuilder(
        [1.0, 1.0],
        mode="no_mask",
        confidence_threshold=0.5,
        grid_size=(4, 4),
    )
    target = _TinyRU(builder)
    assert not any("rscd" in key for key in target.state_dict())
    report = warm_start_rscd_model(
        target, checkpoint_path, expected_sha256=_sha256(checkpoint_path)
    )
    assert report["new_keys"] == ()
    assert report["loaded_keys"] == len(source.state_dict())
    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values()
    ):
        assert torch.equal(source_value, target_value)


def test_strict_ru_warm_start_rejects_any_state_difference(tmp_path):
    source = _TinyRU()
    state = dict(source.state_dict())
    state["unexpected.weight"] = torch.ones(1)
    checkpoint_path = tmp_path / "bad.ckpt"
    torch.save(
        {"hyper_parameters": _ru_hparams(), "state_dict": state}, checkpoint_path
    )
    with pytest.raises(RuntimeError, match="architecture-identical"):
        warm_start_rscd_model(_TinyRU(), checkpoint_path)
