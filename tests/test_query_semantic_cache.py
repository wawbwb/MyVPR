import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.cache_gsv_patch_semantics import (
    build_manifest,
    discover_cities,
    image_name,
)
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.models.query_semantic import verify_query_semantic_cache_hashes
from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_VERSION,
    QUERY_SEMANTIC_SHUFFLE_ALGORITHM,
    build_cross_place_bijection,
)


def _row(place_id: int = 123) -> pd.Series:
    row = pd.Series(
        {
            "city_id": "TestCity",
            "place_id": place_id,
            "year": 2020,
            "month": 7,
            "northdeg": 9,
            "lat": 1.25,
            "lon": -3.5,
            "panoid": "pano",
        }
    )
    row.name = place_id
    return row


def test_cache_image_name_matches_gsv_dataset() -> None:
    row = _row()
    assert image_name(row) == GSVCitiesDataset.get_img_name(row)


def test_cache_array_hashes_are_required_and_verified(tmp_path: Path) -> None:
    filenames = ("labels.npy", "confidence.npy", "shuffled_indices.npy")
    expected = {}
    for index, filename in enumerate(filenames):
        payload = bytes([index, index + 1, index + 2])
        (tmp_path / filename).write_bytes(payload)
        expected[filename] = hashlib.sha256(payload).hexdigest()
    manifest = {"array_sha256": expected}

    assert verify_query_semantic_cache_hashes(tmp_path, manifest) == expected

    (tmp_path / "labels.npy").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_query_semantic_cache_hashes(tmp_path, manifest)


def test_cache_manifest_records_exact_patch_protocol(tmp_path: Path) -> None:
    manifest = build_manifest(
        dataset_root=tmp_path,
        city_entries=[
            {
                "name": "TestCity",
                "offset": 0,
                "count": 8,
                "sha256": "0" * 64,
                "eligible_count": 8,
                "eligible_shuffle_rotation": 4,
            }
        ],
        image_count=8,
        model_name="teacher",
        revision="1" * 40,
        commit="1" * 40,
        transformers_version="4.44.2",
        processor_info={},
        classes=["one", "two"],
        grid_size=(20, 20),
        target_image_size=(280, 280),
        eligible_min_views=4,
        use_amp=True,
        device=torch.device("cuda"),
    )

    assert manifest["teacher_input"] == "clean_rgb"
    assert manifest["pooling"] == (
        "bilinear_logits_to_target_then_softmax_then_nonoverlap_avg_pool"
    )
    assert manifest["confidence_quantization"] == (
        "round(clamp(top1_probability,0,1)*255)"
    )
    assert manifest["inference_precision"] == "amp_float16"
    assert manifest["version"] == QUERY_SEMANTIC_CACHE_VERSION
    assert (
        manifest["shuffle_algorithm"]
        == QUERY_SEMANTIC_SHUFFLE_ALGORITHM
    )


def test_cross_place_bijection_handles_no_raw_circular_shift() -> None:
    # Every raw-order circular shift has at least one same-place collision.
    # A valid bijection still exists because the largest group is only 4/12.
    place_ids = np.asarray([0, 2, 2, 1, 0, 0, 1, 2, 0, 1, 2, 1])
    raw_positions = np.arange(place_ids.size)
    assert all(
        np.any(place_ids == place_ids[(raw_positions + shift) % place_ids.size])
        for shift in range(1, place_ids.size)
    )

    donors, rotation = build_cross_place_bijection(
        place_ids,
        context="Boston-like test city",
    )

    assert rotation == 4
    assert np.array_equal(np.sort(donors), raw_positions)
    assert np.all(place_ids != place_ids[donors])


def test_cross_place_bijection_rejects_mathematically_impossible_case() -> None:
    with pytest.raises(ValueError, match="largest place has 5 of 8"):
        build_cross_place_bijection(
            np.asarray([0, 0, 0, 0, 0, 1, 1, 2]),
            context="unbalanced test city",
        )


def test_shuffled_control_is_bijection_on_training_eligible_rows(
    tmp_path: Path,
) -> None:
    dataframe_dir = tmp_path / "Dataframes"
    dataframe_dir.mkdir()
    # The first 12 rows reproduce the raw-order pattern from the regression
    # above; place 40 is deliberately below K=4 and must remain self-mapped.
    place_ids = np.asarray(
        [10, 30, 30, 20, 10, 10, 20, 30, 10, 20, 30, 20, 40, 40, 40],
        dtype=np.int64,
    )
    pd.DataFrame({"place_id": place_ids}).to_csv(
        dataframe_dir / "TestCity.csv", index=False
    )

    entries, image_count, shuffled = discover_cities(
        tmp_path, eligible_min_views=4
    )

    assert image_count == len(place_ids)
    assert len(entries) == 1
    assert entries[0]["eligible_count"] == 12
    assert entries[0]["eligible_shuffle_rotation"] == 4
    eligible = np.arange(12, dtype=np.int64)
    donors = shuffled[eligible].astype(np.int64)
    assert np.array_equal(np.sort(donors), eligible)
    assert np.all(place_ids[eligible] != place_ids[donors])
    # Rows from places filtered out by K=4 are never sampled during training
    # and remain self-mapped, preserving a full-city bijection on disk.
    assert np.array_equal(shuffled[12:], np.arange(12, 15, dtype=np.int32))
    assert np.unique(shuffled).size == len(place_ids)


def test_gsv_loader_reads_fixed_cross_place_donor(tmp_path: Path) -> None:
    dataframe_dir = tmp_path / "Dataframes"
    dataframe_dir.mkdir()
    place_ids = np.asarray([10] * 4 + [20] * 4 + [30] * 3)
    pd.DataFrame({"place_id": place_ids}).to_csv(
        dataframe_dir / "TestCity.csv", index=False
    )
    entries, image_count, shuffled = discover_cities(
        tmp_path, eligible_min_views=4
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    labels = np.broadcast_to(
        np.arange(image_count, dtype=np.uint8)[:, None, None],
        (image_count, 2, 2),
    ).copy()
    confidence = np.full((image_count, 2, 2), 255, dtype=np.uint8)
    np.save(cache_dir / "labels.npy", labels)
    np.save(cache_dir / "confidence.npy", confidence)
    np.save(cache_dir / "shuffled_indices.npy", shuffled)
    manifest = {
        "schema": "openvpr_ade20k_patch_labels",
        "version": QUERY_SEMANTIC_CACHE_VERSION,
        "complete": True,
        "num_images": image_count,
        "grid_size": [2, 2],
        "num_classes": 20,
        "classes": [f"class-{index}" for index in range(20)],
        "labels_dtype": "uint8",
        "confidence_dtype": "uint8",
        "shuffled_indices_dtype": "int32",
        "eligible_min_views": 4,
        "shuffle_algorithm": QUERY_SEMANTIC_SHUFFLE_ALGORITHM,
        "cities": entries,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    dataset = GSVCitiesDataset(
        dataset_path=tmp_path,
        cities=["TestCity"],
        img_per_place=4,
        transform=None,
        query_semantic_cache_dir=cache_dir,
        query_semantic_selection="shuffled",
        rscd_cache_dir=cache_dir,
    )
    target = dataset._read_query_semantic_cache(
        np.asarray([0], dtype=np.int64)
    )
    donor = int(shuffled[0])
    assert place_ids[0] != place_ids[donor]
    assert int(target["query_semantic_labels"][0, 0, 0]) == donor
    assert int(target["query_semantic_cache_indices"][0]) == 0
    # RSCD always returns both the receiver-aligned map and the same immutable
    # cross-place donor, independent of the legacy query-target selection.
    assert int(target["rscd_labels"][0, 0, 0]) == 0
    assert int(target["rscd_donor_labels"][0, 0, 0]) == donor
    assert target["rscd_confidence"].dtype == torch.float32
    assert target["rscd_donor_confidence"].dtype == torch.float32
    assert int(target["rscd_cache_indices"][0]) == 0
    assert int(target["rscd_donor_cache_indices"][0]) == donor
