from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2 as T2

from scripts.cache_gsv_cc_lsa_targets import (
    target_implementation_sha256,
    validate_static_arrays,
)
from src.cc_lsa_gate_a import file_sha256, string_sequence_sha256
from src.dataloaders.train.cc_lsa import (
    CC_LSA_SOURCE_TRANSFORM,
    CC_LSA_TARGET_SCHEMA,
    CC_LSA_TARGET_VERSION,
    image_to_imagenet_tensor,
    load_target_manifest,
)


def _write_complete_cache(tmp_path):
    dataset_root = tmp_path / "gsv"
    dataframe_dir = dataset_root / "Dataframes"
    dataframe_dir.mkdir(parents=True)
    csv_path = dataframe_dir / "Boston.csv"
    csv_path.write_text("place_id,city_id\n1,Boston\n2,Boston\n", encoding="utf-8")

    cache = tmp_path / "cache"
    cache.mkdir()
    arrays = {
        "paths.npy": np.asarray(
            ["Images/Boston/one.jpg", "Images/Boston/two.jpg"]
        ),
        "city.npy": np.asarray(["Boston", "Boston"]),
        "place_ids.npy": np.asarray([1, 2], dtype=np.int64),
        "split.npy": np.asarray([0, 1], dtype=np.uint8),
        "crop_cls.npy": np.zeros((2, 4, 512), dtype=np.float16),
    }
    for filename, array in arrays.items():
        np.save(cache / filename, array, allow_pickle=False)
    manifest = {
        "schema": CC_LSA_TARGET_SCHEMA,
        "version": CC_LSA_TARGET_VERSION,
        "complete": True,
        "dataset_root": str(dataset_root.resolve()),
        "csv_files": [
            {"city": "Boston", "rows": 2, "sha256": file_sha256(csv_path)}
        ],
        "num_places": 2,
        "num_train_places": 1,
        "num_holdout_places": 1,
        "path_sequence_sha256": string_sequence_sha256(
            arrays["paths.npy"].tolist()
        ),
        "selection": {
            "algorithm": "sha256_one_view_per_place_v1",
            "seed": 42,
            "minimum_views": 4,
            "split_algorithm": "sha256_place_v1",
            "holdout_modulus": 10,
        },
        "image_size": [280, 280],
        "region_grid": [2, 2],
        "source_transform": CC_LSA_SOURCE_TRANSFORM,
        "semantic_dim": 512,
        "teacher": {
            "model_name": "ViT-B-16",
            "pretrained": "openai",
            "base_model_state_sha256": "a" * 64,
        },
        "dtype": "float16",
        "implementation_sha256": target_implementation_sha256(),
        "array_sha256": {
            filename: file_sha256(cache / filename) for filename in arrays
        },
    }
    (cache / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return dataset_root, cache, arrays, manifest


def test_complete_target_manifest_validates_source_and_arrays(tmp_path) -> None:
    dataset_root, cache, _, _ = _write_complete_cache(tmp_path)
    manifest = load_target_manifest(cache, dataset_root=dataset_root)
    assert manifest["num_train_places"] == 1
    assert manifest["num_holdout_places"] == 1


def test_target_source_geometry_matches_canonical_ru_clean_transform() -> None:
    pixels = np.arange(17 * 13 * 3, dtype=np.uint8).reshape(17, 13, 3)
    image = Image.fromarray(pixels, mode="RGB")
    expected = T2.Compose(
        [
            T2.ToImage(),
            T2.Resize(
                size=(280, 280),
                interpolation=T2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )(image)
    actual = image_to_imagenet_tensor(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_target_manifest_rejects_semantically_invalid_split_even_with_new_hash(
    tmp_path,
) -> None:
    dataset_root, cache, _, manifest = _write_complete_cache(tmp_path)
    np.save(cache / "split.npy", np.asarray([0, 0], dtype=np.uint8))
    manifest["array_sha256"]["split.npy"] = file_sha256(cache / "split.npy")
    (cache / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="train-place count"):
        load_target_manifest(cache, dataset_root=dataset_root)


def test_resume_rejects_changed_identity_sidecar(tmp_path) -> None:
    _, cache, arrays, _ = _write_complete_cache(tmp_path)
    changed = arrays["place_ids.npy"].copy()
    changed[0] = 99
    np.save(cache / "place_ids.npy", changed, allow_pickle=False)
    with pytest.raises(ValueError, match="identity values differ"):
        validate_static_arrays(
            cache,
            paths=arrays["paths.npy"],
            cities=arrays["city.npy"],
            place_ids=arrays["place_ids.npy"],
            split=arrays["split.npy"],
        )
