import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compute_rscd_class_reliability import (
    RSCD_CLASS_STATS_SCHEMA,
    compute_rscd_class_stats,
    quantize_min_confidence,
    write_report,
)
from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_SCHEMA,
    QUERY_SEMANTIC_CACHE_VERSION,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_toy_cache(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "gsv"
    dataframe_dir = dataset_root / "Dataframes"
    dataframe_dir.mkdir(parents=True)
    # Places 10 and 20 are eligible at K=3; place 30 is deliberately excluded.
    place_ids = [10, 10, 10, 20, 20, 20, 20, 30, 30]
    csv_path = dataframe_dir / "ToyCity.csv"
    pd.DataFrame({"place_id": place_ids}).to_csv(csv_path, index=False)

    # Two patches are enough to represent the desired per-view presence sets.
    presence = [
        (0, 2),
        (0, 2),
        (1, 2),
        (0, 1),
        (1,),
        (1,),
        (1,),
        (3,),
        (3,),
    ]
    labels = np.full((len(place_ids), 1, 2), 3, dtype=np.uint8)
    confidence = np.zeros_like(labels)
    for row, class_ids in enumerate(presence):
        for column, class_id in enumerate(class_ids):
            labels[row, 0, column] = class_id
            confidence[row, 0, column] = 255

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(cache_dir / "labels.npy", labels)
    np.save(cache_dir / "confidence.npy", confidence)
    np.save(
        cache_dir / "shuffled_indices.npy",
        np.arange(len(place_ids), dtype=np.int32),
    )
    array_hashes = {
        filename: _sha256(cache_dir / filename)
        for filename in (
            "labels.npy",
            "confidence.npy",
            "shuffled_indices.npy",
        )
    }
    manifest = {
        "schema": QUERY_SEMANTIC_CACHE_SCHEMA,
        "version": QUERY_SEMANTIC_CACHE_VERSION,
        "complete": True,
        "num_images": len(place_ids),
        "grid_size": [1, 2],
        "num_classes": 4,
        "classes": ["zero", "one", "two", "unsupported"],
        "eligible_min_views": 3,
        "model_name": "toy-segmenter",
        "requested_revision": "a" * 40,
        "resolved_commit": "a" * 40,
        "cities": [
            {
                "name": "ToyCity",
                "offset": 0,
                "count": len(place_ids),
                "sha256": _sha256(csv_path),
            }
        ],
        "array_sha256": array_hashes,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return dataset_root, cache_dir


def test_confidence_threshold_matches_uint8_decode() -> None:
    assert quantize_min_confidence(0.0) == 0
    assert quantize_min_confidence(0.5) == 128
    assert quantize_min_confidence(0.7) == 179
    assert quantize_min_confidence(1.0) == 255
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        quantize_min_confidence(-0.1)


def test_rscd_stats_use_eligible_place_presence_and_disable_low_support(
    tmp_path: Path,
) -> None:
    dataset_root, cache_dir = _write_toy_cache(tmp_path)
    report = compute_rscd_class_stats(
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        min_confidence=0.5,
        min_support=6,
        chunk_rows=2,
    )

    assert report["schema"] == RSCD_CLASS_STATS_SCHEMA
    assert report["version"] == 1
    assert report["complete"] is True
    assert report["min_confidence_quantized"] == 128
    assert report["totals"] == {
        "cities": 1,
        "eligible_places": 2,
        "eligible_images": 7,
        "cached_images": 9,
        "patches_per_image": 2,
    }
    assert report["source_hashes"]["labels.npy"] == _sha256(
        cache_dir / "labels.npy"
    )
    assert report["source_hashes"]["city_csv"]["ToyCity"] == _sha256(
        dataset_root / "Dataframes" / "ToyCity.csv"
    )

    zero, one, two, unsupported = report["classes"]
    assert [row["id"] for row in report["classes"]] == list(range(4))
    assert zero["repeatability_numerator"] == 2
    assert zero["support"] == 7
    assert zero["repeatability"] == pytest.approx(2.0 / 7.0)
    assert zero["frequency"] == 1.0
    assert zero["nuisance"] == 1.0
    assert zero["valid"] is True

    assert one["repeatability_numerator"] == 12
    assert one["support"] == 14
    assert one["repeatability"] == pytest.approx(6.0 / 7.0)
    assert one["frequency"] == 1.0

    assert two["repeatability_numerator"] == 6
    assert two["support"] == 6
    assert two["repeatability"] == 1.0
    assert two["frequency"] == 0.5
    assert two["nuisance"] == 0.5

    # Class 3 occurs only in the K-ineligible place and must not become a
    # maximal-nuisance class merely because its repeatability is unsupported.
    assert unsupported["views_present"] == 0
    assert unsupported["places_present"] == 0
    assert unsupported["support"] == 0
    assert unsupported["repeatability"] is None
    assert unsupported["nuisance"] is None
    assert unsupported["valid"] is False
    assert "below min_support" in unsupported["invalid_reason"]


def test_rscd_stats_reject_hash_drift_and_report_overwrite(tmp_path: Path) -> None:
    dataset_root, cache_dir = _write_toy_cache(tmp_path)
    report = compute_rscd_class_stats(
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        min_support=1,
    )
    output = tmp_path / "stats.json"
    write_report(output, report, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(output, report, overwrite=False)
    write_report(output, report, overwrite=True)

    labels = np.load(cache_dir / "labels.npy")
    labels[0, 0, 0] = 1
    np.save(cache_dir / "labels.npy", labels)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        compute_rscd_class_stats(
            dataset_root=dataset_root,
            cache_dir=cache_dir,
            min_support=1,
        )
