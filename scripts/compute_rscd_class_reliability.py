"""Compute class reliability for RSCD-BoQ from the frozen ADE20K cache.

The script never reads training images and never changes the existing cache.
It validates the complete v2 cache, its immutable array hashes, and every city
CSV hash before aggregating confidence-filtered class presence by place.

For an eligible place with ``n`` views, let ``m`` be the number of views in
which a class is present.  The class statistics are defined as::

    repeatability = sum(m * (m - 1)) / sum(m * (n - 1))
    frequency     = places containing the class / eligible places
    nuisance      = 1 - repeatability * (1 - frequency)

Classes whose repeatability denominator is below ``--min-support`` are marked
invalid and receive no nuisance value.  The training code must not turn an
unsupported class into a strong negative prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.query_semantic import verify_query_semantic_cache_hashes
from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_SCHEMA,
    QUERY_SEMANTIC_CACHE_VERSION,
)


RSCD_CLASS_STATS_SCHEMA = "openvpr_rscd_class_stats"
RSCD_CLASS_STATS_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_min_confidence(value: float) -> int:
    """Return the uint8 threshold equivalent to ``cached / 255 >= value``."""

    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("min_confidence must be finite and in [0, 1]")
    return int(math.ceil(value * 255.0))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validate_cache_manifest(
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, np.ndarray]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"semantic cache manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != QUERY_SEMANTIC_CACHE_SCHEMA:
        raise ValueError(
            "RSCD requires the ADE20K patch-label cache schema "
            f"{QUERY_SEMANTIC_CACHE_SCHEMA!r}"
        )
    if manifest.get("version") != QUERY_SEMANTIC_CACHE_VERSION:
        raise ValueError(
            "RSCD requires query-semantic cache version "
            f"{QUERY_SEMANTIC_CACHE_VERSION}"
        )
    if manifest.get("complete") is not True:
        raise ValueError("RSCD refuses an incomplete semantic cache")

    num_images = manifest.get("num_images")
    num_classes = manifest.get("num_classes")
    eligible_min_views = manifest.get("eligible_min_views")
    if (
        isinstance(num_images, bool)
        or not isinstance(num_images, int)
        or num_images < 1
    ):
        raise ValueError("cache num_images must be a positive integer")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or not 1 <= num_classes <= 256
    ):
        raise ValueError("cache num_classes must be in [1, 256]")
    if (
        isinstance(eligible_min_views, bool)
        or not isinstance(eligible_min_views, int)
        or eligible_min_views < 2
    ):
        raise ValueError("cache eligible_min_views must be at least 2")

    classes = manifest.get("classes")
    if (
        not isinstance(classes, list)
        or len(classes) != num_classes
        or any(not isinstance(name, str) or not name for name in classes)
        or len(set(classes)) != num_classes
    ):
        raise ValueError("cache classes must be unique non-empty names")
    grid_size = manifest.get("grid_size")
    if (
        not isinstance(grid_size, list)
        or len(grid_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in grid_size)
        or min(grid_size) < 1
    ):
        raise ValueError("cache grid_size must contain two positive integers")

    city_entries = manifest.get("cities")
    if not isinstance(city_entries, list) or not city_entries:
        raise ValueError("cache manifest has no city entries")
    expected_offset = 0
    city_names: set[str] = set()
    for entry in city_entries:
        if not isinstance(entry, dict):
            raise ValueError("cache city entries must be objects")
        name = entry.get("name")
        offset = entry.get("offset")
        count = entry.get("count")
        csv_hash = entry.get("sha256")
        if not isinstance(name, str) or not name or name in city_names:
            raise ValueError("cache city names must be unique and non-empty")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or offset != expected_offset
            or count < 1
        ):
            raise ValueError("cache city offsets/counts must be contiguous")
        if (
            not isinstance(csv_hash, str)
            or len(csv_hash) != 64
            or any(character not in "0123456789abcdef" for character in csv_hash)
        ):
            raise ValueError(f"cache city {name!r} has an invalid CSV SHA256")
        city_names.add(name)
        expected_offset += count
    if expected_offset != num_images:
        raise ValueError("cache city counts do not match num_images")

    array_hashes = verify_query_semantic_cache_hashes(cache_dir, manifest)
    arrays = {
        name: np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in ("labels", "confidence", "shuffled_indices")
    }
    expected_specs = {
        "labels": ((num_images, *grid_size), np.dtype("uint8")),
        "confidence": ((num_images, *grid_size), np.dtype("uint8")),
        "shuffled_indices": ((num_images,), np.dtype("int32")),
    }
    for name, array in arrays.items():
        expected_shape, expected_dtype = expected_specs[name]
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise ValueError(
                f"invalid cache {name}: expected {expected_shape}/{expected_dtype}, "
                f"found {array.shape}/{array.dtype}"
            )
    return manifest, array_hashes, arrays


def _image_class_presence(
    labels: np.ndarray,
    confidence: np.ndarray,
    *,
    num_classes: int,
    threshold: int,
    chunk_rows: int,
) -> np.ndarray:
    """Build an image/class presence matrix while bounding temporary memory."""

    if labels.shape != confidence.shape or labels.ndim != 3:
        raise ValueError("labels/confidence must have matching (N,H,W) shapes")
    result = np.zeros((labels.shape[0], num_classes), dtype=np.bool_)
    patch_count = int(np.prod(labels.shape[1:]))
    for start in range(0, labels.shape[0], chunk_rows):
        stop = min(start + chunk_rows, labels.shape[0])
        label_chunk = np.asarray(labels[start:stop]).reshape(-1, patch_count)
        confidence_chunk = np.asarray(confidence[start:stop]).reshape(
            -1, patch_count
        )
        if label_chunk.size and int(label_chunk.max()) >= num_classes:
            raise ValueError("cached label lies outside the declared class range")
        valid = confidence_chunk >= threshold
        row_indices = np.broadcast_to(
            np.arange(stop - start, dtype=np.int64)[:, None], valid.shape
        )
        result_chunk = result[start:stop]
        result_chunk[row_indices[valid], label_chunk[valid]] = True
    return result


def compute_rscd_class_stats(
    *,
    dataset_root: str | Path,
    cache_dir: str | Path,
    min_confidence: float = 0.5,
    min_support: int = 100,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    """Validate inputs and return the complete RSCD class-statistics report."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not (dataset_root / "Dataframes").is_dir():
        raise FileNotFoundError(
            f"GSV-Cities Dataframes directory not found under {dataset_root}"
        )
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"semantic cache directory not found: {cache_dir}")
    if isinstance(min_support, bool) or int(min_support) != min_support or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    if isinstance(chunk_rows, bool) or int(chunk_rows) != chunk_rows or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    min_support = int(min_support)
    chunk_rows = int(chunk_rows)
    threshold = quantize_min_confidence(min_confidence)

    manifest, array_hashes, arrays = _validate_cache_manifest(cache_dir)
    num_classes = int(manifest["num_classes"])
    eligible_min_views = int(manifest["eligible_min_views"])
    views_present = np.zeros(num_classes, dtype=np.int64)
    places_present = np.zeros(num_classes, dtype=np.int64)
    numerator = np.zeros(num_classes, dtype=np.int64)
    support = np.zeros(num_classes, dtype=np.int64)
    city_csv_hashes: dict[str, str] = {}
    eligible_places = 0
    eligible_images = 0

    for city in manifest["cities"]:
        city_name = str(city["name"])
        csv_path = dataset_root / "Dataframes" / f"{city_name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"GSV-Cities city CSV not found: {csv_path}")
        actual_csv_hash = file_sha256(csv_path)
        expected_csv_hash = str(city["sha256"])
        if actual_csv_hash != expected_csv_hash:
            raise ValueError(
                f"city CSV SHA256 mismatch for {csv_path}: expected "
                f"{expected_csv_hash}, found {actual_csv_hash}"
            )
        city_csv_hashes[city_name] = actual_csv_hash
        frame = pd.read_csv(csv_path, usecols=["place_id"])
        count = int(city["count"])
        if len(frame) != count:
            raise ValueError(
                f"city CSV row-count mismatch for {city_name}: "
                f"manifest={count}, csv={len(frame)}"
            )
        place_ids = frame["place_id"].to_numpy()
        if bool(pd.isna(place_ids).any()):
            raise ValueError(f"city CSV {city_name!r} contains a missing place_id")
        place_codes, unique_places = pd.factorize(place_ids, sort=False)
        place_counts = np.bincount(place_codes, minlength=len(unique_places))
        eligible = place_counts >= eligible_min_views
        if not bool(eligible.any()):
            continue

        offset = int(city["offset"])
        city_slice = slice(offset, offset + count)
        image_presence = _image_class_presence(
            arrays["labels"][city_slice],
            arrays["confidence"][city_slice],
            num_classes=num_classes,
            threshold=threshold,
            chunk_rows=chunk_rows,
        )
        counts_by_place = np.zeros(
            (len(unique_places), num_classes), dtype=np.int64
        )
        np.add.at(counts_by_place, place_codes, image_presence)
        eligible_counts = counts_by_place[eligible]
        eligible_view_counts = place_counts[eligible].astype(np.int64)

        views_present += eligible_counts.sum(axis=0, dtype=np.int64)
        places_present += np.count_nonzero(eligible_counts, axis=0)
        numerator += (
            eligible_counts * (eligible_counts - 1)
        ).sum(axis=0, dtype=np.int64)
        support += (
            eligible_counts * (eligible_view_counts[:, None] - 1)
        ).sum(axis=0, dtype=np.int64)
        eligible_places += int(np.count_nonzero(eligible))
        eligible_images += int(eligible_view_counts.sum(dtype=np.int64))

    if eligible_places < 1:
        raise ValueError("dataset contains no places eligible for RSCD statistics")

    class_rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(manifest["classes"]):
        class_support = int(support[class_id])
        class_numerator = int(numerator[class_id])
        frequency = float(places_present[class_id] / eligible_places)
        valid = class_support >= min_support
        if valid:
            repeatability = float(class_numerator / class_support)
            nuisance = float(1.0 - repeatability * (1.0 - frequency))
            # Clamp harmless floating-point tails while keeping the exact
            # registered formula auditable from the raw counts.
            repeatability = min(1.0, max(0.0, repeatability))
            nuisance = min(1.0, max(0.0, nuisance))
            invalid_reason = None
        else:
            repeatability = None
            nuisance = None
            invalid_reason = (
                f"repeatability support {class_support} is below min_support "
                f"{min_support}"
            )
        class_rows.append(
            {
                "id": class_id,
                "name": str(name),
                "views_present": int(views_present[class_id]),
                "places_present": int(places_present[class_id]),
                "repeatability_numerator": class_numerator,
                "support": class_support,
                "repeatability": repeatability,
                "frequency": frequency,
                "nuisance": nuisance,
                "valid": valid,
                "invalid_reason": invalid_reason,
            }
        )

    manifest_path = cache_dir / "manifest.json"
    return {
        "schema": RSCD_CLASS_STATS_SCHEMA,
        "version": RSCD_CLASS_STATS_VERSION,
        "complete": True,
        "created_utc": utc_now(),
        "cache_schema": str(manifest["schema"]),
        "cache_version": int(manifest["version"]),
        "grid_size": [int(value) for value in manifest["grid_size"]],
        "num_classes": num_classes,
        "min_confidence": float(min_confidence),
        "min_confidence_quantized": threshold,
        "source": {
            "cache_dir": str(cache_dir),
            "dataset_root": str(dataset_root),
            "num_images": int(manifest["num_images"]),
            "eligible_min_views": eligible_min_views,
            "model_name": manifest.get("model_name"),
            "requested_revision": manifest.get("requested_revision"),
            "resolved_commit": manifest.get("resolved_commit"),
            "class_names": [str(name) for name in manifest["classes"]],
        },
        "source_hashes": {
            "manifest.json": file_sha256(manifest_path),
            **array_hashes,
            "city_csv": city_csv_hashes,
        },
        "protocol": {
            "presence": (
                "a class is present in a view iff at least one cached patch "
                "has that top-1 class and passes the confidence threshold"
            ),
            "eligible_place": (
                "original city CSV place_id has at least eligible_min_views rows"
            ),
            "repeatability_numerator": "sum_over_places m*(m-1)",
            "repeatability_support": "sum_over_places m*(n-1)",
            "frequency": "places_present/eligible_places",
            "nuisance": "1-repeatability*(1-frequency)",
            "min_support": min_support,
            "confidence_comparison": (
                "cached_uint8 >= min_confidence_quantized"
            ),
            "confidence_quantization": "ceil(min_confidence*255)",
        },
        "totals": {
            "cities": len(manifest["cities"]),
            "eligible_places": eligible_places,
            "eligible_images": eligible_images,
            "cached_images": int(manifest["num_images"]),
            "patches_per_image": int(np.prod(manifest["grid_size"])),
        },
        "classes": class_rows,
    }


def write_report(path: str | Path, report: Mapping[str, Any], *, overwrite: bool) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing stats file: {path}; pass --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output JSON; defaults to CACHE_DIR/rscd_class_stats.json"
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output
        if args.output is not None
        else args.cache_dir / "rscd_class_stats.json"
    )
    report = compute_rscd_class_stats(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        min_confidence=args.min_confidence,
        min_support=args.min_support,
        chunk_rows=args.chunk_rows,
    )
    write_report(output, report, overwrite=args.overwrite)
    valid_classes = sum(bool(row["valid"]) for row in report["classes"])
    print(
        f"RSCD class stats: {valid_classes}/{report['num_classes']} classes "
        f"valid at min_support={report['protocol']['min_support']}"
    )
    print(
        f"Eligible places/images: {report['totals']['eligible_places']}/"
        f"{report['totals']['eligible_images']}"
    )
    print(f"Wrote: {Path(output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
