"""Immutable cache protocol for AG-SLRD semantic-layout labels.

The layout teacher consumes a deliberately coarse semantic alphabet instead
of the 150 ADE20K category IDs.  The mapping is fixed in source control: it is
not estimated from GSV-Cities or tuned on MSLS.  Dynamic objects remain a
dedicated superclass and are never silently dropped.

This module contains no model code.  It only defines the mapping and validates
portable ``labels.npy``/``shuffled_indices.npy`` caches and their provenance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SEMANTIC_LAYOUT_CACHE_SCHEMA = "openvpr_ade20k_semantic_layout"
SEMANTIC_LAYOUT_CACHE_VERSION = 1
SEMANTIC_LAYOUT_GRID_SIZE = (70, 70)
SEMANTIC_LAYOUT_IGNORE_INDEX = 255
SEMANTIC_LAYOUT_MAPPING_NAME = "ade20k_150_to_vpr_layout_12"
SEMANTIC_LAYOUT_MAPPING_VERSION = 1
ADE20K_PATCH_CACHE_SCHEMA = "openvpr_ade20k_patch_labels"
ADE20K_PATCH_CACHE_VERSION = 2

# This is the exact class order exposed by
# nvidia/segformer-b0-finetuned-ade-512-512.  Requiring the exact order makes a
# teacher/config change fail closed rather than applying a plausible-looking
# but wrong integer mapping.
ADE20K_CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk", "person",
    "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea",
    "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base",
    "box", "column", "signboard", "chest of drawers", "counter", "sand",
    "sink", "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow",
    "screen door", "stairway", "river", "bridge", "bookcase", "blind",
    "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning",
    "streetlight", "booth", "television receiver", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
    "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag",
)

SEMANTIC_LAYOUT_CLASSES = (
    "built_structure",
    "traversable_surface",
    "vegetation",
    "sky",
    "terrain",
    "water",
    "outdoor_infrastructure",
    "dynamic",
    "indoor_surface",
    "furniture_fixture",
    "device_lighting",
    "movable_object",
)
DYNAMIC_SUPERCLASS_ID = SEMANTIC_LAYOUT_CLASSES.index("dynamic")

_MEMBERS_BY_SUPERCLASS = {
    "built_structure": {
        "wall", "building", "windowpane", "door", "house", "column",
        "skyscraper", "screen door", "hovel", "tower",
    },
    "traversable_surface": {
        "floor", "road", "sidewalk", "earth", "path", "stairs", "runway",
        "stairway", "dirt track", "land", "escalator", "stage", "step",
    },
    "vegetation": {"tree", "grass", "plant", "field", "flower", "palm"},
    "sky": {"sky"},
    "terrain": {"mountain", "rock", "sand", "hill"},
    "water": {
        "water", "sea", "river", "swimming pool", "waterfall", "lake",
    },
    "outdoor_infrastructure": {
        "fence", "railing", "signboard", "grandstand", "bridge", "bench",
        "awning", "streetlight", "booth", "pole", "bannister", "fountain",
        "canopy", "tank", "trade name", "sculpture", "traffic light",
        "ashcan", "pier", "bulletin board", "flag",
    },
    "dynamic": {
        "person", "car", "boat", "bus", "truck", "airplane", "van",
        "ship", "minibike", "animal", "bicycle",
    },
    "indoor_surface": {"ceiling", "curtain", "mirror", "rug", "blind", "glass"},
    "furniture_fixture": {
        "bed", "cabinet", "table", "chair", "sofa", "shelf", "armchair",
        "seat", "desk", "wardrobe", "bathtub", "cushion", "base",
        "chest of drawers", "counter", "sink", "fireplace", "refrigerator",
        "pool table", "pillow", "bookcase", "coffee table", "toilet",
        "countertop", "stove", "kitchen island", "swivel chair", "bar",
        "ottoman", "buffet", "conveyer belt", "washer", "stool", "cradle",
        "oven", "microwave", "dishwasher", "hood", "shower", "radiator",
    },
    "device_lighting": {
        "lamp", "computer", "arcade machine", "light", "chandelier",
        "television receiver", "screen", "sconce", "fan", "crt screen",
        "monitor", "clock",
    },
    "movable_object": {
        "painting", "box", "case", "book", "towel", "apparel", "bottle",
        "poster", "plaything", "barrel", "basket", "tent", "bag", "ball",
        "food", "pot", "blanket", "vase", "tray", "plate",
    },
}


def _build_source_to_superclass() -> tuple[int, ...]:
    unknown_groups = set(_MEMBERS_BY_SUPERCLASS) - set(SEMANTIC_LAYOUT_CLASSES)
    if unknown_groups:
        raise RuntimeError(f"unknown semantic-layout groups: {sorted(unknown_groups)}")
    owner: dict[str, int] = {}
    for superclass_id, superclass in enumerate(SEMANTIC_LAYOUT_CLASSES):
        for source_class in _MEMBERS_BY_SUPERCLASS[superclass]:
            if source_class in owner:
                raise RuntimeError(f"duplicate ADE20K mapping for {source_class!r}")
            owner[source_class] = superclass_id
    missing = set(ADE20K_CLASSES) - set(owner)
    extra = set(owner) - set(ADE20K_CLASSES)
    if missing or extra:
        raise RuntimeError(
            "ADE20K superclass mapping is not exhaustive: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return tuple(owner[name] for name in ADE20K_CLASSES)


ADE20K_TO_SEMANTIC_LAYOUT = _build_source_to_superclass()


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_layout_mapping_record() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": SEMANTIC_LAYOUT_MAPPING_NAME,
        "version": SEMANTIC_LAYOUT_MAPPING_VERSION,
        "source_classes": list(ADE20K_CLASSES),
        "superclasses": list(SEMANTIC_LAYOUT_CLASSES),
        "source_to_superclass": list(ADE20K_TO_SEMANTIC_LAYOUT),
        "dynamic_superclass_id": DYNAMIC_SUPERCLASS_ID,
        "ignore_index": SEMANTIC_LAYOUT_IGNORE_INDEX,
    }
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def remap_ade20k_labels(labels: np.ndarray) -> np.ndarray:
    """Map an integer ADE20K label array to the fixed 12-class alphabet."""

    labels = np.asarray(labels)
    if labels.dtype.kind not in "ui":
        raise TypeError("ADE20K labels must have an unsigned/signed integer dtype")
    if labels.size:
        minimum = int(labels.min())
        maximum = int(labels.max())
        if minimum < 0 or maximum >= len(ADE20K_CLASSES):
            raise ValueError(
                "ADE20K label is outside the fixed source range: "
                f"min={minimum}, max={maximum}"
            )
    lookup = np.asarray(ADE20K_TO_SEMANTIC_LAYOUT, dtype=np.uint8)
    return lookup[labels]


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 hex string")
    return value


def validate_mapping_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("semantic-layout manifest has no mapping record")
    mapping = dict(value)
    expected = semantic_layout_mapping_record()
    if mapping != expected:
        raise ValueError(
            "semantic-layout mapping differs from the source-controlled "
            f"{SEMANTIC_LAYOUT_MAPPING_NAME} v{SEMANTIC_LAYOUT_MAPPING_VERSION}"
        )
    return mapping


def validate_ade20k_patch_cache(
    cache_dir: str | Path,
    *,
    verify_array_files: bool = True,
    expected_grid_size: tuple[int, int] = SEMANTIC_LAYOUT_GRID_SIZE,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    """Validate the immutable 150-class cache used as conversion input."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ADE20K patch manifest not found: {manifest_path}")
    manifest = _load_json_mapping(manifest_path)
    if manifest.get("schema") != ADE20K_PATCH_CACHE_SCHEMA:
        raise ValueError("unsupported ADE20K patch-cache schema")
    if manifest.get("version") != ADE20K_PATCH_CACHE_VERSION:
        raise ValueError("unsupported ADE20K patch-cache version")
    if manifest.get("complete") is not True:
        raise ValueError("ADE20K patch cache is incomplete")
    num_images = manifest.get("num_images")
    if not _is_positive_int(num_images):
        raise ValueError("ADE20K patch-cache num_images must be positive")
    if tuple(manifest.get("grid_size", ())) != tuple(expected_grid_size):
        raise ValueError(
            "ADE20K patch cache must be generated at the requested grid; "
            f"expected {list(expected_grid_size)}, found "
            f"{manifest.get('grid_size')!r}. Upsampling a 20x20 cache is forbidden."
        )
    if manifest.get("num_classes") != len(ADE20K_CLASSES):
        raise ValueError("ADE20K patch-cache num_classes must be 150")
    if tuple(manifest.get("classes", ())) != ADE20K_CLASSES:
        raise ValueError(
            "ADE20K patch-cache class names/order differ from the frozen mapping"
        )
    cities = manifest.get("cities")
    if not isinstance(cities, list) or not cities:
        raise ValueError("ADE20K patch-cache manifest has no city entries")
    expected_offset = 0
    seen: set[str] = set()
    for entry in cities:
        if not isinstance(entry, Mapping):
            raise ValueError("ADE20K patch-cache city entry must be an object")
        name, offset, count = entry.get("name"), entry.get("offset"), entry.get("count")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("ADE20K patch-cache city names must be unique")
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or not _is_positive_int(count)
            or offset != expected_offset
        ):
            raise ValueError("ADE20K patch-cache city offsets/counts are invalid")
        _validate_sha256(entry.get("sha256"), field=f"cities.{name}.sha256")
        eligible_count = entry.get("eligible_count")
        if not isinstance(eligible_count, int) or isinstance(eligible_count, bool):
            raise ValueError(f"ADE20K city {name!r} has invalid eligible_count")
        if not 2 <= eligible_count <= count:
            raise ValueError(f"ADE20K city {name!r} eligible_count is out of range")
        seen.add(name)
        expected_offset += count
    if expected_offset != num_images:
        raise ValueError("ADE20K patch-cache city counts do not cover num_images")

    required_arrays = ("labels.npy", "confidence.npy", "shuffled_indices.npy")
    declared = manifest.get("array_sha256")
    if not isinstance(declared, Mapping) or set(declared) != set(required_arrays):
        raise ValueError(
            "ADE20K patch-cache array_sha256 must contain exactly "
            f"{required_arrays}"
        )
    hashes = {
        name: _validate_sha256(declared[name], field=f"array_sha256.{name}")
        for name in required_arrays
    }
    specs = {
        "labels.npy": ((num_images, *expected_grid_size), np.dtype("uint8")),
        "confidence.npy": ((num_images, *expected_grid_size), np.dtype("uint8")),
        "shuffled_indices.npy": ((num_images,), np.dtype("int32")),
    }
    arrays: dict[str, np.ndarray] = {}
    for filename, (shape, dtype) in specs.items():
        path = cache_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"ADE20K patch-cache array not found: {path}")
        if verify_array_files:
            actual = file_sha256(path)
            if actual != hashes[filename]:
                raise ValueError(
                    f"ADE20K patch-cache SHA256 mismatch for {filename}: "
                    f"expected {hashes[filename]}, found {actual}"
                )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"invalid ADE20K patch-cache {filename}: expected {shape}/{dtype}, "
                f"found {array.shape}/{array.dtype}"
            )
        arrays[filename.removesuffix(".npy")] = array
    return manifest, arrays, hashes


def validate_semantic_layout_cache(
    cache_dir: str | Path,
    *,
    verify_array_files: bool = True,
    expected_index_type: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    """Validate a complete portable semantic-layout cache.

    Large array contents are memory-mapped.  SHA256 verification can be
    disabled only for repeated reads after a caller has already verified the
    cache in the same run.
    """

    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"semantic-layout manifest not found: {manifest_path}")
    manifest = _load_json_mapping(manifest_path)
    if manifest.get("schema") != SEMANTIC_LAYOUT_CACHE_SCHEMA:
        raise ValueError("unsupported semantic-layout cache schema")
    if manifest.get("version") != SEMANTIC_LAYOUT_CACHE_VERSION:
        raise ValueError("unsupported semantic-layout cache version")
    if manifest.get("complete") is not True:
        raise ValueError("semantic-layout cache is incomplete")
    num_images = manifest.get("num_images")
    if not _is_positive_int(num_images):
        raise ValueError("semantic-layout num_images must be positive")
    if tuple(manifest.get("grid_size", ())) != SEMANTIC_LAYOUT_GRID_SIZE:
        raise ValueError(
            f"semantic-layout grid_size must be {list(SEMANTIC_LAYOUT_GRID_SIZE)}"
        )
    if manifest.get("num_classes") != len(SEMANTIC_LAYOUT_CLASSES):
        raise ValueError("semantic-layout num_classes is inconsistent")
    if tuple(manifest.get("classes", ())) != SEMANTIC_LAYOUT_CLASSES:
        raise ValueError("semantic-layout class names/order are inconsistent")
    if manifest.get("ignore_index") != SEMANTIC_LAYOUT_IGNORE_INDEX:
        raise ValueError("semantic-layout ignore_index is inconsistent")
    validate_mapping_record(manifest.get("mapping"))

    index = manifest.get("index")
    if not isinstance(index, Mapping) or not isinstance(index.get("type"), str):
        raise ValueError("semantic-layout manifest has no valid index record")
    if expected_index_type is not None and index.get("type") != expected_index_type:
        raise ValueError(
            f"semantic-layout index type {index.get('type')!r} != "
            f"{expected_index_type!r}"
        )

    required_arrays = ("labels.npy", "shuffled_indices.npy")
    declared = manifest.get("array_sha256")
    if not isinstance(declared, Mapping) or set(declared) != set(required_arrays):
        raise ValueError(
            "semantic-layout array_sha256 must contain exactly "
            f"{required_arrays}"
        )
    hashes = {
        name: _validate_sha256(declared[name], field=f"array_sha256.{name}")
        for name in required_arrays
    }
    arrays: dict[str, np.ndarray] = {}
    specs = {
        "labels.npy": ((num_images, *SEMANTIC_LAYOUT_GRID_SIZE), np.dtype("uint8")),
        "shuffled_indices.npy": ((num_images,), np.dtype("int32")),
    }
    for filename, (shape, dtype) in specs.items():
        path = cache_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"semantic-layout array not found: {path}")
        if verify_array_files:
            actual = file_sha256(path)
            if actual != hashes[filename]:
                raise ValueError(
                    f"semantic-layout SHA256 mismatch for {filename}: "
                    f"expected {hashes[filename]}, found {actual}"
                )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"invalid semantic-layout {filename}: expected {shape}/{dtype}, "
                f"found {array.shape}/{array.dtype}"
            )
        arrays[filename.removesuffix(".npy")] = array
    return manifest, arrays, hashes


def validate_derangement(indices: Sequence[int], *, context: str) -> None:
    """Validate a zero-based permutation with no self mapping."""

    values = np.asarray(indices)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{context} must contain at least two indices")
    if values.dtype.kind not in "ui":
        raise TypeError(f"{context} must have an integer dtype")
    expected = np.arange(values.size, dtype=np.int64)
    values64 = values.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(values64), expected):
        raise ValueError(f"{context} must be a permutation")
    if np.any(values64 == expected):
        raise ValueError(f"{context} contains a self donor")


def seeded_derangement(size: int, seed: int) -> np.ndarray:
    """Return a deterministic single-cycle derangement of ``range(size)``."""

    if not _is_positive_int(size) or size < 2:
        raise ValueError("derangement size must be at least two")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(size)
    donors = np.empty(size, dtype=np.int32)
    donors[order] = np.roll(order, -1).astype(np.int32, copy=False)
    validate_derangement(donors, context="seeded donor map")
    return donors
