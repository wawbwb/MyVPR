"""Generate auditable full-database MSLS-val condition query splits.

The standard MSLS-val database is kept unchanged. Query candidates are read
from the official per-city subtask metadata, panorama queries are excluded,
and 25 metre positives are recomputed against the standard database using
city-scoped UTM coordinates. Unlike the standard 740-query manifest, these
custom screening splits retain condition-only queries.

These files support a fast causal screen against one shared full database.
They are not the official MSLS condition subtasks, which also filter the
database by subtask.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_category_prior import (  # noqa: E402
    canonical_image_paths,
    file_sha256,
    string_sequence_sha256,
    validate_ground_truth,
)
from src.dataloaders.valid.msls_condition_protocol import (  # noqa: E402
    CONDITION_COLUMNS,
    CONDITION_FILES as CONDITION_OUTPUTS,
    CONDITION_ORDER,
    CONDITION_UNION_QUERY_FILE as UNION_QUERY_FILENAME,
)


SEASON_OUTPUTS = CONDITION_OUTPUTS["season"]
SEASON_MEMBERS = ("winter2summer", "summer2winter")


@dataclass(frozen=True)
class CityQueryMetadata:
    """Aligned query metadata needed for condition selection and GT."""

    paths: np.ndarray
    coordinates: np.ndarray
    panorama: np.ndarray
    selected: Mapping[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-standard-database night/winter2summer/"
            "summer2winter MSLS-val query splits"
        )
    )
    parser.add_argument(
        "--msls-path", type=Path, default=Path("datasets/msls-val")
    )
    parser.add_argument("--cities", nargs="+", default=("cph", "sf"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_ORDER,
        default=CONDITION_ORDER,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("doc/msls_condition_split_audit.json"),
    )
    parser.add_argument("--distance-threshold", type=float, default=25.0)
    parser.add_argument("--expected-standard-queries", type=int, default=740)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing generated condition manifests/report",
    )
    return parser.parse_args()


def _boolean_array(values: pd.Series, *, source: str) -> np.ndarray:
    """Parse a metadata boolean column without truthiness surprises."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = values.to_numpy()
        if not np.isin(numeric, (0, 1)).all():
            raise ValueError(f"{source} contains values other than 0/1")
        return numeric.astype(bool)

    normalised = values.astype(str).str.strip().str.lower()
    accepted = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalised.tolist()) - set(accepted))
    if unknown:
        raise ValueError(f"{source} contains unsupported values: {unknown}")
    return normalised.map(accepted).to_numpy(dtype=bool)


def _validate_aligned_frames(
    city: str,
    role: str,
    frames: Sequence[tuple[str, pd.DataFrame]],
) -> None:
    for frame_name, frame in frames:
        if not frame.index.is_unique:
            raise ValueError(f"{city} {frame_name} {role} index is not unique")
    reference_name, reference = frames[0]
    for frame_name, frame in frames[1:]:
        if not reference.index.equals(frame.index):
            raise ValueError(
                f"{city} {reference_name}/{frame_name} {role} index order differs"
            )

    if "key" not in reference.columns:
        raise ValueError(f"{city} {reference_name} {role} metadata has no key column")
    reference_keys = reference["key"].astype(str).to_numpy()
    if len(np.unique(reference_keys)) != len(reference_keys):
        raise ValueError(f"{city} {reference_name} {role} keys are not unique")
    for frame_name, frame in frames[1:]:
        if "key" in frame.columns:
            keys = frame["key"].astype(str).to_numpy()
            if not np.array_equal(reference_keys, keys):
                raise ValueError(
                    f"{city} {reference_name}/{frame_name} {role} key order differs"
                )


def _coordinate_array(frame: pd.DataFrame, *, source: Path) -> np.ndarray:
    required = ("easting", "northing")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} has no coordinate columns: {missing}")
    try:
        coordinates = frame.loc[:, list(required)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} contains non-numeric UTM coordinates") from exc
    if coordinates.shape != (len(frame), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"{source} contains invalid UTM coordinates")
    return coordinates


def _load_city_query_metadata(msls_path: Path, city: str) -> CityQueryMetadata:
    query_dir = msls_path / city / "query"
    metadata_path = query_dir / "postprocessed.csv"
    raw_path = query_dir / "raw.csv"
    subtask_path = query_dir / "subtask_index.csv"
    for path in (metadata_path, raw_path, subtask_path):
        if not path.is_file():
            raise FileNotFoundError(f"required MSLS metadata not found: {path}")

    metadata = pd.read_csv(metadata_path, index_col=0, dtype={"key": str})
    raw = pd.read_csv(raw_path, index_col=0, dtype={"key": str})
    subtasks = pd.read_csv(subtask_path, index_col=0, dtype={"key": str})
    if not (len(metadata) == len(raw) == len(subtasks)):
        raise ValueError(f"{city} query metadata/raw/subtask row counts do not match")
    _validate_aligned_frames(
        city,
        "query",
        (
            ("postprocessed", metadata),
            ("raw", raw),
            ("subtask", subtasks),
        ),
    )
    if "pano" not in raw.columns:
        raise ValueError(f"{raw_path} has no pano column")

    keys = metadata["key"].astype(str).to_numpy()
    paths = canonical_image_paths(
        [f"{city}/query/images/{key}.jpg" for key in keys]
    )
    panorama = _boolean_array(raw["pano"], source=f"{raw_path}:pano")
    selected: dict[str, np.ndarray] = {}
    for condition in CONDITION_ORDER:
        column_masks: list[np.ndarray] = []
        for column in CONDITION_COLUMNS[condition]:
            if column not in subtasks.columns:
                raise ValueError(f"{subtask_path} has no {column!r} column")
            column_masks.append(
                _boolean_array(
                    subtasks[column], source=f"{subtask_path}:{column}"
                )
            )
        selected[condition] = np.logical_or.reduce(column_masks)
    return CityQueryMetadata(
        paths=paths,
        coordinates=_coordinate_array(metadata, source=metadata_path),
        panorama=panorama,
        selected=selected,
    )


def _parse_manifest_path(path: str, *, expected_role: str) -> tuple[str, str]:
    canonical = str(path).replace("\\", "/")
    parts = PurePosixPath(canonical).parts
    if (
        len(parts) != 4
        or parts[1] != expected_role
        or parts[2] != "images"
        or PurePosixPath(parts[3]).suffix.lower() != ".jpg"
    ):
        raise ValueError(
            f"unexpected MSLS {expected_role} manifest path: {canonical!r}"
        )
    return parts[0], PurePosixPath(parts[3]).stem


def _load_city_database_coordinate_lookup(
    msls_path: Path, city: str
) -> dict[str, np.ndarray]:
    metadata_path = msls_path / city / "database" / "postprocessed.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"required MSLS metadata not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path, index_col=0, dtype={"key": str})
    _validate_aligned_frames(city, "database", (("postprocessed", metadata),))
    keys = metadata["key"].astype(str).to_numpy()
    coordinates = _coordinate_array(metadata, source=metadata_path)
    return {key: coordinates[index] for index, key in enumerate(keys.tolist())}


def build_standard_database_coordinates(
    msls_path: Path,
    database_paths: Sequence[Any],
    cities: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Align the standard DB manifest to city metadata with global indices."""

    canonical_paths = canonical_image_paths(database_paths)
    metadata_by_city = {
        city: _load_city_database_coordinate_lookup(msls_path, city)
        for city in cities
    }
    indices_by_city: dict[str, list[int]] = {city: [] for city in cities}
    coordinates_by_city: dict[str, list[np.ndarray]] = {city: [] for city in cities}
    seen: set[tuple[str, str]] = set()
    for global_index, path in enumerate(canonical_paths.tolist()):
        city, key = _parse_manifest_path(path, expected_role="database")
        if city not in metadata_by_city:
            raise ValueError(
                f"standard database path belongs to undeclared city {city!r}: {path}"
            )
        identity = (city, key)
        if identity in seen:
            raise ValueError(f"duplicate standard database city/key: {identity}")
        seen.add(identity)
        coordinate = metadata_by_city[city].get(key)
        if coordinate is None:
            raise ValueError(
                f"standard database image has no metadata row: {path!r}"
            )
        indices_by_city[city].append(global_index)
        coordinates_by_city[city].append(coordinate)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for city in cities:
        if not indices_by_city[city]:
            raise ValueError(f"standard database has no images for city {city!r}")
        indices = np.asarray(indices_by_city[city], dtype=np.int64)
        coordinates = np.asarray(coordinates_by_city[city], dtype=np.float64)
        if coordinates.shape != (len(indices), 2):
            raise AssertionError("database coordinate alignment is inconsistent")
        result[city] = indices, coordinates
    return result


def compute_full_database_ground_truth(
    query_path: str,
    query_coordinate: np.ndarray,
    database_by_city: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    distance_threshold: float,
) -> np.ndarray:
    """Return global standard-DB indices within the inclusive UTM radius."""

    city, _ = _parse_manifest_path(query_path, expected_role="query")
    if city not in database_by_city:
        raise ValueError(f"query belongs to city absent from standard DB: {query_path}")
    global_indices, database_coordinates = database_by_city[city]
    coordinate = np.asarray(query_coordinate, dtype=np.float64)
    if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
        raise ValueError(f"invalid query UTM coordinate for {query_path!r}")
    squared_distances = np.square(database_coordinates - coordinate).sum(axis=1)
    positives = global_indices[squared_distances <= distance_threshold**2]
    return np.asarray(positives, dtype=np.int64)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_save_npy(path: Path, values: np.ndarray, *, allow_pickle: bool) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.save(temporary_path, values, allow_pickle=allow_pickle)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _object_array(values: Sequence[np.ndarray]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = np.asarray(value, dtype=np.int64)
    return result


def main() -> None:
    args = parse_args()
    args.msls_path = args.msls_path.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    args.cities = tuple(dict.fromkeys(str(city) for city in args.cities))
    args.conditions = tuple(
        dict.fromkeys(str(condition) for condition in args.conditions)
    )
    if not args.cities:
        raise ValueError("--cities must not be empty")
    if not args.conditions:
        raise ValueError("--conditions must not be empty")
    if set(args.conditions) != set(CONDITION_ORDER):
        raise ValueError(
            "--conditions must include all generated condition manifests: "
            f"{list(CONDITION_ORDER)}"
        )
    # The order is part of the query-union/cache protocol, irrespective of CLI
    # ordering.
    args.conditions = CONDITION_ORDER
    if args.expected_standard_queries <= 0:
        raise ValueError("expected-standard-queries must be positive")
    if not math.isfinite(args.distance_threshold) or args.distance_threshold <= 0:
        raise ValueError("distance-threshold must be finite and positive")
    if args.report.suffix.lower() != ".json":
        raise ValueError("--report must end with .json")
    if not args.msls_path.is_dir():
        raise FileNotFoundError(f"MSLS path not found: {args.msls_path}")

    db_path = args.msls_path / "msls_val_dbImages.npy"
    query_path = args.msls_path / "msls_val_qImages.npy"
    gt_path = args.msls_path / "msls_val_gt_25m.npy"
    for path in (db_path, query_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(f"standard MSLS manifest not found: {path}")

    database_paths = canonical_image_paths(np.load(db_path, allow_pickle=False))
    standard_query_paths = canonical_image_paths(
        np.load(query_path, allow_pickle=False)
    )
    if len(set(database_paths.tolist())) != len(database_paths):
        raise ValueError("standard MSLS database manifest contains duplicate paths")
    if len(set(standard_query_paths.tolist())) != len(standard_query_paths):
        raise ValueError("standard MSLS query manifest contains duplicate paths")
    if set(database_paths.tolist()) & set(standard_query_paths.tolist()):
        raise ValueError("standard MSLS database and query manifests overlap")
    standard_ground_truth = validate_ground_truth(
        np.load(gt_path, allow_pickle=True),
        num_queries=len(standard_query_paths),
        num_references=len(database_paths),
        dataset_name="standard MSLS",
    )
    if len(standard_query_paths) != args.expected_standard_queries:
        raise ValueError(
            f"standard MSLS has {len(standard_query_paths)} queries; expected "
            f"{args.expected_standard_queries}"
        )

    query_cities = {
        _parse_manifest_path(path, expected_role="query")[0]
        for path in standard_query_paths.tolist()
    }
    database_cities = {
        _parse_manifest_path(path, expected_role="database")[0]
        for path in database_paths.tolist()
    }
    if set(args.cities) != query_cities or set(args.cities) != database_cities:
        raise ValueError(
            f"--cities {list(args.cities)} must exactly cover standard query "
            f"cities {sorted(query_cities)} and DB cities {sorted(database_cities)}"
        )

    database_by_city = build_standard_database_coordinates(
        args.msls_path, database_paths, args.cities
    )
    city_query_metadata = {
        city: _load_city_query_metadata(args.msls_path, city)
        for city in args.cities
    }
    standard_gt_by_path = {
        path: np.asarray(standard_ground_truth[index], dtype=np.int64)
        for index, path in enumerate(standard_query_paths.tolist())
    }

    generated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    audit_data: dict[str, dict[str, Any]] = {}
    for condition in args.conditions:
        candidate_paths: list[str] = []
        candidate_coordinates: list[np.ndarray] = []
        panorama_paths: list[str] = []
        for city in args.cities:
            metadata = city_query_metadata[city]
            selected = np.asarray(metadata.selected[condition], dtype=bool)
            selected_indices = np.flatnonzero(selected)
            for row_index in selected_indices.tolist():
                path = str(metadata.paths[row_index])
                if not (args.msls_path / path).is_file():
                    raise FileNotFoundError(
                        f"{condition} candidate image does not exist: "
                        f"{args.msls_path / path}"
                    )
                if bool(metadata.panorama[row_index]):
                    panorama_paths.append(path)
                    continue
                candidate_paths.append(path)
                candidate_coordinates.append(metadata.coordinates[row_index])

        if len(set(candidate_paths)) != len(candidate_paths):
            raise ValueError(f"{condition} contains duplicate candidate paths")
        retained_paths: list[str] = []
        retained_gt: list[np.ndarray] = []
        no_positive_paths: list[str] = []
        overlap_count = 0
        for path, coordinate in zip(candidate_paths, candidate_coordinates):
            positives = compute_full_database_ground_truth(
                path,
                coordinate,
                database_by_city,
                distance_threshold=args.distance_threshold,
            )
            standard_positives = standard_gt_by_path.get(path)
            if standard_positives is not None:
                overlap_count += 1
                if not np.array_equal(
                    np.sort(positives), np.sort(standard_positives)
                ):
                    raise ValueError(
                        f"{condition} recomputed ground truth disagrees with "
                        f"standard MSLS for overlapping query {path!r}: "
                        f"computed={positives.tolist()}, "
                        f"standard={standard_positives.tolist()}"
                    )
            if len(positives) == 0:
                no_positive_paths.append(path)
                continue
            retained_paths.append(path)
            retained_gt.append(positives)

        if not retained_paths:
            raise ValueError(f"{condition} has no non-panorama queries with positives")
        retained_path_array = canonical_image_paths(retained_paths)
        retained_gt_array = _object_array(retained_gt)
        generated[condition] = (retained_path_array, retained_gt_array)
        retained_set = set(retained_paths)
        retained_overlap = sum(
            path in standard_gt_by_path for path in retained_paths
        )
        if retained_overlap != overlap_count:
            raise ValueError(
                f"{condition} overlapping standard query unexpectedly has no positive"
            )
        audit_data[condition] = {
            "subtask_columns": list(CONDITION_COLUMNS[condition]),
            "candidate_queries_before_panorama_exclusion": (
                len(candidate_paths) + len(panorama_paths)
            ),
            "excluded_panorama_queries": len(panorama_paths),
            "excluded_panorama_paths": panorama_paths,
            "candidate_queries_after_panorama_exclusion": len(candidate_paths),
            "excluded_no_positive_queries": len(no_positive_paths),
            "excluded_no_positive_paths": no_positive_paths,
            "retained_queries": len(retained_paths),
            "standard_query_overlap": retained_overlap,
            "condition_only_queries": len(retained_set - set(standard_gt_by_path)),
            "all_candidate_images_verified_present": True,
        }

    season_paths: list[str] = []
    season_ground_truth: list[np.ndarray] = []
    season_index_by_path: dict[str, int] = {}
    duplicate_memberships = 0
    for condition in SEASON_MEMBERS:
        member_paths, member_ground_truth = generated[condition]
        for path, positives in zip(
            member_paths.tolist(), member_ground_truth.tolist()
        ):
            existing_index = season_index_by_path.get(path)
            positives = np.asarray(positives, dtype=np.int64)
            if existing_index is not None:
                duplicate_memberships += 1
                if not np.array_equal(
                    np.sort(season_ground_truth[existing_index]),
                    np.sort(positives),
                ):
                    raise ValueError(
                        "season member manifests disagree on ground truth for "
                        f"shared query {path!r}"
                    )
                continue
            season_index_by_path[path] = len(season_paths)
            season_paths.append(path)
            season_ground_truth.append(positives)
    season_path_array = canonical_image_paths(season_paths)
    season_gt_array = _object_array(season_ground_truth)
    season_overlap = sum(path in standard_gt_by_path for path in season_paths)
    season_audit: dict[str, Any] = {
        "member_conditions": list(SEASON_MEMBERS),
        "merge_order": list(SEASON_MEMBERS),
        "merge_method": "stable first occurrence with exact-GT deduplication",
        "duplicate_query_memberships": duplicate_memberships,
        "retained_queries": len(season_paths),
        "standard_query_overlap": season_overlap,
        "condition_only_queries": len(season_paths) - season_overlap,
    }

    union_paths = standard_query_paths.tolist()
    seen_union = set(union_paths)
    for condition in args.conditions:
        condition_paths = generated[condition][0].tolist()
        for path in condition_paths:
            if path not in seen_union:
                seen_union.add(path)
                union_paths.append(path)
    union_path_array = canonical_image_paths(union_paths)
    if not np.array_equal(
        union_path_array[: len(standard_query_paths)], standard_query_paths
    ):
        raise AssertionError("condition union does not preserve standard-query prefix")
    condition_sets = {
        condition: set(generated[condition][0].tolist())
        for condition in args.conditions
    }
    membership_values: list[int] = []
    for path in union_path_array.tolist():
        membership = 0
        for bit, condition in enumerate(args.conditions):
            if path in condition_sets[condition]:
                membership |= 1 << bit
        membership_values.append(membership)
    membership_counts = {
        str(value): membership_values.count(value)
        for value in sorted(set(membership_values))
    }
    singleton_membership_paths = [
        path
        for path, membership in zip(union_path_array.tolist(), membership_values)
        if membership_counts[str(membership)] == 1
    ]

    union_path = args.msls_path / UNION_QUERY_FILENAME
    output_paths = [union_path]
    for condition in args.conditions:
        query_filename, gt_filename = CONDITION_OUTPUTS[condition]
        output_paths.extend(
            (args.msls_path / query_filename, args.msls_path / gt_filename)
        )
    season_query_filename, season_gt_filename = SEASON_OUTPUTS
    season_query_path = args.msls_path / season_query_filename
    season_gt_path = args.msls_path / season_gt_filename
    output_paths.extend((season_query_path, season_gt_path))
    all_targets = (*output_paths, args.report)
    if len(set(all_targets)) != len(all_targets):
        raise ValueError("generated manifest and report paths must be distinct")
    existing = [path for path in all_targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite generated files; inspect them or rerun with "
            f"--force: {[str(path) for path in existing]}"
        )
    previous_files = {
        str(path): _file_record(path) for path in output_paths if path.is_file()
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(union_path, union_path_array, allow_pickle=False)
    for condition in args.conditions:
        query_filename, gt_filename = CONDITION_OUTPUTS[condition]
        output_query_path = args.msls_path / query_filename
        output_gt_path = args.msls_path / gt_filename
        condition_paths, condition_gt = generated[condition]
        _atomic_save_npy(output_query_path, condition_paths, allow_pickle=False)
        _atomic_save_npy(output_gt_path, condition_gt, allow_pickle=True)
        audit_data[condition]["outputs"] = {
            "queries": _file_record(output_query_path),
            "ground_truth": _file_record(output_gt_path),
        }
        audit_data[condition]["query_paths_sha256"] = string_sequence_sha256(
            condition_paths
        )
        audit_data[condition]["replaced_manifests"] = {
            "queries": previous_files.get(str(output_query_path)),
            "ground_truth": previous_files.get(str(output_gt_path)),
        }
        print(
            f"{condition}: "
            f"{audit_data[condition]['candidate_queries_before_panorama_exclusion']} "
            f"candidates -> {len(condition_paths)} retained "
            f"({audit_data[condition]['excluded_panorama_queries']} panorama, "
            f"{audit_data[condition]['excluded_no_positive_queries']} no-positive; "
            f"{audit_data[condition]['standard_query_overlap']} standard overlap, "
            f"{audit_data[condition]['condition_only_queries']} condition-only)"
        )

    _atomic_save_npy(season_query_path, season_path_array, allow_pickle=False)
    _atomic_save_npy(season_gt_path, season_gt_array, allow_pickle=True)
    season_audit["outputs"] = {
        "queries": _file_record(season_query_path),
        "ground_truth": _file_record(season_gt_path),
    }
    season_audit["query_paths_sha256"] = string_sequence_sha256(
        season_path_array
    )
    season_audit["replaced_manifests"] = {
        "queries": previous_files.get(str(season_query_path)),
        "ground_truth": previous_files.get(str(season_gt_path)),
    }
    print(
        f"season aggregate: {len(season_path_array)} retained "
        f"({duplicate_memberships} duplicate memberships; "
        f"{season_overlap} standard overlap, "
        f"{len(season_path_array) - season_overlap} condition-only)"
    )

    report = {
        "schema_version": 2,
        "method": "full_standard_database_condition_query_splits",
        "protocol": (
            "custom full condition-query sets searched against the standard full "
            "MSLS-val database; not official condition-filtered MSLS subtasks"
        ),
        "msls_path": str(args.msls_path),
        "cities": list(args.cities),
        "distance_threshold_metres": args.distance_threshold,
        "distance_comparison": "inclusive <= threshold",
        "standard": {
            "num_references": len(database_paths),
            "num_queries": len(standard_query_paths),
            "expected_num_queries": args.expected_standard_queries,
            "manifests": {
                "database": _file_record(db_path),
                "queries": _file_record(query_path),
                "ground_truth": _file_record(gt_path),
            },
        },
        "conditions": audit_data,
        "aggregates": {"season": season_audit},
        "query_union": {
            "ordered_conditions": list(args.conditions),
            "condition_bits": {
                str(bit): condition
                for bit, condition in enumerate(args.conditions)
            },
            "condition_membership_counts": membership_counts,
            "singleton_condition_membership_paths": singleton_membership_paths,
            "num_queries": len(union_path_array),
            "num_standard_queries": len(standard_query_paths),
            "num_unique_condition_only_queries": (
                len(union_path_array) - len(standard_query_paths)
            ),
            "standard_queries_are_exact_prefix": True,
            "query_paths_sha256": string_sequence_sha256(union_path_array),
            "output": _file_record(union_path),
            "replaced_manifest": previous_files.get(str(union_path)),
        },
    }
    _atomic_write_json(args.report, report)
    print(
        f"Query union: {len(standard_query_paths)} standard + "
        f"{len(union_path_array) - len(standard_query_paths)} unique extras = "
        f"{len(union_path_array)}"
    )
    print(f"Audit report: {args.report}")


if __name__ == "__main__":
    main()
