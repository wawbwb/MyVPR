"""Generate auditable condition slices of the standard MSLS-val protocol.

The resulting night and season query manifests are strict subsets of
``msls_val_qImages.npy`` and reuse the corresponding rows of
``msls_val_gt_25m.npy``. The database therefore remains the standard full
MSLS-val database. These are custom robustness slices, not the official MSLS
condition subtasks, which also select a condition-specific database.

The previous generator selected rows from ``subtask_index.csv`` without the
official panorama exclusion and recomputed positives from merged city
metadata. That could introduce queries outside the standard 740-query
universe and ambiguous cross-city coordinate mappings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_category_prior import (  # noqa: E402
    canonical_image_paths,
    file_sha256,
    validate_ground_truth,
)


CONDITION_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "night": ("n2d",),
    "season": ("w2s", "s2w"),
}
CONDITION_OUTPUTS: Mapping[str, tuple[str, str]] = {
    "night": ("msls_val_night_qImages.npy", "msls_val_night_gt_25m.npy"),
    "season": ("msls_val_season_qImages.npy", "msls_val_season_gt_25m.npy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate night/season subsets of the standard MSLS-val query "
            "universe and reuse its ground truth"
        )
    )
    parser.add_argument(
        "--msls-path", type=Path, default=Path("datasets/msls-val")
    )
    parser.add_argument("--cities", nargs="+", default=("cph", "sf"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITION_COLUMNS),
        default=tuple(CONDITION_COLUMNS),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("doc/msls_condition_split_audit.json"),
    )
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


def _load_city_candidates(
    msls_path: Path,
    city: str,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return selected paths and selected panoramas for one city."""

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
        raise ValueError(
            f"{city} query metadata/raw/subtask row counts do not match"
        )
    for frame_name, frame in (
        ("postprocessed", metadata),
        ("raw", raw),
        ("subtask", subtasks),
    ):
        if not frame.index.is_unique:
            raise ValueError(f"{city} {frame_name} query index is not unique")
    if not metadata.index.equals(raw.index) or not metadata.index.equals(
        subtasks.index
    ):
        raise ValueError(
            f"{city} postprocessed/raw/subtask query index order differs"
        )
    if "key" not in metadata.columns:
        raise ValueError(f"{metadata_path} has no key column")
    if "pano" not in raw.columns:
        raise ValueError(f"{raw_path} has no pano column")

    keys = metadata["key"].astype(str).to_numpy()
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"{city} query metadata contains duplicate keys")
    if "key" in raw.columns:
        raw_keys = raw["key"].astype(str).to_numpy()
        if not np.array_equal(keys, raw_keys):
            raise ValueError(f"{city} postprocessed/raw query key order differs")
    if "key" in subtasks.columns:
        subtask_keys = subtasks["key"].astype(str).to_numpy()
        if not np.array_equal(keys, subtask_keys):
            raise ValueError(
                f"{city} postprocessed/subtask query key order differs"
            )
    panorama = _boolean_array(raw["pano"], source=f"{raw_path}:pano")

    selected_by_condition: dict[str, set[str]] = {}
    panoramas_by_condition: dict[str, set[str]] = {}
    for condition, columns in CONDITION_COLUMNS.items():
        column_masks = []
        for column in columns:
            if column not in subtasks.columns:
                raise ValueError(f"{subtask_path} has no {column!r} column")
            column_masks.append(
                _boolean_array(
                    subtasks[column], source=f"{subtask_path}:{column}"
                )
            )
        selected = np.logical_or.reduce(column_masks)
        paths = np.asarray(
            [f"{city}/query/images/{key}.jpg" for key in keys],
            dtype=np.str_,
        )
        selected_by_condition[condition] = set(paths[selected].tolist())
        panoramas_by_condition[condition] = set(
            paths[selected & panorama].tolist()
        )
    return selected_by_condition, panoramas_by_condition


def select_standard_condition_subset(
    standard_query_paths: Sequence[Any],
    standard_ground_truth: Sequence[np.ndarray],
    candidate_paths: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intersect candidates with standard queries while preserving their order."""

    standard_paths = canonical_image_paths(standard_query_paths)
    if len(set(standard_paths.tolist())) != len(standard_paths):
        raise ValueError("standard MSLS query manifest contains duplicate paths")
    if len(standard_paths) != len(standard_ground_truth):
        raise ValueError("standard MSLS query/ground-truth count mismatch")
    candidates = set(canonical_image_paths(candidate_paths).tolist())
    standard_set = set(standard_paths.tolist())
    selected_indices = np.asarray(
        [
            query_index
            for query_index, path in enumerate(standard_paths.tolist())
            if path in candidates
        ],
        dtype=np.int64,
    )
    selected_paths = standard_paths[selected_indices]
    selected_ground_truth = np.empty(len(selected_indices), dtype=object)
    for output_index, query_index in enumerate(selected_indices.tolist()):
        selected_ground_truth[output_index] = np.asarray(
            standard_ground_truth[query_index], dtype=np.int64
        ).copy()
    excluded = sorted(candidates - standard_set)
    return selected_paths, selected_ground_truth, excluded


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_save_npy(path: Path, values: np.ndarray, *, allow_pickle: bool) -> None:
    """Replace one generated NPY only after its complete sibling write."""

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


def main() -> None:
    args = parse_args()
    args.msls_path = args.msls_path.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    args.cities = tuple(dict.fromkeys(str(city) for city in args.cities))
    args.conditions = tuple(sorted(set(args.conditions)))
    if args.expected_standard_queries <= 0:
        raise ValueError("expected-standard-queries must be positive")
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
    standard_cities = sorted(
        {path.split("/", maxsplit=1)[0] for path in standard_query_paths.tolist()}
    )
    if set(args.cities) != set(standard_cities):
        raise ValueError(
            f"--cities {list(args.cities)} must exactly cover the standard "
            f"query cities {standard_cities}"
        )

    candidates = {condition: set() for condition in CONDITION_COLUMNS}
    panoramas = {condition: set() for condition in CONDITION_COLUMNS}
    for city in args.cities:
        city_candidates, city_panoramas = _load_city_candidates(
            args.msls_path, city
        )
        for condition in CONDITION_COLUMNS:
            candidates[condition].update(city_candidates[condition])
            panoramas[condition].update(city_panoramas[condition])

    generated: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    output_paths: list[Path] = []
    for condition in args.conditions:
        selected_paths, selected_gt, excluded = select_standard_condition_subset(
            standard_query_paths,
            standard_ground_truth,
            candidates[condition],
        )
        if len(selected_paths) == 0:
            raise ValueError(f"{condition} has no queries in standard MSLS")
        retained_panoramas = set(selected_paths.tolist()) & panoramas[condition]
        if retained_panoramas:
            raise ValueError(
                f"standard MSLS unexpectedly contains {len(retained_panoramas)} "
                f"{condition} panorama queries"
            )
        generated[condition] = (selected_paths, selected_gt, excluded)
        query_filename, gt_filename = CONDITION_OUTPUTS[condition]
        output_paths.extend(
            (args.msls_path / query_filename, args.msls_path / gt_filename)
        )

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
    condition_records: dict[str, Any] = {}
    for condition, (selected_paths, selected_gt, excluded) in generated.items():
        query_filename, gt_filename = CONDITION_OUTPUTS[condition]
        output_query_path = args.msls_path / query_filename
        output_gt_path = args.msls_path / gt_filename
        _atomic_save_npy(
            output_query_path, selected_paths, allow_pickle=False
        )
        _atomic_save_npy(output_gt_path, selected_gt, allow_pickle=True)

        excluded_panoramas = sorted(set(excluded) & panoramas[condition])
        excluded_nonpanoramas = sorted(set(excluded) - panoramas[condition])
        condition_records[condition] = {
            "subtask_columns": list(CONDITION_COLUMNS[condition]),
            "candidate_queries_before_standard_intersection": len(
                candidates[condition]
            ),
            "retained_standard_queries": len(selected_paths),
            "excluded_outside_standard_queries": len(excluded),
            "excluded_panorama_queries": len(excluded_panoramas),
            "excluded_other_queries": len(excluded_nonpanoramas),
            "excluded_paths": excluded,
            "outputs": {
                "queries": _file_record(output_query_path),
                "ground_truth": _file_record(output_gt_path),
            },
            "replaced_manifests": {
                "queries": previous_files.get(str(output_query_path)),
                "ground_truth": previous_files.get(str(output_gt_path)),
            },
        }
        print(
            f"{condition}: {len(candidates[condition])} candidates -> "
            f"{len(selected_paths)} standard queries; excluded {len(excluded)} "
            f"({len(excluded_panoramas)} panorama, "
            f"{len(excluded_nonpanoramas)} other)"
        )

    report = {
        "schema_version": 1,
        "method": "standard_msls_condition_query_slices",
        "protocol": (
            "custom condition subsets of the standard MSLS query universe, "
            "searched against the standard full database; not official "
            "condition-filtered MSLS subtasks"
        ),
        "msls_path": str(args.msls_path),
        "cities": list(args.cities),
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
        "conditions": condition_records,
    }
    _atomic_write_json(args.report, report)
    print(f"Audit report: {args.report}")


if __name__ == "__main__":
    main()
