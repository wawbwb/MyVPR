#!/usr/bin/env python
"""Convert a verified 70x70 ADE20K GSV cache to AG-SLRD layout labels.

The expensive frozen SegFormer pass remains in
``scripts/cache_gsv_patch_semantics.py``.  This converter verifies that source
cache, every source array checksum, every city CSV checksum, and the exact
training-eligible cross-place donor map before applying the source-controlled
150-to-12 class mapping.  It stores only coarse labels and donor indices;
confidence is intentionally not duplicated, but its source SHA256 remains in
the output provenance.

Fresh runs refuse an existing output directory.  ``--resume`` continues only
when the complete immutable input/protocol signature agrees.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_layout_cache import (  # noqa: E402
    SEMANTIC_LAYOUT_CACHE_SCHEMA,
    SEMANTIC_LAYOUT_CACHE_VERSION,
    SEMANTIC_LAYOUT_CLASSES,
    SEMANTIC_LAYOUT_GRID_SIZE,
    SEMANTIC_LAYOUT_IGNORE_INDEX,
    file_sha256,
    remap_ade20k_labels,
    semantic_layout_mapping_record,
    validate_ade20k_patch_cache,
    validate_semantic_layout_cache,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_gsv_index_and_shuffle(
    *,
    dataset_root: Path,
    source_manifest: dict[str, Any],
    shuffled_indices: np.ndarray,
) -> dict[str, Any]:
    """Re-derive eligible rows and prove every donor is cross-place."""

    min_views = source_manifest.get("eligible_min_views")
    if not isinstance(min_views, int) or isinstance(min_views, bool) or min_views < 2:
        raise ValueError("source cache has invalid eligible_min_views")
    num_images = int(source_manifest["num_images"])
    shuffled = np.asarray(shuffled_indices)
    if shuffled.shape != (num_images,) or shuffled.dtype != np.dtype("int32"):
        raise ValueError("source shuffled_indices has an invalid shape/dtype")

    eligible_total = 0
    for city in source_manifest["cities"]:
        name = str(city["name"])
        start = int(city["offset"])
        count = int(city["count"])
        stop = start + count
        csv_path = dataset_root / "Dataframes" / f"{name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"GSV city CSV not found: {csv_path}")
        actual_hash = file_sha256(csv_path)
        if actual_hash != city["sha256"]:
            raise ValueError(
                f"GSV city CSV SHA256 mismatch for {name}: "
                f"expected {city['sha256']}, found {actual_hash}"
            )
        frame = pd.read_csv(csv_path, usecols=["place_id"])
        if len(frame) != count:
            raise ValueError(f"GSV city row count changed for {name}")
        place_ids = frame["place_id"].to_numpy()
        if bool(pd.isna(place_ids).any()):
            raise ValueError(f"GSV city {name} contains a missing place_id")
        place_counts = frame.groupby("place_id")["place_id"].transform("size")
        eligible = np.flatnonzero(place_counts.to_numpy() >= min_views)
        ineligible = np.flatnonzero(place_counts.to_numpy() < min_views)
        if eligible.size != int(city["eligible_count"]):
            raise ValueError(f"GSV eligible row count changed for {name}")

        global_donors = shuffled[start:stop].astype(np.int64, copy=False)
        if np.any(global_donors < start) or np.any(global_donors >= stop):
            raise ValueError(f"GSV donor map crosses city boundaries for {name}")
        local_donors = global_donors - start
        if not np.array_equal(np.sort(local_donors), np.arange(count)):
            raise ValueError(f"GSV donor map is not a city bijection for {name}")
        if ineligible.size and not np.array_equal(local_donors[ineligible], ineligible):
            raise ValueError(f"GSV ineligible rows must self-map for {name}")
        if not np.array_equal(np.sort(local_donors[eligible]), eligible):
            raise ValueError(f"GSV eligible donor set changed for {name}")
        if np.any(place_ids[eligible] == place_ids[local_donors[eligible]]):
            raise ValueError(f"GSV eligible donor is same-place for {name}")
        eligible_total += int(eligible.size)

    return {
        "type": "gsv_city_csv_row",
        "definition": "city CSV offset plus original CSV row ordinal",
        "eligible_min_views": min_views,
        "eligible_rows": eligible_total,
        "shuffle_definition": (
            "within-city bijection; every training-eligible receiver gets a "
            "different-place eligible donor; ineligible rows self-map"
        ),
    }


def build_manifest(
    *,
    dataset_root: Path,
    source_dir: Path,
    source_manifest: dict[str, Any],
    source_hashes: dict[str, str],
    index_record: dict[str, Any],
) -> dict[str, Any]:
    source_manifest_hash = file_sha256(source_dir / "manifest.json")
    return {
        "schema": SEMANTIC_LAYOUT_CACHE_SCHEMA,
        "version": SEMANTIC_LAYOUT_CACHE_VERSION,
        "complete": False,
        "created_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "num_images": int(source_manifest["num_images"]),
        "grid_size": list(SEMANTIC_LAYOUT_GRID_SIZE),
        "num_classes": len(SEMANTIC_LAYOUT_CLASSES),
        "classes": list(SEMANTIC_LAYOUT_CLASSES),
        "ignore_index": SEMANTIC_LAYOUT_IGNORE_INDEX,
        "labels_dtype": "uint8",
        "shuffled_indices_dtype": "int32",
        "label_policy": "argmax_all_patches_no_confidence_threshold",
        "mapping": semantic_layout_mapping_record(),
        "cities": source_manifest["cities"],
        "index": index_record,
        # Kept top-level for the minimal teacher-loader contract.
        "source_manifest_sha256": source_manifest_hash,
        "source": {
            "type": "verified_ade20k_patch_cache",
            "cache_dir": str(source_dir),
            "schema": source_manifest["schema"],
            "version": source_manifest["version"],
            "manifest_sha256": source_manifest_hash,
            "array_sha256": source_hashes,
            "model_name": source_manifest.get("model_name"),
            "requested_revision": source_manifest.get("requested_revision"),
            "resolved_commit": source_manifest.get("resolved_commit"),
            "teacher_input": source_manifest.get("teacher_input"),
            "pooling": source_manifest.get("pooling"),
            "inference_precision": source_manifest.get("inference_precision"),
        },
    }


def resume_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    ignored = {"created_utc", "completed_utc", "complete", "array_sha256", "summary_file"}
    return {key: value for key, value in manifest.items() if key not in ignored}


def summarize_labels(labels: np.ndarray, chunk_rows: int) -> dict[str, Any]:
    counts = np.zeros(len(SEMANTIC_LAYOUT_CLASSES), dtype=np.int64)
    ignored = 0
    for start in range(0, labels.shape[0], chunk_rows):
        chunk = np.asarray(labels[start : start + chunk_rows]).reshape(-1)
        ignored += int(np.count_nonzero(chunk == SEMANTIC_LAYOUT_IGNORE_INDEX))
        valid = chunk != SEMANTIC_LAYOUT_IGNORE_INDEX
        if np.any(chunk[valid] >= len(SEMANTIC_LAYOUT_CLASSES)):
            raise ValueError("converted cache contains an invalid superclass ID")
        counts += np.bincount(
            chunk[valid], minlength=len(SEMANTIC_LAYOUT_CLASSES)
        ).astype(np.int64, copy=False)
    total = int(np.prod(labels.shape, dtype=np.int64))
    return {
        "num_images": int(labels.shape[0]),
        "total_patches": total,
        "ignored_patches": ignored,
        "ignore_fraction": float(ignored / total),
        "classes": [
            {
                "id": index,
                "name": name,
                "count": int(counts[index]),
                "fraction": float(counts[index] / total),
            }
            for index, name in enumerate(SEMANTIC_LAYOUT_CLASSES)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-cache",
        type=Path,
        required=True,
        help="complete v2 ADE20K cache generated directly at grid 70x70",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/ade20k_semantic_layout/gsv_grid70"),
    )
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-source-file-hashes",
        action="store_true",
        help=(
            "skip re-hashing large source arrays only when they were already "
            "verified in the same trusted run; hashes remain bound in output"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_root = args.dataset_root.expanduser().resolve()
    source_dir = args.source_cache.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not (dataset_root / "Dataframes").is_dir():
        raise FileNotFoundError(f"invalid GSV-Cities root: {dataset_root}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source ADE20K cache not found: {source_dir}")
    if output_dir == source_dir:
        raise ValueError("output must differ from source-cache")
    if args.chunk_rows < 1:
        raise ValueError("chunk-rows must be positive")
    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"--resume output not found: {output_dir}")
    elif output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite output: {output_dir}; use --resume only "
            "for an incomplete matching conversion"
        )
    return dataset_root, source_dir, output_dir


def main() -> None:
    args = parse_args()
    dataset_root, source_dir, output_dir = validate_args(args)
    source_manifest, source_arrays, source_hashes = validate_ade20k_patch_cache(
        source_dir,
        verify_array_files=not args.skip_source_file_hashes,
    )
    index_record = validate_gsv_index_and_shuffle(
        dataset_root=dataset_root,
        source_manifest=source_manifest,
        shuffled_indices=source_arrays["shuffled_indices"],
    )
    desired = build_manifest(
        dataset_root=dataset_root,
        source_dir=source_dir,
        source_manifest=source_manifest,
        source_hashes=source_hashes,
        index_record=index_record,
    )
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    label_path = output_dir / "labels.npy"
    shuffle_path = output_dir / "shuffled_indices.npy"
    num_images = int(source_manifest["num_images"])
    shape = (num_images, *SEMANTIC_LAYOUT_GRID_SIZE)

    if args.resume:
        if not manifest_path.is_file() or not progress_path.is_file():
            raise FileNotFoundError("incomplete output lacks manifest/progress JSON")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if resume_signature(manifest) != resume_signature(desired):
            raise ValueError("existing conversion does not match source/protocol")
        if manifest.get("complete") is True:
            validate_semantic_layout_cache(output_dir)
            print(f"Semantic-layout cache is already complete: {output_dir}")
            return
        with progress_path.open("r", encoding="utf-8") as handle:
            next_index = int(json.load(handle)["next_index"])
        labels = np.load(label_path, mmap_mode="r+", allow_pickle=False)
        shuffled = np.load(shuffle_path, mmap_mode="r+", allow_pickle=False)
        if labels.shape != shape or labels.dtype != np.dtype("uint8"):
            raise ValueError("partial labels.npy has an invalid shape/dtype")
        if shuffled.shape != (num_images,) or shuffled.dtype != np.dtype("int32"):
            raise ValueError("partial shuffled_indices.npy is invalid")
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
        manifest = desired
        atomic_json(manifest_path, manifest)
        labels = np.lib.format.open_memmap(
            label_path, mode="w+", dtype="uint8", shape=shape
        )
        shuffled = np.lib.format.open_memmap(
            shuffle_path, mode="w+", dtype="int32", shape=(num_images,)
        )
        shuffled[:] = source_arrays["shuffled_indices"]
        shuffled.flush()
        next_index = 0
        atomic_json(progress_path, {"next_index": 0, "updated_utc": utc_now()})

    if not 0 <= next_index <= num_images:
        raise ValueError(f"invalid conversion cursor {next_index}/{num_images}")
    print(
        f"Convert {num_images} images from ADE20K-150 to "
        f"{len(SEMANTIC_LAYOUT_CLASSES)} layout classes at "
        f"{SEMANTIC_LAYOUT_GRID_SIZE}"
    )
    progress = tqdm(
        total=num_images,
        initial=next_index,
        desc="Build semantic-layout cache",
    )
    for start in range(next_index, num_images, args.chunk_rows):
        stop = min(start + args.chunk_rows, num_images)
        labels[start:stop] = remap_ade20k_labels(source_arrays["labels"][start:stop])
        labels.flush()
        next_index = stop
        atomic_json(
            progress_path,
            {"next_index": next_index, "updated_utc": utc_now()},
        )
        progress.update(stop - start)
    progress.close()

    summary = summarize_labels(labels, args.chunk_rows)
    summary_path = output_dir / "summary.json"
    atomic_json(summary_path, summary)
    manifest["complete"] = True
    manifest["completed_utc"] = utc_now()
    manifest["summary_file"] = summary_path.name
    manifest["array_sha256"] = {
        "labels.npy": file_sha256(label_path),
        "shuffled_indices.npy": file_sha256(shuffle_path),
    }
    atomic_json(manifest_path, manifest)
    validate_semantic_layout_cache(output_dir)
    print(f"Mapping SHA256: {manifest['mapping']['sha256']}")
    print(f"Ignored patches: {summary['ignored_patches']} (expected 0)")
    print(f"Wrote complete semantic-layout cache to: {output_dir}")


if __name__ == "__main__":
    main()
