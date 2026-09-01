#!/usr/bin/env python
"""Cache 70x70 coarse ADE20K layouts in canonical MSLS evaluation order.

The cache contains the standard MSLS database followed by the standard
queries, exactly matching ``MapillarySLSDataset``.  SegFormer is frozen and
used only during this offline step.  Its 150 ADE20K classes are converted to
the source-controlled 12-class AG-SLRD alphabet; no class is suppressed and
the dynamic superclass remains explicit.

Fresh runs refuse to overwrite an output directory.  ``--resume`` continues
an incomplete cache only when the dataset, model, mapping and inference
protocol still match its manifest.
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
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_gsv_patch_semantics import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_REVISION,
    load_teacher,
    model_classes,
    processor_record,
)
from src.semantic_layout_cache import (  # noqa: E402
    ADE20K_CLASS_NAME_NORMALIZATION,
    ADE20K_CLASSES,
    ADE20K_TO_SEMANTIC_LAYOUT,
    SEMANTIC_LAYOUT_CACHE_SCHEMA,
    SEMANTIC_LAYOUT_CACHE_VERSION,
    SEMANTIC_LAYOUT_CLASSES,
    SEMANTIC_LAYOUT_GRID_SIZE,
    SEMANTIC_LAYOUT_IGNORE_INDEX,
    file_sha256,
    seeded_derangement,
    semantic_layout_mapping_record,
    validate_ade20k_class_names,
)


INDEX_TYPE = "msls_standard_db_queries_v1"
SHUFFLE_ALGORITHM = "role_preserving_known_place_derangement_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def string_sequence_sha256(values: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values.tolist():
        encoded = str(value).replace("\\", "/").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_relative_paths(values: np.ndarray, *, role: str) -> np.ndarray:
    result: list[str] = []
    for raw_value in values.tolist():
        value = str(raw_value).replace("\\", "/")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value:
            raise ValueError(f"unsafe {role} image path: {value!r}")
        result.append(value)
    normalized = np.asarray(result)
    if len(set(result)) != len(result):
        raise ValueError(f"{role} image manifest contains duplicate paths")
    return normalized


def load_msls_index(
    dataset_root: Path,
) -> tuple[np.ndarray, int, int, tuple[np.ndarray, ...], dict[str, Any]]:
    files = {
        "database": dataset_root / "msls_val_dbImages.npy",
        "queries": dataset_root / "msls_val_qImages.npy",
        "ground_truth": dataset_root / "msls_val_gt_25m.npy",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MSLS index files are missing: {missing}")
    database = normalize_relative_paths(
        np.load(files["database"], allow_pickle=False), role="database"
    )
    queries = normalize_relative_paths(
        np.load(files["queries"], allow_pickle=False), role="query"
    )
    ground_truth = np.load(files["ground_truth"], allow_pickle=True)
    if ground_truth.ndim != 1 or len(ground_truth) != len(queries):
        raise ValueError("MSLS ground truth length does not match query manifest")
    if set(database.tolist()) & set(queries.tolist()):
        raise ValueError("MSLS database and query manifests overlap")
    normalized_ground_truth: list[np.ndarray] = []
    for query_index, positives in enumerate(ground_truth):
        positive_array = np.asarray(positives)
        if (
            positive_array.ndim != 1
            or positive_array.dtype.kind not in "ui"
            or len(positive_array) == 0
            or bool(np.any(positive_array < 0))
            or bool(np.any(positive_array >= len(database)))
        ):
            raise ValueError(f"invalid ground truth for query {query_index}")
        normalized_ground_truth.append(
            np.unique(positive_array.astype(np.int64, copy=False))
        )
    paths = np.concatenate((database, queries))
    records = {
        name: {
            "path": path.name,
            "sha256": file_sha256(path),
        }
        for name, path in files.items()
    }
    records["path_sequence_sha256"] = string_sequence_sha256(paths)
    return (
        paths,
        len(database),
        len(queries),
        tuple(normalized_ground_truth),
        records,
    )


def _repair_forbidden_donors(
    donors: np.ndarray,
    forbidden: list[set[int]],
    *,
    seed: int,
    context: str,
) -> np.ndarray:
    """Keep a bijection while removing every known same-place pairing."""

    donors = np.asarray(donors, dtype=np.int32).copy()
    if len(donors) != len(forbidden):
        raise ValueError(f"{context} forbidden set count is inconsistent")
    rng = np.random.default_rng(seed)
    for receiver in range(len(donors)):
        if int(donors[receiver]) not in forbidden[receiver]:
            continue
        repaired = False
        # The forbidden graph is extremely sparse.  Seeded random trials keep
        # construction linear in practice; a deterministic scan is a complete
        # fallback rather than silently accepting an overlap.
        candidates = rng.integers(0, len(donors), size=min(4096, len(donors)))
        for donor_receiver in candidates.tolist() + list(range(len(donors))):
            donor_receiver = int(donor_receiver)
            if donor_receiver == receiver:
                continue
            left_donor = int(donors[receiver])
            right_donor = int(donors[donor_receiver])
            if (
                right_donor in forbidden[receiver]
                or left_donor in forbidden[donor_receiver]
            ):
                continue
            donors[receiver], donors[donor_receiver] = (
                donors[donor_receiver],
                donors[receiver],
            )
            repaired = True
            break
        if not repaired:
            raise ValueError(
                f"{context} cannot construct a known-cross-place donor for "
                f"receiver {receiver}"
            )
    violations = [
        receiver
        for receiver, donor in enumerate(donors.tolist())
        if int(donor) in forbidden[receiver]
    ]
    if violations:
        raise RuntimeError(
            f"internal error: {context} retains forbidden donors: "
            f"{violations[:5]}"
        )
    return donors


def build_role_preserving_shuffle(
    num_references: int,
    num_queries: int,
    seed: int,
    ground_truth: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    references = seeded_derangement(num_references, seed)
    queries = seeded_derangement(num_queries, seed + 1)
    if ground_truth is not None:
        if len(ground_truth) != num_queries:
            raise ValueError("ground truth count does not match MSLS queries")
        reference_forbidden = [{index} for index in range(num_references)]
        for positives in ground_truth:
            positive_set = set(np.asarray(positives, dtype=np.int64).tolist())
            for reference in positive_set:
                reference_forbidden[reference].update(positive_set)
        query_positive_sets = [set(values.tolist()) for values in ground_truth]
        query_forbidden = [{index} for index in range(num_queries)]
        for left in range(num_queries):
            for right in range(left + 1, num_queries):
                if not query_positive_sets[left].isdisjoint(
                    query_positive_sets[right]
                ):
                    query_forbidden[left].add(right)
                    query_forbidden[right].add(left)
        references = _repair_forbidden_donors(
            references,
            reference_forbidden,
            seed=seed + 2,
            context="MSLS database shuffle",
        )
        queries = _repair_forbidden_donors(
            queries,
            query_forbidden,
            seed=seed + 3,
            context="MSLS query shuffle",
        )
    queries = queries.astype(np.int64)
    queries += num_references
    combined = np.concatenate((references, queries.astype(np.int32)))
    expected = np.arange(num_references + num_queries, dtype=np.int64)
    if (
        combined.dtype != np.dtype("int32")
        or np.unique(combined).size != combined.size
        or bool(np.any(combined.astype(np.int64) == expected))
        or bool(np.any(combined[:num_references] >= num_references))
        or bool(np.any(combined[num_references:] < num_references))
    ):
        raise RuntimeError("internal error: invalid role-preserving shuffle")
    return combined


class MSLSImages(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        paths: np.ndarray,
        start_index: int,
        processor: Any,
    ) -> None:
        self.dataset_root = dataset_root
        self.paths = paths
        self.start_index = int(start_index)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths) - self.start_index

    def __getitem__(self, offset: int) -> tuple[int, torch.Tensor]:
        index = self.start_index + int(offset)
        image_path = self.dataset_root / str(self.paths[index])
        if not image_path.is_file():
            raise FileNotFoundError(f"MSLS image not found: {image_path}")
        with Image.open(image_path) as image:
            processed = self.processor(
                images=image.convert("RGB"), return_tensors="pt"
            )
        pixel_values = processed.get("pixel_values")
        if (
            not torch.is_tensor(pixel_values)
            or pixel_values.ndim != 4
            or pixel_values.shape[0] != 1
            or pixel_values.shape[1] != 3
        ):
            raise ValueError(
                "AutoImageProcessor must return pixel_values with shape "
                "(1,3,H,W)"
            )
        return index, pixel_values[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--msls-path", type=Path, default=Path("datasets/msls-val")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--target-image-size", type=int, nargs=2, default=(280, 280)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            raise ValueError("specify an explicit CUDA device, e.g. cuda:1")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    return device


def resume_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "version",
        "num_images",
        "grid_size",
        "num_classes",
        "classes",
        "ignore_index",
        "mapping",
        "index",
        "source",
        "teacher",
        "protocol",
    )
    return {field: manifest.get(field) for field in fields}


def main() -> None:
    args = parse_args()
    dataset_root = args.msls_path.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MSLS root not found: {dataset_root}")
    if args.batch_size < 1 or args.num_workers < 0 or args.flush_every < 1:
        raise ValueError("invalid batch-size, num-workers or flush-every")
    target_image_size = tuple(int(value) for value in args.target_image_size)
    if len(target_image_size) != 2 or min(target_image_size) < 1:
        raise ValueError("target-image-size must contain two positive integers")
    if any(
        target % grid
        for target, grid in zip(target_image_size, SEMANTIC_LAYOUT_GRID_SIZE)
    ):
        raise ValueError("target-image-size must be divisible by 70x70")
    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError("--resume requires an existing output directory")
    elif output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")

    device = choose_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    (
        paths,
        num_references,
        num_queries,
        ground_truth,
        source_files,
    ) = load_msls_index(
        dataset_root
    )
    shuffle = build_role_preserving_shuffle(
        num_references,
        num_queries,
        int(args.seed),
        ground_truth=ground_truth,
    )
    processor, teacher, transformers_version, commit = load_teacher(
        args.model_name, args.revision, device
    )
    source_classes = tuple(model_classes(teacher))
    validate_ade20k_class_names(source_classes)

    desired_manifest: dict[str, Any] = {
        "schema": SEMANTIC_LAYOUT_CACHE_SCHEMA,
        "version": SEMANTIC_LAYOUT_CACHE_VERSION,
        "complete": False,
        "created_utc": utc_now(),
        "num_images": len(paths),
        "grid_size": list(SEMANTIC_LAYOUT_GRID_SIZE),
        "num_classes": len(SEMANTIC_LAYOUT_CLASSES),
        "classes": list(SEMANTIC_LAYOUT_CLASSES),
        "ignore_index": SEMANTIC_LAYOUT_IGNORE_INDEX,
        "mapping": semantic_layout_mapping_record(),
        "index": {
            "type": INDEX_TYPE,
            "num_references": num_references,
            "num_queries": num_queries,
            "dataset_root": str(dataset_root),
            "files": source_files,
        },
        "source": {
            "kind": "direct_frozen_segformer",
            "image_order": "database_then_standard_queries",
        },
        "teacher": {
            "model_name": args.model_name,
            "requested_revision": args.revision,
            "resolved_commit": commit,
            "source_classes": list(source_classes),
            "class_name_normalization": ADE20K_CLASS_NAME_NORMALIZATION,
            "transformers_version": transformers_version,
            "torch_version": str(torch.__version__),
            "processor": processor_record(processor),
        },
        "protocol": {
            "teacher_input": "clean_rgb",
            "target_image_size": list(target_image_size),
            "pooling": (
                "bilinear_logits_to_target_then_softmax_then_"
                "nonoverlap_avg_pool"
            ),
            "label_rule": "argmax_pooled_probability_then_fixed_mapping",
            "inference_precision": (
                "amp_float16" if args.amp else "float32"
            ),
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
            "shuffle_seed": int(args.seed),
            "shuffle_definition": (
                "independent deterministic single-cycle derangements for "
                "database and query roles, repaired so no donor shares a "
                "known MSLS positive-place set with its receiver"
            ),
        },
        "labels_dtype": "uint8",
        "shuffled_indices_dtype": "int32",
    }

    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    labels_path = output_dir / "labels.npy"
    shuffle_path = output_dir / "shuffled_indices.npy"
    if args.resume:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("complete") is True:
            print(f"Cache is already complete: {output_dir}")
            return
        if resume_signature(existing) != resume_signature(desired_manifest):
            raise ValueError("existing MSLS cache does not match this run")
        with progress_path.open("r", encoding="utf-8") as handle:
            next_index = int(json.load(handle)["next_index"])
        labels = np.load(labels_path, mmap_mode="r+", allow_pickle=False)
        cached_shuffle = np.load(shuffle_path, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(cached_shuffle, shuffle):
            raise ValueError("cached shuffled_indices.npy is inconsistent")
        manifest = existing
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
        manifest = desired_manifest
        atomic_json(manifest_path, manifest)
        labels = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype="uint8",
            shape=(len(paths), *SEMANTIC_LAYOUT_GRID_SIZE),
        )
        np.save(shuffle_path, shuffle, allow_pickle=False)
        next_index = 0
        atomic_json(progress_path, {"next_index": 0, "updated_utc": utc_now()})

    expected_shape = (len(paths), *SEMANTIC_LAYOUT_GRID_SIZE)
    if labels.shape != expected_shape or labels.dtype != np.dtype("uint8"):
        raise ValueError("partial labels.npy has an invalid shape or dtype")
    if not 0 <= next_index <= len(paths):
        raise ValueError("partial cache cursor is outside the image index")

    dataset = MSLSImages(dataset_root, paths, next_index, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    lookup = torch.tensor(
        ADE20K_TO_SEMANTIC_LAYOUT, dtype=torch.long, device=device
    )
    patch_size = tuple(
        target // grid
        for target, grid in zip(target_image_size, SEMANTIC_LAYOUT_GRID_SIZE)
    )
    progress = tqdm(
        total=len(paths), initial=next_index, desc="Cache MSLS semantic layouts"
    )
    batches_since_flush = 0
    try:
        for indices, pixel_values in loader:
            index_array = indices.numpy().astype(np.int64, copy=False)
            expected = np.arange(
                next_index, next_index + len(index_array), dtype=np.int64
            )
            if not np.array_equal(index_array, expected):
                raise RuntimeError("DataLoader changed canonical MSLS order")
            pixel_values = pixel_values.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(args.amp),
            ):
                output = teacher(pixel_values=pixel_values)
            logits = getattr(output, "logits", None)
            if (
                not torch.is_tensor(logits)
                or logits.ndim != 4
                or logits.shape[1] != len(ADE20K_CLASSES)
            ):
                raise ValueError("SegFormer returned invalid logits")
            resized = F.interpolate(
                logits.float(),
                size=target_image_size,
                mode="bilinear",
                align_corners=False,
            )
            pooled = F.avg_pool2d(
                resized.softmax(dim=1),
                kernel_size=patch_size,
                stride=patch_size,
            )
            coarse = lookup[pooled.argmax(dim=1)]
            labels[index_array] = coarse.to(torch.uint8).cpu().numpy()
            next_index = int(index_array[-1]) + 1
            progress.update(len(index_array))
            batches_since_flush += 1
            if batches_since_flush >= args.flush_every:
                labels.flush()
                atomic_json(
                    progress_path,
                    {"next_index": next_index, "updated_utc": utc_now()},
                )
                batches_since_flush = 0
    finally:
        progress.close()

    if next_index != len(paths):
        raise RuntimeError(f"cache stopped early at {next_index}/{len(paths)}")
    labels.flush()
    atomic_json(
        progress_path,
        {"next_index": len(paths), "updated_utc": utc_now()},
    )
    manifest["array_sha256"] = {
        "labels.npy": file_sha256(labels_path),
        "shuffled_indices.npy": file_sha256(shuffle_path),
    }
    manifest["complete"] = True
    manifest["completed_utc"] = utc_now()
    atomic_json(manifest_path, manifest)
    print(
        f"Wrote {len(paths)} layouts ({num_references} database + "
        f"{num_queries} queries) to {output_dir}"
    )


if __name__ == "__main__":
    main()
