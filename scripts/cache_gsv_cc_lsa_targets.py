#!/usr/bin/env python
"""Cache frozen crop-CLS targets for CC-LSA teacher pre-training.

Exactly one SHA256-selected image is used per eligible GSV place.  The clean
280x280 image is split into a fixed 2x2 grid and each crop is encoded by the
pinned frozen OpenCLIP CLS encoder.  The cache is resumable and immutable once
complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cc_lsa_gate_a import (  # noqa: E402
    file_sha256,
    implementation_sha256,
    string_sequence_sha256,
)
from src.dataloaders.train.cc_lsa import (  # noqa: E402
    CC_LSA_IMAGE_SIZE,
    CC_LSA_MODEL_NAME,
    CC_LSA_PRETRAINED,
    CC_LSA_SOURCE_TRANSFORM,
    CC_LSA_TARGET_SCHEMA,
    CC_LSA_TARGET_VERSION,
    GSVPlaceViewDataset,
    discover_gsv_place_views,
    load_target_manifest,
)
from src.models.cc_lsa import crop_image_grid, module_state_sha256  # noqa: E402
from src.models.clip_teacher import CLIPTeacherEncoder  # noqa: E402


TARGET_IMPLEMENTATION_FILES = (
    "src/models/cc_lsa.py",
    "src/dataloaders/train/cc_lsa.py",
    "src/models/clip_teacher.py",
    "scripts/cache_gsv_cc_lsa_targets.py",
)


def target_implementation_sha256() -> dict[str, str]:
    fingerprints = implementation_sha256()
    return {name: fingerprints[name] for name in TARGET_IMPLEMENTATION_FILES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            raise ValueError("specify an explicit CUDA device, e.g. cuda:1")
        if device.index == 0:
            raise ValueError("GPU 0 is faulty on the training machine; use cuda:1")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    return device


def configure_reproducibility() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_static_arrays(
    output: Path,
    *,
    paths: np.ndarray,
    cities: np.ndarray,
    place_ids: np.ndarray,
    split: np.ndarray,
) -> None:
    """Refuse a resume whose immutable identity arrays were altered."""

    expected = {
        "paths.npy": paths,
        "city.npy": cities,
        "place_ids.npy": place_ids,
        "split.npy": split,
    }
    for filename, reference in expected.items():
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(f"partial CC-LSA cache is missing {path}")
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        if cached.shape != reference.shape or cached.dtype != reference.dtype:
            raise ValueError(f"partial CC-LSA identity array differs: {filename}")
        if not np.array_equal(cached, reference):
            raise ValueError(f"partial CC-LSA identity values differ: {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/gsv_cities"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--teacher-chunk-size", type=int, default=32)
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--hf-mirror", default="https://hf-mirror.com")
    parser.add_argument("--image-size", type=int, nargs=2, default=(280, 280))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-views", type=int, default=4)
    parser.add_argument("--holdout-modulus", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility()
    if min(args.batch_size, args.teacher_chunk_size, args.flush_every) < 1:
        raise ValueError("batch/chunk/flush values must be positive")
    if tuple(args.image_size) != CC_LSA_IMAGE_SIZE:
        raise ValueError("registered CC-LSA target image size is exactly 280 280")
    if args.seed != 42 or args.minimum_views != 4 or args.holdout_modulus != 10:
        raise ValueError("registered target protocol requires seed=42, min_views=4, modulus=10")
    if args.model_name != CC_LSA_MODEL_NAME or args.pretrained != CC_LSA_PRETRAINED:
        raise ValueError("registered target teacher is exactly ViT-B-16/openai")

    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = choose_device(args.device)
    records, csv_records = discover_gsv_place_views(
        dataset_root,
        seed=args.seed,
        minimum_views=args.minimum_views,
        holdout_modulus=args.holdout_modulus,
    )
    paths = np.asarray([record.relative_path for record in records])
    cities = np.asarray([record.city for record in records])
    place_ids = np.asarray([record.place_id for record in records], dtype=np.int64)
    split = np.asarray([record.split for record in records], dtype=np.uint8)

    teacher = CLIPTeacherEncoder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        hf_mirror=args.hf_mirror,
    ).to(device)
    teacher.eval()
    semantic_dim = int(teacher.global_dim)
    base_hash = module_state_sha256(teacher.visual)
    desired = {
        "schema": CC_LSA_TARGET_SCHEMA,
        "version": CC_LSA_TARGET_VERSION,
        "complete": False,
        "created_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "csv_files": csv_records,
        "num_places": len(records),
        "num_train_places": int((split == 0).sum()),
        "num_holdout_places": int((split == 1).sum()),
        "path_sequence_sha256": string_sequence_sha256(paths.tolist()),
        "selection": {
            "algorithm": "sha256_one_view_per_place_v1",
            "seed": args.seed,
            "minimum_views": args.minimum_views,
            "split_algorithm": "sha256_place_v1",
            "holdout_modulus": args.holdout_modulus,
        },
        "image_size": list(args.image_size),
        "region_grid": [2, 2],
        "source_transform": CC_LSA_SOURCE_TRANSFORM,
        "semantic_dim": semantic_dim,
        "teacher": {
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "base_model_state_sha256": base_hash,
        },
        "dtype": "float16",
        "software": {
            "torch": str(torch.__version__),
            "open_clip_torch": version("open_clip_torch"),
        },
        "numeric_protocol": {
            "inference": "float32",
            "deterministic_algorithms": True,
            "allow_tf32": False,
        },
        "implementation_sha256": target_implementation_sha256(),
    }
    signature_fields = (
        "schema",
        "version",
        "dataset_root",
        "csv_files",
        "num_places",
        "path_sequence_sha256",
        "selection",
        "image_size",
        "region_grid",
        "source_transform",
        "semantic_dim",
        "teacher",
        "dtype",
        "software",
        "numeric_protocol",
        "implementation_sha256",
    )
    signature = {name: desired[name] for name in signature_fields}
    manifest_path = output / "manifest.json"
    progress_path = output / "progress.json"
    target_path = output / "crop_cls.npy"
    if output.exists():
        if not manifest_path.is_file() or not progress_path.is_file():
            raise FileExistsError(f"existing output is not a resumable cache: {output}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        existing_signature = {name: manifest.get(name) for name in signature_fields}
        if existing_signature != signature:
            raise ValueError("existing CC-LSA target cache signature differs")
        if manifest.get("complete") is True:
            load_target_manifest(output, dataset_root=dataset_root)
            print(f"Complete CC-LSA target cache already exists: {output}")
            return
        if manifest.get("complete") is not False:
            raise ValueError("partial CC-LSA target manifest has invalid state")
        with progress_path.open("r", encoding="utf-8") as handle:
            next_index = int(json.load(handle)["next_index"])
        validate_static_arrays(
            output,
            paths=paths,
            cities=cities,
            place_ids=place_ids,
            split=split,
        )
        targets = np.load(target_path, mmap_mode="r+", allow_pickle=False)
    else:
        output.mkdir(parents=True)
        manifest = desired
        atomic_json(manifest_path, manifest)
        np.save(output / "paths.npy", paths, allow_pickle=False)
        np.save(output / "city.npy", cities, allow_pickle=False)
        np.save(output / "place_ids.npy", place_ids, allow_pickle=False)
        np.save(output / "split.npy", split, allow_pickle=False)
        targets = np.lib.format.open_memmap(
            target_path,
            mode="w+",
            dtype="float16",
            shape=(len(records), 4, semantic_dim),
        )
        next_index = 0
        atomic_json(progress_path, {"next_index": 0, "updated_utc": utc_now()})
    if targets.shape != (len(records), 4, semantic_dim) or targets.dtype != np.float16:
        raise ValueError("partial crop_cls.npy has invalid shape or dtype")
    if not 0 <= next_index <= len(records):
        raise ValueError("partial cache cursor is outside the record range")

    dataset = GSVPlaceViewDataset(dataset_root, records, image_size=tuple(args.image_size))
    subset = Subset(dataset, range(next_index, len(dataset)))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    progress = tqdm(total=len(records), initial=next_index, desc="Cache CC-LSA crop CLS")
    batches_since_flush = 0
    with torch.inference_mode():
        for images, indices in loader:
            index_array = np.asarray(indices, dtype=np.int64)
            expected = np.arange(next_index, next_index + len(index_array))
            if not np.array_equal(index_array, expected):
                raise RuntimeError("CC-LSA target DataLoader changed canonical order")
            images = images.to(device=device, non_blocking=True)
            crops = crop_image_grid(images, rows=2, columns=2)
            embeddings = []
            for chunk in crops.split(args.teacher_chunk_size):
                global_features, _ = teacher(chunk)
                embeddings.append(global_features.float().cpu())
            values = torch.cat(embeddings).reshape(len(images), 4, semantic_dim)
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError("frozen CLIP returned non-finite crop targets")
            targets[index_array] = values.numpy().astype(np.float16)
            next_index += len(index_array)
            progress.update(len(index_array))
            batches_since_flush += 1
            if batches_since_flush >= args.flush_every:
                targets.flush()
                atomic_json(
                    progress_path,
                    {"next_index": next_index, "updated_utc": utc_now()},
                )
                batches_since_flush = 0
    progress.close()
    targets.flush()
    del targets
    if next_index != len(records):
        raise RuntimeError(f"target cache stopped at {next_index}/{len(records)}")
    atomic_json(progress_path, {"next_index": next_index, "updated_utc": utc_now()})
    manifest["complete"] = True
    manifest["completed_utc"] = utc_now()
    filenames = ("paths.npy", "city.npy", "place_ids.npy", "split.npy", "crop_cls.npy")
    manifest["array_sha256"] = {
        filename: file_sha256(output / filename) for filename in filenames
    }
    atomic_json(manifest_path, manifest)
    load_target_manifest(output, dataset_root=dataset_root)
    print(
        f"Wrote {len(records)} crop-CLS rows "
        f"({manifest['num_train_places']} train / {manifest['num_holdout_places']} holdout) "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
