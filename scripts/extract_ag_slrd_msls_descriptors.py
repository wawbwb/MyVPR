#!/usr/bin/env python
"""Extract canonical MSLS descriptors for the AG-SLRD Phase-0 audit.

Two subcommands deliberately share the same database-then-query order:

``ru``
    Extract the frozen RGB RU+BoQ checkpoint from standard MSLS images.
``semantic``
    Extract an AG-SLRD layout teacher from the immutable MSLS layout cache,
    using either aligned layouts or the registered shuffled-layout control.

Each command writes one ``.npy`` descriptor matrix plus an adjacent JSON
provenance record.  Existing outputs are never overwritten.
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
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders.train.semantic_layout import (  # noqa: E402
    SemanticLayoutIndexDataset,
)
from src.models.ag_slrd import (  # noqa: E402
    file_sha256,
    load_semantic_layout_teacher,
)
from src.semantic_layout_cache import (  # noqa: E402
    SEMANTIC_LAYOUT_CLASSES,
    semantic_layout_mapping_record,
    validate_semantic_layout_cache,
)


MSLS_INDEX_TYPE = "msls_standard_db_queries_v1"
OUTPUT_SCHEMA = "openvpr_ag_slrd_msls_descriptors"
OUTPUT_VERSION = 1


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


def metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def validate_common(args: argparse.Namespace) -> tuple[Path, Path, torch.device]:
    output = args.output.expanduser().resolve()
    sidecar = metadata_path(output)
    if output.suffix.lower() != ".npy":
        raise ValueError("--output must end with .npy")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {sidecar}")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    device = choose_device(args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, sidecar, device


def write_descriptor_batches(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    total_rows: int,
    output: Path,
    device: torch.device,
    description: str,
) -> tuple[int, int]:
    model.eval()
    descriptor_memmap: np.memmap | None = None
    next_index = 0
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with torch.inference_mode():
            for inputs, indices in tqdm(loader, desc=description):
                index_array = np.asarray(indices, dtype=np.int64)
                expected = np.arange(
                    next_index, next_index + len(index_array), dtype=np.int64
                )
                if not np.array_equal(index_array, expected):
                    raise RuntimeError("descriptor DataLoader changed canonical order")
                inputs = inputs.to(device=device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    descriptors = model(inputs)
                if isinstance(descriptors, (tuple, list)):
                    descriptors = descriptors[0]
                if not torch.is_tensor(descriptors) or descriptors.ndim != 2:
                    raise ValueError("model must return a 2-D descriptor tensor")
                values = descriptors.detach().float().cpu().numpy()
                if not bool(np.isfinite(values).all()):
                    raise ValueError("model returned non-finite descriptors")
                if descriptor_memmap is None:
                    descriptor_memmap = np.lib.format.open_memmap(
                        temporary,
                        mode="w+",
                        dtype="float32",
                        shape=(total_rows, values.shape[1]),
                    )
                elif values.shape[1] != descriptor_memmap.shape[1]:
                    raise ValueError("descriptor width changed between batches")
                descriptor_memmap[index_array] = values
                next_index += len(index_array)
        if descriptor_memmap is None or next_index != total_rows:
            raise RuntimeError(
                f"descriptor extraction stopped at {next_index}/{total_rows}"
            )
        descriptor_dim = int(descriptor_memmap.shape[1])
        descriptor_memmap.flush()
        del descriptor_memmap
        os.replace(temporary, output)
        return next_index, descriptor_dim
    except BaseException:
        if descriptor_memmap is not None:
            del descriptor_memmap
        if temporary.exists():
            temporary.unlink()
        raise


def extract_ru(args: argparse.Namespace) -> None:
    # Keep the RGB stack lazy: semantic-only extraction and ``--help`` do not
    # need FAISS, torchvision, DINOv2 or the historical checkpoint loader.
    from scripts.eval_condition_robustness import (
        build_transform,
        load_inference_model_from_ckpt,
    )
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset

    output, sidecar, device = validate_common(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    msls_root = args.msls_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RU checkpoint not found: {checkpoint}")
    dataset = MapillarySLSDataset(
        dataset_path=msls_root,
        input_transform=build_transform(tuple(args.image_size)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_inference_model_from_ckpt(checkpoint, device)
    rows, dimension = write_descriptor_batches(
        model=model,
        loader=loader,
        total_rows=len(dataset),
        output=output,
        device=device,
        description="Extract frozen RU",
    )
    record = {
        "schema": OUTPUT_SCHEMA,
        "version": OUTPUT_VERSION,
        "complete": True,
        "created_utc": utc_now(),
        "kind": "ru",
        "order": "msls_standard_database_then_queries",
        "num_rows": rows,
        "num_references": dataset.num_references,
        "num_queries": dataset.num_queries,
        "descriptor_dim": dimension,
        "descriptor_sha256": file_sha256(output),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "msls_path": str(msls_root),
        "image_size": list(args.image_size),
    }
    atomic_json(sidecar, record)
    print(f"Wrote RU descriptors: {output}")


def extract_semantic(args: argparse.Namespace) -> None:
    output, sidecar, device = validate_common(args)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    cache_dir = args.layout_cache.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"semantic teacher checkpoint not found: {checkpoint_path}"
        )
    manifest, _, cache_hashes = validate_semantic_layout_cache(
        cache_dir,
        verify_array_files=True,
        expected_index_type=MSLS_INDEX_TYPE,
    )
    index = manifest["index"]
    num_references = int(index.get("num_references", -1))
    num_queries = int(index.get("num_queries", -1))
    if num_references < 1 or num_queries < 1:
        raise ValueError("MSLS layout cache has invalid role counts")
    if num_references + num_queries != int(manifest["num_images"]):
        raise ValueError("MSLS layout cache role counts do not cover all rows")

    model, checkpoint = load_semantic_layout_teacher(
        checkpoint_path, map_location="cpu"
    )
    cache_record = checkpoint.get("cache")
    if not isinstance(cache_record, dict):
        raise ValueError("semantic teacher checkpoint has no cache provenance")
    expected_mapping = semantic_layout_mapping_record()["sha256"]
    if cache_record.get("mapping_sha256") != expected_mapping:
        raise ValueError("teacher mapping differs from the MSLS layout mapping")
    if model.num_classes != len(SEMANTIC_LAYOUT_CLASSES):
        raise ValueError("teacher class count differs from MSLS layout cache")
    model = model.to(device)
    receiver_indices = np.arange(manifest["num_images"], dtype=np.int64)
    dataset = SemanticLayoutIndexDataset(
        cache_dir,
        receiver_indices,
        mode=args.selection,
        verify_cache_hashes=False,
        expected_index_type=MSLS_INDEX_TYPE,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    rows, dimension = write_descriptor_batches(
        model=model,
        loader=loader,
        total_rows=len(dataset),
        output=output,
        device=device,
        description=(
            f"Extract {checkpoint.get('mode', 'unknown')} teacher / "
            f"{args.selection} layouts"
        ),
    )
    record = {
        "schema": OUTPUT_SCHEMA,
        "version": OUTPUT_VERSION,
        "complete": True,
        "created_utc": utc_now(),
        "kind": "semantic_layout",
        "teacher_training_mode": checkpoint.get("mode"),
        "layout_selection": args.selection,
        "order": "msls_standard_database_then_queries",
        "num_rows": rows,
        "num_references": num_references,
        "num_queries": num_queries,
        "descriptor_dim": dimension,
        "descriptor_sha256": file_sha256(output),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "layout_cache": {
            "path": str(cache_dir),
            "manifest_sha256": file_sha256(cache_dir / "manifest.json"),
            "array_sha256": cache_hashes,
        },
    }
    atomic_json(sidecar, record)
    print(f"Wrote semantic-layout descriptors: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    ru = subparsers.add_parser("ru", help="extract the frozen RGB RU checkpoint")
    ru.add_argument("--checkpoint", type=Path, required=True)
    ru.add_argument("--msls-path", type=Path, default=Path("datasets/msls-val"))
    ru.add_argument("--image-size", type=int, nargs=2, default=(280, 280))
    ru.add_argument("--output", type=Path, required=True)
    ru.add_argument("--device", default="cuda:1")
    ru.add_argument("--batch-size", type=int, default=32)
    ru.add_argument("--num-workers", type=int, default=8)

    semantic = subparsers.add_parser(
        "semantic", help="extract a semantic-layout teacher checkpoint"
    )
    semantic.add_argument("--checkpoint", type=Path, required=True)
    semantic.add_argument("--layout-cache", type=Path, required=True)
    semantic.add_argument(
        "--selection", choices=("aligned", "shuffled"), required=True
    )
    semantic.add_argument("--output", type=Path, required=True)
    semantic.add_argument("--device", default="cuda:1")
    semantic.add_argument("--batch-size", type=int, default=256)
    semantic.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "ru":
        extract_ru(args)
    elif args.kind == "semantic":
        extract_semantic(args)
    else:  # pragma: no cover - argparse enforces the choices.
        raise RuntimeError(f"unsupported descriptor kind: {args.kind}")


if __name__ == "__main__":
    main()
