#!/usr/bin/env python
"""Precompute sparse CLIP semantic-spatial affinities for GSV-Cities.

The cache is keyed by ``city CSV offset + original CSV row ordinal``.  It is
therefore stable across the per-city DataFrame shuffling performed during
training.  Dense 196x196 affinities exist only for the current GPU batch; the
stored representation contains top-k indices/weights and one entropy-based
confidence per patch.

Run this script on clean photographs.  Cached targets must only be paired with
spatially preserving (photometric-only) training augmentation.
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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clip_teacher import CLIPTeacherEncoder  # noqa: E402
from src.utils import config_manager  # noqa: E402


SCHEMA = "openvpr_clip_sparse_affinity"
VERSION = 1
MODEL_NAME = "ViT-B-16"
PRETRAINED = "openai"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def image_name(row: pd.Series) -> str:
    city = row["city_id"]
    place_id = str(int(row["place_id"]) % 10**5).zfill(7)
    year = str(row["year"]).zfill(4)
    month = str(row["month"]).zfill(2)
    northdeg = str(row["northdeg"]).zfill(3)
    lat, lon = str(row["lat"]), str(row["lon"])
    return (
        f"{city}_{place_id}_{year}_{month}_{northdeg}_{lat}_{lon}_"
        f"{row['panoid']}.jpg"
    )


class CityRows(Dataset):
    """A contiguous original-CSV slice with its global cache indices."""

    def __init__(
        self,
        dataset_root: Path,
        dataframe: pd.DataFrame,
        city_offset: int,
        start_row: int,
        transform,
    ) -> None:
        self.dataset_root = dataset_root
        self.dataframe = dataframe.reset_index(drop=True)
        self.city_offset = int(city_offset)
        self.start_row = int(start_row)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe) - self.start_row

    def __getitem__(self, index: int):
        row_ordinal = self.start_row + int(index)
        row = self.dataframe.iloc[row_ordinal]
        path = (
            self.dataset_root
            / "Images"
            / str(row["city_id"])
            / image_name(row)
        )
        if not path.is_file():
            raise FileNotFoundError(f"GSV-Cities image not found: {path}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return self.city_offset + row_ordinal, tensor


def discover_cities(dataset_root: Path) -> tuple[list[dict], int]:
    dataframe_dir = dataset_root / "Dataframes"
    csv_paths = sorted(dataframe_dir.glob("*.csv"), key=lambda path: path.name)
    if not csv_paths:
        raise FileNotFoundError(f"no city CSV files found in {dataframe_dir}")
    entries = []
    offset = 0
    for csv_path in csv_paths:
        # ``chunksize`` yields DataFrames, so count their rows rather than
        # keeping an entire city metadata table resident.  Only place_id is
        # parsed because the remaining columns are irrelevant to the offset.
        count = sum(
            len(chunk)
            for chunk in pd.read_csv(
                csv_path, usecols=["place_id"], chunksize=100_000
            )
        )
        entries.append(
            {
                "name": csv_path.stem,
                "offset": offset,
                "count": count,
                "sha256": sha256(csv_path),
            }
        )
        offset += count
    return entries, offset


def build_manifest(
    dataset_root: Path,
    city_entries: list[dict],
    image_count: int,
    patch_count: int,
    topk: int,
    temperature: float,
    spatial_sigma: float,
) -> dict:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "num_images": image_count,
        "cities": city_entries,
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "teacher_input_size": 224,
        "patch_count": patch_count,
        "grid_size": math.isqrt(patch_count),
        "topk": topk,
        "semantic_temperature": temperature,
        "spatial_sigma": spatial_sigma,
        "diagonal_masked": True,
        "indices_dtype": "uint8",
        "weights_dtype": "float16",
        "confidence_dtype": "float16",
    }


def resume_signature(manifest: dict) -> dict:
    """Fields whose mismatch makes an existing partial cache unsafe."""
    names = (
        "schema",
        "version",
        "num_images",
        "cities",
        "model_name",
        "pretrained",
        "teacher_input_size",
        "patch_count",
        "grid_size",
        "topk",
        "semantic_temperature",
        "spatial_sigma",
        "diagonal_masked",
        "indices_dtype",
        "weights_dtype",
        "confidence_dtype",
    )
    return {name: manifest.get(name) for name in names}


def validate_resume_prefix(
    next_index: int, city_entries: list[dict]
) -> None:
    """Only accept a cursor in the contiguous global city-row ordering."""
    if next_index < 0:
        raise ValueError("resume cursor cannot be negative")
    for city in city_entries:
        start = int(city["offset"])
        stop = start + int(city["count"])
        if start <= next_index <= stop:
            return
    if city_entries:
        final_stop = int(city_entries[-1]["offset"]) + int(
            city_entries[-1]["count"]
        )
        if next_index == final_stop:
            return
    raise ValueError("resume cursor does not lie in the cache city ordering")


def open_arrays(
    output_dir: Path,
    image_count: int,
    patch_count: int,
    topk: int,
    resume: bool,
) -> dict[str, np.memmap]:
    specs = {
        "indices": ("uint8", (image_count, patch_count, topk)),
        "weights": ("float16", (image_count, patch_count, topk)),
        "confidence": ("float16", (image_count, patch_count)),
    }
    arrays = {}
    for name, (dtype, shape) in specs.items():
        path = output_dir / f"{name}.npy"
        if resume:
            if not path.is_file():
                raise FileNotFoundError(f"partial cache is missing {path}")
            array = np.load(path, mmap_mode="r+")
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError(
                    f"partial cache {path} has {array.shape}/{array.dtype}, "
                    f"expected {shape}/{dtype}"
                )
        else:
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing cache array: {path}"
                )
            array = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape
            )
        arrays[name] = array
    return arrays


def semantic_affinity(
    patches: torch.Tensor,
    spatial_penalty: torch.Tensor,
    diagonal_mask: torch.Tensor,
    temperature: float,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = torch.nn.functional.normalize(patches.float(), dim=-1)
    logits = torch.bmm(patches, patches.transpose(1, 2)) / temperature
    logits = logits - spatial_penalty
    logits.masked_fill_(diagonal_mask.unsqueeze(0), -torch.inf)
    affinity = logits.softmax(dim=-1)
    entropy = -(affinity.clamp_min(1e-12) * affinity.clamp_min(1e-12).log()).sum(-1)
    confidence = 1.0 - entropy / math.log(affinity.shape[-1] - 1)
    weights, indices = affinity.topk(topk, dim=-1, largest=True, sorted=True)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return indices, weights, confidence.clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="GSV-Cities root; defaults to config/data/config.yaml:gsv-cities",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--semantic-temperature", type=float, default=0.07)
    parser.add_argument("--spatial-sigma", type=float, default=0.25)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="CUDA device for cache generation (default: cuda:1; GPU 0 is not used)",
    )
    parser.add_argument("--hf-mirror", default="https://hf-mirror.com")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.flush_every < 1:
        raise ValueError("batch-size/flush-every must be positive; num-workers non-negative")
    if args.semantic_temperature <= 0 or args.spatial_sigma <= 0:
        raise ValueError("semantic-temperature and spatial-sigma must be positive")

    dataset_root = (
        Path(config_manager.get_dataset_path("gsv-cities", "train"))
        if args.dataset_root is None
        else args.dataset_root
    ).expanduser().resolve()
    if not (dataset_root / "Dataframes").is_dir() or not (
        dataset_root / "Images"
    ).is_dir():
        raise FileNotFoundError(
            f"invalid GSV-Cities root (need Dataframes/ and Images/): {dataset_root}"
        )
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    city_entries, image_count = discover_cities(dataset_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is None:
        raise ValueError("specify --device cuda:1; GPU 0 is faulty")
    if device.type == "cuda" and device.index == 0:
        raise ValueError("GPU 0 is faulty; use --device cuda:1")
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device} is unavailable")
    torch.set_float32_matmul_precision("high")
    teacher = CLIPTeacherEncoder(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        hf_mirror=args.hf_mirror,
    ).to(device).eval()
    teacher.visual.requires_grad_(False)
    patch_count = int(teacher.visual.positional_embedding.shape[0] - 1)
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise RuntimeError(f"CLIP patch count is not square: {patch_count}")
    if patch_count > 256:
        raise RuntimeError("uint8 sparse indices require at most 256 patches")
    if not 1 <= args.topk < patch_count:
        raise ValueError(f"topk must be in [1, {patch_count - 1}]")

    desired_manifest = build_manifest(
        dataset_root,
        city_entries,
        image_count,
        patch_count,
        args.topk,
        args.semantic_temperature,
        args.spatial_sigma,
    )
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    resume = manifest_path.is_file()
    if resume:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if resume_signature(manifest) != resume_signature(desired_manifest):
            raise ValueError(
                "existing cache manifest does not match this dataset/config; "
                "choose a new --output directory"
            )
        if manifest.get("complete", False):
            print(f"Cache is already complete: {output_dir}")
            return
        if not progress_path.is_file():
            raise FileNotFoundError(
                f"partial cache has no resume cursor: {progress_path}"
            )
        with progress_path.open("r", encoding="utf-8") as handle:
            next_index = int(json.load(handle)["next_index"])
    else:
        manifest = desired_manifest
        atomic_json(manifest_path, manifest)
        next_index = 0
        atomic_json(progress_path, {"next_index": 0})

    if not 0 <= next_index <= image_count:
        raise ValueError(f"invalid resume cursor {next_index}/{image_count}")
    validate_resume_prefix(next_index, city_entries)
    # progress.json is committed only after array.flush(), so every row below
    # this cursor is known durable; later rows are safely overwritten.
    arrays = open_arrays(
        output_dir, image_count, patch_count, args.topk, resume=resume
    )

    transform = T2.Compose(
        [
            T2.ToImage(),
            T2.Resize((224, 224), interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    coords_1d = torch.linspace(0.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    coords = torch.stack((yy, xx), dim=-1).flatten(0, 1)
    spatial_penalty = torch.cdist(coords, coords).square()
    spatial_penalty = spatial_penalty / (2.0 * args.spatial_sigma**2)
    diagonal_mask = torch.eye(patch_count, device=device, dtype=torch.bool)

    batches_since_flush = 0
    for city in city_entries:
        city_start = int(city["offset"])
        city_stop = city_start + int(city["count"])
        if next_index >= city_stop:
            continue
        start_row = max(0, next_index - city_start)
        dataframe = pd.read_csv(
            dataset_root / "Dataframes" / f"{city['name']}.csv"
        )
        dataset = CityRows(
            dataset_root, dataframe, city_start, start_row, transform
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for global_indices, images in loader:
            global_indices_np = global_indices.numpy()
            images = images.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                _, patch_tokens = teacher(images)
                projected = teacher.project_patch_tokens(patch_tokens)
            sparse_indices, sparse_weights, confidence = semantic_affinity(
                projected,
                spatial_penalty,
                diagonal_mask,
                args.semantic_temperature,
                args.topk,
            )
            arrays["indices"][global_indices_np] = sparse_indices.byte().cpu().numpy()
            arrays["weights"][global_indices_np] = sparse_weights.half().cpu().numpy()
            arrays["confidence"][global_indices_np] = confidence.half().cpu().numpy()
            next_index = int(global_indices_np[-1]) + 1
            batches_since_flush += 1
            if batches_since_flush >= args.flush_every:
                for array in arrays.values():
                    array.flush()
                atomic_json(progress_path, {"next_index": next_index})
                batches_since_flush = 0
                print(f"cached {next_index}/{image_count} images", flush=True)

    for array in arrays.values():
        array.flush()
    atomic_json(progress_path, {"next_index": image_count})
    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(f"Wrote complete semantic cache to {output_dir}")


if __name__ == "__main__":
    main()
