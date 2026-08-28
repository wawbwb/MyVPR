#!/usr/bin/env python
"""Cache frozen ADE20K patch labels for Query-conditioned Semantic BoQ.

The cache is keyed by ``city CSV offset + original CSV row ordinal`` and is
therefore stable across the per-city DataFrame shuffling used by GSV-Cities
training. A frozen SegFormer teacher sees each clean photograph. Its logits
are resized to the VPR image size, converted to probabilities and averaged
over exact non-overlapping DINO patch cells; the cache stores the winning
class as uint8 and its probability quantised to uint8.

For the wrong-image semantic control, ``shuffled_indices.npy`` contains a
fixed bijection within every city.  Training-eligible rows are first stably
grouped by ``place_id`` and then rotated so every donor belongs to a different
place than its receiver, including when the original CSV row order admits no
single valid circular shift.

Fresh runs refuse to reuse an existing output directory.  Interrupted runs
can be continued with ``--resume``.  Array data is flushed before the durable
progress cursor is advanced, so rows at or beyond the cursor are always safe
to overwrite.
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
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_SCHEMA,
    QUERY_SEMANTIC_CACHE_VERSION,
    QUERY_SEMANTIC_SHUFFLE_ALGORITHM,
    build_cross_place_bijection,
)

SCHEMA = QUERY_SEMANTIC_CACHE_SCHEMA
VERSION = QUERY_SEMANTIC_CACHE_VERSION
DEFAULT_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def image_name(row: pd.Series) -> str:
    """Match ``GSVCitiesDataset.get_img_name`` on an original CSV row."""

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


def discover_cities(
    dataset_root: Path,
    eligible_min_views: int,
) -> tuple[list[dict[str, Any]], int, np.ndarray]:
    """Build the stable index and a training-eligible shuffle bijection."""

    dataframe_dir = dataset_root / "Dataframes"
    csv_paths = sorted(dataframe_dir.glob("*.csv"), key=lambda path: path.name)
    if not csv_paths:
        raise FileNotFoundError(f"no city CSV files found in {dataframe_dir}")

    entries: list[dict[str, Any]] = []
    shuffled_parts: list[np.ndarray] = []
    offset = 0
    for csv_path in csv_paths:
        place_frame = pd.read_csv(csv_path, usecols=["place_id"])
        count = len(place_frame)
        if count == 0:
            raise ValueError(f"city CSV is empty: {csv_path}")
        place_ids = place_frame["place_id"].to_numpy()
        place_counts = place_frame.groupby("place_id")["place_id"].transform(
            "size"
        )
        eligible_positions = np.flatnonzero(
            place_counts.to_numpy() >= eligible_min_views
        )
        if eligible_positions.size < 2:
            raise ValueError(
                f"city {csv_path.stem!r} has fewer than two training-eligible "
                f"images at min_views={eligible_min_views}"
            )
        eligible_place_ids = place_ids[eligible_positions]
        eligible_donors, rotation = build_cross_place_bijection(
            eligible_place_ids,
            context=f"city {csv_path.stem!r}",
        )
        local_donors = np.arange(count, dtype=np.int64)
        local_donors[eligible_positions] = eligible_positions[eligible_donors]
        if bool(
            np.any(
                place_ids[eligible_positions]
                == place_ids[local_donors[eligible_positions]]
            )
        ):
            raise RuntimeError(
                f"internal error: eligible shuffle for {csv_path.stem} "
                "contains a same-place donor"
            )
        shuffled_parts.append((offset + local_donors).astype(np.int32))
        entries.append(
            {
                "name": csv_path.stem,
                "offset": offset,
                "count": count,
                "sha256": file_sha256(csv_path),
                "eligible_count": int(eligible_positions.size),
                "eligible_shuffle_rotation": rotation,
            }
        )
        offset += count

    shuffled_indices = np.concatenate(shuffled_parts)
    if shuffled_indices.shape != (offset,):
        raise RuntimeError("internal error: shuffled index length mismatch")
    if offset > np.iinfo(np.int32).max:
        raise ValueError("GSV cache is too large for int32 shuffled indices")
    return entries, offset, shuffled_indices


class CityRows(Dataset):
    """An original-CSV city suffix with its global cache row indices."""

    def __init__(
        self,
        dataset_root: Path,
        dataframe: pd.DataFrame,
        city_offset: int,
        start_row: int,
        processor: Any,
    ) -> None:
        self.dataset_root = dataset_root
        self.dataframe = dataframe.reset_index(drop=True)
        self.city_offset = int(city_offset)
        self.start_row = int(start_row)
        self.processor = processor
        if not 0 <= self.start_row <= len(self.dataframe):
            raise ValueError("start_row lies outside the city DataFrame")

    def __len__(self) -> int:
        return len(self.dataframe) - self.start_row

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        row_ordinal = self.start_row + int(index)
        row = self.dataframe.iloc[row_ordinal]
        image_path = (
            self.dataset_root
            / "Images"
            / str(row["city_id"])
            / image_name(row)
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"GSV-Cities image not found: {image_path}")
        with Image.open(image_path) as image:
            processed = self.processor(
                images=image.convert("RGB"), return_tensors="pt"
            )
        pixel_values = processed.get("pixel_values")
        if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
            raise TypeError("AutoImageProcessor did not return 4D pixel_values")
        if pixel_values.shape[0] != 1 or pixel_values.shape[1] != 3:
            raise ValueError(
                "AutoImageProcessor pixel_values must have shape (1,3,H,W), "
                f"got {tuple(pixel_values.shape)}"
            )
        return self.city_offset + row_ordinal, pixel_values[0]


def normalise_processor_size(size: Any) -> tuple[int, int]:
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            height, width = int(size["height"]), int(size["width"])
        elif "shortest_edge" in size:
            height = width = int(size["shortest_edge"])
        else:
            raise ValueError(f"unsupported AutoImageProcessor size: {size!r}")
    elif isinstance(size, int):
        height = width = int(size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        height, width = int(size[0]), int(size[1])
    else:
        raise ValueError(f"unsupported AutoImageProcessor size: {size!r}")
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid AutoImageProcessor size: {(height, width)}")
    return height, width


def processor_record(processor: Any) -> dict[str, Any]:
    size = normalise_processor_size(getattr(processor, "size", None))
    image_mean = [float(value) for value in processor.image_mean]
    image_std = [float(value) for value in processor.image_std]
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("SegFormer processor must provide three-channel mean/std")
    if not bool(getattr(processor, "do_resize", True)):
        raise ValueError("SegFormer processor must enable resizing")
    if not bool(getattr(processor, "do_rescale", True)):
        raise ValueError("SegFormer processor must enable image rescaling")
    if not bool(getattr(processor, "do_normalize", True)):
        raise ValueError("SegFormer processor must enable image normalisation")
    return {
        "class": processor.__class__.__name__,
        "size": list(size),
        "image_mean": image_mean,
        "image_std": image_std,
        "rescale_factor": float(getattr(processor, "rescale_factor", 1.0 / 255.0)),
        "resample": str(getattr(processor, "resample", "unknown")),
    }


def model_classes(model: torch.nn.Module) -> list[str]:
    num_classes = int(model.config.num_labels)
    if not 1 <= num_classes <= 256:
        raise ValueError(
            f"uint8 labels require 1 <= num_classes <= 256, got {num_classes}"
        )
    id2label = {
        int(class_id): str(name)
        for class_id, name in dict(model.config.id2label).items()
    }
    expected = set(range(num_classes))
    if set(id2label) != expected:
        raise ValueError(
            "SegFormer id2label must contain contiguous IDs "
            f"0..{num_classes - 1}"
        )
    return [id2label[class_id] for class_id in range(num_classes)]


def resolved_commit(processor: Any, model: torch.nn.Module) -> str | None:
    processor_dict = (
        processor.to_dict() if callable(getattr(processor, "to_dict", None)) else {}
    )
    candidates = (
        getattr(model.config, "_commit_hash", None),
        getattr(processor, "_commit_hash", None),
        getattr(processor, "init_kwargs", {}).get("_commit_hash"),
        processor_dict.get("_commit_hash"),
    )
    commits = [str(value) for value in candidates if value]
    if len(set(commits)) > 1:
        raise ValueError(
            "processor and model resolved to different Hugging Face commits: "
            f"{sorted(set(commits))}"
        )
    return commits[0] if commits else None


def load_teacher(
    model_name: str,
    revision: str,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, str, str | None]:
    try:
        import transformers
        from transformers import (
            AutoImageProcessor,
            SegformerForSemanticSegmentation,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install it in the VPR environment "
            "before generating the ADE20K cache"
        ) from exc

    processor = AutoImageProcessor.from_pretrained(
        model_name, revision=revision
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, revision=revision
    )
    commit = resolved_commit(processor, model)
    immutable_revision = (
        len(revision) == 40
        and all(character in "0123456789abcdefABCDEF" for character in revision)
    )
    if immutable_revision:
        if commit is not None and commit.lower() != revision.lower():
            raise ValueError(
                "Hugging Face resolved a different commit than the requested "
                f"immutable revision: {commit} != {revision}"
            )
        # Older transformers versions do not always expose _commit_hash, but
        # a full 40-hex revision is already an immutable identity.
        commit = revision.lower()
    model.to(device).eval().requires_grad_(False)
    return processor, model, str(transformers.__version__), commit


def build_manifest(
    *,
    dataset_root: Path,
    city_entries: list[dict[str, Any]],
    image_count: int,
    model_name: str,
    revision: str,
    commit: str | None,
    transformers_version: str,
    processor_info: dict[str, Any],
    classes: list[str],
    grid_size: tuple[int, int],
    target_image_size: tuple[int, int],
    eligible_min_views: int,
    use_amp: bool,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "complete": False,
        "created_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "num_images": image_count,
        "cities": city_entries,
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_commit": commit,
        "transformers_version": transformers_version,
        "torch_version": str(torch.__version__),
        "processor": processor_info,
        "num_classes": len(classes),
        "classes": classes,
        "grid_size": list(grid_size),
        "target_image_size": list(target_image_size),
        "eligible_min_views": eligible_min_views,
        "labels_dtype": "uint8",
        "confidence_dtype": "uint8",
        "shuffled_indices_dtype": "int32",
        "shuffle_algorithm": QUERY_SEMANTIC_SHUFFLE_ALGORITHM,
        "confidence_quantization": "round(clamp(top1_probability,0,1)*255)",
        "shuffle_definition": (
            "rows from places with at least eligible_min_views form a fixed "
            "within-city bijection made by stable place grouping followed by "
            "a largest-place-size rotation; other rows map to themselves"
        ),
        "teacher_input": "clean_rgb",
        "pooling": (
            "bilinear_logits_to_target_then_softmax_then_nonoverlap_avg_pool"
        ),
        "inference_precision": "amp_float16" if use_amp else "float32",
        "device_type": device.type,
    }


def resume_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return every field whose change could alter a cached target row."""

    names = (
        "schema",
        "version",
        "num_images",
        "cities",
        "model_name",
        "transformers_version",
        "torch_version",
        "processor",
        "num_classes",
        "classes",
        "grid_size",
        "target_image_size",
        "eligible_min_views",
        "labels_dtype",
        "confidence_dtype",
        "shuffled_indices_dtype",
        "shuffle_algorithm",
        "confidence_quantization",
        "shuffle_definition",
        "teacher_input",
        "pooling",
        "inference_precision",
        "device_type",
    )
    signature = {name: manifest.get(name) for name in names}
    # Different revision spellings (for example ``main`` and its immutable
    # commit hash) are equivalent when they resolve to the same weights.  If
    # the installed transformers version cannot expose the resolved commit,
    # fall back to the requested revision so resume remains conservative.
    signature["model_revision_identity"] = (
        manifest.get("resolved_commit") or manifest.get("requested_revision")
    )
    return signature


def open_target_arrays(
    output_dir: Path,
    image_count: int,
    grid_size: tuple[int, int],
    resume: bool,
) -> dict[str, np.memmap]:
    specs = {
        "labels": (np.dtype("uint8"), (image_count, *grid_size)),
        "confidence": (np.dtype("uint8"), (image_count, *grid_size)),
    }
    arrays: dict[str, np.memmap] = {}
    for name, (dtype, shape) in specs.items():
        path = output_dir / f"{name}.npy"
        if resume:
            if not path.is_file():
                raise FileNotFoundError(f"partial cache is missing {path}")
            array = np.load(path, mmap_mode="r+")
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"partial cache {path} has {array.shape}/{array.dtype}, "
                    f"expected {shape}/{dtype}"
                )
        else:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite cache array: {path}")
            array = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape
            )
        arrays[name] = array
    return arrays


def create_or_validate_shuffle_array(
    output_dir: Path,
    expected: np.ndarray,
    resume: bool,
) -> None:
    path = output_dir / "shuffled_indices.npy"
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"partial cache is missing {path}")
        cached = np.load(path, mmap_mode="r")
        if cached.shape != expected.shape or cached.dtype != np.dtype("int32"):
            raise ValueError(
                f"invalid shuffled index array {cached.shape}/{cached.dtype}; "
                f"expected {expected.shape}/int32"
            )
        if not np.array_equal(cached, expected):
            raise ValueError(
                "shuffled_indices.npy does not match the current CSV place IDs"
            )
        return

    if path.exists():
        raise FileExistsError(f"refusing to overwrite shuffle array: {path}")
    shuffled = np.lib.format.open_memmap(
        path, mode="w+", dtype="int32", shape=expected.shape
    )
    shuffled[:] = expected
    shuffled.flush()


def summarize_cache(
    labels: np.ndarray,
    confidence: np.ndarray,
    classes: list[str],
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    """Compute compact cache diagnostics without loading all rows into RAM."""
    class_counts = np.zeros(len(classes), dtype=np.int64)
    confidence_sum = 0
    threshold_counts = {threshold: 0 for threshold in (0.5, 0.6, 0.7)}
    total_patches = int(np.prod(labels.shape, dtype=np.int64))
    for start in range(0, labels.shape[0], chunk_rows):
        stop = min(start + chunk_rows, labels.shape[0])
        label_chunk = np.asarray(labels[start:stop]).reshape(-1)
        confidence_chunk = np.asarray(confidence[start:stop]).reshape(-1)
        if label_chunk.size and int(label_chunk.max()) >= len(classes):
            raise ValueError("cached label lies outside the teacher class range")
        class_counts += np.bincount(
            label_chunk, minlength=len(classes)
        ).astype(np.int64, copy=False)
        confidence_sum += int(confidence_chunk.sum(dtype=np.uint64))
        for threshold in threshold_counts:
            quantized_threshold = math.ceil(threshold * 255.0)
            threshold_counts[threshold] += int(
                np.count_nonzero(confidence_chunk >= quantized_threshold)
            )
    class_rows = [
        {
            "id": class_id,
            "name": name,
            "count": int(class_counts[class_id]),
            "fraction": float(class_counts[class_id] / total_patches),
        }
        for class_id, name in enumerate(classes)
    ]
    top_classes = sorted(
        class_rows, key=lambda row: row["count"], reverse=True
    )[:15]
    return {
        "num_images": int(labels.shape[0]),
        "total_patches": total_patches,
        "mean_confidence": float(confidence_sum / (255.0 * total_patches)),
        "confidence_coverage": {
            str(threshold): float(count / total_patches)
            for threshold, count in threshold_counts.items()
        },
        "top_classes": top_classes,
        "classes": class_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="GSV-Cities root; defaults to config/data/config.yaml:gsv-cities",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--grid-size", type=int, nargs=2, default=(20, 20))
    parser.add_argument(
        "--target-image-size",
        type=int,
        nargs=2,
        default=(280, 280),
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "VPR input size whose non-overlapping patches define the cache "
            "grid (default: 280 280)"
        ),
    )
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument(
        "--eligible-min-views",
        type=int,
        default=4,
        help="match GSVCities img_per_place for the shuffled control",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing incomplete cache after exact validation",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="run the frozen teacher with CUDA float16 autocast",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, torch.device]:
    if args.dataset_root is None:
        from src.utils import config_manager

        dataset_root = Path(
            config_manager.get_dataset_path("gsv-cities", "train")
        )
    else:
        dataset_root = args.dataset_root
    dataset_root = dataset_root.expanduser().resolve()
    if not (dataset_root / "Dataframes").is_dir() or not (
        dataset_root / "Images"
    ).is_dir():
        raise FileNotFoundError(
            "invalid GSV-Cities root; expected Dataframes/ and Images/: "
            f"{dataset_root}"
        )

    output_dir = args.output.expanduser().resolve()
    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"--resume requires an existing cache directory: {output_dir}"
            )
    elif output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output; use a new path or --resume: "
            f"{output_dir}"
        )

    if (
        args.batch_size < 1
        or args.num_workers < 0
        or args.flush_every < 1
        or args.eligible_min_views < 2
    ):
        raise ValueError(
            "batch-size/flush-every must be positive, eligible-min-views at "
            "least 2, and num-workers non-negative"
        )
    if len(args.grid_size) != 2 or min(args.grid_size) <= 0:
        raise ValueError("grid-size must contain two positive integers")
    if len(args.target_image_size) != 2 or min(args.target_image_size) <= 0:
        raise ValueError(
            "target-image-size must contain two positive integers"
        )
    if any(
        target % grid
        for target, grid in zip(args.target_image_size, args.grid_size)
    ):
        raise ValueError(
            "target-image-size must be exactly divisible by grid-size so "
            "cached patches are non-overlapping"
        )
    if not args.model_name or not args.revision:
        raise ValueError("model-name and revision must be non-empty")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp currently requires a CUDA device")
    return dataset_root, output_dir, device


def main() -> None:
    args = parse_args()
    dataset_root, output_dir, device = validate_args(args)
    grid_size = tuple(int(value) for value in args.grid_size)
    target_image_size = tuple(
        int(value) for value in args.target_image_size
    )
    use_amp = bool(args.amp)

    city_entries, image_count, expected_shuffle = discover_cities(
        dataset_root, eligible_min_views=args.eligible_min_views
    )
    processor, teacher, transformers_version, commit = load_teacher(
        args.model_name, args.revision, device
    )
    processor_info = processor_record(processor)
    classes = model_classes(teacher)
    desired_manifest = build_manifest(
        dataset_root=dataset_root,
        city_entries=city_entries,
        image_count=image_count,
        model_name=args.model_name,
        revision=args.revision,
        commit=commit,
        transformers_version=transformers_version,
        processor_info=processor_info,
        classes=classes,
        grid_size=grid_size,
        target_image_size=target_image_size,
        eligible_min_views=args.eligible_min_views,
        use_amp=use_amp,
        device=device,
    )

    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"partial cache has no manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if resume_signature(manifest) != resume_signature(desired_manifest):
            raise ValueError(
                "existing cache manifest does not match this dataset/model/config; "
                "choose a new --output directory"
            )
        if manifest.get("complete", False):
            print(f"Cache is already complete: {output_dir}")
            return
        if not progress_path.is_file():
            raise FileNotFoundError(
                f"incomplete cache has no durable cursor: {progress_path}"
            )
        with progress_path.open("r", encoding="utf-8") as handle:
            next_index = int(json.load(handle)["next_index"])
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
        manifest = desired_manifest
        atomic_json(manifest_path, manifest)
        next_index = 0

    if not 0 <= next_index <= image_count:
        raise ValueError(f"invalid resume cursor {next_index}/{image_count}")

    arrays = open_target_arrays(
        output_dir,
        image_count=image_count,
        grid_size=grid_size,
        resume=args.resume,
    )
    create_or_validate_shuffle_array(
        output_dir, expected=expected_shuffle, resume=args.resume
    )
    if not args.resume:
        atomic_json(
            progress_path,
            {"next_index": 0, "updated_utc": utc_now()},
        )

    estimated_mib = image_count * grid_size[0] * grid_size[1] * 2 / 2**20
    print(f"Device: {device}; precision: {manifest['inference_precision']}")
    print(
        f"Teacher: {args.model_name}@{commit or args.revision}; "
        f"classes: {len(classes)}"
    )
    print(
        f"Images: {image_count}; grid: {grid_size}; "
        f"labels+confidence: {estimated_mib:.1f} MiB"
    )
    print(f"Resume cursor: {next_index}/{image_count}")

    batches_since_flush = 0
    progress = tqdm(
        total=image_count,
        initial=next_index,
        desc="Cache ADE20K patch labels",
    )
    try:
        for city in city_entries:
            city_start = int(city["offset"])
            city_stop = city_start + int(city["count"])
            if next_index >= city_stop:
                continue
            dataframe = pd.read_csv(
                dataset_root / "Dataframes" / f"{city['name']}.csv"
            )
            if len(dataframe) != int(city["count"]):
                raise ValueError(
                    f"city row count changed after manifest validation: {city['name']}"
                )
            start_row = max(0, next_index - city_start)
            dataset = CityRows(
                dataset_root=dataset_root,
                dataframe=dataframe,
                city_offset=city_start,
                start_row=start_row,
                processor=processor,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
                drop_last=False,
            )
            for global_indices, pixel_values in loader:
                indices = global_indices.numpy().astype(np.int64, copy=False)
                expected_indices = np.arange(
                    next_index, next_index + len(indices), dtype=np.int64
                )
                if not np.array_equal(indices, expected_indices):
                    raise RuntimeError(
                        "DataLoader violated the contiguous original-CSV cache order"
                    )
                pixel_values = pixel_values.to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    teacher_output = teacher(pixel_values=pixel_values)
                logits = getattr(teacher_output, "logits", None)
                if not torch.is_tensor(logits) or logits.ndim != 4:
                    raise TypeError("SegFormer did not return 4D logits")
                if logits.shape[1] != len(classes):
                    raise ValueError(
                        "SegFormer logits class count does not match config.id2label"
                    )

                # Match standard segmentation post-processing first, then
                # average class probabilities over the exact, non-overlapping
                # VPR/DINO patch cells (14x14 for 280 -> 20).
                resized_logits = F.interpolate(
                    logits.float(),
                    size=target_image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                probabilities = resized_logits.softmax(dim=1)
                patch_size = tuple(
                    target // grid
                    for target, grid in zip(target_image_size, grid_size)
                )
                patch_probabilities = F.avg_pool2d(
                    probabilities,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                confidence, labels = patch_probabilities.max(dim=1)
                if not bool(torch.isfinite(confidence).all()):
                    raise ValueError("teacher produced non-finite patch confidence")
                labels_numpy = labels.to(torch.uint8).cpu().numpy()
                confidence_numpy = (
                    confidence.clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                arrays["labels"][indices] = labels_numpy
                arrays["confidence"][indices] = confidence_numpy

                next_index = int(indices[-1]) + 1
                progress.update(len(indices))
                batches_since_flush += 1
                if batches_since_flush >= args.flush_every:
                    for array in arrays.values():
                        array.flush()
                    atomic_json(
                        progress_path,
                        {"next_index": next_index, "updated_utc": utc_now()},
                    )
                    batches_since_flush = 0
    finally:
        progress.close()

    if next_index != image_count:
        raise RuntimeError(
            f"cache traversal stopped early at {next_index}/{image_count}"
        )
    for array in arrays.values():
        array.flush()
    atomic_json(
        progress_path,
        {"next_index": image_count, "updated_utc": utc_now()},
    )
    summary = summarize_cache(
        arrays["labels"], arrays["confidence"], classes
    )
    atomic_json(output_dir / "summary.json", summary)
    manifest["complete"] = True
    manifest["completed_utc"] = utc_now()
    manifest["summary_file"] = "summary.json"
    manifest["array_sha256"] = {
        filename: file_sha256(output_dir / filename)
        for filename in (
            "labels.npy",
            "confidence.npy",
            "shuffled_indices.npy",
        )
    }
    atomic_json(manifest_path, manifest)
    print(
        "Cache audit: mean confidence="
        f"{summary['mean_confidence']:.4f}; coverage@0.5="
        f"{summary['confidence_coverage']['0.5']:.4f}"
    )
    print(f"Wrote complete ADE20K patch cache to {output_dir}")


if __name__ == "__main__":
    main()
