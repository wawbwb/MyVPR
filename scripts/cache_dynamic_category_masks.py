"""Cache dynamic-category area masks for the standard MSLS validation set.

The cache is intentionally generated before VPR inference so the segmentation
teacher can be removed from GPU memory.  Pixel labels come from a frozen
torchvision DeepLabV3-MobileNetV3 teacher.  Hard dynamic pixels are area-pooled
to the 20x20 DINOv2 patch grid; there is no confidence threshold and no
per-image standardisation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)
from torchvision.transforms import v2 as T2
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_category_prior import (  # noqa: E402
    DEFAULT_DYNAMIC_CLASSES,
    dynamic_patch_coverage,
    file_sha256,
    resolve_dynamic_class_ids,
    save_mask_cache,
    string_sequence_sha256,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("specify an explicit CUDA index; use --device cuda:1")
        if device.index == 0:
            raise ValueError("GPU 0 is faulty; use --device cuda:1")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    return device


class MSLSImageIndex:
    """Minimal standard-MSLS index without importing retrieval dependencies."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        db_file = dataset_path / "msls_val_dbImages.npy"
        query_file = dataset_path / "msls_val_qImages.npy"
        if not db_file.is_file() or not query_file.is_file():
            raise FileNotFoundError(
                "MSLS path must contain msls_val_dbImages.npy and "
                "msls_val_qImages.npy"
            )
        self.dbImages = np.load(db_file, allow_pickle=False)
        self.qImages = np.load(query_file, allow_pickle=False)
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __len__(self) -> int:
        return len(self.image_paths)


class SegmentationImageDataset(Dataset):
    def __init__(self, dataset: MSLSImageIndex, transform: Any) -> None:
        self.dataset_path = dataset.dataset_path
        self.image_paths = dataset.image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.dataset_path / self.image_paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen dynamic-category masks for MSLS-val"
    )
    parser.add_argument("--msls-path", type=Path, default=Path("datasets/msls-val"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="new directory for run.json and a 12-image unbiased mask montage",
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=None,
        help="optional local DeepLabV3-MobileNetV3 state_dict",
    )
    parser.add_argument(
        "--dynamic-classes", nargs="+", default=DEFAULT_DYNAMIC_CLASSES
    )
    parser.add_argument("--seg-size", type=int, nargs=2, default=(520, 520))
    parser.add_argument("--grid-size", type=int, nargs=2, default=(20, 20))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="opt into teacher autocast; default FP32 avoids argmax boundary drift",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.msls_path = args.msls_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.report_dir = args.report_dir.expanduser().resolve()
    if not args.msls_path.is_dir():
        raise FileNotFoundError(f"MSLS path not found: {args.msls_path}")
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end with .npz")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cache: {args.output}")
    if args.report_dir.exists():
        raise FileExistsError(f"refusing to overwrite report dir: {args.report_dir}")
    if args.report_dir == args.output or args.report_dir in args.output.parents:
        raise ValueError("report-dir must not be an ancestor of the cache output")
    if min(args.seg_size) <= 0 or min(args.grid_size) <= 0:
        raise ValueError("seg-size and grid-size must be positive")
    if any(
        seg_extent % grid_extent != 0
        for seg_extent, grid_extent in zip(args.seg_size, args.grid_size)
    ):
        raise ValueError(
            "each seg-size extent must be divisible by the corresponding grid-size"
        )
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid batch-size or num-workers")
    if args.teacher_weights is not None:
        args.teacher_weights = args.teacher_weights.expanduser().resolve()
        if not args.teacher_weights.is_file():
            raise FileNotFoundError(
                f"teacher weights not found: {args.teacher_weights}"
            )


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("teacher checkpoint must contain a state_dict mapping")
    for key in ("state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break
    result: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = str(key)
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        result[clean_key] = value
    if not result:
        raise ValueError("teacher checkpoint contains no tensor parameters")
    return result


def build_teacher(
    weights_path: Path | None, device: torch.device
) -> tuple[torch.nn.Module, tuple[str, ...], dict[str, Any]]:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    categories = tuple(str(name).lower() for name in weights.meta["categories"])
    if weights_path is None:
        model = deeplabv3_mobilenet_v3_large(weights=weights)
        cached_name = Path(urlparse(weights.url).path).name
        cached_path = Path(torch.hub.get_dir()) / "checkpoints" / cached_name
        provenance = {
            "source": "torchvision_default",
            "weights_name": str(weights),
            "weights_url": weights.url,
            "cached_path": str(cached_path),
            "weights_sha256": (
                file_sha256(cached_path) if cached_path.is_file() else None
            ),
        }
    else:
        model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=len(categories),
            aux_loss=True,
        )
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)
        provenance = {
            "source": "local_state_dict",
            "weights_name": weights_path.name,
            "weights_url": weights.url,
            "cached_path": str(weights_path),
            "weights_sha256": file_sha256(weights_path),
        }
    model.to(device).eval().requires_grad_(False)
    provenance["architecture"] = "deeplabv3_mobilenet_v3_large"
    return model, categories, provenance


def build_transform(seg_size: tuple[int, int]) -> T2.Compose:
    return T2.Compose(
        [
            T2.ToImage(),
            T2.Resize(
                size=list(seg_size),
                interpolation=T2.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _partition_statistics(masks: np.ndarray) -> dict[str, Any]:
    flat = masks.astype(np.float32, copy=False).reshape(len(masks), -1)
    per_image = flat.mean(axis=1)
    return {
        "image_count": int(len(masks)),
        "patch_value_mean": float(flat.mean()),
        "patch_value_p50": float(np.quantile(flat, 0.50)),
        "patch_value_p90": float(np.quantile(flat, 0.90)),
        "patch_value_p99": float(np.quantile(flat, 0.99)),
        "per_image_coverage_mean": float(per_image.mean()),
        "per_image_coverage_p50": float(np.quantile(per_image, 0.50)),
        "per_image_coverage_p90": float(np.quantile(per_image, 0.90)),
        "per_image_coverage_max": float(per_image.max()),
        "empty_image_fraction": float(np.mean(per_image == 0.0)),
    }


def _save_montage(
    output: Path,
    dataset: MSLSImageIndex,
    masks: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    cell_width, cell_height = 280, 308
    columns = 4
    rows = int(np.ceil(len(sample_indices) / columns))
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for cell, sample_index in enumerate(sample_indices.tolist()):
        with Image.open(
            Path(dataset.dataset_path) / dataset.image_paths[sample_index]
        ) as source:
            source = source.convert("RGB").resize(
                (cell_width, 280), resample=Image.Resampling.BILINEAR
            )
        alpha = Image.fromarray(
            np.uint8(np.clip(masks[sample_index], 0.0, 1.0) * 255.0), mode="L"
        ).resize((cell_width, 280), resample=Image.Resampling.BILINEAR)
        alpha = alpha.point(lambda value: int(round(0.65 * value)))
        overlay = Image.new("RGBA", source.size, (255, 32, 32, 0))
        overlay.putalpha(alpha)
        visual = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(visual)
        for x in range(0, cell_width, cell_width // masks.shape[2]):
            draw.line((x, 0, x, 280), fill=(255, 255, 255), width=1)
        for y in range(0, 280, 280 // masks.shape[1]):
            draw.line((0, y, cell_width, y), fill=(255, 255, 255), width=1)
        x0 = (cell % columns) * cell_width
        y0 = (cell // columns) * cell_height
        canvas.paste(visual, (x0, y0))
        label = (
            f"idx={sample_index}  mean={float(masks[sample_index].mean()):.3f}"
        )
        ImageDraw.Draw(canvas).text((x0 + 5, y0 + 286), label, fill="black")
    canvas.save(output, quality=92)


def main() -> None:
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    base_dataset = MSLSImageIndex(args.msls_path)
    dataset = SegmentationImageDataset(
        base_dataset, build_transform(tuple(args.seg_size))
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    teacher, categories, teacher_record = build_teacher(
        args.teacher_weights, device
    )
    class_ids, class_names = resolve_dynamic_class_ids(
        categories, args.dynamic_classes
    )
    print(f"Device: {device}")
    print(f"Dynamic classes: {dict(zip(class_names, class_ids))}")
    print(f"Images: {len(dataset)}; output grid: {tuple(args.grid_size)}")

    masks = np.empty((len(dataset), *args.grid_size), dtype=np.float16)
    class_pixel_counts = np.zeros(len(class_ids), dtype=np.int64)
    total_pixels = 0
    use_amp = device.type == "cuda" and args.amp
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Cache dynamic masks"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                output = teacher(images)
            if not isinstance(output, Mapping) or "out" not in output:
                raise TypeError("segmentation teacher did not return an 'out' tensor")
            coverage, labels = dynamic_patch_coverage(
                output["out"], class_ids, tuple(args.grid_size)
            )
            numpy_indices = torch.as_tensor(indices, dtype=torch.long).numpy()
            masks[numpy_indices] = coverage.cpu().numpy().astype(np.float16)
            for position, class_id in enumerate(class_ids):
                class_pixel_counts[position] += int((labels == class_id).sum().item())
            total_pixels += labels.numel()

    save_mask_cache(
        args.output,
        masks=masks,
        image_paths=base_dataset.image_paths,
        num_references=base_dataset.num_references,
        grid_size=tuple(args.grid_size),
        segmentation_size=tuple(args.seg_size),
        model_name=teacher_record["architecture"],
        weights_name=teacher_record["weights_name"],
        weights_url=teacher_record["weights_url"],
        dynamic_class_names=class_names,
        dynamic_class_ids=class_ids,
    )

    args.report_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    montage_indices = np.sort(
        rng.choice(len(base_dataset), size=min(12, len(base_dataset)), replace=False)
    )
    _save_montage(
        args.report_dir / "mask_montage.jpg",
        base_dataset,
        masks,
        montage_indices,
    )
    report = {
        "schema_version": 1,
        "method": "dynamic_category_mask_cache",
        "cache": {
            "path": str(args.output),
            "sha256": file_sha256(args.output),
            "size_bytes": args.output.stat().st_size,
            "image_paths_sha256": string_sequence_sha256(
                [str(path).replace("\\", "/") for path in base_dataset.image_paths]
            ),
        },
        "dataset": {
            "path": str(args.msls_path),
            "num_images": len(base_dataset),
            "num_references": base_dataset.num_references,
            "num_queries": base_dataset.num_queries,
        },
        "teacher": teacher_record,
        "segmentation_size": list(args.seg_size),
        "grid_size": list(args.grid_size),
        "mask_definition": "hard argmax dynamic-pixel area fraction per patch",
        "dynamic_classes": dict(zip(class_names, class_ids)),
        "missing_cityscapes_dynamic_classes": ["rider", "truck"],
        "class_pixel_fractions": {
            name: float(count / total_pixels)
            for name, count in zip(class_names, class_pixel_counts.tolist())
        },
        "mask_statistics": {
            "all": _partition_statistics(masks),
            "references": _partition_statistics(
                masks[: base_dataset.num_references]
            ),
            "queries": _partition_statistics(masks[base_dataset.num_references :]),
        },
        "montage": {
            "selection": "seeded uniform random without replacement",
            "seed": args.seed,
            "indices": montage_indices.tolist(),
            "paths": [
                str(base_dataset.image_paths[index]).replace("\\", "/")
                for index in montage_indices.tolist()
            ],
            "file": "mask_montage.jpg",
        },
        "amp": use_amp,
        "limitations": [
            "The Pascal-VOC label space has no rider or truck category.",
            "The mask is a segmentation argmax area prior, not place evidence.",
            "The cache contains no per-image z-score or confidence threshold.",
        ],
    }
    with (args.report_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Mask cache: {args.output}")
    print(f"Audit report: {args.report_dir}")


if __name__ == "__main__":
    main()
