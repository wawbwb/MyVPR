"""Visualize aligned and shuffled semantic propagation on one fixed DINO batch.

This diagnostic deliberately loads a single repeatability+uniqueness checkpoint,
computes its raw DINO feature map once, and changes only the cached CLIP sparse
affinity.  It therefore isolates the spatial effect of aligned semantic
propagation from differences learned by separately trained backbones.

Example:
    python scripts/visualize_semantic_region_delta.py \
      --feature-ckpt /path/to/repeatability_uniqueness_best.ckpt \
      --device cuda:1 \
      --clean-input \
      --output doc/semantic_region_delta_batch0

The output directory contains per-sample PNG panels, a CSV for the complete
batch, a tensor bundle, and run.json with the exact diagnostic settings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import random
import sys
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.semantic_region_gate import SemanticRegionReliabilityTarget


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
IMAGENET_MEAN_STD = {
    "mean": IMAGENET_MEAN.tolist(),
    "std": IMAGENET_STD.tolist(),
}
EXPECTED_CONTROL_MODE = "repeatability_uniqueness_only"
DEFAULT_CONFIG = REPO_ROOT / "config/boq_dinov2_semantic_region_full.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aligned and place-shuffled CLIP propagation deltas while "
            "holding the image batch and raw DINO features fixed."
        )
    )
    parser.add_argument(
        "--feature-ckpt",
        type=Path,
        required=True,
        help=(
            "repeatability+uniqueness-only checkpoint used for the one shared "
            "raw DINO feature map"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="full semantic-region YAML supplying cache and target settings",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="override distillation.semantic_region.cache_dir",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("doc/semantic_region_delta"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--clean-input",
        action="store_true",
        help=(
            "replace random photometric augmentation with the deterministic "
            "validation transform; geometry is unchanged either way"
        ),
    )
    parser.add_argument(
        "--save-raw-features",
        action="store_true",
        help="also store the complete raw feature batch (roughly 100 MB for ViT-B)",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=280,
        help="pixel size of each cell in the 4x4 diagnostic panel",
    )
    return parser.parse_args()


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive copy of ``base`` updated by ``override``."""

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_diagnostic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(f"{spec} requested, but CUDA is not available")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{spec} requested, but only {torch.cuda.device_count()} CUDA devices exist"
        )
    return torch.device("cuda", index)


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("hyper_parameters")
    if not isinstance(config, Mapping):
        raise KeyError("checkpoint has no mapping-valued hyper_parameters")
    if "backbone" not in config and isinstance(config.get("config_dict"), Mapping):
        config = config["config_dict"]
    if "backbone" not in config or "datamodule" not in config:
        raise KeyError("checkpoint hyper_parameters lack backbone/datamodule config")
    return copy.deepcopy(dict(config))


def _get_instance(module_name: str, class_name: str, params: Mapping[str, Any]) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**dict(params))


def _strip_compiled_model_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return OrderedDict(
        (
            key[len(prefix) :] if key.startswith(prefix) else key,
            value,
        )
        for key, value in state_dict.items()
    )


def _extract_required_submodule_state(
    state_dict: Mapping[str, torch.Tensor], prefix: str, module_name: str
) -> OrderedDict[str, torch.Tensor]:
    result = OrderedDict(
        (key[len(prefix) :], value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )
    if not result:
        raise RuntimeError(
            f"checkpoint contains no {module_name} weights with prefix {prefix!r}"
        )
    return result


def load_control_backbone(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only the shared raw-feature backbone from the neutral control."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = _checkpoint_config(checkpoint)
    semantic_config = (
        config.get("distillation", {}).get("semantic_region", {}) or {}
    )
    checkpoint_mode = str(semantic_config.get("mode", "")).lower()
    if checkpoint_mode != EXPECTED_CONTROL_MODE:
        raise ValueError(
            "--feature-ckpt must be the neutral "
            f"{EXPECTED_CONTROL_MODE!r} run, found mode={checkpoint_mode!r}. "
            "Do not compare raw features from separately trained full and "
            "shuffled checkpoints."
        )

    backbone_config = config["backbone"]
    backbone = _get_instance(
        backbone_config["module"],
        backbone_config["class"],
        copy.deepcopy(backbone_config.get("params", {})),
    )
    state = _strip_compiled_model_prefix(checkpoint["state_dict"])
    backbone_state = _extract_required_submodule_state(
        state, "backbone.", "backbone"
    )
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(device).eval()
    return backbone, config


def load_diagnostic_config(
    checkpoint_config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    if not isinstance(override, Mapping):
        raise TypeError("diagnostic YAML root must be a mapping")
    return deep_update(checkpoint_config, override)


def _resolve_repo_relative(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"semantic cache manifest not found: {manifest_path}")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("semantic cache manifest root must be an object")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content": manifest,
    }


def validate_diagnostic_config(
    checkpoint_config: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    cache_override: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    checkpoint_data = checkpoint_config["datamodule"]
    data_config = dict(diagnostic_config["datamodule"])
    for name in ("batch_size", "img_per_place"):
        expected = int(checkpoint_data[name])
        actual = int(data_config[name])
        if actual != expected:
            raise ValueError(
                f"diagnostic {name}={actual} differs from checkpoint {expected}; "
                "P and K are part of the reliability target definition"
            )
    for name in ("train_set_name", "train_image_size", "cities"):
        if name in checkpoint_data and data_config.get(name) != checkpoint_data[name]:
            raise ValueError(
                f"diagnostic {name}={data_config.get(name)!r} differs from "
                f"checkpoint {checkpoint_data[name]!r}; use the checkpoint's "
                "training data geometry and place population"
            )
    place_count = int(data_config["batch_size"])
    views_per_place = int(data_config["img_per_place"])
    if place_count < 2 or views_per_place < 2:
        raise ValueError("diagnostic requires at least two places and two views")
    augmentation_mode = str(
        data_config.get("augmentation_mode", "photometric")
    ).lower()
    if augmentation_mode != "photometric":
        raise ValueError("semantic cache diagnosis requires photometric augmentation")

    semantic_config = dict(
        diagnostic_config.get("distillation", {}).get("semantic_region", {}) or {}
    )
    if not bool(semantic_config.get("enabled", False)):
        raise ValueError("diagnostic config must enable semantic_region")
    if str(semantic_config.get("mode", "")).lower() != "full":
        raise ValueError(
            "diagnostic config must be the aligned full config; shuffled is "
            "constructed internally from the same cache"
        )
    cache_value = cache_override or semantic_config.get("cache_dir")
    if not cache_value:
        raise ValueError("semantic cache path is missing from config and CLI")
    cache_dir = _resolve_repo_relative(Path(cache_value))
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"semantic cache directory not found: {cache_dir}")
    return data_config, semantic_config, cache_dir


def build_diagnostic_loader(
    data_config: Mapping[str, Any],
    cache_dir: Path,
    device: torch.device,
    num_workers: int,
    clean_input: bool,
    seed: int,
) -> tuple[DataLoader, Any]:
    from src.core.vpr_datamodule import VPRDataModule

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    data_module = VPRDataModule(
        train_set_name=data_config["train_set_name"],
        cities=data_config.get("cities", "all"),
        train_image_size=data_config["train_image_size"],
        val_image_size=data_config.get("val_image_size"),
        batch_size=int(data_config["batch_size"]),
        img_per_place=int(data_config["img_per_place"]),
        shuffle_all=False,
        random_sample_from_each_place=True,
        num_workers=num_workers,
        batch_sampler=None,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=[],
        return_augmented=False,
        return_metadata=True,
        return_teacher_view=False,
        augmentation_mode="photometric",
        semantic_cache_dir=str(cache_dir),
    )
    dataset = data_module._get_train_dataset()
    if clean_input:
        if tuple(data_module.val_image_size) != tuple(data_module.train_image_size):
            raise ValueError(
                "--clean-input requires val_image_size == train_image_size so "
                "the DINO/cache grid geometry stays fixed"
            )
        dataset.transform = data_module.val_transform

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=data_module.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    return loader, data_module


def get_fixed_batch(loader: DataLoader, batch_index: int) -> tuple[Any, Any, Any]:
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    for current_index, batch in enumerate(loader):
        if current_index == batch_index:
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise ValueError(
                    "expected (images, labels, metadata); check cache/metadata setup"
                )
            return batch[0], batch[1], batch[2]
    raise IndexError(
        f"batch_index {batch_index} is outside loader range of {len(loader)} batches"
    )


def flatten_and_validate_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    metadata: Mapping[str, torch.Tensor],
    place_count: int,
    views_per_place: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if images.ndim != 5:
        raise ValueError("images must have shape (P,K,C,H,W)")
    if tuple(images.shape[:2]) != (place_count, views_per_place):
        raise ValueError(
            f"batch has P,K={tuple(images.shape[:2])}, expected "
            f"{place_count, views_per_place}; choose a non-final full batch"
        )
    if tuple(labels.shape[:2]) != (place_count, views_per_place):
        raise ValueError("labels must preserve the same (P,K) ordering as images")

    required = {
        "semantic_indices": 4,
        "semantic_weights": 4,
        "semantic_confidence": 3,
    }
    flat_metadata: dict[str, torch.Tensor] = {}
    for name, ndim in required.items():
        value = metadata.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"metadata[{name!r}] must be a {ndim}-D tensor")
        if tuple(value.shape[:2]) != (place_count, views_per_place):
            raise ValueError(f"metadata[{name!r}] does not preserve (P,K)")
        flat_metadata[name] = value.flatten(0, 1)
    if flat_metadata["semantic_indices"].dtype != torch.uint8:
        raise ValueError("semantic_indices must retain uint8 cache dtype")

    for name, value in metadata.items():
        if name in required or not isinstance(value, torch.Tensor):
            continue
        if value.ndim >= 2 and tuple(value.shape[:2]) == (
            place_count,
            views_per_place,
        ):
            flat_metadata[name] = value.flatten(0, 1)

    flat_images = images.flatten(0, 1)
    flat_labels = labels.flatten(0, 1)
    return flat_images, flat_labels, flat_metadata


def roll_cache_by_place(
    semantic_indices: torch.Tensor,
    semantic_weights: torch.Tensor,
    semantic_confidence: torch.Tensor,
    place_count: int,
    views_per_place: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll all cache fields together: current place p receives p-1, same view k."""

    batch_size = place_count * views_per_place
    if semantic_indices.shape[0] != batch_size:
        raise ValueError("cache batch size must equal place_count * views_per_place")
    if semantic_weights.shape != semantic_indices.shape:
        raise ValueError("semantic_weights must match semantic_indices")
    if semantic_confidence.shape != semantic_indices.shape[:2]:
        raise ValueError("semantic_confidence must match cache batch and patch axes")

    cache_tail = semantic_indices.shape[1:]
    confidence_tail = semantic_confidence.shape[1:]
    rolled_indices = (
        semantic_indices.reshape(place_count, views_per_place, *cache_tail)
        .roll(1, 0)
        .flatten(0, 1)
    )
    rolled_weights = (
        semantic_weights.reshape(place_count, views_per_place, *cache_tail)
        .roll(1, 0)
        .flatten(0, 1)
    )
    rolled_confidence = (
        semantic_confidence.reshape(
            place_count, views_per_place, *confidence_tail
        )
        .roll(1, 0)
        .flatten(0, 1)
    )
    return rolled_indices, rolled_weights, rolled_confidence


def sparse_propagation_debug(
    builder: SemanticRegionReliabilityTarget,
    base_reliability: torch.Tensor,
    semantic_indices: torch.Tensor,
    semantic_weights: torch.Tensor,
    semantic_confidence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose each term of the production sparse semantic propagation formula."""

    if base_reliability.ndim != 2:
        raise ValueError("base_reliability must have shape (B,N)")
    if semantic_indices.ndim != 3:
        raise ValueError("semantic_indices must have shape (B,N,topk)")
    if semantic_indices.dtype != torch.uint8:
        raise ValueError("semantic_indices must be uint8")
    if semantic_weights.shape != semantic_indices.shape:
        raise ValueError("semantic_weights must match semantic_indices")
    if semantic_confidence.shape != semantic_indices.shape[:2]:
        raise ValueError("semantic_confidence must have shape (B,N)")
    if semantic_indices.shape[0] != base_reliability.shape[0]:
        raise ValueError("feature and cache batch sizes differ")

    patch_count = semantic_indices.shape[1]
    patch_side = builder._square_side(patch_count, "semantic cache")
    reliability_side = builder._square_side(
        base_reliability.shape[1], "reliability"
    )
    base14 = F.interpolate(
        base_reliability.view(-1, 1, reliability_side, reliability_side),
        size=(patch_side, patch_side),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    if patch_count > 256:
        raise ValueError("uint8 semantic cache cannot address more than 256 patches")

    weights = semantic_weights.float().clamp_min(0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(builder.eps)
    gathered = base14.gather(1, semantic_indices.long().flatten(1)).view_as(
        weights
    )
    smoothed14 = (gathered * weights).sum(dim=-1)
    confidence14 = semantic_confidence.float().clamp(0.0, 1.0)
    delta14 = confidence14 * (smoothed14 - base14)
    propagated14 = base14 + delta14
    out10 = F.interpolate(
        propagated14.view(-1, 1, patch_side, patch_side),
        size=(reliability_side, reliability_side),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    return {
        "base14": base14,
        "smoothed14": smoothed14,
        "confidence14": confidence14,
        "delta14": delta14,
        "propagated14": propagated14,
        "out10": out10,
        "normalized_weights": weights,
    }


def make_final_target(
    builder: SemanticRegionReliabilityTarget,
    reliability: torch.Tensor,
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    """Apply the production resize -> standardize -> tanh target transform."""

    side = builder._square_side(reliability.shape[1], "target")
    target = F.interpolate(
        reliability.view(-1, 1, side, side),
        size=feature_hw,
        mode="bilinear",
        align_corners=False,
    )
    standardized = builder._standardize(target.flatten(1))
    return torch.tanh(builder.target_scale * standardized).view_as(target)


def _builder_kwargs(semantic_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "match_grid": int(semantic_config.get("match_grid", 10)),
        "target_scale": float(semantic_config.get("target_scale", 2.0)),
        "place_chunk_size": int(semantic_config.get("place_chunk_size", 8)),
        "min_spatial_std": float(
            semantic_config.get("min_spatial_std", 1e-3)
        ),
    }


def build_propagation_diagnostics(
    raw_featmap: torch.Tensor,
    semantic_indices: torch.Tensor,
    semantic_weights: torch.Tensor,
    semantic_confidence: torch.Tensor,
    place_count: int,
    views_per_place: int,
    semantic_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute aligned and shuffled outputs from one shared raw feature map."""

    if raw_featmap.ndim != 4:
        raise ValueError("raw_featmap must have shape (B,C,H,W)")
    if raw_featmap.shape[0] != place_count * views_per_place:
        raise ValueError("raw feature batch does not match P*K")
    builder_kwargs = _builder_kwargs(semantic_config)
    aligned_builder = SemanticRegionReliabilityTarget(
        mode="full", **builder_kwargs
    ).to(raw_featmap.device)
    shuffled_builder = SemanticRegionReliabilityTarget(
        mode="shuffled", **builder_kwargs
    ).to(raw_featmap.device)

    with torch.inference_mode(), torch.autocast(
        device_type=raw_featmap.device.type, enabled=False
    ):
        repeatability, uniqueness, component_stats = (
            aligned_builder._vpr_components(
                raw_featmap.detach(), place_count, views_per_place
            )
        )
        base10 = 0.5 * (
            aligned_builder._standardize(repeatability)
            + aligned_builder._standardize(uniqueness)
        )
        rolled_cache = roll_cache_by_place(
            semantic_indices,
            semantic_weights,
            semantic_confidence,
            place_count,
            views_per_place,
        )
        aligned = sparse_propagation_debug(
            aligned_builder,
            base10,
            semantic_indices,
            semantic_weights,
            semantic_confidence,
        )
        shuffled = sparse_propagation_debug(
            aligned_builder, base10, *rolled_cache
        )

        reference_aligned, _ = aligned_builder._sparse_semantic_smooth(
            base10,
            semantic_indices,
            semantic_weights,
            semantic_confidence,
            place_count,
            views_per_place,
        )
        reference_shuffled, _ = shuffled_builder._sparse_semantic_smooth(
            base10,
            semantic_indices,
            semantic_weights,
            semantic_confidence,
            place_count,
            views_per_place,
        )
        torch.testing.assert_close(aligned["out10"], reference_aligned)
        torch.testing.assert_close(shuffled["out10"], reference_shuffled)

        feature_hw = tuple(raw_featmap.shape[-2:])
        aligned["target"] = make_final_target(
            aligned_builder, aligned["out10"], feature_hw
        )
        shuffled["target"] = make_final_target(
            shuffled_builder, shuffled["out10"], feature_hw
        )

    for name, branch in (("aligned", aligned), ("shuffled", shuffled)):
        for tensor_name, tensor in branch.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite values in {name}.{tensor_name}")
    return {
        "repeatability10": repeatability,
        "uniqueness10": uniqueness,
        "base10": base10,
        "aligned": aligned,
        "shuffled": shuffled,
        "rolled_cache": {
            "semantic_indices": rolled_cache[0],
            "semantic_weights": rolled_cache[1],
            "semantic_confidence": rolled_cache[2],
        },
        "component_stats": component_stats,
        "builder_kwargs": builder_kwargs,
    }


def select_sample_indices(
    aligned_delta: torch.Tensor,
    target_difference: torch.Tensor,
    num_samples: int,
    seed: int,
) -> tuple[list[int], dict[int, list[str]]]:
    """Mix high-effect, high-contrast, and random samples without duplicates."""

    if aligned_delta.ndim < 2 or target_difference.ndim < 2:
        raise ValueError("selection scores require a batch axis and spatial axes")
    batch_size = aligned_delta.shape[0]
    if target_difference.shape[0] != batch_size:
        raise ValueError("selection tensors must have the same batch size")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    num_samples = min(num_samples, batch_size)
    aligned_score = aligned_delta.abs().flatten(1).mean(1)
    contrast_score = target_difference.abs().flatten(1).mean(1)
    aligned_order = torch.argsort(aligned_score, descending=True).cpu().tolist()
    contrast_order = torch.argsort(contrast_score, descending=True).cpu().tolist()

    random_slots = 1 if num_samples >= 3 else 0
    ranked_slots = num_samples - random_slots
    aligned_slots = (ranked_slots + 1) // 2
    contrast_slots = ranked_slots - aligned_slots
    selected: list[int] = []
    reasons: dict[int, list[str]] = {}

    def add_from(order: list[int], count: int, reason: str) -> None:
        if count <= 0:
            return
        added = 0
        for index in order:
            if index in reasons:
                if reason not in reasons[index]:
                    reasons[index].append(reason)
                continue
            selected.append(index)
            reasons[index] = [reason]
            added += 1
            if added == count:
                break

    add_from(aligned_order, aligned_slots, "top_aligned_delta")
    add_from(contrast_order, contrast_slots, "top_target_contrast")
    if len(selected) < ranked_slots:
        add_from(contrast_order + aligned_order, ranked_slots - len(selected), "rank_fill")

    if random_slots:
        remaining = [index for index in range(batch_size) if index not in reasons]
        generator = random.Random(seed)
        random_index = generator.choice(remaining) if remaining else selected[-1]
        if random_index not in reasons:
            selected.append(random_index)
            reasons[random_index] = ["random_audit"]
        elif "random_audit" not in reasons[random_index]:
            reasons[random_index].append("random_audit")

    if len(selected) < num_samples:
        add_from(contrast_order + aligned_order, num_samples - len(selected), "rank_fill")
    return selected[:num_samples], reasons


def _quantile_scale(tensors: list[torch.Tensor], quantile: float = 0.99) -> float:
    values = torch.cat([tensor.detach().float().abs().flatten() for tensor in tensors])
    scale = float(torch.quantile(values, quantile).cpu())
    return max(scale, 1e-6)


def _image_to_rgb(image: torch.Tensor) -> Image.Image:
    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]
    image = image.detach().float().cpu() * std + mean
    array = (
        image.clamp(0.0, 1.0).permute(1, 2, 0).mul(255).round().byte().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _diverging_rgb(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values / scale, -1.0, 1.0)
    amount = np.abs(normalized)[..., None]
    white = np.full((*normalized.shape, 3), 245.0, dtype=np.float32)
    negative = np.array([40.0, 90.0, 210.0], dtype=np.float32)
    positive = np.array([215.0, 55.0, 45.0], dtype=np.float32)
    endpoint = np.where((normalized >= 0)[..., None], positive, negative)
    return np.clip(white * (1.0 - amount) + endpoint * amount, 0, 255).astype(
        np.uint8
    )


def _confidence_rgb(values: np.ndarray) -> np.ndarray:
    normalized = np.clip(values, 0.0, 1.0)[..., None]
    low = np.array([35.0, 20.0, 75.0], dtype=np.float32)
    high = np.array([250.0, 220.0, 55.0], dtype=np.float32)
    return np.clip(low * (1.0 - normalized) + high * normalized, 0, 255).astype(
        np.uint8
    )


def _map_array(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu().squeeze()
    if value.ndim != 2:
        side = math.isqrt(value.numel())
        if side * side != value.numel():
            raise ValueError("visualized map must be a square grid")
        value = value.reshape(side, side)
    return value.numpy()


def _resize_rgb(array: np.ndarray, size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return Image.fromarray(array, mode="RGB").resize((size, size), resampling)


def _heatmap(value: torch.Tensor, size: int, scale: float) -> Image.Image:
    return _resize_rgb(_diverging_rgb(_map_array(value), scale), size)


def _confidence_map(value: torch.Tensor, size: int) -> Image.Image:
    return _resize_rgb(_confidence_rgb(_map_array(value)), size)


def _overlay(
    base_image: Image.Image,
    value: torch.Tensor,
    size: int,
    scale: float,
) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    base = np.asarray(base_image.resize((size, size), resampling), dtype=np.float32)
    values = _map_array(value)
    color = np.asarray(_resize_rgb(_diverging_rgb(values, scale), size), dtype=np.float32)
    magnitude = np.abs(values) / max(scale, 1e-6)
    alpha_image = Image.fromarray(
        np.clip(magnitude * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize((size, size), resampling)
    alpha = np.asarray(alpha_image, dtype=np.float32)[..., None] / 255.0
    alpha *= 0.72
    blended = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _cell(
    content: Image.Image,
    title: str,
    footer: str,
    size: int,
) -> Image.Image:
    title_height = 26
    footer_height = 22
    cell = Image.new("RGB", (size, title_height + size + footer_height), "white")
    cell.paste(content.resize((size, size)), (0, title_height))
    draw = ImageDraw.Draw(cell)
    draw.text((6, 7), title, fill="black")
    draw.text((6, title_height + size + 5), footer, fill=(55, 55, 55))
    return cell


def _format_range(scale: float) -> str:
    return f"shared range [-{scale:.3f}, +{scale:.3f}]"


def render_sample_panel(
    images: torch.Tensor,
    diagnostics: Mapping[str, Any],
    flat_index: int,
    place_count: int,
    views_per_place: int,
    reliability_scale: float,
    delta_scale: float,
    delta_difference_scale: float,
    target_difference_scale: float,
    panel_size: int,
) -> Image.Image:
    if panel_size < 160:
        raise ValueError("panel_size must be at least 160 pixels")
    place_index, view_index = divmod(flat_index, views_per_place)
    donor_place = (place_index - 1) % place_count
    donor_index = donor_place * views_per_place + view_index
    current = _image_to_rgb(images[flat_index])
    donor = _image_to_rgb(images[donor_index])
    aligned = diagnostics["aligned"]
    shuffled = diagnostics["shuffled"]
    delta_difference = aligned["delta14"][flat_index] - shuffled["delta14"][flat_index]
    target_difference = (
        aligned["target"][flat_index] - shuffled["target"][flat_index]
    )
    patch_side = math.isqrt(aligned["base14"][flat_index].numel())
    target_height, target_width = aligned["target"].shape[-2:]

    cells = [
        _cell(current, "Current clean/augmented image", f"p={place_index}, k={view_index}, flat={flat_index}", panel_size),
        _cell(_heatmap(aligned["base14"][flat_index], panel_size, reliability_scale), f"RU base reliability ({patch_side}x{patch_side})", _format_range(reliability_scale), panel_size),
        _cell(_heatmap(aligned["smoothed14"][flat_index], panel_size, reliability_scale), "Aligned neighbor average", _format_range(reliability_scale), panel_size),
        _cell(_confidence_map(aligned["confidence14"][flat_index], panel_size), "Aligned cache confidence", "fixed range [0, 1]", panel_size),
        _cell(_heatmap(aligned["delta14"][flat_index], panel_size, delta_scale), "Aligned propagation delta", _format_range(delta_scale), panel_size),
        _cell(_overlay(current, aligned["delta14"][flat_index], panel_size, delta_scale), "Aligned delta on current image", f"mean|d|={aligned['delta14'][flat_index].abs().mean().item():.4f}", panel_size),
        _cell(_heatmap(aligned["target"][flat_index], panel_size, 1.0), f"Aligned final target ({target_height}x{target_width})", "fixed range [-1, +1]", panel_size),
        _cell(donor, "Shuffled semantic donor", f"p={donor_place}, k={view_index}, flat={donor_index}", panel_size),
        _cell(_heatmap(shuffled["smoothed14"][flat_index], panel_size, reliability_scale), "Shuffled neighbor average", _format_range(reliability_scale), panel_size),
        _cell(_confidence_map(shuffled["confidence14"][flat_index], panel_size), "Shuffled cache confidence", "fixed range [0, 1]", panel_size),
        _cell(_heatmap(shuffled["delta14"][flat_index], panel_size, delta_scale), "Shuffled propagation delta", _format_range(delta_scale), panel_size),
        _cell(_overlay(current, shuffled["delta14"][flat_index], panel_size, delta_scale), "Shuffled delta on current image", f"mean|d|={shuffled['delta14'][flat_index].abs().mean().item():.4f}", panel_size),
        _cell(_heatmap(delta_difference, panel_size, delta_difference_scale), "Aligned - shuffled delta", _format_range(delta_difference_scale), panel_size),
        _cell(_overlay(current, delta_difference, panel_size, delta_difference_scale), "Delta contrast on current image", f"mean|diff|={delta_difference.abs().mean().item():.4f}", panel_size),
        _cell(_heatmap(target_difference, panel_size, target_difference_scale), "Aligned - shuffled target", _format_range(target_difference_scale), panel_size),
        _cell(_overlay(current, target_difference, panel_size, target_difference_scale), "Target contrast on current image", f"mean|diff|={target_difference.abs().mean().item():.4f}", panel_size),
    ]
    columns = 4
    rows = 4
    gap = 8
    cell_width, cell_height = cells[0].size
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns - 1) * gap,
            rows * cell_height + (rows - 1) * gap,
        ),
        (230, 230, 230),
    )
    for index, cell in enumerate(cells):
        row, column = divmod(index, columns)
        canvas.paste(cell, (column * (cell_width + gap), row * (cell_height + gap)))
    return canvas


def _flat_metadata_value(
    metadata: Mapping[str, torch.Tensor], name: str, index: int, component: int | None = None
) -> float:
    value = metadata.get(name)
    if not isinstance(value, torch.Tensor):
        return float("nan")
    selected = value[index]
    if component is not None:
        selected = selected[component]
    return float(selected.detach().cpu())


def _spatial_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().float().flatten()
    second = second.detach().float().flatten()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if denominator <= 1e-12:
        return float("nan")
    return float((first @ second / denominator).cpu())


def make_summary_records(
    labels: torch.Tensor,
    metadata: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, Any],
    selected: list[int],
    reasons: Mapping[int, list[str]],
    place_count: int,
    views_per_place: int,
) -> list[dict[str, Any]]:
    selected_set = set(selected)
    aligned = diagnostics["aligned"]
    shuffled = diagnostics["shuffled"]
    records: list[dict[str, Any]] = []
    for index in range(labels.numel()):
        place_index, view_index = divmod(index, views_per_place)
        donor_place = (place_index - 1) % place_count
        donor_index = donor_place * views_per_place + view_index
        delta_difference = aligned["delta14"][index] - shuffled["delta14"][index]
        target_difference = aligned["target"][index] - shuffled["target"][index]
        records.append(
            {
                "flat_index": index,
                "place_index": place_index,
                "view_index": view_index,
                "place_label": int(labels[index].detach().cpu()),
                "donor_flat_index": donor_index,
                "donor_place_index": donor_place,
                "donor_place_label": int(labels[donor_index].detach().cpu()),
                "latitude": _flat_metadata_value(metadata, "coordinates", index, 0),
                "longitude": _flat_metadata_value(metadata, "coordinates", index, 1),
                "year": _flat_metadata_value(metadata, "years", index),
                "month": _flat_metadata_value(metadata, "months", index),
                "heading": _flat_metadata_value(metadata, "headings", index),
                "aligned_confidence_mean": float(aligned["confidence14"][index].mean().cpu()),
                "shuffled_confidence_mean": float(shuffled["confidence14"][index].mean().cpu()),
                "aligned_delta_mean": float(aligned["delta14"][index].mean().cpu()),
                "aligned_delta_abs_mean": float(aligned["delta14"][index].abs().mean().cpu()),
                "aligned_delta_abs_max": float(aligned["delta14"][index].abs().max().cpu()),
                "shuffled_delta_mean": float(shuffled["delta14"][index].mean().cpu()),
                "shuffled_delta_abs_mean": float(shuffled["delta14"][index].abs().mean().cpu()),
                "shuffled_delta_abs_max": float(shuffled["delta14"][index].abs().max().cpu()),
                "delta_abs_difference_mean": float(delta_difference.abs().mean().cpu()),
                "delta_abs_difference_max": float(delta_difference.abs().max().cpu()),
                "target_abs_difference_mean": float(target_difference.abs().mean().cpu()),
                "target_spatial_correlation": _spatial_correlation(
                    aligned["target"][index], shuffled["target"][index]
                ),
                "aligned_target_std": float(aligned["target"][index].std(unbiased=False).cpu()),
                "shuffled_target_std": float(shuffled["target"][index].std(unbiased=False).cpu()),
                "selected": index in selected_set,
                "selection_reason": "|".join(reasons.get(index, [])),
            }
        )
    return records


def save_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot save an empty summary")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


def prepare_output_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}; choose a new directory"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if args.panel_size < 160:
        raise ValueError("panel_size must be at least 160")
    output_dir = prepare_output_directory(args.output_dir)
    device = resolve_device(args.device)
    set_diagnostic_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    checkpoint_path = args.feature_ckpt.expanduser().resolve()
    backbone, checkpoint_config = load_control_backbone(checkpoint_path, device)
    diagnostic_config = load_diagnostic_config(checkpoint_config, args.config)
    data_config, semantic_config, cache_dir = validate_diagnostic_config(
        checkpoint_config, diagnostic_config, args.cache_dir
    )
    cache_manifest = load_cache_manifest(cache_dir)
    place_count = int(data_config["batch_size"])
    views_per_place = int(data_config["img_per_place"])
    loader, _ = build_diagnostic_loader(
        data_config,
        cache_dir,
        device,
        args.num_workers,
        args.clean_input,
        args.seed,
    )
    images_pk, labels_pk, metadata_pk = get_fixed_batch(loader, args.batch_index)
    images, labels, metadata = flatten_and_validate_batch(
        images_pk,
        labels_pk,
        metadata_pk,
        place_count,
        views_per_place,
    )
    images_device = images.to(device, non_blocking=True)
    semantic_indices = metadata["semantic_indices"].to(device, non_blocking=True)
    semantic_weights = metadata["semantic_weights"].to(device, non_blocking=True)
    semantic_confidence = metadata["semantic_confidence"].to(
        device, non_blocking=True
    )

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        raw_output = backbone(images_device)
    raw_featmap = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output
    if not isinstance(raw_featmap, torch.Tensor) or raw_featmap.ndim != 4:
        raise ValueError("backbone must return a 4-D local feature map")
    if raw_featmap.shape[0] != place_count * views_per_place:
        raise ValueError("backbone output batch does not match P*K")

    diagnostics = build_propagation_diagnostics(
        raw_featmap,
        semantic_indices,
        semantic_weights,
        semantic_confidence,
        place_count,
        views_per_place,
        semantic_config,
    )
    target_difference = (
        diagnostics["aligned"]["target"]
        - diagnostics["shuffled"]["target"]
    )
    selected, reasons = select_sample_indices(
        diagnostics["aligned"]["delta14"],
        target_difference,
        args.num_samples,
        args.seed,
    )
    reliability_scale = _quantile_scale(
        [
            diagnostics["aligned"]["base14"],
            diagnostics["aligned"]["smoothed14"],
            diagnostics["shuffled"]["smoothed14"],
        ]
    )
    delta_scale = _quantile_scale(
        [diagnostics["aligned"]["delta14"], diagnostics["shuffled"]["delta14"]]
    )
    delta_difference = (
        diagnostics["aligned"]["delta14"]
        - diagnostics["shuffled"]["delta14"]
    )
    delta_difference_scale = _quantile_scale([delta_difference])
    target_difference_scale = _quantile_scale([target_difference])
    diagnostics_cpu = _cpu_tree(diagnostics)

    records = make_summary_records(
        labels,
        metadata,
        diagnostics_cpu,
        selected,
        reasons,
        place_count,
        views_per_place,
    )
    save_summary_csv(output_dir / "summary.csv", records)

    for rank, flat_index in enumerate(selected):
        panel = render_sample_panel(
            images,
            diagnostics_cpu,
            flat_index,
            place_count,
            views_per_place,
            reliability_scale,
            delta_scale,
            delta_difference_scale,
            target_difference_scale,
            args.panel_size,
        )
        reason = "-".join(reasons.get(flat_index, ["selected"]))
        panel.save(
            output_dir / f"sample_{rank:02d}_flat_{flat_index:03d}_{reason}.png",
            format="PNG",
        )

    tensor_bundle = {
        "schema_version": 1,
        "selected_indices": torch.tensor(selected, dtype=torch.long),
        "labels": labels.detach().cpu(),
        "selected_images_normalized": images[selected].detach().cpu(),
        "selected_raw_featmap": raw_featmap[selected].detach().cpu(),
        "metadata": {
            name: value.detach().cpu()
            for name, value in metadata.items()
            if name not in {
                "semantic_indices",
                "semantic_weights",
                "semantic_confidence",
            }
        },
        "cache": {
            "semantic_indices": semantic_indices.detach().cpu(),
            "semantic_weights": semantic_weights.detach().cpu(),
            "semantic_confidence": semantic_confidence.detach().cpu(),
        },
        "diagnostics": diagnostics_cpu,
    }
    if args.save_raw_features:
        tensor_bundle["raw_featmap"] = raw_featmap.detach().cpu()
    torch.save(tensor_bundle, output_dir / "diagnostic_tensors.pt")

    component_stats = {
        name: float(value.detach().cpu())
        for name, value in diagnostics_cpu["component_stats"].items()
    }
    run_metadata = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_mode": EXPECTED_CONTROL_MODE,
        "config": str(args.config.expanduser().resolve()),
        "cache_dir": str(cache_dir),
        "cache_manifest": cache_manifest,
        "device": str(device),
        "seed": args.seed,
        "batch_index": args.batch_index,
        "place_count": place_count,
        "views_per_place": views_per_place,
        "batch_size_images": place_count * views_per_place,
        "clean_input": args.clean_input,
        "num_workers": args.num_workers,
        "image_shape": list(images.shape),
        "raw_featmap_shape": list(raw_featmap.shape),
        "cache_shape": list(semantic_indices.shape),
        "builder": diagnostics_cpu["builder_kwargs"],
        "shared_scales": {
            "reliability_abs_p99": reliability_scale,
            "propagation_delta_abs_p99": delta_scale,
            "delta_difference_abs_p99": delta_difference_scale,
            "target_difference_abs_p99": target_difference_scale,
        },
        "selected_indices": selected,
        "selection_reasons": {str(key): value for key, value in reasons.items()},
        "component_stats": component_stats,
        "production_consistency_check": "passed",
        "saved_full_raw_features": args.save_raw_features,
    }
    with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    print(f"Shared raw feature map: {tuple(raw_featmap.shape)} from one control checkpoint")
    print(f"Fixed batch: P={place_count}, K={views_per_place}, batch_index={args.batch_index}")
    print("Production aligned/shuffled propagation consistency: passed")
    print(f"Selected flat indices: {selected}")
    print(f"Saved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
