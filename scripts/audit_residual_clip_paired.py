"""Paired inference interventions for a trained Residual-CLIP checkpoint.

The audit keeps one aligned-trained checkpoint fixed and reuses the same raw
DINO feature map for every intervention.  It therefore answers a narrower
mechanistic question than the matched training controls: does the learned
adapter react specifically to the aligned CLIP tokens at inference time?

The cheap default samples images and reports descriptor drift only.  Pass
``--full-retrieval`` to extract the complete standard MSLS-val database/query
set and additionally report paired query outcomes and capped positive ranks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_condition_robustness import (  # noqa: E402
    build_transform,
    choose_device,
    load_inference_model_from_ckpt,
    strip_compiled_model_prefix,
)
from src.dataloaders.valid.mapillary_sls import (  # noqa: E402
    MapillarySLSDataset,
)
from src.models.aggregators.boq import BoQ  # noqa: E402


VARIANTS = (
    "bypass",
    "aligned",
    "zero_clip",
    "global_only",
    "wrong_region",
    "wrong_image_cross_city",
)
KEY_CONTROLS = (
    "bypass",
    "zero_clip",
    "global_only",
    "wrong_region",
    "wrong_image_cross_city",
)
BASE_PREFIXES = ("backbone.", "aggregator.", "semantic_region_gate.")
RESIDUAL_PREFIX = "backbone.residual_clip_fusion."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired inference interventions on one aligned Residual-CLIP "
            "checkpoint"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ru-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--msls-path", type=Path, default=Path("datasets/msls-val")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="parent for temporary descriptor memmaps in --full-retrieval mode",
    )
    parser.add_argument(
        "--keep-descriptors",
        action="store_true",
        help="retain full descriptor memmaps below output/descriptors",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=(280, 280)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=512,
        help="balanced DB/query sample size for the cheap descriptor-only audit",
    )
    parser.add_argument(
        "--full-retrieval",
        action="store_true",
        help="evaluate all standard MSLS-val images and paired query outcomes",
    )
    parser.add_argument(
        "--descriptor-dtype",
        choices=("float32", "float16"),
        default="float32",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=(1, 5, 10, 15))
    parser.add_argument(
        "--rank-k",
        type=int,
        default=100,
        help="cap for the reported first-positive rank",
    )
    parser.add_argument(
        "--minimum-net-queries",
        type=int,
        default=4,
        help="aligned must net at least this many R@1 queries over every control",
    )
    parser.add_argument(
        "--expected-bypass-r1",
        type=float,
        default=91.22,
        help="historical RU R@1 in percent",
    )
    parser.add_argument(
        "--baseline-tolerance-pp", type=float, default=0.15
    )
    parser.add_argument(
        "--equivalence-tolerance",
        type=float,
        default=1e-5,
        help="first-batch aligned shared-path vs ordinary-forward tolerance",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_paths(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [str(value).replace("\\", "/") for value in values], dtype=np.str_
    )


def validate_args(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "ru_checkpoint"):
        value = getattr(args, name).expanduser().resolve()
        setattr(args, name, value)
        if not value.is_file():
            raise FileNotFoundError(f"{name.replace('_', ' ')} not found: {value}")
    args.msls_path = args.msls_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.msls_path.is_dir():
        raise FileNotFoundError(f"MSLS path not found: {args.msls_path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.scratch_dir is not None:
        args.scratch_dir = args.scratch_dir.expanduser().resolve()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if len(args.image_size) != 2 or min(args.image_size) <= 0:
        raise ValueError("image-size must contain two positive integers")
    if args.sample_count < 2:
        raise ValueError("sample-count must be at least two")
    if any(int(k) <= 0 for k in args.k_values):
        raise ValueError("k-values must be positive")
    args.k_values = tuple(sorted(set(int(k) for k in args.k_values)))
    if 1 not in args.k_values:
        raise ValueError("k-values must include 1 for paired outcomes")
    if args.rank_k < max(args.k_values):
        raise ValueError("rank-k must be at least max(k-values)")
    if args.minimum_net_queries < 1:
        raise ValueError("minimum-net-queries must be positive")
    if not math.isfinite(args.expected_bypass_r1) or not (
        0.0 <= args.expected_bypass_r1 <= 100.0
    ):
        raise ValueError("expected-bypass-r1 must be a percent in [0,100]")
    if args.baseline_tolerance_pp < 0:
        raise ValueError("baseline-tolerance-pp must be non-negative")
    if not math.isfinite(args.equivalence_tolerance) or (
        args.equivalence_tolerance < 0
    ):
        raise ValueError("equivalence-tolerance must be finite and non-negative")


def validate_aligned_checkpoint(
    checkpoint_path: Path, image_size: tuple[int, int]
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("hyper_parameters")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no hyper_parameters mapping")
    if (config.get("aggregator", {}) or {}).get("class") != "BoQ":
        raise ValueError("paired audit currently requires a BoQ checkpoint")
    backbone_params = (
        (config.get("backbone", {}) or {}).get("params", {}) or {}
    )
    fusion = backbone_params.get("residual_clip_fusion", {}) or {}
    distill_fusion = (
        (config.get("distillation", {}) or {}).get("residual_clip", {}) or {}
    )
    if not fusion.get("enabled", False) or not distill_fusion.get(
        "enabled", False
    ):
        raise ValueError("checkpoint does not enable Residual-CLIP")
    if fusion.get("mode") != "aligned" or distill_fusion.get("mode") != "aligned":
        raise ValueError(
            "paired attribution audit requires an aligned-trained checkpoint"
        )
    configured_size = tuple(
        int(value)
        for value in (config.get("datamodule", {}) or {}).get(
            "val_image_size", ()
        )
    )
    if configured_size and configured_size != tuple(image_size):
        raise ValueError(
            f"checkpoint val size {configured_size} != requested {image_size}"
        )
    state = strip_compiled_model_prefix(checkpoint["state_dict"])
    residual_keys = sorted(
        key for key in state if key.startswith(RESIDUAL_PREFIX)
    )
    if not residual_keys:
        raise ValueError("checkpoint contains no Residual-CLIP adapter weights")
    residual_norms = {
        key: float(state[key].detach().float().norm().item())
        for key in residual_keys
    }
    return {
        "backbone_class": (config.get("backbone", {}) or {}).get("class"),
        "aggregator_class": (config.get("aggregator", {}) or {}).get("class"),
        "val_image_size": list(configured_size),
        "fusion": dict(fusion),
        "distillation_residual_clip": dict(distill_fusion),
        "residual_parameter_norms": residual_norms,
    }


def verify_frozen_base(
    aligned_checkpoint: Path, ru_checkpoint: Path
) -> dict[str, Any]:
    aligned_raw = torch.load(
        aligned_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    ru_raw = torch.load(ru_checkpoint, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    aligned = strip_compiled_model_prefix(aligned_raw)
    ru = strip_compiled_model_prefix(ru_raw)
    aligned_keys = {
        key
        for key in aligned
        if key.startswith(BASE_PREFIXES) and not key.startswith(RESIDUAL_PREFIX)
    }
    ru_keys = {key for key in ru if key.startswith(BASE_PREFIXES)}
    if aligned_keys != ru_keys:
        raise RuntimeError(
            "aligned/RU frozen-state keys differ: missing_from_aligned="
            f"{sorted(ru_keys - aligned_keys)[:10]}, extra_in_aligned="
            f"{sorted(aligned_keys - ru_keys)[:10]}"
        )
    unequal = [
        key for key in sorted(aligned_keys) if not torch.equal(aligned[key], ru[key])
    ]
    if unequal:
        raise RuntimeError(
            "frozen RU/BoQ tensors changed in aligned checkpoint: "
            f"{unequal[:10]}"
        )
    return {
        "equal": True,
        "compared_tensors": len(aligned_keys),
        "aligned_checkpoint_sha256": _sha256_file(aligned_checkpoint),
        "ru_checkpoint_sha256": _sha256_file(ru_checkpoint),
    }


def _path_city(path: Any) -> str:
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    if not parts:
        raise ValueError(f"cannot infer city from empty path: {path!r}")
    return parts[0].lower()


def cross_city_donor_indices(
    image_paths: Sequence[Any], num_references: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a fixed DB/Q-preserving donor map whose city always differs."""

    paths = _canonical_paths(image_paths)
    image_count = len(paths)
    if not 1 < num_references < image_count - 1:
        raise ValueError("reference and query partitions both need two images")
    rng = np.random.default_rng(int(seed))
    donors = np.full(image_count, -1, dtype=np.int64)
    pair_counts: dict[str, int] = {}

    for role, start, stop in (
        ("reference", 0, num_references),
        ("query", num_references, image_count),
    ):
        indices = np.arange(start, stop, dtype=np.int64)
        cities = np.asarray([_path_city(paths[index]) for index in indices])
        unique_cities = sorted(set(cities.tolist()))
        if len(unique_cities) < 2:
            raise ValueError(
                f"{role} partition has fewer than two path-derived cities"
            )
        for source_city in unique_cities:
            sources = indices[cities == source_city].copy()
            eligible = indices[cities != source_city].copy()
            rng.shuffle(sources)
            rng.shuffle(eligible)
            for offset, source in enumerate(sources.tolist()):
                donor = int(eligible[offset % len(eligible)])
                donors[source] = donor
                donor_city = _path_city(paths[donor])
                key = f"{role}:{source_city}->{donor_city}"
                pair_counts[key] = pair_counts.get(key, 0) + 1

    indices = np.arange(image_count, dtype=np.int64)
    if np.any(donors < 0) or np.any(donors == indices):
        raise AssertionError("cross-city donor map contains missing/self donors")
    if np.any(donors[:num_references] >= num_references) or np.any(
        donors[num_references:] < num_references
    ):
        raise AssertionError("cross-city donor map crosses DB/query roles")
    for source, donor in enumerate(donors.tolist()):
        if _path_city(paths[source]) == _path_city(paths[donor]):
            raise AssertionError("cross-city donor map contains a same-city pair")
    return donors, pair_counts


class PairedImageDataset(Dataset):
    def __init__(self, dataset: Dataset, donor_indices: np.ndarray) -> None:
        self.dataset = dataset
        self.donor_indices = np.asarray(donor_indices, dtype=np.int64)
        if self.donor_indices.shape != (len(dataset),):
            raise ValueError("donor map length does not match dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        source, source_index = self.dataset[index]
        donor_index = int(self.donor_indices[index])
        donor, returned_donor_index = self.dataset[donor_index]
        if int(source_index) != int(index) or int(returned_donor_index) != donor_index:
            raise RuntimeError("base dataset returned inconsistent sample indices")
        return source, donor, int(source_index), donor_index


def balanced_sample_indices(
    num_references: int, num_queries: int, sample_count: int, seed: int
) -> np.ndarray:
    total = num_references + num_queries
    count = min(int(sample_count), total)
    query_count = min(num_queries, max(1, count // 2))
    reference_count = min(num_references, count - query_count)
    if reference_count + query_count < count:
        query_count = min(num_queries, count - reference_count)
    rng = np.random.default_rng(int(seed))
    references = rng.choice(
        num_references, size=reference_count, replace=False
    ).astype(np.int64)
    queries = num_references + rng.choice(
        num_queries, size=query_count, replace=False
    ).astype(np.int64)
    return np.sort(np.concatenate((references, queries)))


def aggregate_feature_map(model: torch.nn.Module, feature_map: torch.Tensor) -> torch.Tensor:
    if model.semantic_region_gate is not None:
        feature_map, _, _ = model.semantic_region_gate(feature_map)
    if model.spatial_attn_head is not None:
        feature_map, _ = model.spatial_attn_head(feature_map)
    output = model.aggregator(feature_map)
    descriptor = output[0] if isinstance(output, (tuple, list)) else output
    if descriptor.ndim != 2 or not bool(torch.isfinite(descriptor).all()):
        raise ValueError("aggregator returned invalid descriptors")
    return descriptor


def matched_variant_descriptors(
    model: torch.nn.Module,
    images: torch.Tensor,
    donor_images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    backbone = model.backbone
    fusion = getattr(backbone, "residual_clip_fusion", None)
    provider = getattr(backbone, "_residual_clip_provider", None)
    if fusion is None or provider is None:
        raise RuntimeError("model has no usable Residual-CLIP branch/provider")
    if not hasattr(backbone, "extract_residual_clip_components"):
        raise RuntimeError(
            "DinoV2 lacks extract_residual_clip_components; synchronize the "
            "paired-audit code before running"
        )
    raw, _x_cls, clip_global, clip_patches = (
        backbone.extract_residual_clip_components(images)
    )
    donor_global, donor_patches = provider.encode(donor_images)
    zero_global = torch.zeros_like(clip_global)
    zero_patches = torch.zeros_like(clip_patches)

    feature_maps = {"bypass": raw}
    feature_maps["aligned"], _ = fusion(
        raw,
        clip_patches,
        clip_global,
        intervention_mode="aligned",
    )
    feature_maps["zero_clip"], _ = fusion(
        raw,
        zero_patches,
        zero_global,
        intervention_mode="aligned",
    )
    feature_maps["global_only"], _ = fusion(
        raw,
        clip_patches,
        clip_global,
        intervention_mode="global_only",
    )
    feature_maps["wrong_region"], _ = fusion(
        raw,
        clip_patches,
        clip_global,
        intervention_mode="wrong_region",
    )
    # Donor CLIP tokens are deliberately passed through the aligned token path.
    # The donor never enters DINO; source DINO is therefore identical across all
    # variants.  The manifest-derived city differs and DB/query roles are kept.
    feature_maps["wrong_image_cross_city"], _ = fusion(
        raw,
        donor_patches,
        donor_global,
        intervention_mode="aligned",
    )
    return {
        name: aggregate_feature_map(model, feature_maps[name])
        for name in VARIANTS
    }


def verify_first_batch_equivalence(
    model: torch.nn.Module,
    images: torch.Tensor,
    aligned_descriptor: torch.Tensor,
    tolerance: float,
) -> float:
    ordinary = model(images)
    if isinstance(ordinary, (tuple, list)):
        ordinary = ordinary[0]
    error = float(
        (ordinary.detach().float() - aligned_descriptor.detach().float())
        .abs()
        .amax()
        .item()
    )
    if error > tolerance:
        raise RuntimeError(
            "shared paired path does not reproduce ordinary aligned forward: "
            f"max_abs_error={error:.3e} > {tolerance:.3e}"
        )
    return error


def descriptor_drift_rows(
    descriptors: Mapping[str, torch.Tensor], sample_indices: Sequence[int]
) -> list[dict[str, Any]]:
    detached = {
        name: value.detach().float().cpu() for name, value in descriptors.items()
    }
    indices = np.asarray(sample_indices, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for reference in ("bypass", "aligned"):
        base = detached[reference]
        for variant in VARIANTS:
            other = detached[variant]
            delta = other - base
            cosine = torch.nn.functional.cosine_similarity(other, base, dim=1)
            l2 = delta.norm(dim=1)
            rms = delta.square().mean(dim=1).sqrt()
            max_abs = delta.abs().amax(dim=1)
            for row_index, sample_index in enumerate(indices.tolist()):
                rows.append(
                    {
                        "image_index": int(sample_index),
                        "reference": reference,
                        "variant": variant,
                        "cosine_similarity": float(cosine[row_index].item()),
                        "cosine_distance": float(1.0 - cosine[row_index].item()),
                        "l2": float(l2[row_index].item()),
                        "rms": float(rms[row_index].item()),
                        "max_abs": float(max_abs[row_index].item()),
                    }
                )
    return rows


def summarise_drift(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["reference"]), str(row["variant"])) for row in rows})
    for reference, variant in keys:
        selected = [
            row
            for row in rows
            if row["reference"] == reference and row["variant"] == variant
        ]
        result: dict[str, Any] = {
            "reference": reference,
            "variant": variant,
            "num_images": len(selected),
        }
        for metric in ("cosine_distance", "l2", "rms", "max_abs"):
            values = np.asarray([float(row[metric]) for row in selected])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_p50"] = float(np.percentile(values, 50))
            result[f"{metric}_p95"] = float(np.percentile(values, 95))
            result[f"{metric}_max"] = float(values.max())
        output.append(result)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class DescriptorStore:
    def __init__(
        self,
        directory: Path,
        image_count: int,
        variant_names: Sequence[str],
        dtype: str,
    ) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.image_count = int(image_count)
        self.variant_names = tuple(variant_names)
        self.dtype = np.dtype(dtype)
        self.arrays: dict[str, np.memmap] = {}
        self.paths: dict[str, Path] = {}
        self.seen = np.zeros(self.image_count, dtype=bool)
        self.capacity_checked = False

    def _check_capacity(self, descriptor_dim: int) -> None:
        required = (
            len(self.variant_names)
            * self.image_count
            * int(descriptor_dim)
            * self.dtype.itemsize
        )
        free = shutil.disk_usage(self.directory).free
        print(
            f"Descriptor scratch estimate: {required / 2**30:.2f} GiB; "
            f"free: {free / 2**30:.2f} GiB"
        )
        if free < int(required * 1.10):
            raise OSError(
                "insufficient scratch space: need at least 110% of the "
                f"{required / 2**30:.2f} GiB estimate"
            )
        self.capacity_checked = True

    def mark_indices(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError("batch indices must be one-dimensional and unique")
        if np.any(indices < 0) or np.any(indices >= self.image_count):
            raise IndexError("batch descriptor index outside dataset")
        if np.any(self.seen[indices]):
            raise RuntimeError("descriptor extraction produced duplicate indices")
        self.seen[indices] = True

    def write(self, name: str, indices: np.ndarray, values: np.ndarray) -> None:
        if name not in self.variant_names:
            raise KeyError(f"unknown descriptor variant: {name}")
        if values.ndim != 2 or len(values) != len(indices):
            raise ValueError("descriptor batch has the wrong shape")
        if not self.capacity_checked:
            self._check_capacity(values.shape[1])
        if name not in self.arrays:
            path = self.directory / f"{name}.npy"
            self.paths[name] = path
            self.arrays[name] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=self.dtype,
                shape=(self.image_count, values.shape[1]),
            )
        self.arrays[name][indices] = values.astype(self.dtype, copy=False)

    def finish(self) -> None:
        if not bool(self.seen.all()):
            missing = np.flatnonzero(~self.seen)
            raise RuntimeError(
                f"descriptor extraction missed {len(missing)} images; "
                f"first missing={missing[:10].tolist()}"
            )
        if set(self.arrays) != set(self.variant_names):
            raise RuntimeError("not every descriptor variant was written")
        for array in self.arrays.values():
            array.flush()

    def close(self) -> None:
        for array in self.arrays.values():
            array.flush()
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays.clear()


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def run_sample_audit(
    *,
    model: torch.nn.Module,
    paired_dataset: PairedImageDataset,
    base_dataset: MapillarySLSDataset,
    donor_indices: np.ndarray,
    sample_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    equivalence_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    loader = make_loader(
        Subset(paired_dataset, sample_indices.tolist()),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    all_rows: list[dict[str, Any]] = []
    equivalence_error: float | None = None
    with torch.inference_mode():
        for images, donor_images, indices, returned_donors in tqdm(
            loader, desc="Paired descriptor sample"
        ):
            indices_np = np.asarray(indices, dtype=np.int64)
            donor_np = np.asarray(returned_donors, dtype=np.int64)
            if not np.array_equal(donor_np, donor_indices[indices_np]):
                raise RuntimeError("DataLoader donor indices do not match donor map")
            images = images.to(device, non_blocking=True)
            donor_images = donor_images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                descriptors = matched_variant_descriptors(
                    model, images, donor_images
                )
                if equivalence_error is None:
                    equivalence_error = verify_first_batch_equivalence(
                        model,
                        images,
                        descriptors["aligned"],
                        equivalence_tolerance,
                    )
            batch_rows = descriptor_drift_rows(descriptors, indices_np)
            for row in batch_rows:
                image_index = int(row["image_index"])
                donor_index = int(donor_indices[image_index])
                row.update(
                    {
                        "role": (
                            "reference"
                            if image_index < base_dataset.num_references
                            else "query"
                        ),
                        "image_path": str(
                            base_dataset.image_paths[image_index]
                        ).replace("\\", "/"),
                        "donor_index": donor_index,
                        "donor_path": str(
                            base_dataset.image_paths[donor_index]
                        ).replace("\\", "/"),
                    }
                )
            all_rows.extend(batch_rows)
    if equivalence_error is None:
        raise RuntimeError("sample audit produced no batches")
    return all_rows, summarise_drift(all_rows), equivalence_error


def extract_full_descriptors(
    *,
    model: torch.nn.Module,
    paired_dataset: PairedImageDataset,
    donor_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    equivalence_tolerance: float,
    directory: Path,
    dtype: str,
) -> tuple[DescriptorStore, float]:
    loader = make_loader(
        paired_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    store = DescriptorStore(directory, len(paired_dataset), VARIANTS, dtype)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    equivalence_error: float | None = None
    try:
        with torch.inference_mode():
            for images, donor_images, indices, returned_donors in tqdm(
                loader, desc="Extract six paired variants"
            ):
                indices_np = np.asarray(indices, dtype=np.int64)
                donor_np = np.asarray(returned_donors, dtype=np.int64)
                if not np.array_equal(donor_np, donor_indices[indices_np]):
                    raise RuntimeError("DataLoader donor indices do not match map")
                images = images.to(device, non_blocking=True)
                donor_images = donor_images.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    descriptors = matched_variant_descriptors(
                        model, images, donor_images
                    )
                    if equivalence_error is None:
                        equivalence_error = verify_first_batch_equivalence(
                            model,
                            images,
                            descriptors["aligned"],
                            equivalence_tolerance,
                        )
                store.mark_indices(indices_np)
                for name, descriptor in descriptors.items():
                    store.write(
                        name,
                        indices_np,
                        descriptor.detach().float().cpu().numpy(),
                    )
        store.finish()
        if equivalence_error is None:
            raise RuntimeError("full extraction produced no batches")
        return store, equivalence_error
    except Exception:
        store.close()
        raise


def scan_descriptor_drift(
    left_path: Path,
    right_path: Path,
    *,
    start: int,
    stop: int,
    chunk_size: int = 256,
) -> dict[str, float]:
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    if left.shape != right.shape:
        raise ValueError("descriptor arrays have different shapes")
    cosine_distances: list[np.ndarray] = []
    l2_values: list[np.ndarray] = []
    rms_values: list[np.ndarray] = []
    max_values: list[np.ndarray] = []
    for offset in range(start, stop, chunk_size):
        end = min(offset + chunk_size, stop)
        a = np.asarray(left[offset:end], dtype=np.float32)
        b = np.asarray(right[offset:end], dtype=np.float32)
        delta = b - a
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)
        cosine_distances.append(1.0 - cosine)
        l2_values.append(np.linalg.norm(delta, axis=1))
        rms_values.append(np.sqrt(np.mean(np.square(delta), axis=1)))
        max_values.append(np.max(np.abs(delta), axis=1))
    del left, right
    metrics = {
        "cosine_distance": np.concatenate(cosine_distances),
        "l2": np.concatenate(l2_values),
        "rms": np.concatenate(rms_values),
        "max_abs": np.concatenate(max_values),
    }
    result: dict[str, float] = {}
    for name, values in metrics.items():
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_p50"] = float(np.percentile(values, 50))
        result[f"{name}_p95"] = float(np.percentile(values, 95))
        result[f"{name}_max"] = float(values.max())
    return result


def search_and_score(
    descriptor_path: Path,
    *,
    num_references: int,
    ground_truth: Sequence[np.ndarray],
    k_values: Sequence[int],
    rank_k: int,
) -> dict[str, Any]:
    descriptors = np.load(descriptor_path, mmap_mode="r")
    num_queries = len(ground_truth)
    if len(descriptors) != num_references + num_queries:
        raise ValueError("descriptor count does not match standard MSLS")
    references = np.ascontiguousarray(
        descriptors[:num_references], dtype=np.float32
    )
    queries = np.ascontiguousarray(
        descriptors[num_references:], dtype=np.float32
    )
    max_gt = max(len(np.asarray(value)) for value in ground_truth)
    search_k = min(num_references, max(rank_k, max(k_values), max_gt + 1))
    index = faiss.IndexFlatL2(references.shape[1])
    index.add(references)
    distances, predictions = index.search(queries, search_k)

    hits: dict[int, np.ndarray] = {}
    recalls: dict[int, float] = {}
    for k in k_values:
        hit = np.asarray(
            [
                np.any(np.isin(predictions[i, :k], ground_truth[i]))
                for i in range(num_queries)
            ],
            dtype=bool,
        )
        hits[int(k)] = hit
        recalls[int(k)] = float(hit.mean())

    first_positive_rank = np.full(num_queries, rank_k + 1, dtype=np.int64)
    positive_found = np.zeros(num_queries, dtype=bool)
    best_positive_distance = np.empty(num_queries, dtype=np.float32)
    nearest_negative_distance = np.empty(num_queries, dtype=np.float32)
    for query_index in range(num_queries):
        positives = np.asarray(ground_truth[query_index], dtype=np.int64)
        capped_predictions = predictions[query_index, :rank_k]
        matches = np.flatnonzero(np.isin(capped_predictions, positives))
        if len(matches):
            first_positive_rank[query_index] = int(matches[0]) + 1
            positive_found[query_index] = True
        positive_descriptors = references[positives]
        difference = positive_descriptors - queries[query_index][None]
        best_positive_distance[query_index] = float(
            np.square(difference).sum(axis=1).min()
        )
        negative_positions = np.flatnonzero(
            ~np.isin(predictions[query_index], positives)
        )
        if not len(negative_positions):
            raise RuntimeError("retrieval prefix unexpectedly contains no negative")
        nearest_negative_distance[query_index] = float(
            distances[query_index, negative_positions[0]]
        )

    result = {
        "recalls": recalls,
        "hits": hits,
        "top1": predictions[:, 0].copy(),
        "top1_distance": distances[:, 0].copy(),
        "first_positive_rank": first_positive_rank,
        "positive_found_within_rank_k": positive_found,
        "best_positive_distance": best_positive_distance,
        "nearest_negative_distance": nearest_negative_distance,
        "positive_negative_margin": (
            nearest_negative_distance - best_positive_distance
        ),
    }
    del index, references, queries, distances, predictions, descriptors
    return result


def full_evaluation_rows(
    store: DescriptorStore,
    dataset: MapillarySLSDataset,
    k_values: Sequence[int],
    rank_k: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    results = {
        name: search_and_score(
            store.paths[name],
            num_references=dataset.num_references,
            ground_truth=dataset.ground_truth,
            k_values=k_values,
            rank_k=rank_k,
        )
        for name in VARIANTS
    }
    drift_rows: list[dict[str, Any]] = []
    for reference in ("bypass", "aligned"):
        for variant in VARIANTS:
            for role, start, stop in (
                ("reference", 0, dataset.num_references),
                ("query", dataset.num_references, len(dataset)),
                ("all", 0, len(dataset)),
            ):
                drift_rows.append(
                    {
                        "reference": reference,
                        "variant": variant,
                        "role": role,
                        "num_images": stop - start,
                        **scan_descriptor_drift(
                            store.paths[reference],
                            store.paths[variant],
                            start=start,
                            stop=stop,
                        ),
                    }
                )

    query_rows: list[dict[str, Any]] = []
    query_paths = _canonical_paths(dataset.qImages)
    reference_paths = _canonical_paths(dataset.dbImages)
    for variant in VARIANTS:
        result = results[variant]
        for query_index, query_path in enumerate(query_paths.tolist()):
            prediction = int(result["top1"][query_index])
            row: dict[str, Any] = {
                "query_index": query_index,
                "query_path": query_path,
                "variant": variant,
                "top1_reference_index": prediction,
                "top1_reference_path": reference_paths[prediction],
                "top1_squared_l2": float(result["top1_distance"][query_index]),
                "first_positive_rank_capped": int(
                    result["first_positive_rank"][query_index]
                ),
                "positive_found_within_rank_k": int(
                    result["positive_found_within_rank_k"][query_index]
                ),
                "best_positive_squared_l2": float(
                    result["best_positive_distance"][query_index]
                ),
                "nearest_negative_squared_l2": float(
                    result["nearest_negative_distance"][query_index]
                ),
                "positive_negative_margin": float(
                    result["positive_negative_margin"][query_index]
                ),
            }
            for k in k_values:
                row[f"hit@{k}"] = int(result["hits"][int(k)][query_index])
            query_rows.append(row)

    paired_rows: list[dict[str, Any]] = []
    aligned = results["aligned"]
    for comparator in KEY_CONTROLS:
        other = results[comparator]
        aligned_hits = aligned["hits"][1]
        other_hits = other["hits"][1]
        aligned_only = int(np.sum(aligned_hits & ~other_hits))
        other_only = int(np.sum(~aligned_hits & other_hits))
        aligned_rank = aligned["first_positive_rank"]
        other_rank = other["first_positive_rank"]
        paired_rows.append(
            {
                "left": "aligned",
                "right": comparator,
                "left_only_correct_r@1": aligned_only,
                "right_only_correct_r@1": other_only,
                "net_queries_r@1": aligned_only - other_only,
                "both_correct_r@1": int(np.sum(aligned_hits & other_hits)),
                "both_wrong_r@1": int(np.sum(~aligned_hits & ~other_hits)),
                "top1_reference_changed": int(
                    np.sum(aligned["top1"] != other["top1"])
                ),
                "left_better_first_positive_rank": int(
                    np.sum(aligned_rank < other_rank)
                ),
                "right_better_first_positive_rank": int(
                    np.sum(other_rank < aligned_rank)
                ),
                "rank_tied": int(np.sum(aligned_rank == other_rank)),
                "mean_capped_rank_advantage": float(
                    np.mean(other_rank.astype(float) - aligned_rank.astype(float))
                ),
                "mean_margin_advantage": float(
                    np.mean(
                        aligned["positive_negative_margin"]
                        - other["positive_negative_margin"]
                    )
                ),
            }
        )
    return query_rows, paired_rows, drift_rows, results


def build_summary_rows(
    results: Mapping[str, Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> list[dict[str, Any]]:
    bypass = results["bypass"]
    aligned = results["aligned"]
    net_by_variant = {
        str(row["right"]): int(row["net_queries_r@1"]) for row in paired_rows
    }
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        result = results[variant]
        row: dict[str, Any] = {
            "variant": variant,
            "num_queries": len(result["top1"]),
            "aligned_net_queries_r@1": (
                0 if variant == "aligned" else net_by_variant.get(variant, 0)
            ),
        }
        for k in k_values:
            recall = float(result["recalls"][int(k)])
            row[f"r@{k}"] = recall
            row[f"delta_vs_bypass_r@{k}_pp"] = 100.0 * (
                recall - float(bypass["recalls"][int(k)])
            )
            row[f"delta_vs_aligned_r@{k}_pp"] = 100.0 * (
                recall - float(aligned["recalls"][int(k)])
            )
        rows.append(row)
    return rows


def print_sample_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\nPaired descriptor-only audit")
    print("=" * 86)
    print(
        f"{'Reference':<10} {'Variant':<25} {'cos-dist mean':>14} "
        f"{'L2 mean':>12} {'RMS mean':>12}"
    )
    for row in rows:
        print(
            f"{row['reference']:<10} {row['variant']:<25} "
            f"{float(row['cosine_distance_mean']):>14.6e} "
            f"{float(row['l2_mean']):>12.6e} "
            f"{float(row['rms_mean']):>12.6e}"
        )


def print_full_summary(
    rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> None:
    print("\nPaired Residual-CLIP retrieval audit")
    print("=" * 100)
    header = f"{'Variant':<25}" + "".join(
        f" R@{k:>2}".rjust(10) for k in k_values
    )
    print(header)
    for row in rows:
        metrics = "".join(
            f"{100.0 * float(row[f'r@{k}']):>9.2f}%" for k in k_values
        )
        print(f"{row['variant']:<25}{metrics}")
    print("-" * 100)
    for row in paired_rows:
        print(
            "aligned vs "
            f"{row['right']:<24}: net R@1 "
            f"{int(row['net_queries_r@1']):+d} queries; "
            f"top1 changed={int(row['top1_reference_changed'])}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    checkpoint_config = validate_aligned_checkpoint(
        args.checkpoint, tuple(args.image_size)
    )
    frozen_base = verify_frozen_base(args.checkpoint, args.ru_checkpoint)
    transform = build_transform(tuple(args.image_size))
    dataset = MapillarySLSDataset(
        dataset_path=str(args.msls_path), input_transform=transform
    )
    if max(args.k_values) > dataset.num_references:
        raise ValueError("largest k-value exceeds MSLS reference count")
    donor_indices, city_pair_counts = cross_city_donor_indices(
        dataset.image_paths, dataset.num_references, args.seed
    )
    paired_dataset = PairedImageDataset(dataset, donor_indices)

    print(f"Device: {device}")
    print(f"Aligned checkpoint: {args.checkpoint}")
    print(f"Frozen RU checkpoint: {args.ru_checkpoint}")
    print(
        f"MSLS: {dataset.num_references} references, "
        f"{dataset.num_queries} queries"
    )
    print(f"Variants: {VARIANTS}")
    print("Loading one model and its frozen CLIP provider...")
    model = load_inference_model_from_ckpt(args.checkpoint, device)
    if not isinstance(model.aggregator, BoQ):
        raise TypeError("checkpoint aggregator is not repository BoQ")
    if model.semantic_region_gate is None:
        raise RuntimeError("checkpoint loader did not restore the RU gate")
    model.backbone.prepare_residual_clip_provider(device)

    args.output.mkdir(parents=True, exist_ok=False)
    common_record: dict[str, Any] = {
        "schema_version": 1,
        "method": "residual_clip_paired_inference_intervention",
        "mode": "full_retrieval" if args.full_retrieval else "descriptor_sample",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256_file(args.checkpoint),
            "size_bytes": args.checkpoint.stat().st_size,
            "validated_config": checkpoint_config,
        },
        "ru_checkpoint": {
            "path": str(args.ru_checkpoint),
            "sha256": _sha256_file(args.ru_checkpoint),
            "size_bytes": args.ru_checkpoint.stat().st_size,
        },
        "frozen_base_verification": frozen_base,
        "dataset": {
            "path": str(args.msls_path),
            "num_references": dataset.num_references,
            "num_queries": dataset.num_queries,
        },
        "variants": {
            "bypass": "raw DINO -> frozen RU gate -> frozen BoQ",
            "aligned": "source DINO plus source aligned local CLIP residual",
            "zero_clip": "source DINO plus W(0 - normalized DINO anchor)",
            "global_only": "source CLIP global token repeated spatially",
            "wrong_region": "source local CLIP tokens shifted by half the grid",
            "wrong_image_cross_city": (
                "source DINO plus a fixed cross-city donor image's CLIP tokens; "
                "DB/query roles preserved; proxy for wrong place"
            ),
        },
        "donor_mapping": {
            "seed": args.seed,
            "sha256": hashlib.sha256(donor_indices.tobytes()).hexdigest(),
            "self_pairs": int(np.sum(donor_indices == np.arange(len(dataset)))),
            "city_pair_counts": city_pair_counts,
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "image_size": list(args.image_size),
            "amp": use_amp,
        },
        "limitations": [
            "Inference interventions are out-of-distribution probes and do not "
            "replace matched training controls.",
            "wrong_image_cross_city is a deterministic wrong-image proxy; no "
            "MSLS place ID is asserted for donor images.",
        ],
    }

    temporary: tempfile.TemporaryDirectory[str] | None = None
    store: DescriptorStore | None = None
    try:
        if not args.full_retrieval:
            sample_indices = balanced_sample_indices(
                dataset.num_references,
                dataset.num_queries,
                args.sample_count,
                args.seed,
            )
            per_image, summary, equivalence_error = run_sample_audit(
                model=model,
                paired_dataset=paired_dataset,
                base_dataset=dataset,
                donor_indices=donor_indices,
                sample_indices=sample_indices,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_amp=use_amp,
                equivalence_tolerance=args.equivalence_tolerance,
            )
            write_csv(args.output / "per_image.csv", per_image)
            write_csv(args.output / "summary.csv", summary)
            common_record["sample"] = {
                "count": len(sample_indices),
                "indices_sha256": hashlib.sha256(
                    sample_indices.tobytes()
                ).hexdigest(),
                "num_references": int(
                    np.sum(sample_indices < dataset.num_references)
                ),
                "num_queries": int(
                    np.sum(sample_indices >= dataset.num_references)
                ),
            }
            common_record["aligned_equivalence_max_abs_error"] = equivalence_error
            common_record["descriptor_drift_summary"] = summary
            print_sample_summary(summary)
        else:
            if args.keep_descriptors:
                descriptor_directory = args.output / "descriptors"
            else:
                if args.scratch_dir is not None:
                    args.scratch_dir.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.TemporaryDirectory(
                    prefix="residual-clip-paired-",
                    dir=(
                        str(args.scratch_dir)
                        if args.scratch_dir is not None
                        else None
                    ),
                )
                descriptor_directory = Path(temporary.name)
            store, equivalence_error = extract_full_descriptors(
                model=model,
                paired_dataset=paired_dataset,
                donor_indices=donor_indices,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_amp=use_amp,
                equivalence_tolerance=args.equivalence_tolerance,
                directory=descriptor_directory,
                dtype=args.descriptor_dtype,
            )
            query_rows, paired_rows, drift_rows, results = full_evaluation_rows(
                store, dataset, args.k_values, args.rank_k
            )
            summary_rows = build_summary_rows(
                results, paired_rows, args.k_values
            )
            bypass_error_pp = abs(
                100.0 * float(results["bypass"]["recalls"][1])
                - args.expected_bypass_r1
            )
            net_queries = {
                str(row["right"]): int(row["net_queries_r@1"])
                for row in paired_rows
            }
            verdict = {
                "status": (
                    "pass"
                    if bypass_error_pp <= args.baseline_tolerance_pp
                    and all(
                        net_queries[name] >= args.minimum_net_queries
                        for name in KEY_CONTROLS
                    )
                    else "fail"
                ),
                "bypass_reproduced": (
                    bypass_error_pp <= args.baseline_tolerance_pp
                ),
                "bypass_error_pp": bypass_error_pp,
                "minimum_net_queries": args.minimum_net_queries,
                "aligned_net_queries_r@1": net_queries,
                "decision_rule": (
                    "bypass must reproduce historical RU within tolerance and "
                    "aligned must net at least minimum_net_queries R@1 queries "
                    "over bypass, zero_clip, global_only, wrong_region, and "
                    "wrong_image_cross_city"
                ),
                "scope": (
                    "mechanism screen only; a pass still requires matched "
                    "training controls"
                ),
            }
            write_csv(args.output / "summary.csv", summary_rows)
            write_csv(args.output / "descriptor_drift.csv", drift_rows)
            write_csv(args.output / "query_outcomes.csv", query_rows)
            write_csv(args.output / "paired_comparisons.csv", paired_rows)
            common_record["aligned_equivalence_max_abs_error"] = equivalence_error
            common_record["retrieval_summary"] = summary_rows
            common_record["paired_comparisons"] = paired_rows
            common_record["descriptor_drift_summary"] = drift_rows
            common_record["verdict"] = verdict
            common_record["descriptor_storage"] = {
                "dtype": args.descriptor_dtype,
                "kept": args.keep_descriptors,
                "directory": (
                    str(args.output / "descriptors")
                    if args.keep_descriptors
                    else None
                ),
            }
            common_record["rank_reporting"] = {
                "rank_k": args.rank_k,
                "not_found_value": args.rank_k + 1,
            }
            print_full_summary(summary_rows, paired_rows, args.k_values)
            print(f"Mechanism verdict: {verdict['status'].upper()}")

        (args.output / "run.json").write_text(
            json.dumps(common_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results written to: {args.output}")
    finally:
        if store is not None:
            store.close()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
