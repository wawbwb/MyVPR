"""Shared, side-effect-free helpers for dynamic-category prior screening."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CACHE_SCHEMA_VERSION = 1
DEFAULT_DYNAMIC_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorbike",
    "bus",
    "train",
)
CLASS_ALIASES = {
    "bike": "bicycle",
    "motorcycle": "motorbike",
    "pedestrian": "person",
}


def canonical_image_paths(paths: Sequence[Any]) -> np.ndarray:
    """Return stable slash-normalised path strings without touching the FS."""

    return np.asarray(
        [str(path).replace("\\", "/") for path in paths],
        dtype=np.str_,
    )


def build_query_union(
    standard_query_paths: Sequence[Any],
    condition_query_groups: Sequence[Sequence[Any]],
) -> np.ndarray:
    """Build a stable standard-first union of retrieval query manifests.

    Standard MSLS queries keep their exact manifest order.  Condition-only
    queries are appended on first occurrence while preserving the order of
    both the condition groups and each manifest.  Duplicate paths inside any
    individual manifest are rejected because they make query outcomes and
    mask donors ambiguous.
    """

    standard = canonical_image_paths(standard_query_paths)
    if len(set(standard.tolist())) != len(standard):
        raise ValueError("standard MSLS queries contain duplicate paths")

    union = standard.tolist()
    seen = set(union)
    for group_index, group in enumerate(condition_query_groups):
        condition = canonical_image_paths(group)
        if len(set(condition.tolist())) != len(condition):
            raise ValueError(
                f"condition query manifest {group_index} contains duplicate paths"
            )
        for path in condition.tolist():
            if path not in seen:
                seen.add(path)
                union.append(path)
    return np.asarray(union, dtype=np.str_)


def validate_ground_truth(
    ground_truth: Sequence[Any],
    *,
    num_queries: int,
    num_references: int,
    dataset_name: str,
) -> list[np.ndarray]:
    """Validate and normalise one retrieval ground-truth list.

    Every query must have at least one unique integer reference index in the
    shared database.  Failing closed here prevents invalid condition manifests
    from producing plausible-looking recall values.
    """

    if num_queries <= 0:
        raise ValueError(f"{dataset_name} has no queries")
    if num_references <= 0:
        raise ValueError(f"{dataset_name} has no references")
    if len(ground_truth) != num_queries:
        raise ValueError(
            f"{dataset_name} ground truth/query count mismatch: "
            f"{len(ground_truth)} vs {num_queries}"
        )

    normalised: list[np.ndarray] = []
    for query_index, positives in enumerate(ground_truth):
        indices = np.asarray(positives)
        if indices.ndim != 1:
            raise ValueError(
                f"{dataset_name} query {query_index} ground truth must be 1-D"
            )
        if indices.size == 0:
            raise ValueError(
                f"{dataset_name} query {query_index} has empty ground truth"
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                f"{dataset_name} query {query_index} ground truth is not integer"
            )
        indices = indices.astype(np.int64, copy=False)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                f"{dataset_name} query {query_index} ground truth has duplicates"
            )
        if int(indices.min()) < 0 or int(indices.max()) >= num_references:
            raise ValueError(
                f"{dataset_name} query {query_index} ground truth is outside "
                f"[0, {num_references})"
            )
        normalised.append(indices)
    return normalised


def validate_overlapping_ground_truth(
    standard_query_paths: Sequence[Any],
    standard_ground_truth: Sequence[np.ndarray],
    condition_query_paths: Sequence[Any],
    condition_ground_truth: Sequence[np.ndarray],
    *,
    condition_name: str,
) -> tuple[int, int]:
    """Require identical positive sets for queries shared by two manifests."""

    standard_paths = canonical_image_paths(standard_query_paths)
    condition_paths = canonical_image_paths(condition_query_paths)
    if len(standard_paths) != len(standard_ground_truth):
        raise ValueError("standard query paths/ground truth count mismatch")
    if len(condition_paths) != len(condition_ground_truth):
        raise ValueError(
            f"{condition_name} query paths/ground truth count mismatch"
        )
    if len(set(standard_paths.tolist())) != len(standard_paths):
        raise ValueError("standard MSLS queries contain duplicate paths")
    if len(set(condition_paths.tolist())) != len(condition_paths):
        raise ValueError(f"{condition_name} queries contain duplicate paths")

    standard_lookup = {
        path: query_index
        for query_index, path in enumerate(standard_paths.tolist())
    }
    overlap = 0
    for condition_index, path in enumerate(condition_paths.tolist()):
        standard_index = standard_lookup.get(path)
        if standard_index is None:
            continue
        overlap += 1
        standard_positives = np.sort(
            np.asarray(standard_ground_truth[standard_index], dtype=np.int64)
        )
        condition_positives = np.sort(
            np.asarray(condition_ground_truth[condition_index], dtype=np.int64)
        )
        if not np.array_equal(standard_positives, condition_positives):
            raise ValueError(
                f"{condition_name} ground truth disagrees with standard MSLS "
                f"for overlapping query {path!r}; regenerate the condition "
                "manifests before evaluation"
            )
    return overlap, len(condition_paths) - overlap


def string_sequence_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dynamic_class_ids(
    categories: Sequence[str], requested: Sequence[str]
) -> tuple[list[int], list[str]]:
    """Resolve class names from the teacher metadata and reject omissions."""

    category_to_id = {
        str(category).strip().lower(): index
        for index, category in enumerate(categories)
    }
    names = []
    for requested_name in requested:
        name = str(requested_name).strip().lower()
        name = CLASS_ALIASES.get(name, name)
        if name not in names:
            names.append(name)
    unknown = sorted(set(names) - set(category_to_id))
    if unknown:
        raise ValueError(
            f"segmentation teacher has no categories {unknown}; supported "
            f"categories are {list(categories)}"
        )
    return [category_to_id[name] for name in names], names


def dynamic_patch_coverage(
    logits: torch.Tensor,
    dynamic_class_ids: Sequence[int],
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert hard pixel labels to dynamic-pixel fraction per DINO patch.

    The returned mask is in ``[0, 1]`` and has shape ``(B, grid_h, grid_w)``.
    No confidence threshold or per-image normalisation is used.
    """

    if logits.ndim != 4:
        raise ValueError(f"logits must have shape (B, C, H, W), got {logits.shape}")
    if not dynamic_class_ids:
        raise ValueError("at least one dynamic class is required")
    if len(grid_size) != 2 or min(grid_size) <= 0:
        raise ValueError(f"invalid grid_size: {grid_size}")
    if min(dynamic_class_ids) < 0 or max(dynamic_class_ids) >= logits.shape[1]:
        raise ValueError("dynamic class id is outside the teacher output channels")

    labels = logits.argmax(dim=1)
    class_ids = torch.as_tensor(
        dynamic_class_ids,
        dtype=labels.dtype,
        device=labels.device,
    )
    dynamic_pixels = (labels.unsqueeze(-1) == class_ids).any(dim=-1).float()
    coverage = F.adaptive_avg_pool2d(
        dynamic_pixels.unsqueeze(1), output_size=grid_size
    )[:, 0]
    return coverage, labels


def role_preserving_derangement(
    num_references: int,
    num_queries: int,
    seed: int,
    query_strata: Sequence[Any] | None = None,
) -> np.ndarray:
    """Map images to wrong-image masks without crossing roles/conditions.

    References form one seeded random cycle.  Queries form one cycle per value
    in ``query_strata`` (or one shared query cycle when it is omitted).  This
    yields a permutation with no fixed points that is independent of inference
    batch size/order.  Condition membership can therefore be preserved exactly.
    """

    if num_references < 2 or num_queries < 2:
        raise ValueError("both reference and query partitions need at least 2 images")
    if query_strata is None:
        strata = np.zeros(num_queries, dtype=np.int64)
    else:
        strata = np.asarray(query_strata)
        if strata.ndim != 1 or len(strata) != num_queries:
            raise ValueError("query_strata must have one value per query")
    rng = np.random.default_rng(int(seed))
    result = np.empty(num_references + num_queries, dtype=np.int64)

    def assign_cycle(indices: np.ndarray, name: str) -> None:
        if len(indices) < 2:
            raise ValueError(
                f"cannot derange {name}: stratum contains only {len(indices)} image"
            )
        cycle = rng.permutation(indices).astype(np.int64, copy=False)
        result[cycle] = np.roll(cycle, 1)

    assign_cycle(np.arange(num_references, dtype=np.int64), "reference partition")
    query_global = num_references + np.arange(num_queries, dtype=np.int64)
    for stratum in np.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        assign_cycle(query_global[positions], f"query stratum {stratum!r}")

    indices = np.arange(len(result), dtype=np.int64)
    if np.any(result == indices):
        raise AssertionError("derangement unexpectedly contains a fixed point")
    if np.any(result[:num_references] >= num_references):
        raise AssertionError("reference mask donor crossed into the query partition")
    if np.any(result[num_references:] < num_references):
        raise AssertionError("query mask donor crossed into the reference partition")
    if len(np.unique(result)) != len(result):
        raise AssertionError("mask donors are not a permutation")
    return result


def _sample_seed(seed: int, sample_index: int) -> int:
    return (int(seed) * 1_000_003 + int(sample_index) * 97_409) % (2**63 - 1)


def spatially_permute_masks(
    masks: torch.Tensor,
    sample_indices: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Per-image random control preserving every mask value exactly."""

    if masks.ndim != 3:
        raise ValueError("masks must have shape (B, H, W)")
    indices = torch.as_tensor(sample_indices, dtype=torch.long).cpu()
    if indices.numel() != masks.shape[0]:
        raise ValueError("sample_indices length must equal the mask batch size")

    flat_cpu = masks.detach().float().cpu().flatten(1)
    permuted = torch.empty_like(flat_cpu)
    for row, sample_index in enumerate(indices.tolist()):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_sample_seed(seed, int(sample_index)))
        order = torch.randperm(flat_cpu.shape[1], generator=generator)
        permuted[row] = flat_cpu[row, order]
    return permuted.reshape_as(masks).to(device=masks.device, dtype=masks.dtype)


def save_mask_cache(
    output: Path,
    *,
    masks: np.ndarray,
    image_paths: Sequence[Any],
    num_references: int,
    grid_size: tuple[int, int],
    segmentation_size: tuple[int, int],
    model_name: str,
    weights_name: str,
    weights_url: str,
    dynamic_class_names: Sequence[str],
    dynamic_class_ids: Sequence[int],
) -> None:
    """Write a portable, metadata-bound compressed mask cache."""

    canonical_paths = canonical_image_paths(image_paths)
    if masks.shape != (len(canonical_paths), *grid_size):
        raise ValueError(
            "mask tensor shape does not match image count/grid: "
            f"{masks.shape} vs {(len(canonical_paths), *grid_size)}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        masks=np.asarray(masks, dtype=np.float16),
        image_paths=canonical_paths,
        image_paths_sha256=np.asarray(string_sequence_sha256(canonical_paths)),
        num_references=np.asarray(num_references, dtype=np.int64),
        grid_size=np.asarray(grid_size, dtype=np.int64),
        segmentation_size=np.asarray(segmentation_size, dtype=np.int64),
        mask_definition=np.asarray("hard_argmax_dynamic_pixel_area_fraction"),
        model_name=np.asarray(model_name),
        weights_name=np.asarray(weights_name),
        weights_url=np.asarray(weights_url),
        dynamic_class_names=np.asarray(dynamic_class_names, dtype=np.str_),
        dynamic_class_ids=np.asarray(dynamic_class_ids, dtype=np.int64),
    )


def load_and_validate_mask_cache(
    cache_path: Path,
    *,
    expected_image_paths: Sequence[Any],
    expected_num_references: int,
    expected_grid_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a cache only when its dataset identity matches exactly."""

    if not cache_path.is_file():
        raise FileNotFoundError(f"mask cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {
            "schema_version",
            "masks",
            "image_paths",
            "image_paths_sha256",
            "num_references",
            "grid_size",
            "segmentation_size",
            "mask_definition",
            "model_name",
            "weights_name",
            "weights_url",
            "dynamic_class_names",
            "dynamic_class_ids",
        }
        missing = sorted(required - set(cache.files))
        if missing:
            raise ValueError(f"mask cache is missing fields: {missing}")
        schema_version = int(cache["schema_version"].item())
        if schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported mask cache schema {schema_version}; "
                f"expected {CACHE_SCHEMA_VERSION}"
            )
        cached_paths = canonical_image_paths(cache["image_paths"].tolist())
        expected_paths = canonical_image_paths(expected_image_paths)
        if not np.array_equal(cached_paths, expected_paths):
            raise ValueError(
                "mask cache image paths/order do not exactly match this MSLS dataset"
            )
        cached_hash = str(cache["image_paths_sha256"].item())
        expected_hash = string_sequence_sha256(expected_paths)
        if cached_hash != expected_hash:
            raise ValueError("mask cache path hash is inconsistent")
        if int(cache["num_references"].item()) != int(expected_num_references):
            raise ValueError("mask cache reference/query boundary does not match")
        grid_size = tuple(int(value) for value in cache["grid_size"].tolist())
        if grid_size != tuple(expected_grid_size):
            raise ValueError(
                f"mask cache grid {grid_size} does not match {expected_grid_size}"
            )
        masks = np.asarray(cache["masks"], dtype=np.float32)
        if masks.shape != (len(expected_paths), *expected_grid_size):
            raise ValueError(f"unexpected cached mask shape: {masks.shape}")
        if not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
            raise ValueError("cached masks must be finite and lie in [0, 1]")
        metadata = {
            "schema_version": schema_version,
            "image_paths_sha256": cached_hash,
            "num_references": int(cache["num_references"].item()),
            "grid_size": list(grid_size),
            "segmentation_size": [
                int(value) for value in cache["segmentation_size"].tolist()
            ],
            "mask_definition": str(cache["mask_definition"].item()),
            "model_name": str(cache["model_name"].item()),
            "weights_name": str(cache["weights_name"].item()),
            "weights_url": str(cache["weights_url"].item()),
            "dynamic_class_names": cache["dynamic_class_names"].tolist(),
            "dynamic_class_ids": [
                int(value) for value in cache["dynamic_class_ids"].tolist()
            ],
        }
    return masks, metadata


def map_query_indices(
    union_query_paths: Sequence[Any], subset_query_paths: Sequence[Any]
) -> np.ndarray:
    """Map a unique query subset to offsets in a unique query union."""

    full = canonical_image_paths(union_query_paths)
    condition = canonical_image_paths(subset_query_paths)
    mapping: dict[str, int] = {}
    duplicates = set()
    for index, path in enumerate(full.tolist()):
        if path in mapping:
            duplicates.add(path)
        mapping[path] = index
    if duplicates:
        raise ValueError(f"query union contains duplicate paths: {duplicates}")
    if len(set(condition.tolist())) != len(condition):
        raise ValueError("query subset contains duplicate paths")
    missing = [path for path in condition.tolist() if path not in mapping]
    if missing:
        raise ValueError(
            f"query subset contains paths missing from the query union "
            f"({len(missing)} missing); regenerate all condition manifests "
            "and msls_val_condition_union_qImages.npy"
        )
    return np.asarray([mapping[path] for path in condition.tolist()], dtype=np.int64)
