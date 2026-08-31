"""Training-only reliability-calibrated semantic counterfactual dropout.

The module deliberately adds no inference-time parameters.  ADE20K labels and
class statistics are used only to choose non-overlapping 2x2 token blocks
during training; validation and exported checkpoints retain the historical
DINOv2 -> RU gate -> BoQ path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.query_semantic import (
    _canonical_checkpoint_state,
    _validate_ru_checkpoint_config,
)
from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_SCHEMA,
    QUERY_SEMANTIC_CACHE_VERSION,
)


RSCD_STATS_SCHEMA = "openvpr_rscd_class_stats"
RSCD_STATS_VERSION = 1
RSCD_MODES = frozenset(
    {"no_mask", "aligned", "uniform", "shuffled"}
)
_CACHE_ARRAYS = ("labels.npy", "confidence.npy", "shuffled_indices.npy")
_SHA256_HEX = frozenset("0123456789abcdef")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return value


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return value


def _load_json_mapping(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


@dataclass(frozen=True)
class RSCDStats:
    """Validated immutable class statistics used by :class:`RSCDMaskBuilder`."""

    path: Path
    sha256: str
    grid_size: tuple[int, int]
    num_classes: int
    min_confidence: float
    min_confidence_quantized: int
    class_names: tuple[str, ...]
    nuisance_scores: tuple[float, ...]
    valid_classes: tuple[bool, ...]
    source_hashes: Mapping[str, Any]


def _validate_stats_document(document: Mapping[str, Any], path: Path) -> RSCDStats:
    if document.get("schema") != RSCD_STATS_SCHEMA:
        raise ValueError(f"unsupported RSCD stats schema in {path}")
    if document.get("version") != RSCD_STATS_VERSION:
        raise ValueError(f"unsupported RSCD stats version in {path}")
    if document.get("complete") is not True:
        raise ValueError(f"RSCD stats are incomplete: {path}")
    if document.get("cache_schema") != QUERY_SEMANTIC_CACHE_SCHEMA:
        raise ValueError("RSCD stats cache_schema is unsupported")
    if document.get("cache_version") != QUERY_SEMANTIC_CACHE_VERSION:
        raise ValueError("RSCD stats cache_version is unsupported")

    grid = document.get("grid_size")
    if (
        not isinstance(grid, list)
        or len(grid) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in grid)
        or any(value < 2 or value % 2 for value in grid)
    ):
        raise ValueError("RSCD stats grid_size must contain two positive even integers")
    grid_size = (int(grid[0]), int(grid[1]))
    num_classes = _require_int(
        document.get("num_classes"), field="num_classes", minimum=2
    )
    min_confidence = _require_probability(
        document.get("min_confidence"), field="min_confidence"
    )
    min_confidence_quantized = _require_int(
        document.get("min_confidence_quantized"),
        field="min_confidence_quantized",
    )
    expected_quantized = int(math.ceil(min_confidence * 255.0))
    if min_confidence_quantized != expected_quantized:
        raise ValueError(
            "min_confidence_quantized must equal ceil(min_confidence*255)"
        )
    if min_confidence_quantized > 255:
        raise ValueError("min_confidence_quantized must not exceed 255")

    source = document.get("source")
    source_hashes = document.get("source_hashes")
    protocol = document.get("protocol")
    totals = document.get("totals")
    if not all(isinstance(value, Mapping) for value in (source, source_hashes, protocol, totals)):
        raise ValueError("RSCD stats source/source_hashes/protocol/totals must be mappings")

    source_num_images = _require_int(
        source.get("num_images"), field="source.num_images", minimum=1
    )
    _require_int(
        source.get("eligible_min_views"),
        field="source.eligible_min_views",
        minimum=2,
    )
    source_class_names = source.get("class_names")
    if (
        not isinstance(source_class_names, list)
        or len(source_class_names) != num_classes
        or any(not isinstance(name, str) or not name for name in source_class_names)
        or len(set(source_class_names)) != num_classes
    ):
        raise ValueError("source.class_names must list every unique class in id order")

    required_source_hashes = {"manifest.json", *_CACHE_ARRAYS, "city_csv"}
    if set(source_hashes) != required_source_hashes:
        raise ValueError(
            "source_hashes must contain manifest, all immutable arrays, and city_csv"
        )
    for filename in ("manifest.json", *_CACHE_ARRAYS):
        _require_sha256(source_hashes.get(filename), field=f"source_hashes.{filename}")
    city_hashes = source_hashes.get("city_csv")
    if not isinstance(city_hashes, Mapping) or not city_hashes:
        raise ValueError("source_hashes.city_csv must be a non-empty mapping")
    for city, digest in city_hashes.items():
        if not isinstance(city, str) or not city:
            raise ValueError("source_hashes.city_csv contains an invalid city name")
        _require_sha256(digest, field=f"source_hashes.city_csv.{city}")

    min_support = _require_int(
        protocol.get("min_support"), field="protocol.min_support", minimum=1
    )
    for field in (
        "presence",
        "eligible_place",
        "repeatability_numerator",
        "repeatability_support",
        "frequency",
        "nuisance",
        "confidence_comparison",
        "confidence_quantization",
    ):
        if not isinstance(protocol.get(field), str) or not protocol[field]:
            raise ValueError(f"protocol.{field} must be a non-empty string")

    eligible_places = _require_int(
        totals.get("eligible_places"), field="totals.eligible_places", minimum=1
    )
    eligible_images = _require_int(
        totals.get("eligible_images"), field="totals.eligible_images", minimum=1
    )
    cached_images = _require_int(
        totals.get("cached_images"), field="totals.cached_images", minimum=1
    )
    _require_int(totals.get("cities"), field="totals.cities", minimum=1)
    patches_per_image = _require_int(
        totals.get("patches_per_image"),
        field="totals.patches_per_image",
        minimum=1,
    )
    if cached_images != source_num_images:
        raise ValueError("totals.cached_images disagrees with source.num_images")
    if patches_per_image != grid_size[0] * grid_size[1]:
        raise ValueError("totals.patches_per_image disagrees with grid_size")
    if eligible_images > cached_images:
        raise ValueError("eligible_images cannot exceed cached_images")

    classes = document.get("classes")
    if not isinstance(classes, list) or len(classes) != num_classes:
        raise ValueError("classes must contain exactly num_classes entries")
    nuisance_scores: list[float] = []
    valid_classes: list[bool] = []
    for expected_id, row in enumerate(classes):
        if not isinstance(row, Mapping):
            raise ValueError(f"classes[{expected_id}] must be a mapping")
        if row.get("id") != expected_id:
            raise ValueError("class ids must be contiguous and in ascending order")
        if row.get("name") != source_class_names[expected_id]:
            raise ValueError(f"classes[{expected_id}].name disagrees with source")
        views_present = _require_int(
            row.get("views_present"),
            field=f"classes[{expected_id}].views_present",
        )
        places_present = _require_int(
            row.get("places_present"),
            field=f"classes[{expected_id}].places_present",
        )
        numerator = _require_int(
            row.get("repeatability_numerator"),
            field=f"classes[{expected_id}].repeatability_numerator",
        )
        support = _require_int(
            row.get("support"), field=f"classes[{expected_id}].support"
        )
        if views_present > eligible_images or places_present > eligible_places:
            raise ValueError(f"classes[{expected_id}] presence count is impossible")
        if numerator > support:
            raise ValueError(
                f"classes[{expected_id}] repeatability_numerator exceeds support"
            )
        frequency = _require_probability(
            row.get("frequency"), field=f"classes[{expected_id}].frequency"
        )
        expected_frequency = places_present / eligible_places
        if not math.isclose(frequency, expected_frequency, abs_tol=1e-9):
            raise ValueError(f"classes[{expected_id}] frequency is inconsistent")

        valid = row.get("valid")
        if not isinstance(valid, bool):
            raise ValueError(f"classes[{expected_id}].valid must be boolean")
        invalid_reason = row.get("invalid_reason")
        if valid:
            if support < min_support:
                raise ValueError(f"classes[{expected_id}] is valid below min_support")
            if invalid_reason is not None:
                raise ValueError(f"classes[{expected_id}] valid row has invalid_reason")
            repeatability = _require_probability(
                row.get("repeatability"),
                field=f"classes[{expected_id}].repeatability",
            )
            expected_repeatability = numerator / support
            if not math.isclose(repeatability, expected_repeatability, abs_tol=1e-9):
                raise ValueError(
                    f"classes[{expected_id}] repeatability is inconsistent"
                )
            nuisance = _require_probability(
                row.get("nuisance"), field=f"classes[{expected_id}].nuisance"
            )
            expected_nuisance = 1.0 - repeatability * (1.0 - frequency)
            if not math.isclose(nuisance, expected_nuisance, abs_tol=1e-9):
                raise ValueError(f"classes[{expected_id}] nuisance is inconsistent")
            nuisance_scores.append(nuisance)
            valid_classes.append(True)
        else:
            if support >= min_support:
                raise ValueError(f"classes[{expected_id}] invalid row meets min_support")
            if row.get("repeatability") is not None or row.get("nuisance") is not None:
                raise ValueError(
                    f"classes[{expected_id}] invalid row must use null reliability"
                )
            if not isinstance(invalid_reason, str) or not invalid_reason:
                raise ValueError(
                    f"classes[{expected_id}] invalid row needs invalid_reason"
                )
            nuisance_scores.append(0.0)
            valid_classes.append(False)

    return RSCDStats(
        path=path,
        sha256=_file_sha256(path),
        grid_size=grid_size,
        num_classes=num_classes,
        min_confidence=min_confidence,
        min_confidence_quantized=min_confidence_quantized,
        class_names=tuple(source_class_names),
        nuisance_scores=tuple(nuisance_scores),
        valid_classes=tuple(valid_classes),
        source_hashes=dict(source_hashes),
    )


def _validate_stats_against_cache(
    stats: RSCDStats,
    cache_manifest_path: Path,
    *,
    verify_array_files: bool,
) -> None:
    manifest = _load_json_mapping(
        cache_manifest_path, description="query-semantic cache manifest"
    )
    if manifest.get("complete") is not True:
        raise ValueError("query-semantic cache manifest is incomplete")
    if manifest.get("schema") != QUERY_SEMANTIC_CACHE_SCHEMA:
        raise ValueError("query-semantic cache schema disagrees with RSCD stats")
    if manifest.get("version") != QUERY_SEMANTIC_CACHE_VERSION:
        raise ValueError("query-semantic cache version disagrees with RSCD stats")
    if tuple(manifest.get("grid_size", ())) != stats.grid_size:
        raise ValueError("query-semantic cache grid disagrees with RSCD stats")
    if manifest.get("num_classes") != stats.num_classes:
        raise ValueError("query-semantic cache class count disagrees with RSCD stats")
    if tuple(manifest.get("classes", ())) != stats.class_names:
        raise ValueError("query-semantic cache classes disagree with RSCD stats")
    if _file_sha256(cache_manifest_path) != stats.source_hashes["manifest.json"]:
        raise ValueError("query-semantic manifest SHA256 disagrees with RSCD stats")

    declared_arrays = manifest.get("array_sha256")
    if not isinstance(declared_arrays, Mapping) or set(declared_arrays) != set(
        _CACHE_ARRAYS
    ):
        raise ValueError("query-semantic cache manifest has invalid array hashes")
    cache_dir = cache_manifest_path.parent
    for filename in _CACHE_ARRAYS:
        declared = _require_sha256(
            declared_arrays.get(filename), field=f"manifest.array_sha256.{filename}"
        )
        if declared != stats.source_hashes[filename]:
            raise ValueError(f"{filename} SHA256 disagrees with RSCD stats")
        if verify_array_files:
            array_path = cache_dir / filename
            if not array_path.is_file():
                raise FileNotFoundError(f"query-semantic cache array not found: {array_path}")
            if _file_sha256(array_path) != declared:
                raise ValueError(f"query-semantic cache file SHA256 mismatch: {filename}")

    city_entries = manifest.get("cities")
    if not isinstance(city_entries, list) or not city_entries:
        raise ValueError("query-semantic cache manifest has no city entries")
    declared_city_hashes = {}
    for entry in city_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("query-semantic cache city entry must be a mapping")
        city = entry.get("name")
        if not isinstance(city, str) or not city or city in declared_city_hashes:
            raise ValueError("query-semantic cache has invalid/duplicate cities")
        declared_city_hashes[city] = _require_sha256(
            entry.get("sha256"), field=f"manifest.cities.{city}.sha256"
        )
    if declared_city_hashes != dict(stats.source_hashes["city_csv"]):
        raise ValueError("city CSV hashes disagree with RSCD stats")


def load_rscd_stats(
    stats_path: str | Path,
    *,
    cache_manifest: str | Path | None = None,
    verify_array_files: bool = True,
    expected_min_confidence: float | None = None,
) -> RSCDStats:
    """Load statistics and optionally bind them to an immutable cache.

    Passing ``cache_manifest`` is strongly recommended for every training run;
    it verifies schema, classes, grid and cryptographic provenance.  When a
    manifest is supplied, immutable array files are also re-hashed by default.
    """

    path = Path(stats_path).expanduser().resolve()
    document = _load_json_mapping(path, description="RSCD stats")
    stats = _validate_stats_document(document, path)
    if expected_min_confidence is not None:
        expected = _require_probability(
            expected_min_confidence, field="expected_min_confidence"
        )
        if not math.isclose(stats.min_confidence, expected, abs_tol=1e-12):
            raise ValueError("RSCD stats min_confidence disagrees with runtime")
    if cache_manifest is not None:
        manifest_path = Path(cache_manifest).expanduser().resolve()
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        _validate_stats_against_cache(
            stats,
            manifest_path,
            verify_array_files=bool(verify_array_files),
        )
    return stats


class RSCDMaskBuilder(nn.Module):
    """Build matched 2x2 token-block masks without trainable/persistent state."""

    def __init__(
        self,
        nuisance_scores: RSCDStats | Sequence[float] | torch.Tensor,
        *,
        mode: str,
        confidence_threshold: float | None = None,
        max_coverage: float = 0.15,
        seed: int = 42,
        grid_size: tuple[int, int] | None = None,
        valid_classes: Sequence[bool] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in RSCD_MODES:
            raise ValueError(f"mode must be one of {sorted(RSCD_MODES)}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(float(max_coverage)) or not 0.0 < max_coverage <= 1.0:
            raise ValueError("max_coverage must be finite and in (0, 1]")

        if isinstance(nuisance_scores, RSCDStats):
            stats = nuisance_scores
            scores = torch.tensor(stats.nuisance_scores, dtype=torch.float32)
            valid = torch.tensor(stats.valid_classes, dtype=torch.bool)
            if confidence_threshold is None:
                confidence_threshold = stats.min_confidence
            elif not math.isclose(
                float(confidence_threshold), stats.min_confidence, abs_tol=1e-12
            ):
                raise ValueError("confidence_threshold disagrees with RSCD stats")
            if grid_size is None:
                grid_size = stats.grid_size
            elif tuple(grid_size) != stats.grid_size:
                raise ValueError("grid_size disagrees with RSCD stats")
            if valid_classes is not None:
                raise ValueError("valid_classes must not override validated RSCD stats")
        else:
            scores = torch.as_tensor(nuisance_scores, dtype=torch.float32)
            if scores.ndim != 1 or scores.numel() < 2:
                raise ValueError("nuisance_scores must be a 1D vector with >=2 classes")
            if valid_classes is None:
                valid = torch.ones_like(scores, dtype=torch.bool)
            else:
                valid = torch.as_tensor(valid_classes, dtype=torch.bool)
                if valid.shape != scores.shape:
                    raise ValueError("valid_classes must match nuisance_scores")
            if confidence_threshold is None:
                raise ValueError("confidence_threshold is required without RSCDStats")
            if grid_size is None:
                raise ValueError("grid_size is required without RSCDStats")

        threshold = _require_probability(
            confidence_threshold, field="confidence_threshold"
        )
        if scores.ndim != 1 or scores.numel() < 2:
            raise ValueError("nuisance_scores must be a 1D vector with >=2 classes")
        if not bool(torch.isfinite(scores).all()) or bool(
            ((scores < 0) | (scores > 1)).any()
        ):
            raise ValueError("nuisance_scores must be finite and in [0, 1]")
        if valid.shape != scores.shape:
            raise ValueError("valid_classes must match nuisance_scores")
        if bool((scores[~valid] != 0).any()):
            raise ValueError("invalid classes must have zero nuisance weight")
        grid_size = tuple(int(value) for value in grid_size)
        if len(grid_size) != 2 or any(value < 2 or value % 2 for value in grid_size):
            raise ValueError("grid_size must contain two positive even integers")
        max_blocks = int(math.floor(max_coverage * math.prod(grid_size) / 4.0))
        if max_blocks < 1:
            raise ValueError("max_coverage permits no complete 2x2 token block")

        self.mode = mode
        self.confidence_threshold = threshold
        self.max_coverage = float(max_coverage)
        self.seed = int(seed)
        self.grid_size = grid_size
        self.block_size = 2
        self.max_blocks = max_blocks
        # Non-persistent buffers follow the Lightning device while adding no
        # checkpoint keys.  The class is intentionally parameter-free.
        self.register_buffer("_nuisance", scores.clone(), persistent=False)
        self.register_buffer("_valid_classes", valid.clone(), persistent=False)

    @property
    def num_classes(self) -> int:
        return int(self._nuisance.numel())

    @staticmethod
    def _is_integer_tensor(value: torch.Tensor) -> bool:
        return value.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }

    def _validate_runtime_map(
        self,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        *,
        prefix: str,
    ) -> None:
        if not torch.is_tensor(labels) or not torch.is_tensor(confidence):
            raise TypeError(f"{prefix} labels/confidence must be tensors")
        if labels.ndim != 3 or tuple(labels.shape[1:]) != self.grid_size:
            raise ValueError(
                f"{prefix} labels must have shape (B,{self.grid_size[0]},"
                f"{self.grid_size[1]})"
            )
        if confidence.shape != labels.shape:
            raise ValueError(f"{prefix} confidence must match labels")
        if not self._is_integer_tensor(labels):
            raise TypeError(f"{prefix} labels must use an integer dtype")
        if not confidence.is_floating_point():
            raise TypeError(f"{prefix} confidence must be decoded floating point")
        if labels.device != confidence.device:
            raise ValueError(f"{prefix} labels/confidence must share a device")
        if labels.numel() and (
            int(labels.min().item()) < 0
            or int(labels.max().item()) >= self.num_classes
        ):
            raise ValueError(f"{prefix} labels are outside the class range")
        if not bool(torch.isfinite(confidence).all()) or bool(
            ((confidence < 0) | (confidence > 1)).any()
        ):
            raise ValueError(f"{prefix} confidence must be finite and in [0, 1]")

    def _block_candidates(
        self, labels: torch.Tensor, confidence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = labels.shape
        blocks = (
            labels.view(batch, height // 2, 2, width // 2, 2)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        confidence_blocks = (
            confidence.view(batch, height // 2, 2, width // 2, 2)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        block_labels = blocks[..., 0, 0]
        same_label = (blocks == block_labels[..., None, None]).all(dim=(-1, -2))
        high_confidence = (
            confidence_blocks >= self.confidence_threshold
        ).all(dim=(-1, -2))
        nuisance = self._nuisance[block_labels.long()]
        valid = self._valid_classes[block_labels.long()]
        candidates = same_label & high_confidence & valid & (nuisance > 0)
        return candidates.flatten(1), nuisance.flatten(1)

    def _stateless_uniform(
        self,
        cache_indices: torch.Tensor,
        global_step: int,
        count: int,
        *,
        stream: int,
    ) -> torch.Tensor:
        # Integer-only SplitMix-style hashing makes every random key a pure
        # function of seed, stable cache row, optimizer step and block ordinal.
        # It neither consumes nor depends on a process-global RNG stream.
        device = cache_indices.device
        modulus_mask = (1 << 63) - 1
        block = torch.arange(count, device=device, dtype=torch.int64)[None]
        base_constant = (
            self.seed * 3935559000370003845
            + global_step * 3202034522624059733
            + stream * 2691343689449507681
        ) & modulus_mask
        value = (
            cache_indices.to(torch.int64)[:, None] * 6364136223846793005
            + block * 1442695040888963407
            + base_constant
        )
        value = torch.bitwise_and(value, modulus_mask)
        value = torch.bitwise_xor(value, value >> 30)
        value = torch.bitwise_and(value * 6364136223846793005, modulus_mask)
        value = torch.bitwise_xor(value, value >> 27)
        value = torch.bitwise_and(value * 1442695040888963407, modulus_mask)
        value = torch.bitwise_xor(value, value >> 31)
        # Float64 preserves 53 useful random bits.  Clamp away from zero for
        # the exponential-race weighted sampling below.
        uniform = (value.to(torch.float64) + 1.0) / float(1 << 63)
        return uniform.clamp_(min=torch.finfo(torch.float64).eps, max=1.0)

    @staticmethod
    def _select_blocks(
        candidates: torch.Tensor,
        weights: torch.Tensor,
        quota: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.shape != weights.shape or candidates.shape != noise.shape:
            raise ValueError("candidate, weight and noise shapes must match")
        if quota.shape != (candidates.shape[0],):
            raise ValueError("quota must have shape (B,)")
        priority = -torch.log(noise) / weights.double().clamp_min(1e-12)
        priority = priority.masked_fill(~candidates, float("inf"))
        order = priority.argsort(dim=1)
        take = torch.arange(
            candidates.shape[1], device=candidates.device
        )[None] < quota[:, None]
        selected = torch.zeros_like(candidates)
        selected.scatter_(1, order, take)
        return selected & candidates

    def build(
        self,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        cache_indices: torch.Tensor,
        global_step: int,
        *,
        donor_labels: torch.Tensor | None = None,
        donor_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a boolean ``(B,H,W)`` mask and detached diagnostics.

        Every active mode uses the same per-image quota: the minimum of the
        configured cap and the numbers of valid aligned/donor 2x2 blocks.
        Consequently aligned, uniform and shuffled masks have exactly equal
        coverage for a given batch, cache row and step.
        """

        self._validate_runtime_map(labels, confidence, prefix="aligned")
        batch = labels.shape[0]
        if (
            not torch.is_tensor(cache_indices)
            or cache_indices.ndim != 1
            or cache_indices.shape[0] != batch
            or not self._is_integer_tensor(cache_indices)
        ):
            raise ValueError("cache_indices must be an integer tensor with shape (B,)")
        if cache_indices.device != labels.device:
            raise ValueError("cache_indices and labels must share a device")
        if cache_indices.numel() and int(cache_indices.min().item()) < 0:
            raise ValueError("cache_indices must be non-negative")
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")

        if self.mode != "no_mask":
            if donor_labels is None or donor_confidence is None:
                raise ValueError(
                    "active RSCD modes require donor maps for a common quota"
                )
            self._validate_runtime_map(
                donor_labels, donor_confidence, prefix="donor"
            )
            if donor_labels.shape[0] != batch or donor_labels.device != labels.device:
                raise ValueError("donor maps must match the aligned batch/device")

        aligned_candidates, aligned_weights = self._block_candidates(
            labels, confidence
        )
        if donor_labels is None:
            donor_candidates = torch.zeros_like(aligned_candidates)
            donor_weights = torch.zeros_like(aligned_weights)
        else:
            donor_candidates, donor_weights = self._block_candidates(
                donor_labels, donor_confidence
            )

        aligned_count = aligned_candidates.sum(dim=1)
        donor_count = donor_candidates.sum(dim=1)
        if self.mode == "no_mask":
            quota = torch.zeros_like(aligned_count)
        else:
            cap = torch.full_like(aligned_count, self.max_blocks)
            quota = torch.minimum(cap, torch.minimum(aligned_count, donor_count))

        block_count = aligned_candidates.shape[1]
        if self.mode == "aligned":
            candidates = aligned_candidates
            weights = aligned_weights
            diagnostic_weights = aligned_weights
            stream = 1
        elif self.mode == "shuffled":
            candidates = donor_candidates
            weights = donor_weights
            diagnostic_weights = donor_weights
            stream = 2
        elif self.mode == "uniform":
            candidates = torch.ones_like(aligned_candidates)
            weights = torch.ones_like(aligned_weights)
            # Selection is uniform, but report the nuisance of the receiver
            # regions actually removed so the placebo remains interpretable.
            diagnostic_weights = aligned_weights
            stream = 3
        else:
            candidates = torch.zeros_like(aligned_candidates)
            weights = torch.ones_like(aligned_weights)
            diagnostic_weights = torch.zeros_like(aligned_weights)
            stream = 4

        noise = self._stateless_uniform(
            cache_indices, global_step, block_count, stream=stream
        )
        selected = self._select_blocks(candidates, weights, quota, noise)
        block_height, block_width = self.grid_size[0] // 2, self.grid_size[1] // 2
        mask = (
            selected.view(batch, block_height, block_width)
            .repeat_interleave(2, dim=1)
            .repeat_interleave(2, dim=2)
        )

        selected_weight_sum = (diagnostic_weights * selected).sum(dim=1)
        selected_weight_mean = torch.where(
            quota > 0,
            selected_weight_sum / quota.clamp_min(1).to(weights.dtype),
            torch.zeros_like(selected_weight_sum),
        )
        mask_tokens = mask.sum(dim=(1, 2))
        stats = {
            "rscd_mask_coverage": mask.float().mean().detach(),
            "rscd_mask_tokens": mask_tokens.float().mean().detach(),
            "rscd_quota_blocks": quota.float().mean().detach(),
            "rscd_aligned_candidate_blocks": aligned_count.float().mean().detach(),
            "rscd_shuffled_candidate_blocks": donor_count.float().mean().detach(),
            "rscd_selected_nuisance": selected_weight_mean.mean().detach(),
            "rscd_zero_quota_frac": (quota == 0).float().mean().detach(),
        }
        return mask, stats


def apply_token_mask(featmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked tokens by a detached per-image spatial mean."""

    if not torch.is_tensor(featmap) or featmap.ndim != 4:
        raise ValueError("featmap must have shape (B,C,H,W)")
    if not torch.is_tensor(mask):
        raise TypeError("mask must be a tensor")
    if mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError("4D mask must have shape (B,1,H,W)")
        mask = mask[:, 0]
    if mask.ndim != 3 or tuple(mask.shape) != (
        featmap.shape[0],
        featmap.shape[2],
        featmap.shape[3],
    ):
        raise ValueError("mask must have shape (B,H,W) matching featmap")
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have boolean dtype")
    if mask.device != featmap.device:
        raise ValueError("mask and featmap must share a device")
    if not bool(torch.isfinite(featmap).all()):
        raise ValueError("featmap must contain only finite values")
    detached_mean = featmap.mean(dim=(-2, -1), keepdim=True).detach()
    return torch.where(mask[:, None], detached_mean, featmap)


def pairwise_relation_loss(
    masked_descriptors: torch.Tensor,
    clean_descriptors: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 loss between off-diagonal batch cosine relations.

    The clean relation matrix is always detached, even if the caller forgot to
    construct the clean branch under ``torch.no_grad``.
    """

    if not torch.is_tensor(masked_descriptors) or not torch.is_tensor(
        clean_descriptors
    ):
        raise TypeError("descriptors must be tensors")
    if (
        masked_descriptors.ndim != 2
        or clean_descriptors.shape != masked_descriptors.shape
    ):
        raise ValueError("descriptors must share shape (B,D)")
    if masked_descriptors.shape[0] < 2 or masked_descriptors.shape[1] < 1:
        raise ValueError("pairwise relation loss requires B>=2 and D>=1")
    if masked_descriptors.device != clean_descriptors.device:
        raise ValueError("descriptor tensors must share a device")
    if not bool(torch.isfinite(masked_descriptors).all()) or not bool(
        torch.isfinite(clean_descriptors).all()
    ):
        raise ValueError("descriptors must contain only finite values")

    masked = F.normalize(masked_descriptors.float(), p=2, dim=1)
    clean = F.normalize(clean_descriptors.detach().float(), p=2, dim=1)
    masked_relations = masked @ masked.transpose(0, 1)
    clean_relations = clean @ clean.transpose(0, 1)
    off_diagonal = ~torch.eye(
        masked.shape[0], dtype=torch.bool, device=masked.device
    )
    return F.smooth_l1_loss(
        masked_relations[off_diagonal], clean_relations[off_diagonal]
    )


def warm_start_rscd_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Strictly load the audited RU checkpoint into an unchanged model.

    Unlike earlier semantic branches, RSCD adds no learned tensors.  Therefore
    any missing or unexpected state key is an error rather than an allowed new
    parameter.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {path}")
    checkpoint_sha256 = _file_sha256(path)
    if expected_sha256 is not None:
        expected = _require_sha256(
            str(expected_sha256).lower(), field="expected RU checkpoint SHA256"
        )
        if checkpoint_sha256 != expected:
            raise RuntimeError(
                "RU checkpoint SHA256 mismatch: expected "
                f"{expected}, found {checkpoint_sha256}"
            )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("initial checkpoint must be a mapping")
    provenance = _validate_ru_checkpoint_config(checkpoint)
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("initial checkpoint contains no state_dict mapping")
    state = _canonical_checkpoint_state(raw_state)
    for prefix in ("backbone.", "aggregator.", "semantic_region_gate."):
        if not any(key.startswith(prefix) for key in state):
            raise RuntimeError(
                f"RU warm start requires checkpoint weights with prefix {prefix!r}"
            )
    model_keys = set(model.state_dict())
    rscd_keys = sorted(key for key in model_keys if "rscd" in key.lower())
    if rscd_keys:
        raise RuntimeError(
            "RSCD must not add persistent checkpoint state: " f"{rscd_keys}"
        )
    if model_keys != set(state):
        missing = sorted(model_keys - set(state))
        unexpected = sorted(set(state) - model_keys)
        raise RuntimeError(
            "RSCD requires an architecture-identical RU warm start: "
            f"missing={missing}, unexpected={unexpected}"
        )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RU checkpoint is incompatible with the configured RSCD model: {exc}"
        ) from exc
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_keys": len(state),
        "new_keys": (),
        **provenance,
    }


__all__ = [
    "RSCD_MODES",
    "RSCD_STATS_SCHEMA",
    "RSCD_STATS_VERSION",
    "RSCDMaskBuilder",
    "RSCDStats",
    "apply_token_mask",
    "load_rscd_stats",
    "pairwise_relation_loss",
    "warm_start_rscd_model",
]
