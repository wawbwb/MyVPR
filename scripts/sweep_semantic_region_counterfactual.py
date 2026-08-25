"""Sweep sparse-confidence and target-normalization counterfactuals offline.

The input is ``diagnostic_tensors.pt`` produced by
``visualize_semantic_region_delta.py``.  No checkpoint, dataset, CLIP cache, or
GPU is required: the bundle already contains the shared DINO reliability and
the aligned/shuffled semantic neighbour averages.

This script answers a target-design question, not a retrieval question.  It
measures how hard confidence masks and alternatives to per-image unit-variance
normalization change the teacher targets.  A promising row still requires a
matched aligned/shuffled retraining run before making an R@1 claim.

Example:
    python scripts/sweep_semantic_region_counterfactual.py \
      --input doc/semantic_region_delta_batch0_clean/diagnostic_tensors.pt \
      --output doc/semantic_region_counterfactual_batch0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


TRANSFORMS = ("per_image_zscore", "shared_base", "center_only")
COMPOSITIONS = ("production_roundtrip", "ru_additive")
TRANSFORM_DESCRIPTIONS = {
    "per_image_zscore": (
        "Production transform: independently center and unit-standardize "
        "every image before tanh."
    ),
    "shared_base": (
        "Recommended diagnostic: use one mean/std measured from the RU base "
        "for every image and every counterfactual."
    ),
    "center_only": (
        "Center each image but do not divide by its spatial standard deviation."
    ),
}
COMPOSITION_DESCRIPTIONS = {
    "production_roundtrip": (
        "Reproduce production full/shuffled: resize base 10->14, add delta, "
        "then resize the complete map 14->10."
    ),
    "ru_additive": (
        "RU-preserving counterfactual: resize only delta 14->10 and add it "
        "to the untouched base10; a zero mask is exactly the RU base."
    ),
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


@dataclass(frozen=True)
class MaskSpec:
    """One confidence-coverage intervention."""

    name: str
    threshold: float | None = None
    top_fraction: float | None = None
    disabled: bool = False


@dataclass(frozen=True)
class TargetScaler:
    """A fixed affine scale estimated once from the no-semantics RU base."""

    mean: float
    std: float

    @classmethod
    def from_base(
        cls, base_feature: torch.Tensor, eps: float = 1e-6
    ) -> "TargetScaler":
        _require_feature_map(base_feature, "base_feature")
        mean = float(base_feature.float().mean())
        std = float(base_feature.float().std(unbiased=False))
        return cls(mean=mean, std=max(std, eps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline sweep of hard confidence masks and target normalization "
            "using a saved semantic-region diagnostic tensor bundle."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="diagnostic_tensors.pt from visualize_semantic_region_delta.py",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        required=True,
        help="new or empty output directory",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.5, 0.6, 0.7],
        help="absolute confidence thresholds; retained patches keep confidence",
    )
    parser.add_argument(
        "--top-fractions",
        type=float,
        nargs="*",
        default=[0.2],
        help="per-image top confidence fractions, e.g. 0.2 for top 20%%",
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=TRANSFORMS,
        default=list(TRANSFORMS),
    )
    parser.add_argument(
        "--compositions",
        nargs="+",
        choices=COMPOSITIONS,
        default=list(COMPOSITIONS),
        help="include production interpolation and/or RU-preserving delta add",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="tanh multiplier; default reads builder target_scale from bundle",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def _require_feature_map(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must have shape (B,1,H,W)")
    if value.shape[1] != 1:
        raise ValueError(f"{name} must have exactly one channel")


def _square_side(token_count: int, name: str) -> int:
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise ValueError(f"{name} token count must form a square grid")
    return side


def make_hard_mask(
    confidence: torch.Tensor,
    threshold: float | None = None,
    top_fraction: float | None = None,
) -> torch.Tensor:
    """Return an exact Boolean coverage mask for a ``(B,N)`` confidence map.

    Retained locations keep their original confidence later; the mask does not
    turn confidence into one.  For a top fraction, exactly ``ceil(N*f)``
    locations are selected per image, even when confidence values are tied.
    """

    if not isinstance(confidence, torch.Tensor) or confidence.ndim != 2:
        raise ValueError("confidence must have shape (B,N)")
    if threshold is not None and top_fraction is not None:
        raise ValueError("choose threshold or top_fraction, not both")
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0,1]")
        return confidence >= threshold
    if top_fraction is not None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0,1]")
        keep = max(1, math.ceil(confidence.shape[1] * top_fraction))
        indices = confidence.topk(keep, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros_like(confidence, dtype=torch.bool)
        return mask.scatter(1, indices, True)
    return torch.ones_like(confidence, dtype=torch.bool)


def resize_reliability(
    reliability: torch.Tensor, feature_hw: tuple[int, int]
) -> torch.Tensor:
    """Bilinearly resize a flattened square reliability grid."""

    if not isinstance(reliability, torch.Tensor) or reliability.ndim != 2:
        raise ValueError("reliability must have shape (B,N)")
    if len(feature_hw) != 2 or min(feature_hw) < 1:
        raise ValueError("feature_hw must contain two positive integers")
    side = _square_side(reliability.shape[1], "reliability")
    return F.interpolate(
        reliability.float().view(-1, 1, side, side),
        size=feature_hw,
        mode="bilinear",
        align_corners=False,
    )


def apply_target_transform(
    feature_map: torch.Tensor,
    mode: str,
    target_scale: float,
    scaler: TargetScaler | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Transform one pre-target feature map into the bounded teacher target."""

    _require_feature_map(feature_map, "feature_map")
    if mode not in TRANSFORMS:
        raise ValueError(f"unknown target transform: {mode}")
    if target_scale <= 0.0 or eps <= 0.0:
        raise ValueError("target_scale and eps must be positive")
    values = feature_map.float()
    if mode == "per_image_zscore":
        flat = values.flatten(1)
        centre = flat.mean(dim=1, keepdim=True)
        scale = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        normalized = ((flat - centre) / scale).view_as(values)
    elif mode == "shared_base":
        if scaler is None:
            raise ValueError("shared_base transform requires a TargetScaler")
        normalized = (values - scaler.mean) / max(scaler.std, eps)
    else:
        normalized = values - values.mean(dim=(-2, -1), keepdim=True)
    return torch.tanh(target_scale * normalized)


def build_counterfactual_branch(
    base14: torch.Tensor,
    smoothed14: torch.Tensor,
    confidence14: torch.Tensor,
    base_side: int,
    feature_hw: tuple[int, int],
    mask: torch.Tensor,
    transform: str,
    target_scale: float,
    scaler: TargetScaler | None = None,
    eps: float = 1e-6,
    composition: str = "production_roundtrip",
    base_reliability: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Rebuild one branch after a mask and target-transform intervention."""

    expected = base14.shape
    if base14.ndim != 2 or smoothed14.shape != expected:
        raise ValueError("base14 and smoothed14 must share shape (B,N)")
    if confidence14.shape != expected or mask.shape != expected:
        raise ValueError("confidence14 and mask must match base14")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be Boolean")
    if base_side < 1:
        raise ValueError("base_side must be positive")
    if composition not in COMPOSITIONS:
        raise ValueError(f"unknown composition: {composition}")

    confidence = confidence14.float().clamp(0.0, 1.0)
    effective_confidence = confidence * mask.float()
    delta14 = effective_confidence * (smoothed14.float() - base14.float())
    propagated14 = base14.float() + delta14
    patch_side = _square_side(propagated14.shape[1], "propagated14")
    resized_delta = F.interpolate(
        delta14.view(-1, 1, patch_side, patch_side),
        size=(base_side, base_side),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    if composition == "production_roundtrip":
        out_base = F.interpolate(
            propagated14.view(-1, 1, patch_side, patch_side),
            size=(base_side, base_side),
            mode="bilinear",
            align_corners=False,
        ).flatten(1)
    else:
        if base_reliability is None:
            raise ValueError("ru_additive composition requires base_reliability")
        if base_reliability.shape != (base14.shape[0], base_side * base_side):
            raise ValueError("base_reliability shape does not match base_side")
        out_base = base_reliability.float() + resized_delta
    pre_target = resize_reliability(out_base, feature_hw)
    target = apply_target_transform(
        pre_target,
        mode=transform,
        target_scale=target_scale,
        scaler=scaler,
        eps=eps,
    )
    return {
        "mask": mask,
        "effective_confidence": effective_confidence,
        "delta14": delta14,
        "propagated14": propagated14,
        "resized_delta": resized_delta,
        "out_base": out_base,
        "pre_target": pre_target,
        "target": target,
    }


def _spatial_correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.shape[0] < 1:
        raise ValueError("correlation inputs must share a non-empty batch shape")
    left = first.float().flatten(1)
    right = second.float().flatten(1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    denominator = left.norm(dim=1) * right.norm(dim=1)
    result = (left * right).sum(dim=1) / denominator.clamp_min(1e-12)
    return torch.where(
        denominator > 1e-12,
        result,
        torch.full_like(result, float("nan")),
    )


def _per_sample_abs_mean(value: torch.Tensor) -> torch.Tensor:
    return value.float().abs().flatten(1).mean(dim=1)


def _per_sample_std(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).std(dim=1, unbiased=False)


def _per_sample_fraction(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).mean(dim=1)


def _finite_mean(value: torch.Tensor) -> float:
    finite = value[torch.isfinite(value)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _mean_field(records: list[dict[str, Any]], field: str) -> float:
    values = [float(record[field]) for record in records]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _mask_specs(
    thresholds: list[float], top_fractions: list[float]
) -> list[MaskSpec]:
    specs = [MaskSpec(name="semantic_off", disabled=True), MaskSpec(name="dense")]
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("every threshold must lie in [0,1]")
        specs.append(
            MaskSpec(
                name=f"confidence_ge_{threshold:g}", threshold=threshold
            )
        )
    for fraction in top_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("every top fraction must lie in (0,1]")
        specs.append(
            MaskSpec(
                name=f"top_{100.0 * fraction:g}pct_per_image",
                top_fraction=fraction,
            )
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("threshold/top-fraction arguments produce duplicate masks")
    return specs


def _load_bundle(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic tensor bundle not found: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic tensor bundle schema")
    diagnostics = bundle.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise KeyError("bundle has no diagnostics mapping")
    for name in ("base10", "aligned", "shuffled", "builder_kwargs"):
        if name not in diagnostics:
            raise KeyError(f"bundle diagnostics missing {name!r}")
    for branch_name in ("aligned", "shuffled"):
        branch = diagnostics[branch_name]
        if not isinstance(branch, dict):
            raise TypeError(f"diagnostics[{branch_name!r}] must be a mapping")
        for name in ("base14", "smoothed14", "confidence14", "target"):
            if not isinstance(branch.get(name), torch.Tensor):
                raise KeyError(f"diagnostics.{branch_name} missing tensor {name}")
    return bundle


def _prepare_output_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}; choose a new directory"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_and_describe_bundle(
    bundle: dict[str, Any], target_scale_override: float | None, eps: float
) -> dict[str, Any]:
    diagnostics = bundle["diagnostics"]
    base10 = diagnostics["base10"].float()
    if base10.ndim != 2:
        raise ValueError("diagnostics.base10 must have shape (B,N)")
    base_side = _square_side(base10.shape[1], "base10")
    aligned = diagnostics["aligned"]
    shuffled = diagnostics["shuffled"]
    if aligned["target"].shape != shuffled["target"].shape:
        raise ValueError("aligned and shuffled saved targets differ in shape")
    _require_feature_map(aligned["target"], "aligned.target")
    feature_hw = tuple(int(value) for value in aligned["target"].shape[-2:])
    source_target_scale = float(
        diagnostics["builder_kwargs"].get("target_scale", 2.0)
    )
    target_scale = (
        float(target_scale_override)
        if target_scale_override is not None
        else source_target_scale
    )
    if target_scale <= 0.0 or eps <= 0.0:
        raise ValueError("target_scale and eps must be positive")
    base_feature = resize_reliability(base10, feature_hw)
    scaler = TargetScaler.from_base(base_feature, eps=eps)

    if not torch.allclose(
        aligned["base14"].float(), shuffled["base14"].float(), rtol=0, atol=0
    ):
        raise ValueError("aligned and shuffled do not share the same base14")
    if aligned["base14"].shape[0] != base10.shape[0]:
        raise ValueError("base10/base14 batch sizes differ")
    return {
        "base10": base10,
        "base_side": base_side,
        "feature_hw": feature_hw,
        "target_scale": target_scale,
        "source_target_scale": source_target_scale,
        "base_feature": base_feature,
        "scaler": scaler,
    }


def _production_reproduction_check(
    bundle: dict[str, Any], context: dict[str, Any], eps: float
) -> None:
    diagnostics = bundle["diagnostics"]
    for branch_name in ("aligned", "shuffled"):
        branch = diagnostics[branch_name]
        mask = make_hard_mask(branch["confidence14"])
        rebuilt = build_counterfactual_branch(
            branch["base14"],
            branch["smoothed14"],
            branch["confidence14"],
            context["base_side"],
            context["feature_hw"],
            mask,
            "per_image_zscore",
            context["source_target_scale"],
            context["scaler"],
            eps,
        )
        torch.testing.assert_close(
            rebuilt["target"],
            branch["target"].float(),
            rtol=2e-5,
            atol=2e-6,
            msg=f"dense production reconstruction failed for {branch_name}",
        )
        if isinstance(branch.get("delta14"), torch.Tensor):
            torch.testing.assert_close(
                rebuilt["delta14"], branch["delta14"].float()
            )


def _sample_identity(bundle: dict[str, Any], index: int) -> dict[str, int]:
    labels = bundle.get("labels")
    label = int(labels[index]) if isinstance(labels, torch.Tensor) else -1
    return {"flat_index": index, "place_label": label}


def _make_sample_records(
    bundle: dict[str, Any],
    context: dict[str, Any],
    spec: MaskSpec,
    transform: str,
    composition: str,
    eps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics = bundle["diagnostics"]
    base_target = apply_target_transform(
        context["base_feature"],
        transform,
        context["target_scale"],
        context["scaler"],
        eps,
    )
    built: dict[str, dict[str, torch.Tensor]] = {}
    masks: dict[str, torch.Tensor] = {}
    for branch_name in ("aligned", "shuffled"):
        branch = diagnostics[branch_name]
        if spec.disabled:
            mask = torch.zeros_like(branch["confidence14"], dtype=torch.bool)
        else:
            mask = make_hard_mask(
                branch["confidence14"].float(),
                spec.threshold,
                spec.top_fraction,
            )
        masks[branch_name] = mask
        built[branch_name] = build_counterfactual_branch(
            branch["base14"],
            branch["smoothed14"],
            branch["confidence14"],
            context["base_side"],
            context["feature_hw"],
            mask,
            transform,
            context["target_scale"],
            context["scaler"],
            eps,
            composition,
            context["base10"],
        )

    aligned = built["aligned"]
    shuffled = built["shuffled"]
    batch_size = aligned["target"].shape[0]
    values = {
        "aligned_coverage": _per_sample_fraction(masks["aligned"]),
        "shuffled_coverage": _per_sample_fraction(masks["shuffled"]),
        "aligned_zero_active_image": (
            masks["aligned"].sum(dim=1) == 0
        ).float(),
        "shuffled_zero_active_image": (
            masks["shuffled"].sum(dim=1) == 0
        ).float(),
        "aligned_effective_confidence_mean": _per_sample_fraction(
            aligned["effective_confidence"]
        ),
        "shuffled_effective_confidence_mean": _per_sample_fraction(
            shuffled["effective_confidence"]
        ),
        "aligned_delta_abs_mean": _per_sample_abs_mean(aligned["delta14"]),
        "shuffled_delta_abs_mean": _per_sample_abs_mean(shuffled["delta14"]),
        "aligned_pre_target_std": _per_sample_std(aligned["pre_target"]),
        "shuffled_pre_target_std": _per_sample_std(shuffled["pre_target"]),
        "aligned_target_std": _per_sample_std(aligned["target"]),
        "shuffled_target_std": _per_sample_std(shuffled["target"]),
        "aligned_target_change_from_ru": _per_sample_abs_mean(
            aligned["target"] - base_target
        ),
        "shuffled_target_change_from_ru": _per_sample_abs_mean(
            shuffled["target"] - base_target
        ),
        "aligned_target_correlation_with_ru": _spatial_correlation(
            aligned["target"], base_target
        ),
        "shuffled_target_correlation_with_ru": _spatial_correlation(
            shuffled["target"], base_target
        ),
        "aligned_saturation_fraction": _per_sample_fraction(
            aligned["target"].abs() > 0.95
        ),
        "shuffled_saturation_fraction": _per_sample_fraction(
            shuffled["target"].abs() > 0.95
        ),
        "pre_transform_pair_difference": _per_sample_abs_mean(
            aligned["pre_target"] - shuffled["pre_target"]
        ),
        "target_pair_difference": _per_sample_abs_mean(
            aligned["target"] - shuffled["target"]
        ),
        "target_pair_correlation": _spatial_correlation(
            aligned["target"], shuffled["target"]
        ),
        "target_pair_sign_disagreement": _per_sample_fraction(
            torch.sign(aligned["target"]) != torch.sign(shuffled["target"])
        ),
    }
    denominator = values["pre_transform_pair_difference"].clamp_min(eps)
    values["pair_difference_amplification"] = (
        values["target_pair_difference"] / denominator
    )

    records: list[dict[str, Any]] = []
    for index in range(batch_size):
        record: dict[str, Any] = {
            "mask": spec.name,
            "transform": transform,
            "composition": composition,
            **_sample_identity(bundle, index),
        }
        for name, tensor in values.items():
            record[name] = float(tensor[index])
        records.append(record)

    summary: dict[str, Any] = {
        "mask": spec.name,
        "transform": transform,
        "composition": composition,
        "threshold": spec.threshold,
        "top_fraction": spec.top_fraction,
        "semantic_disabled": spec.disabled,
        "sample_count": batch_size,
    }
    for name, tensor in values.items():
        summary[name] = _finite_mean(tensor)
    for branch_name in ("aligned", "shuffled"):
        coverage = values[f"{branch_name}_coverage"]
        summary[f"{branch_name}_coverage_min"] = float(coverage.min())
        summary[f"{branch_name}_coverage_max"] = float(coverage.max())
    return records, summary


def run_sweep(
    bundle: dict[str, Any],
    context: dict[str, Any],
    specs: list[MaskSpec],
    transforms: list[str],
    compositions: list[str],
    eps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for composition in compositions:
        for transform in transforms:
            for spec in specs:
                records, summary = _make_sample_records(
                    bundle, context, spec, transform, composition, eps
                )
                per_sample.extend(records)
                summaries.append(summary)

    dense_by_transform = {
        (row["composition"], row["transform"]): row
        for row in summaries
        if row["mask"] == "dense"
    }
    for row in summaries:
        dense = dense_by_transform[(row["composition"], row["transform"])]
        for branch in ("aligned", "shuffled"):
            numerator = row[f"{branch}_delta_abs_mean"]
            denominator = max(dense[f"{branch}_delta_abs_mean"], eps)
            row[f"{branch}_delta_retained_ratio"] = numerator / denominator
    return per_sample, summaries


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot save empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _render_summary_table(
    path: Path, summaries: list[dict[str, Any]]
) -> None:
    columns = [
        ("composition / mask / transform", None, 430),
        ("coverage A", "aligned_coverage", 150),
        ("delta kept A", "aligned_delta_retained_ratio", 160),
        ("target dRU A", "aligned_target_change_from_ru", 170),
        ("target dRU S", "shuffled_target_change_from_ru", 170),
        ("A-S target", "target_pair_difference", 160),
        ("saturation A", "aligned_saturation_fraction", 160),
    ]
    row_height = 34
    header_height = 48
    width = sum(item[2] for item in columns) + 24
    height = header_height + row_height * len(summaries) + 24
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_positions: list[int] = []
    x = 12
    for title, _, column_width in columns:
        x_positions.append(x)
        draw.text((x + 4, 12), title, fill="black")
        x += column_width
        draw.line((x, 0, x, height), fill=(220, 220, 220), width=1)
    draw.line((0, header_height, width, header_height), fill=(80, 80, 80), width=1)

    for row_index, row in enumerate(summaries):
        y = header_height + row_index * row_height
        if row_index % 2:
            draw.rectangle((0, y, width, y + row_height), fill=(247, 247, 247))
        label = (
            f"{row['composition']} / {row['mask']} / {row['transform']}"
        )
        draw.text((x_positions[0] + 4, y + 9), label, fill="black")
        for column_index, (_, field, column_width) in enumerate(columns[1:], 1):
            value = float(row[field])
            cell_x = x_positions[column_index]
            maximum = 1.0
            fraction = max(0.0, min(1.0, value / maximum))
            bar_width = int((column_width - 64) * fraction)
            draw.rectangle(
                (cell_x + 4, y + 9, cell_x + 4 + bar_width, y + 24),
                fill=(224, 126, 55),
            )
            draw.text((cell_x + column_width - 56, y + 9), f"{value:.3f}", fill="black")
    canvas.save(path, format="PNG")


def _image_from_normalized(tensor: torch.Tensor, size: int) -> Image.Image:
    image = tensor.float() * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    array = (
        image.clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)


def _overlay_mask(image: Image.Image, mask: torch.Tensor, size: int) -> Image.Image:
    side = _square_side(mask.numel(), "mask")
    mask_array = mask.view(side, side).byte().numpy() * 255
    mask_image = Image.fromarray(mask_array, mode="L").resize(
        (size, size), Image.Resampling.NEAREST
    )
    colour = Image.new("RGB", (size, size), (255, 40, 180))
    return Image.composite(
        Image.blend(image, colour, alpha=0.48), image, mask_image
    )


def _confidence_image(confidence: torch.Tensor, size: int) -> Image.Image:
    side = _square_side(confidence.numel(), "confidence")
    values = confidence.view(side, side).float().clamp(0.0, 1.0).numpy()
    rgb = np.stack(
        [values, 0.25 + 0.55 * values, 1.0 - 0.85 * values], axis=-1
    )
    array = (rgb.clip(0.0, 1.0) * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _add_caption(image: Image.Image, title: str) -> Image.Image:
    caption_height = 26
    cell = Image.new("RGB", (image.width, image.height + caption_height), "white")
    cell.paste(image, (0, caption_height))
    ImageDraw.Draw(cell).text((5, 7), title, fill="black")
    return cell


def _render_selected_mask_montage(
    path: Path, bundle: dict[str, Any], specs: list[MaskSpec]
) -> bool:
    selected = bundle.get("selected_indices")
    images = bundle.get("selected_images_normalized")
    if not isinstance(selected, torch.Tensor) or not isinstance(images, torch.Tensor):
        return False
    if selected.numel() != images.shape[0] or images.ndim != 4:
        return False
    aligned_confidence = bundle["diagnostics"]["aligned"]["confidence14"].float()
    non_dense = [spec for spec in specs if spec.name != "dense"]
    cell_size = 170
    gap = 5
    rows: list[list[Image.Image]] = []
    for rank, flat_index_tensor in enumerate(selected):
        flat_index = int(flat_index_tensor)
        original = _image_from_normalized(images[rank], cell_size)
        confidence = aligned_confidence[flat_index]
        cells = [
            _add_caption(original, f"flat {flat_index}"),
            _add_caption(_confidence_image(confidence, cell_size), "aligned confidence"),
        ]
        for spec in non_dense:
            if spec.disabled:
                mask = torch.zeros_like(confidence, dtype=torch.bool)
            else:
                mask = make_hard_mask(
                    confidence.unsqueeze(0), spec.threshold, spec.top_fraction
                )[0]
            cells.append(
                _add_caption(
                    _overlay_mask(original, mask, cell_size),
                    f"{spec.name} ({mask.float().mean():.3f})",
                )
            )
        rows.append(cells)
    if not rows:
        return False
    columns = len(rows[0])
    cell_width, cell_height = rows[0][0].size
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns - 1) * gap,
            len(rows) * cell_height + (len(rows) - 1) * gap,
        ),
        (225, 225, 225),
    )
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            canvas.paste(
                cell,
                (
                    column_index * (cell_width + gap),
                    row_index * (cell_height + gap),
                ),
            )
    canvas.save(path, format="PNG")
    return True


def _print_summary(summaries: list[dict[str, Any]]) -> None:
    print(
        "composition | mask | transform | covA | keptA | target-dRU A/S | "
        "target A-S | satA/S"
    )
    for row in summaries:
        print(
            f"{row['composition']} | {row['mask']} | "
            f"{row['transform']} | "
            f"{row['aligned_coverage']:.3f} | "
            f"{row['aligned_delta_retained_ratio']:.3f} | "
            f"{row['aligned_target_change_from_ru']:.4f}/"
            f"{row['shuffled_target_change_from_ru']:.4f} | "
            f"{row['target_pair_difference']:.4f} | "
            f"{row['aligned_saturation_fraction']:.3f}/"
            f"{row['shuffled_saturation_fraction']:.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.eps <= 0.0:
        raise ValueError("--eps must be positive")
    input_path = args.input.expanduser().resolve()
    output_dir = _prepare_output_directory(args.output_dir)
    specs = _mask_specs(args.thresholds, args.top_fractions)
    bundle = _load_bundle(input_path)
    context = _validate_and_describe_bundle(bundle, args.target_scale, args.eps)
    _production_reproduction_check(bundle, context, args.eps)

    per_sample, summaries = run_sweep(
        bundle,
        context,
        specs,
        list(args.transforms),
        list(args.compositions),
        args.eps,
    )
    _write_csv(output_dir / "counterfactual_summary.csv", summaries)
    _write_csv(output_dir / "counterfactual_per_sample.csv", per_sample)
    _render_summary_table(output_dir / "counterfactual_summary.png", summaries)
    montage_saved = _render_selected_mask_montage(
        output_dir / "aligned_mask_montage.png", bundle, specs
    )

    payload = {
        "schema_version": 1,
        "input": str(input_path),
        "production_reproduction_check": "passed",
        "batch_size": int(context["base10"].shape[0]),
        "base_grid_side": context["base_side"],
        "target_hw": list(context["feature_hw"]),
        "target_scale": context["target_scale"],
        "source_target_scale": context["source_target_scale"],
        "shared_base_scaler": {
            "mean": context["scaler"].mean,
            "std": context["scaler"].std,
        },
        "mask_definition": (
            "Retained patches keep their original confidence; rejected patches "
            "receive confidence zero. Top fractions retain exactly ceil(N*f) "
            "patches independently per image."
        ),
        "transforms": {
            name: TRANSFORM_DESCRIPTIONS[name] for name in args.transforms
        },
        "compositions": {
            name: COMPOSITION_DESCRIPTIONS[name] for name in args.compositions
        },
        "summaries": summaries,
        "aligned_mask_montage_saved": montage_saved,
        "limitations": [
            "This is a fixed-feature teacher-target counterfactual, not a retrieval evaluation.",
            "It cannot determine which target improves R@1 without matched retraining.",
            "One fixed batch is descriptive; repeat across batches/cities before retraining.",
        ],
    }
    with (output_dir / "counterfactual_run.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    print("Dense production target reproduction: passed")
    _print_summary(summaries)
    print(f"Saved counterfactual sweep to: {output_dir}")


if __name__ == "__main__":
    main()
