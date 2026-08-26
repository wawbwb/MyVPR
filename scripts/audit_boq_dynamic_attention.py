"""Audit baseline BoQ cross-attention overlap with dynamic-category masks.

The audited baseline is the repeatability+uniqueness checkpoint used by
``eval_dynamic_category_prior.py`` with ``attention_bias=None``.  Its trained
RU gate remains active.  The script performs one frozen forward pass over the
standard database plus the complete condition-query union, extracts per-head
BoQ cross-attention without changing descriptors, and measures overlap with:

* aligned: the image's own dynamic mask;
* shuffled: a role/condition-preserving wrong-image mask;
* random: an exact spatial permutation of the image's own mask values.

Attention maps are routing probabilities, not causal pixel attribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.boq_attention_audit import (  # noqa: E402
    compute_attention_components,
    compute_mask_overlap,
    fc_energy_slot_weights,
    force_per_head_cross_attention,
)
from scripts.dynamic_category_prior import (  # noqa: E402
    file_sha256,
    load_and_validate_mask_cache,
    role_preserving_derangement,
    spatially_permute_masks,
)
from scripts.eval_condition_robustness import (  # noqa: E402
    build_transform,
    choose_device,
    load_inference_model_from_ckpt,
)
from scripts.eval_dynamic_category_prior import (  # noqa: E402
    build_evaluation_sets,
    build_query_condition_strata,
    extract_ru_feature_map,
    validate_ru_checkpoint_configuration,
)
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset  # noqa: E402
from src.dataloaders.valid.msls_condition import (  # noqa: E402
    MSLSConditionUnionDataset,
)
from src.dataloaders.valid.msls_condition_protocol import (  # noqa: E402
    CONDITION_ORDER,
)
from src.models.aggregators.boq import BoQ  # noqa: E402


MASK_VARIANTS = ("aligned", "shuffled", "random")
DEFAULT_TOP_FRACTIONS = (0.1, 0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure frozen RU-BoQ baseline attention on dynamic masks"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--msls-path", type=Path, default=Path("datasets/msls-val"))
    parser.add_argument("--mask-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="*",
        choices=CONDITION_ORDER,
        default=CONDITION_ORDER,
        help="condition groups included in reports and balanced visual samples",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(280, 280))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--top-fractions",
        type=float,
        nargs="+",
        default=DEFAULT_TOP_FRACTIONS,
        help="attention concentration fractions (default: 0.1 0.2)",
    )
    parser.add_argument(
        "--num-random-visualizations",
        type=int,
        default=12,
        help="number of deterministic role/condition-balanced image rows",
    )
    parser.add_argument(
        "--num-extreme-visualizations",
        type=int,
        default=6,
        help="rows for each aligned-minus-random high/low montage",
    )
    parser.add_argument(
        "--num-head-detail-images",
        type=int,
        default=3,
        help="balanced samples receiving a layer-by-head detail image",
    )
    parser.add_argument(
        "--min-dynamic-area-for-extremes",
        type=float,
        default=0.01,
        help="exclude almost-empty masks from enrichment-based image selection",
    )
    parser.add_argument(
        "--visual-size",
        type=int,
        default=280,
        help="square panel size for audit montages",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    requested = set(args.conditions)
    args.conditions = tuple(name for name in CONDITION_ORDER if name in requested)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.msls_path = args.msls_path.expanduser().resolve()
    args.mask_cache = args.mask_cache.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.msls_path.is_dir():
        raise FileNotFoundError(f"MSLS path not found: {args.msls_path}")
    if not args.mask_cache.is_file():
        raise FileNotFoundError(f"mask cache not found: {args.mask_cache}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid batch-size or num-workers")
    if len(args.image_size) != 2 or min(args.image_size) <= 0:
        raise ValueError("image-size must contain two positive integers")
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in args.top_fractions):
        raise ValueError("top-fractions must lie in (0,1]")
    args.top_fractions = tuple(sorted(set(float(value) for value in args.top_fractions)))
    for name in (
        "num_random_visualizations",
        "num_extreme_visualizations",
        "num_head_detail_images",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if not 0 <= args.min_dynamic_area_for_extremes <= 1:
        raise ValueError("min-dynamic-area-for-extremes must lie in [0,1]")
    if args.visual_size < 64:
        raise ValueError("visual-size must be at least 64 pixels")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _friendly_condition_name(dataset_name: str) -> str:
    prefix = "msls-val-"
    suffix = "-full-db"
    if dataset_name.startswith(prefix) and dataset_name.endswith(suffix):
        return dataset_name[len(prefix) : -len(suffix)]
    return dataset_name


def build_report_groups(
    dataset: MSLSConditionUnionDataset,
    evaluations: Sequence[Mapping[str, Any]],
) -> tuple[OrderedDict[str, np.ndarray], list[str], list[str]]:
    """Build global image-index groups and per-image role/condition labels."""

    image_count = len(dataset)
    num_references = dataset.num_references
    groups: OrderedDict[str, np.ndarray] = OrderedDict()
    groups["all"] = np.arange(image_count, dtype=np.int64)
    groups["references"] = np.arange(num_references, dtype=np.int64)
    groups["union_queries"] = num_references + np.arange(
        dataset.num_queries, dtype=np.int64
    )

    standard = evaluations[0]
    groups["standard_queries"] = num_references + np.asarray(
        standard["query_offsets"], dtype=np.int64
    )
    query_conditions: list[list[str]] = [[] for _ in range(dataset.num_queries)]
    for evaluation in evaluations[1:]:
        name = _friendly_condition_name(str(evaluation["name"]))
        offsets = np.asarray(evaluation["query_offsets"], dtype=np.int64)
        groups[name] = num_references + offsets
        for offset in offsets.tolist():
            query_conditions[offset].append(name)

    roles = []
    condition_labels = []
    for image_index in range(image_count):
        if image_index < num_references:
            roles.append("reference")
            condition_labels.append("")
            continue
        query_offset = image_index - num_references
        roles.append(
            "standard_query"
            if query_offset < dataset.num_standard_queries
            else "condition_only_query"
        )
        condition_labels.append("+".join(query_conditions[query_offset]))
    return groups, roles, condition_labels


def _initialise_group_accumulators(
    groups: Mapping[str, np.ndarray],
    *,
    component_count: int,
    num_layers: int,
    num_heads: int,
    num_queries: int,
    num_tokens: int,
    focus_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group_name in groups:
        result[group_name] = {
            "count": 0,
            "component_map_sum": np.zeros(
                (component_count, num_tokens), dtype=np.float64
            ),
            "head_map_sum": np.zeros(
                (num_layers, num_heads, num_tokens), dtype=np.float64
            ),
            "query_map_sum": np.zeros(
                (num_layers, num_queries, num_tokens), dtype=np.float64
            ),
            "focus_sum": {
                name: np.zeros(component_count, dtype=np.float64)
                for name in focus_names
            },
            "variants": {
                variant: {
                    "area_sum": 0.0,
                    "component_mass_sum": np.zeros(
                        component_count, dtype=np.float64
                    ),
                    "head_mass_sum": np.zeros(
                        (num_layers, num_heads), dtype=np.float64
                    ),
                    "query_mass_sum": np.zeros(
                        (num_layers, num_queries), dtype=np.float64
                    ),
                }
                for variant in MASK_VARIANTS
            },
        }
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _finite_quantile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, quantile))


def build_summary_rows(
    accumulators: Mapping[str, Mapping[str, Any]],
    *,
    component_names: Sequence[str],
    per_image_rows: Sequence[Mapping[str, Any]],
    groups: Mapping[str, np.ndarray],
    min_macro_area: float,
) -> list[dict[str, Any]]:
    rows = []
    for group_name, accumulator in accumulators.items():
        count = int(accumulator["count"])
        if count != len(groups[group_name]):
            raise AssertionError(
                f"group {group_name} accumulated {count} images, expected "
                f"{len(groups[group_name])}"
            )
        group_indices = groups[group_name].tolist()
        for component_index, component_name in enumerate(component_names):
            row: dict[str, Any] = {
                "group": group_name,
                "component": component_name,
                "image_count": count,
            }
            for focus_name, values in accumulator["focus_sum"].items():
                row[f"{focus_name}_mean"] = float(values[component_index] / count)
            for variant in MASK_VARIANTS:
                variant_acc = accumulator["variants"][variant]
                area_sum = float(variant_acc["area_sum"])
                mass_sum = float(variant_acc["component_mass_sum"][component_index])
                row[f"{variant}_area_mean"] = area_sum / count
                row[f"{variant}_attention_mass_mean"] = mass_sum / count
                row[f"{variant}_micro_enrichment"] = _safe_ratio(
                    mass_sum, area_sum
                )
                macro_values = [
                    float(per_image_rows[index][f"{component_name}_{variant}_enrichment"])
                    for index in group_indices
                    if float(per_image_rows[index][f"{variant}_area_fraction"])
                    >= min_macro_area
                ]
                row[f"{variant}_macro_eligible_count"] = len(macro_values)
                for label, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                    row[f"{variant}_enrichment_{label}"] = _finite_quantile(
                        macro_values, quantile
                    )
            for comparator in ("random", "shuffled"):
                aligned = accumulator["variants"]["aligned"]
                other = accumulator["variants"][comparator]
                row[f"aligned_minus_{comparator}_mass_mean"] = float(
                    (
                        aligned["component_mass_sum"][component_index]
                        - other["component_mass_sum"][component_index]
                    )
                    / count
                )
                row[f"aligned_minus_{comparator}_micro_enrichment"] = float(
                    row["aligned_micro_enrichment"]
                    - row[f"{comparator}_micro_enrichment"]
                )
            rows.append(row)
    return rows


def build_unit_summary_rows(
    accumulators: Mapping[str, Mapping[str, Any]],
    *,
    unit: str,
    num_layers: int,
    units_per_layer: int,
) -> list[dict[str, Any]]:
    if unit not in {"head", "query"}:
        raise ValueError("unit must be head or query")
    mass_key = f"{unit}_mass_sum"
    rows = []
    for group_name, accumulator in accumulators.items():
        count = int(accumulator["count"])
        for layer_index in range(num_layers):
            for unit_index in range(units_per_layer):
                row: dict[str, Any] = {
                    "group": group_name,
                    "layer": layer_index + 1,
                    unit: unit_index,
                    "image_count": count,
                }
                for variant in MASK_VARIANTS:
                    variant_acc = accumulator["variants"][variant]
                    area_sum = float(variant_acc["area_sum"])
                    mass_sum = float(
                        variant_acc[mass_key][layer_index, unit_index]
                    )
                    row[f"{variant}_area_mean"] = area_sum / count
                    row[f"{variant}_attention_mass_mean"] = mass_sum / count
                    row[f"{variant}_micro_enrichment"] = _safe_ratio(
                        mass_sum, area_sum
                    )
                for comparator in ("random", "shuffled"):
                    row[f"aligned_minus_{comparator}_mass_mean"] = float(
                        (
                            accumulator["variants"]["aligned"][mass_key][
                                layer_index, unit_index
                            ]
                            - accumulator["variants"][comparator][mass_key][
                                layer_index, unit_index
                            ]
                        )
                        / count
                    )
                    row[f"aligned_minus_{comparator}_micro_enrichment"] = float(
                        row["aligned_micro_enrichment"]
                        - row[f"{comparator}_micro_enrichment"]
                    )
                rows.append(row)
    return rows


def balanced_random_indices(
    groups: Mapping[str, np.ndarray],
    *,
    total: int,
    seed: int,
) -> list[int]:
    if total <= 0:
        return []
    preferred = [
        name
        for name in (
            "references",
            "standard_queries",
            *CONDITION_ORDER,
        )
        if name in groups and len(groups[name]) > 0
    ]
    rng = np.random.default_rng(int(seed))
    shuffled = {
        name: rng.permutation(groups[name]).astype(np.int64).tolist()
        for name in preferred
    }
    positions = {name: 0 for name in preferred}
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < total:
        progressed = False
        for name in preferred:
            values = shuffled[name]
            while positions[name] < len(values):
                candidate = int(values[positions[name]])
                positions[name] += 1
                if candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                progressed = True
                break
            if len(selected) >= total:
                break
        if not progressed:
            break
    return selected


def select_extreme_indices(
    per_image_rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    min_area: float,
) -> tuple[list[int], list[int]]:
    eligible = []
    key = "consensus_raw_aligned_minus_random_mass"
    for row in per_image_rows:
        if float(row["aligned_area_fraction"]) < min_area:
            continue
        eligible.append((float(row[key]), int(row["image_index"])))
    eligible.sort()
    low = [index for _, index in eligible[:count]]
    high = [index for _, index in reversed(eligible[-count:])]
    return low, high


_COLOUR_ANCHORS = np.asarray(
    [
        [32, 36, 96],
        [29, 117, 188],
        [38, 190, 162],
        [246, 215, 70],
        [231, 76, 60],
    ],
    dtype=np.float32,
)


def _colourise(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    scaled = clipped * (len(_COLOUR_ANCHORS) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(_COLOUR_ANCHORS) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = _COLOUR_ANCHORS[lower] * (1.0 - fraction) + _COLOUR_ANCHORS[upper] * fraction
    return np.uint8(np.clip(rgb, 0, 255))


def _resize_float_grid(grid: np.ndarray, size: int, *, nearest: bool) -> np.ndarray:
    image = Image.fromarray(np.asarray(grid, dtype=np.float32), mode="F")
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = image.resize((size, size), resample=resampling)
    return np.asarray(image, dtype=np.float32)


def _caption_panel(image: Image.Image, caption: str, width: int) -> Image.Image:
    caption_height = 42
    panel = Image.new("RGB", (width, image.height + caption_height), "white")
    panel.paste(image, (0, 0))
    draw = ImageDraw.Draw(panel)
    first, second = (caption.split("\n", 1) + [""])[:2]
    draw.text((5, image.height + 4), first[:90], fill="black")
    draw.text((5, image.height + 20), second[:90], fill="black")
    return panel


def _dynamic_overlay(original: Image.Image, mask: np.ndarray, size: int) -> Image.Image:
    coverage = np.clip(_resize_float_grid(mask, size, nearest=False), 0.0, 1.0)
    source = np.asarray(original, dtype=np.float32)
    red = np.zeros_like(source)
    red[..., 0] = 255.0
    alpha = (0.72 * coverage)[..., None]
    return Image.fromarray(np.uint8(source * (1.0 - alpha) + red * alpha), mode="RGB")


def _attention_overlay(
    original: Image.Image,
    attention: np.ndarray,
    *,
    size: int,
    density_scale: float,
) -> Image.Image:
    density = attention * attention.size
    resized = _resize_float_grid(density, size, nearest=False)
    colours = _colourise(resized / density_scale).astype(np.float32)
    source = np.asarray(original, dtype=np.float32)
    alpha = 0.56
    return Image.fromarray(
        np.uint8(source * (1.0 - alpha) + colours * alpha), mode="RGB"
    )


def _load_raw_image(dataset: MSLSConditionUnionDataset, index: int, size: int) -> Image.Image:
    path = dataset.dataset_path / str(dataset.image_paths[index])
    with Image.open(path) as image:
        return image.convert("RGB").resize(
            (size, size), resample=Image.Resampling.BICUBIC
        )


def save_sample_montage(
    output: Path,
    *,
    indices: Sequence[int],
    dataset: MSLSConditionUnionDataset,
    masks: np.ndarray,
    layer_maps: np.ndarray,
    fc_maps: np.ndarray,
    per_image_rows: Sequence[Mapping[str, Any]],
    density_scale: float,
    visual_size: int,
) -> None:
    if not indices:
        return
    rows = []
    num_layers = layer_maps.shape[1]
    for index in indices:
        original = _load_raw_image(dataset, int(index), visual_size)
        row = per_image_rows[int(index)]
        panels = [
            _caption_panel(
                original,
                f"idx={index} {row['role']}\n{str(row['image_path'])[-72:]}",
                visual_size,
            ),
            _caption_panel(
                _dynamic_overlay(original, masks[index], visual_size),
                "dynamic mask (red)\n"
                f"area={float(row['aligned_area_fraction']):.3f}",
                visual_size,
            ),
        ]
        for layer_index in range(num_layers):
            name = f"layer_{layer_index + 1}"
            panels.append(
                _caption_panel(
                    _attention_overlay(
                        original,
                        np.asarray(layer_maps[index, layer_index]),
                        size=visual_size,
                        density_scale=density_scale,
                    ),
                    f"{name} raw attention\n"
                    f"mass={float(row[f'{name}_aligned_attention_mass']):.3f} "
                    f"E={float(row[f'{name}_aligned_enrichment']):.2f}",
                    visual_size,
                )
            )
        consensus = np.asarray(layer_maps[index], dtype=np.float32).mean(axis=0)
        panels.append(
            _caption_panel(
                _attention_overlay(
                    original,
                    consensus,
                    size=visual_size,
                    density_scale=density_scale,
                ),
                "consensus raw attention\n"
                f"A-R mass={float(row['consensus_raw_aligned_minus_random_mass']):+.4f}",
                visual_size,
            )
        )
        panels.append(
            _caption_panel(
                _attention_overlay(
                    original,
                    np.asarray(fc_maps[index]),
                    size=visual_size,
                    density_scale=density_scale,
                ),
                "FC-energy proxy (secondary)\nnot causal attribution",
                visual_size,
            )
        )
        canvas_row = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), panels[0].height),
            "white",
        )
        x = 0
        for panel in panels:
            canvas_row.paste(panel, (x, 0))
            x += panel.width
        rows.append(canvas_row)
    canvas = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height for row in rows)),
        "white",
    )
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(output, quality=92)


def save_mean_position_montage(
    output: Path,
    *,
    accumulators: Mapping[str, Mapping[str, Any]],
    component_names: Sequence[str],
    grid_size: tuple[int, int],
    density_scale: float,
    visual_size: int,
) -> None:
    rows = []
    for group_name, accumulator in accumulators.items():
        mean_maps = accumulator["component_map_sum"] / accumulator["count"]
        panels = []
        for component_index, component_name in enumerate(component_names):
            grid = mean_maps[component_index].reshape(grid_size)
            density = _resize_float_grid(grid * grid.size, visual_size, nearest=False)
            image = Image.fromarray(
                _colourise(density / density_scale), mode="RGB"
            )
            panels.append(
                _caption_panel(
                    image,
                    f"{group_name}: {component_name}\n"
                    f"density scale 0..{density_scale:.2f}; uniform=1",
                    visual_size,
                )
            )
        row = Image.new(
            "RGB", (visual_size * len(panels), panels[0].height), "white"
        )
        for column, panel in enumerate(panels):
            row.paste(panel, (column * visual_size, 0))
        rows.append(row)
    canvas = Image.new(
        "RGB",
        (rows[0].width, sum(row.height for row in rows)),
        "white",
    )
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(output, quality=92)


def save_head_detail(
    output: Path,
    *,
    original: Image.Image,
    head_maps: np.ndarray,
    density_scale: float,
    visual_size: int,
    image_index: int,
    image_path: str,
) -> None:
    rows = []
    num_layers, num_heads = head_maps.shape[:2]
    for layer_index in range(num_layers):
        panels = []
        for head_index in range(num_heads):
            overlay = _attention_overlay(
                original,
                head_maps[layer_index, head_index],
                size=visual_size,
                density_scale=density_scale,
            )
            panels.append(
                _caption_panel(
                    overlay,
                    f"layer={layer_index + 1} head={head_index}\n"
                    f"idx={image_index}",
                    visual_size,
                )
            )
        row = Image.new(
            "RGB", (visual_size * num_heads, panels[0].height), "white"
        )
        for column, panel in enumerate(panels):
            row.paste(panel, (column * visual_size, 0))
        rows.append(row)
    title_height = 28
    canvas = Image.new(
        "RGB", (rows[0].width, title_height + sum(row.height for row in rows)), "white"
    )
    ImageDraw.Draw(canvas).text(
        (5, 6), f"idx={image_index} {image_path}", fill="black"
    )
    y = title_height
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(output, quality=92)


def _output_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    grid_size = (args.image_size[0] // 14, args.image_size[1] // 14)
    if tuple(value * 14 for value in grid_size) != tuple(args.image_size):
        raise ValueError("DINOv2 image dimensions must be divisible by patch size 14")

    checkpoint_config = validate_ru_checkpoint_configuration(
        args.checkpoint, tuple(args.image_size)
    )
    transform = build_transform(tuple(args.image_size))
    standard_dataset = MapillarySLSDataset(
        dataset_path=str(args.msls_path), input_transform=transform
    )
    dataset = MSLSConditionUnionDataset(
        dataset_path=str(args.msls_path), input_transform=transform
    )
    evaluations, condition_memberships = build_evaluation_sets(
        standard_dataset,
        dataset,
        args.conditions,
        transform,
        args.msls_path,
    )
    masks, cache_metadata = load_and_validate_mask_cache(
        args.mask_cache,
        expected_image_paths=dataset.image_paths,
        expected_num_references=dataset.num_references,
        expected_grid_size=grid_size,
    )
    query_strata, query_stratum_counts = build_query_condition_strata(
        condition_memberships, dataset.num_queries
    )
    donor_indices = role_preserving_derangement(
        dataset.num_references,
        dataset.num_queries,
        args.seed,
        query_strata=query_strata,
    )
    groups, roles, condition_labels = build_report_groups(dataset, evaluations)
    group_memberships = {
        name: np.isin(np.arange(len(dataset), dtype=np.int64), indices)
        for name, indices in groups.items()
    }

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mask cache: {args.mask_cache}")
    print(
        "Baseline path: DINO -> RU semantic_region_gate -> "
        "BoQ(attention_bias=None)"
    )
    print(f"Images: {len(dataset)}; grid: {grid_size}")

    model = load_inference_model_from_ckpt(args.checkpoint, device)
    if not isinstance(model.aggregator, BoQ):
        raise TypeError("checkpoint aggregator is not the repository BoQ class")
    if model.semantic_region_gate is None:
        raise RuntimeError("RU semantic_region_gate was not restored")
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda" and not args.no_amp

    args.output.mkdir(parents=True, exist_ok=False)
    per_image_rows: list[dict[str, Any] | None] = [None] * len(dataset)
    layer_map_store: np.memmap | None = None
    fc_map_store: np.memmap | None = None
    accumulators: dict[str, dict[str, Any]] | None = None
    component_names: list[str] | None = None
    slot_weights: torch.Tensor | None = None
    dimensions: dict[str, int] | None = None
    validation_record: dict[str, Any] | None = None
    max_attention_sum_error = 0.0

    with torch.inference_mode():
        for images, sample_indices in tqdm(loader, desc="Audit baseline BoQ attention"):
            images = images.to(device, non_blocking=True)
            indices = torch.as_tensor(sample_indices, dtype=torch.long)
            numpy_indices = indices.numpy()
            aligned = torch.from_numpy(masks[numpy_indices]).to(
                device=device, dtype=torch.float32
            )
            shuffled = torch.from_numpy(masks[donor_indices[numpy_indices]]).to(
                device=device, dtype=torch.float32
            )
            random = spatially_permute_masks(aligned, indices, args.seed)
            mask_stack = torch.stack((aligned, shuffled, random), dim=1)

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                feature_map = extract_ru_feature_map(model, images)
                if tuple(feature_map.shape[-2:]) != grid_size:
                    raise ValueError(
                        f"DINO grid {tuple(feature_map.shape[-2:])} != mask grid "
                        f"{grid_size}"
                    )
                legacy_descriptor = None
                legacy_attention = None
                if validation_record is None:
                    legacy_descriptor, legacy_attention = model.aggregator(
                        feature_map, attention_bias=None
                    )
                with force_per_head_cross_attention(model.aggregator):
                    descriptor, attentions = model.aggregator(
                        feature_map, attention_bias=None
                    )

            if slot_weights is None:
                if not attentions or attentions[0].ndim != 4:
                    raise ValueError("per-head BoQ extraction did not return B,H,Q,N")
                num_layers = len(attentions)
                num_queries = int(attentions[0].shape[2])
                slot_weights = fc_energy_slot_weights(
                    model.aggregator.fc.weight,
                    num_layers=num_layers,
                    num_queries=num_queries,
                ).to(device)

            diagnostics = compute_attention_components(
                attentions,
                grid_size=grid_size,
                fc_slot_weights=slot_weights,
                top_fractions=args.top_fractions,
            )
            batch_dimensions = diagnostics["dimensions"]
            static_dimensions = {
                name: int(batch_dimensions[name])
                for name in ("num_layers", "num_heads", "num_queries", "num_tokens")
            }
            if dimensions is None:
                dimensions = static_dimensions
            elif dimensions != static_dimensions:
                raise ValueError(
                    "BoQ attention dimensions changed between batches: "
                    f"{dimensions} vs {static_dimensions}"
                )
            component_names = diagnostics["component_names"]
            component_maps = diagnostics["component_maps"]
            head_maps = diagnostics["head_maps"]
            query_maps = diagnostics["query_maps"]

            if validation_record is None:
                assert legacy_descriptor is not None and legacy_attention is not None
                descriptor_delta = float(
                    (descriptor.float() - legacy_descriptor.float()).abs().max()
                )
                attention_delta = max(
                    float(
                        (
                            per_head.float().mean(dim=1)
                            - legacy.float()
                        )
                        .abs()
                        .max()
                    )
                    for per_head, legacy in zip(attentions, legacy_attention)
                )
                if not torch.equal(descriptor, legacy_descriptor):
                    torch.testing.assert_close(
                        descriptor,
                        legacy_descriptor,
                        rtol=1e-6,
                        atol=1e-7,
                    )
                if attention_delta > 2e-3:
                    raise AssertionError(
                        "per-head mean does not reproduce legacy BoQ attention: "
                        f"max delta {attention_delta}"
                    )
                validation_record = {
                    "descriptor_exact_equal": bool(
                        torch.equal(descriptor, legacy_descriptor)
                    ),
                    "descriptor_max_abs_delta": descriptor_delta,
                    "per_head_mean_vs_legacy_attention_max_abs_delta": attention_delta,
                }

            attention_sums = diagnostics["stack"].sum(dim=-1)
            max_attention_sum_error = max(
                max_attention_sum_error,
                float((attention_sums - 1.0).abs().max()),
            )
            overlap = compute_mask_overlap(component_maps, mask_stack)
            head_overlap = compute_mask_overlap(
                head_maps.flatten(1, 2), mask_stack
            )
            query_overlap = compute_mask_overlap(
                query_maps.flatten(1, 2), mask_stack
            )

            component_np = component_maps.detach().cpu().numpy()
            head_np = head_maps.detach().cpu().numpy()
            query_np = query_maps.detach().cpu().numpy()
            focus_np = {
                name: values.detach().cpu().numpy()
                for name, values in diagnostics["focus"].items()
            }
            area_np = overlap["area"].detach().cpu().numpy()
            mass_np = overlap["mass"].detach().cpu().numpy()
            enrichment_np = overlap["enrichment"].detach().cpu().numpy()
            support_area_np = overlap["support_area"].detach().cpu().numpy()
            support_mass_np = overlap["support_mass"].detach().cpu().numpy()
            support_enrichment_np = (
                overlap["support_enrichment"].detach().cpu().numpy()
            )
            head_mass_np = head_overlap["mass"].detach().cpu().numpy().reshape(
                len(numpy_indices),
                len(MASK_VARIANTS),
                dimensions["num_layers"],
                dimensions["num_heads"],
            )
            query_mass_np = query_overlap["mass"].detach().cpu().numpy().reshape(
                len(numpy_indices),
                len(MASK_VARIANTS),
                dimensions["num_layers"],
                dimensions["num_queries"],
            )

            if layer_map_store is None:
                layer_map_store = np.lib.format.open_memmap(
                    args.output / "attention_layer_maps.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(
                        len(dataset),
                        dimensions["num_layers"],
                        *grid_size,
                    ),
                )
                fc_map_store = np.lib.format.open_memmap(
                    args.output / "attention_fc_energy_proxy_maps.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(dataset), *grid_size),
                )
                accumulators = _initialise_group_accumulators(
                    groups,
                    component_count=len(component_names),
                    num_layers=dimensions["num_layers"],
                    num_heads=dimensions["num_heads"],
                    num_queries=dimensions["num_queries"],
                    num_tokens=dimensions["num_tokens"],
                    focus_names=list(focus_np),
                )
            assert layer_map_store is not None
            assert fc_map_store is not None
            assert accumulators is not None
            layer_map_store[numpy_indices] = diagnostics["layer_maps"].reshape(
                len(numpy_indices), dimensions["num_layers"], *grid_size
            ).detach().cpu().numpy().astype(np.float16)
            fc_map_store[numpy_indices] = diagnostics["fc_proxy_map"].reshape(
                len(numpy_indices), *grid_size
            ).detach().cpu().numpy().astype(np.float16)

            for local_index, global_index in enumerate(numpy_indices.tolist()):
                row: dict[str, Any] = {
                    "image_index": global_index,
                    "role": roles[global_index],
                    "condition_membership": condition_labels[global_index],
                    "image_path": str(dataset.image_paths[global_index]).replace(
                        "\\", "/"
                    ),
                }
                for variant_index, variant in enumerate(MASK_VARIANTS):
                    row[f"{variant}_area_fraction"] = float(
                        area_np[local_index, variant_index]
                    )
                    row[f"{variant}_support_patch_fraction"] = float(
                        support_area_np[local_index, variant_index]
                    )
                    for component_index, component_name in enumerate(component_names):
                        row[f"{component_name}_{variant}_attention_mass"] = float(
                            mass_np[local_index, variant_index, component_index]
                        )
                        row[f"{component_name}_{variant}_enrichment"] = float(
                            enrichment_np[
                                local_index, variant_index, component_index
                            ]
                        )
                        row[
                            f"{component_name}_{variant}_support_attention_mass"
                        ] = float(
                            support_mass_np[
                                local_index, variant_index, component_index
                            ]
                        )
                        row[
                            f"{component_name}_{variant}_support_enrichment"
                        ] = float(
                            support_enrichment_np[
                                local_index, variant_index, component_index
                            ]
                        )
                for component_index, component_name in enumerate(component_names):
                    for focus_name, values in focus_np.items():
                        row[f"{component_name}_{focus_name}"] = float(
                            values[local_index, component_index]
                        )
                    for comparator in ("random", "shuffled"):
                        row[f"{component_name}_aligned_minus_{comparator}_mass"] = float(
                            mass_np[local_index, 0, component_index]
                            - mass_np[
                                local_index,
                                MASK_VARIANTS.index(comparator),
                                component_index,
                            ]
                        )
                per_image_rows[global_index] = row

            for group_name, membership in group_memberships.items():
                selected = membership[numpy_indices]
                if not bool(selected.any()):
                    continue
                accumulator = accumulators[group_name]
                selected_count = int(selected.sum())
                accumulator["count"] += selected_count
                accumulator["component_map_sum"] += component_np[selected].sum(axis=0)
                accumulator["head_map_sum"] += head_np[selected].sum(axis=0)
                accumulator["query_map_sum"] += query_np[selected].sum(axis=0)
                for focus_name, values in focus_np.items():
                    accumulator["focus_sum"][focus_name] += values[selected].sum(axis=0)
                for variant_index, variant in enumerate(MASK_VARIANTS):
                    variant_acc = accumulator["variants"][variant]
                    variant_acc["area_sum"] += float(
                        area_np[selected, variant_index].sum()
                    )
                    variant_acc["component_mass_sum"] += mass_np[
                        selected, variant_index
                    ].sum(axis=0)
                    variant_acc["head_mass_sum"] += head_mass_np[
                        selected, variant_index
                    ].sum(axis=0)
                    variant_acc["query_mass_sum"] += query_mass_np[
                        selected, variant_index
                    ].sum(axis=0)

    if (
        layer_map_store is None
        or fc_map_store is None
        or accumulators is None
        or component_names is None
        or dimensions is None
        or slot_weights is None
        or validation_record is None
        or any(row is None for row in per_image_rows)
    ):
        raise RuntimeError("attention audit produced no complete output")
    layer_map_store.flush()
    fc_map_store.flush()
    complete_rows = [row for row in per_image_rows if row is not None]

    summary_rows = build_summary_rows(
        accumulators,
        component_names=component_names,
        per_image_rows=complete_rows,
        groups=groups,
        min_macro_area=args.min_dynamic_area_for_extremes,
    )
    head_rows = build_unit_summary_rows(
        accumulators,
        unit="head",
        num_layers=dimensions["num_layers"],
        units_per_layer=dimensions["num_heads"],
    )
    query_rows = build_unit_summary_rows(
        accumulators,
        unit="query",
        num_layers=dimensions["num_layers"],
        units_per_layer=dimensions["num_queries"],
    )
    slot_weight_rows = [
        {
            "layer": layer_index + 1,
            "query": query_index,
            "fc_energy_weight": float(slot_weights[layer_index, query_index].cpu()),
        }
        for layer_index in range(dimensions["num_layers"])
        for query_index in range(dimensions["num_queries"])
    ]

    write_csv(args.output / "summary.csv", summary_rows)
    write_csv(args.output / "per_image.csv", complete_rows)
    write_csv(args.output / "head_summary.csv", head_rows)
    write_csv(args.output / "query_slot_summary.csv", query_rows)
    write_csv(args.output / "fc_energy_slot_weights.csv", slot_weight_rows)

    mean_component_maps = {}
    mean_head_maps = {}
    mean_query_maps = {}
    for group_name, accumulator in accumulators.items():
        safe_name = group_name.replace("-", "_")
        count = accumulator["count"]
        mean_component_maps[safe_name] = (
            accumulator["component_map_sum"] / count
        ).reshape(len(component_names), *grid_size).astype(np.float32)
        mean_head_maps[safe_name] = (
            accumulator["head_map_sum"] / count
        ).reshape(
            dimensions["num_layers"],
            dimensions["num_heads"],
            *grid_size,
        ).astype(np.float32)
        mean_query_maps[safe_name] = (
            accumulator["query_map_sum"] / count
        ).reshape(
            dimensions["num_layers"],
            dimensions["num_queries"],
            *grid_size,
        ).astype(np.float32)
    np.savez_compressed(
        args.output / "mean_component_attention_maps.npz", **mean_component_maps
    )
    np.savez_compressed(
        args.output / "mean_head_attention_maps.npz", **mean_head_maps
    )
    np.savez_compressed(
        args.output / "mean_query_slot_attention_maps.npz", **mean_query_maps
    )

    layer_density = np.asarray(layer_map_store, dtype=np.float32) * dimensions[
        "num_tokens"
    ]
    fc_density = np.asarray(fc_map_store, dtype=np.float32) * dimensions["num_tokens"]
    density_scale = float(
        max(
            1.0,
            np.quantile(layer_density, 0.99),
            np.quantile(fc_density, 0.99),
        )
    )

    random_indices = balanced_random_indices(
        groups, total=args.num_random_visualizations, seed=args.seed
    )
    low_indices, high_indices = select_extreme_indices(
        complete_rows,
        count=args.num_extreme_visualizations,
        min_area=args.min_dynamic_area_for_extremes,
    )
    visual_groups = {
        "balanced_random": random_indices,
        "high_aligned_minus_random": high_indices,
        "low_aligned_minus_random": low_indices,
    }
    visual_rows = []
    for reason, indices in visual_groups.items():
        montage_path = args.output / f"attention_{reason}.jpg"
        save_sample_montage(
            montage_path,
            indices=indices,
            dataset=dataset,
            masks=masks,
            layer_maps=layer_map_store,
            fc_maps=fc_map_store,
            per_image_rows=complete_rows,
            density_scale=density_scale,
            visual_size=args.visual_size,
        )
        for rank, index in enumerate(indices):
            visual_rows.append(
                {
                    "reason": reason,
                    "rank": rank,
                    "image_index": index,
                    "image_path": complete_rows[index]["image_path"],
                    "aligned_area_fraction": complete_rows[index][
                        "aligned_area_fraction"
                    ],
                    "consensus_raw_aligned_minus_random_mass": complete_rows[index][
                        "consensus_raw_aligned_minus_random_mass"
                    ],
                    "montage": montage_path.name,
                }
            )
    if visual_rows:
        write_csv(args.output / "visualization_selection.csv", visual_rows)
    save_mean_position_montage(
        args.output / "mean_position_attention.jpg",
        accumulators=accumulators,
        component_names=component_names,
        grid_size=grid_size,
        density_scale=density_scale,
        visual_size=min(args.visual_size, 220),
    )

    head_detail_dir = args.output / "head_details"
    detail_indices = random_indices[: args.num_head_detail_images]
    if detail_indices:
        head_detail_dir.mkdir(parents=False, exist_ok=False)
        detail_images = torch.stack([dataset[index][0] for index in detail_indices]).to(
            device
        )
        with torch.inference_mode(), torch.amp.autocast(
            device_type=amp_device, enabled=use_amp
        ):
            detail_features = extract_ru_feature_map(model, detail_images)
            with force_per_head_cross_attention(model.aggregator):
                _, detail_attentions = model.aggregator(
                    detail_features, attention_bias=None
                )
        detail_stack = torch.stack(
            [attention.detach().float() for attention in detail_attentions], dim=1
        )
        detail_head_maps = detail_stack.mean(dim=3).reshape(
            len(detail_indices),
            dimensions["num_layers"],
            dimensions["num_heads"],
            *grid_size,
        ).cpu().numpy()
        for row_index, image_index in enumerate(detail_indices):
            original = _load_raw_image(dataset, image_index, args.visual_size)
            save_head_detail(
                head_detail_dir / f"index_{image_index:05d}.jpg",
                original=original,
                head_maps=detail_head_maps[row_index],
                density_scale=density_scale,
                visual_size=args.visual_size,
                image_index=image_index,
                image_path=str(dataset.image_paths[image_index]).replace("\\", "/"),
            )

    output_paths = [
        args.output / "summary.csv",
        args.output / "per_image.csv",
        args.output / "head_summary.csv",
        args.output / "query_slot_summary.csv",
        args.output / "fc_energy_slot_weights.csv",
        args.output / "attention_layer_maps.npy",
        args.output / "attention_fc_energy_proxy_maps.npy",
        args.output / "mean_component_attention_maps.npz",
        args.output / "mean_head_attention_maps.npz",
        args.output / "mean_query_slot_attention_maps.npz",
        args.output / "mean_position_attention.jpg",
        *(
            args.output / f"attention_{name}.jpg"
            for name, indices in visual_groups.items()
            if indices
        ),
    ]
    if visual_rows:
        output_paths.append(args.output / "visualization_selection.csv")
    run_record = {
        "schema_version": 1,
        "method": "baseline_ru_boq_dynamic_attention_audit",
        "baseline_definition": (
            "repeatability+uniqueness checkpoint with its trained "
            "semantic_region_gate active and BoQ attention_bias=None"
        ),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "size_bytes": args.checkpoint.stat().st_size,
            "validated_config": checkpoint_config,
        },
        "mask_cache": {
            "path": str(args.mask_cache),
            "sha256": file_sha256(args.mask_cache),
            "size_bytes": args.mask_cache.stat().st_size,
            **cache_metadata,
        },
        "dataset": {
            "path": str(args.msls_path),
            "num_images": len(dataset),
            "num_references": dataset.num_references,
            "num_standard_queries": dataset.num_standard_queries,
            "num_union_queries": dataset.num_queries,
            "groups": {name: len(indices) for name, indices in groups.items()},
            "query_condition_stratum_counts": query_stratum_counts,
        },
        "attention": {
            **dimensions,
            "grid_size": list(grid_size),
            "returned_weights": (
                "post-softmax BoQ cross-attention probabilities, per head; "
                "each learned query sums to one across spatial keys"
            ),
            "primary_layer_map": "equal mean over heads and learned queries",
            "primary_consensus_map": "equal mean over BoQ layers",
            "density_visualisation": (
                "num_tokens * attention; uniform routing equals 1; one shared "
                "p99-derived scale, no per-image min-max"
            ),
            "density_scale_max": density_scale,
            "max_spatial_sum_error": max_attention_sum_error,
            "top_fractions": list(args.top_fractions),
            "fc_energy_proxy": (
                "query-slot weights proportional to squared final-FC column "
                "norm; secondary heuristic, not causal contribution"
            ),
        },
        "mask_overlap": {
            "primary_mass": "sum_n attention[n] * dynamic_patch_fraction[n]",
            "uniform_expected_mass": "mean_n dynamic_patch_fraction[n]",
            "micro_enrichment": "sum_images mass / sum_images area",
            "aligned": "the image's own cached mask",
            "shuffled": (
                "seeded no-fixed-point donor preserving reference/query role "
                "and exact query condition-membership stratum"
            ),
            "random": "same image mask values spatially permuted exactly",
            "donor_indices_sha256": hashlib.sha256(donor_indices.tobytes()).hexdigest(),
        },
        "per_head_extraction_validation": validation_record,
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "image_size": list(args.image_size),
            "amp": use_amp,
            "seed": args.seed,
            "torch_version": torch.__version__,
        },
        "visualisations": {
            "balanced_random_indices": random_indices,
            "high_aligned_minus_random_indices": high_indices,
            "low_aligned_minus_random_indices": low_indices,
            "head_detail_indices": detail_indices,
            "selection_min_dynamic_area": args.min_dynamic_area_for_extremes,
        },
        "files": [_output_file_record(path) for path in output_paths],
        "limitations": [
            (
                "BoQ cross-attention describes learned-query routing, not a "
                "causal pixel saliency or complete descriptor contribution."
            ),
            (
                "The baseline contains the trained RU semantic_region_gate; "
                "it is not the photometric-only visual checkpoint."
            ),
            (
                "The second BoQ layer attends to spatially indexed tokens that "
                "have already undergone encoder mixing."
            ),
            (
                "The dynamic mask is a Pascal-VOC hard-argmax class prior and "
                "does not label roads, sky, vegetation, or buildings."
            ),
        ],
    }
    with (args.output / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    consensus_rows = [
        row
        for row in summary_rows
        if row["component"] == "consensus_raw"
    ]
    print("\nBaseline BoQ dynamic-attention audit")
    print("=" * 112)
    print(
        f"{'Group':<22} {'N':>7} {'area':>9} {'aligned mass':>13} "
        f"{'E_align':>9} {'A-R mass':>11} {'A-S mass':>11}"
    )
    for row in consensus_rows:
        print(
            f"{row['group']:<22} {int(row['image_count']):>7d} "
            f"{float(row['aligned_area_mean']):>9.4f} "
            f"{float(row['aligned_attention_mass_mean']):>13.4f} "
            f"{float(row['aligned_micro_enrichment']):>9.3f} "
            f"{float(row['aligned_minus_random_mass_mean']):>+11.5f} "
            f"{float(row['aligned_minus_shuffled_mass_mean']):>+11.5f}"
        )
    print("-" * 112)
    print("E_align < 1 means relative avoidance; > 1 means enrichment.")
    print("A-R controls per-image mask values; A-S controls the population position prior.")
    print("Content-specific alignment requires aligned to beat both controls.")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
