"""Sweep the CLIP-only contribution of one trained Residual-CLIP adapter.

For a selected CLIP control ``v``, the trained affine residual is decomposed
before the frozen RU gate and BoQ aggregator::

    R_v   = W(P(C_v) - norm(D))
    R_0   = W(P(0)   - norm(D))
    R_sem = R_v - R_0
    Z     = D + R_0 + gamma * R_sem

Thus gamma=0 is the existing zero-CLIP path, gamma=1 is the recorded model,
and gamma>1 amplifies only the CLIP-dependent component.  The DINO anchor and
the residual adapter bias remain unscaled.  Every non-zero gamma is evaluated
with aligned, global-only, wrong-region, and fixed DB/query-role-preserving
wrong-image CLIP tokens from the same raw DINO extraction.  The sweep is
explicitly exploratory: selecting gamma on MSLS-val cannot itself establish a
validated gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_residual_clip_paired import (  # noqa: E402
    DescriptorStore,
    PairedImageDataset,
    _canonical_paths,
    _sha256_file,
    aggregate_feature_map,
    search_and_score,
    validate_aligned_checkpoint,
    verify_first_batch_equivalence,
    verify_frozen_base,
    write_csv,
)
from scripts.dynamic_category_prior import (  # noqa: E402
    role_preserving_derangement,
)
from scripts.eval_condition_robustness import (  # noqa: E402
    build_transform,
    choose_device,
    load_inference_model_from_ckpt,
)
from src.dataloaders.valid.mapillary_sls import (  # noqa: E402
    MapillarySLSDataset,
)
from src.models.aggregators.boq import BoQ  # noqa: E402


SEMANTIC_MODES = (
    "aligned",
    "global_only",
    "wrong_region",
    "wrong_image_role_derangement",
)
COMPARATOR_MODES = (
    "bypass",
    "zero_clip",
    "global_only",
    "wrong_region",
    "wrong_image_role_derangement",
)


@dataclass(frozen=True)
class SweepVariant:
    name: str
    mode: str
    semantic_gamma: float | None


def gamma_label(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def build_variant_specs(scales: list[float] | tuple[float, ...]) -> tuple[SweepVariant, ...]:
    canonical = tuple(sorted(set(float(value) for value in scales)))
    if not canonical or canonical[0] != 0.0 or 1.0 not in canonical:
        raise ValueError("semantic scales must include exact endpoints 0 and 1")
    specs = [
        SweepVariant("bypass", "bypass", None),
        SweepVariant("zero_clip_g0", "zero_clip", 0.0),
    ]
    for gamma in canonical:
        if gamma == 0.0:
            continue
        label = gamma_label(gamma)
        specs.extend(
            SweepVariant(f"{mode}_g{label}", mode, gamma)
            for mode in SEMANTIC_MODES
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise AssertionError("semantic-scale variant names are not unique")
    return tuple(specs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the CLIP-only gain of a trained Residual-CLIP adapter"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ru-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--msls-path", type=Path, default=Path("datasets/msls-val")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--keep-descriptors", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(280, 280))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--semantic-gammas",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0, 2.0, 4.0),
    )
    parser.add_argument(
        "--descriptor-dtype",
        choices=("float32", "float16"),
        default="float32",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=(1, 5, 10, 15))
    parser.add_argument("--rank-k", type=int, default=100)
    parser.add_argument("--minimum-net-queries", type=int, default=8)
    parser.add_argument("--expected-bypass-r1", type=float, default=91.22)
    parser.add_argument("--baseline-tolerance-pp", type=float, default=0.15)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-5)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[float, ...]:
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
    if any(int(k) <= 0 for k in args.k_values):
        raise ValueError("k-values must be positive")
    args.k_values = tuple(sorted(set(int(k) for k in args.k_values)))
    if 1 not in args.k_values:
        raise ValueError("k-values must include 1")
    if 5 not in args.k_values:
        raise ValueError("k-values must include 5 for the preregistered tie-break")
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

    scales: list[float] = []
    for raw in args.semantic_gammas:
        if isinstance(raw, bool) or not math.isfinite(float(raw)):
            raise ValueError("semantic gammas must be finite real numbers")
        value = float(raw)
        if value < 0.0 or value > 8.0:
            raise ValueError("semantic gammas must lie in [0,8]")
        scales.append(value)
    canonical = tuple(sorted(set(scales)))
    if 0.0 not in canonical or 1.0 not in canonical:
        raise ValueError("semantic gammas must include exact endpoints 0 and 1")
    return canonical


def _fusion_inputs(
    mode: str,
    *,
    clip_global: torch.Tensor,
    clip_patches: torch.Tensor,
    donor_global: torch.Tensor,
    donor_patches: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if mode == "wrong_image_role_derangement":
        return donor_patches, donor_global, "aligned"
    if mode not in ("aligned", "global_only", "wrong_region"):
        raise ValueError(f"unsupported semantic sweep mode: {mode}")
    return clip_patches, clip_global, mode


def _fuse_cached_semantic_gain(
    dino_features: torch.Tensor,
    base_residual: torch.Tensor,
    variant_residual: torch.Tensor,
    semantic_gamma: float,
) -> torch.Tensor:
    """Apply a cached ``R0 + gamma * (Rv - R0)`` decomposition."""

    if isinstance(semantic_gamma, bool) or not isinstance(
        semantic_gamma, Real
    ):
        raise TypeError("semantic_gamma must be a finite real scalar")
    semantic_gamma = float(semantic_gamma)
    if not math.isfinite(semantic_gamma) or not 0.0 <= semantic_gamma <= 8.0:
        raise ValueError("semantic_gamma must be finite and in [0,8]")
    if dino_features.ndim != 4:
        raise ValueError("DINO features must have shape (B,C,H,W)")
    if not dino_features.is_floating_point():
        raise TypeError("DINO features must be floating point")
    if base_residual.shape != variant_residual.shape:
        raise ValueError("base and variant residual shapes do not match")
    if base_residual.device != variant_residual.device:
        raise ValueError("base and variant residual devices do not match")
    if base_residual.dtype != variant_residual.dtype:
        raise ValueError("base and variant residual dtypes do not match")
    if not base_residual.is_floating_point():
        raise TypeError("cached residuals must be floating point")
    batch_size, channels, height, width = dino_features.shape
    dino_tokens = dino_features.permute(0, 2, 3, 1).reshape(
        batch_size, height * width, channels
    )
    if base_residual.shape != dino_tokens.shape:
        raise ValueError("cached residual shape does not match DINO tokens")
    if base_residual.device != dino_tokens.device:
        raise ValueError("cached residual and DINO devices do not match")
    if semantic_gamma == 0.0:
        scaled_residual = base_residual
    elif semantic_gamma == 1.0:
        scaled_residual = variant_residual
    else:
        semantic_fp32 = variant_residual.float() - base_residual.float()
        scaled_residual = (
            base_residual.float()
            + semantic_fp32.new_tensor(semantic_gamma) * semantic_fp32
        ).to(dtype=variant_residual.dtype)
    if not bool(torch.isfinite(scaled_residual).all()):
        raise FloatingPointError("scaled semantic residual is non-finite")
    return (
        (dino_tokens + scaled_residual)
        .reshape(batch_size, height, width, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def sweep_variant_descriptors(
    model: torch.nn.Module,
    images: torch.Tensor,
    donor_images: torch.Tensor,
    specs: tuple[SweepVariant, ...],
    *,
    check_endpoints: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    backbone = model.backbone
    fusion = getattr(backbone, "residual_clip_fusion", None)
    provider = getattr(backbone, "_residual_clip_provider", None)
    if fusion is None or provider is None:
        raise RuntimeError("model has no usable Residual-CLIP branch/provider")
    raw, _x_cls, clip_global, clip_patches = (
        backbone.extract_residual_clip_components(images)
    )
    donor_global, donor_patches = provider.encode(donor_images)

    if fusion.bypassed:
        raise RuntimeError("semantic-gamma sweep cannot run inside fusion bypass")
    zero_feature, base_residual = fusion(
        raw,
        torch.zeros_like(clip_patches),
        torch.zeros_like(clip_global),
        intervention_mode="aligned",
    )
    mode_inputs: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
    mode_variant_features: dict[str, torch.Tensor] = {}
    mode_variant_residuals: dict[str, torch.Tensor] = {}
    for mode in SEMANTIC_MODES:
        patches, global_features, intervention = _fusion_inputs(
            mode,
            clip_global=clip_global,
            clip_patches=clip_patches,
            donor_global=donor_global,
            donor_patches=donor_patches,
        )
        mode_inputs[mode] = (patches, global_features, intervention)
        feature_map, variant_residual = fusion(
            raw,
            patches,
            global_features,
            intervention_mode=intervention,
        )
        mode_variant_features[mode] = feature_map
        mode_variant_residuals[mode] = variant_residual

    descriptors: dict[str, torch.Tensor] = {
        "bypass": aggregate_feature_map(model, raw),
        "zero_clip_g0": aggregate_feature_map(model, zero_feature),
    }
    endpoint_diagnostics: dict[str, Any] = {}
    cached_direct_max_error = 0.0
    for spec in specs:
        if spec.mode in ("bypass", "zero_clip"):
            continue
        assert spec.semantic_gamma is not None
        if spec.semantic_gamma == 1.0:
            feature_map = mode_variant_features[spec.mode]
        else:
            feature_map = _fuse_cached_semantic_gain(
                raw,
                base_residual,
                mode_variant_residuals[spec.mode],
                spec.semantic_gamma,
            )
        if check_endpoints:
            patches, global_features, intervention = mode_inputs[spec.mode]
            direct_feature, _ = fusion(
                raw,
                patches,
                global_features,
                intervention_mode=intervention,
                semantic_gamma=spec.semantic_gamma,
            )
            direct_error = float(
                (feature_map.detach().float() - direct_feature.detach().float())
                .abs()
                .amax()
                .item()
            )
            endpoint_diagnostics[
                f"{spec.name}_cached_direct_max_abs_error"
            ] = direct_error
            cached_direct_max_error = max(cached_direct_max_error, direct_error)
        descriptors[spec.name] = aggregate_feature_map(model, feature_map)
    if set(descriptors) != {spec.name for spec in specs}:
        raise RuntimeError("semantic sweep did not produce every descriptor variant")

    if check_endpoints:
        endpoint_diagnostics["component_contract"] = {
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "base_residual_shape": list(base_residual.shape),
            "base_residual_dtype": str(base_residual.dtype),
            "device": str(raw.device),
        }
        gamma_zero_feature, _ = fusion(
            raw,
            clip_patches,
            clip_global,
            intervention_mode="aligned",
            semantic_gamma=0.0,
        )
        endpoint_diagnostics["gamma0_zero_clip_max_abs_error"] = float(
            (gamma_zero_feature.detach().float() - zero_feature.detach().float())
            .abs()
            .amax()
            .item()
        )
        endpoint_diagnostics["cached_direct_max_abs_error"] = (
            cached_direct_max_error
        )
        zero_descriptor = descriptors["zero_clip_g0"].detach().float()
        descriptor_drift: dict[str, dict[str, float]] = {}
        for name, descriptor in descriptors.items():
            candidate = descriptor.detach().float()
            delta = candidate - zero_descriptor
            descriptor_drift[name] = {
                "cosine_distance_mean": float(
                    (
                        1.0
                        - torch.nn.functional.cosine_similarity(
                            candidate, zero_descriptor, dim=1
                        )
                    )
                    .mean()
                    .item()
                ),
                "l2_mean": float(delta.norm(dim=1).mean().item()),
                "max_abs": float(delta.abs().amax().item()),
            }
        endpoint_diagnostics["first_batch_descriptor_drift_vs_zero_clip"] = (
            descriptor_drift
        )
        endpoint_diagnostics["first_batch_semantic_residual_by_mode"] = {
            mode: {
                "rms": float(
                    (
                        mode_variant_residuals[mode].detach().float()
                        - base_residual.detach().float()
                    )
                    .square()
                    .mean()
                    .sqrt()
                    .item()
                ),
                "max_abs": float(
                    (
                        mode_variant_residuals[mode].detach().float()
                        - base_residual.detach().float()
                    )
                    .abs()
                    .amax()
                    .item()
                ),
            }
            for mode in SEMANTIC_MODES
        }
    return descriptors, endpoint_diagnostics


def extract_descriptors(
    *,
    model: torch.nn.Module,
    paired_dataset: PairedImageDataset,
    donor_indices: np.ndarray,
    specs: tuple[SweepVariant, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    equivalence_tolerance: float,
    directory: Path,
    dtype: str,
) -> tuple[DescriptorStore, dict[str, Any]]:
    loader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    names = tuple(spec.name for spec in specs)
    store = DescriptorStore(directory, len(paired_dataset), names, dtype)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    endpoint_diagnostics: dict[str, Any] | None = None
    try:
        with torch.inference_mode():
            for images, donor_images, indices, returned_donors in tqdm(
                loader,
                desc=f"Extract {len(specs)} semantic-gamma variants",
            ):
                indices_np = np.asarray(indices, dtype=np.int64)
                donor_np = np.asarray(returned_donors, dtype=np.int64)
                if not np.array_equal(donor_np, donor_indices[indices_np]):
                    raise RuntimeError("DataLoader donor indices do not match map")
                images = images.to(device, non_blocking=True)
                donor_images = donor_images.to(device, non_blocking=True)
                first_batch = endpoint_diagnostics is None
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    descriptors, diagnostics = sweep_variant_descriptors(
                        model,
                        images,
                        donor_images,
                        specs,
                        check_endpoints=first_batch,
                    )
                    if first_batch:
                        aligned_g1 = next(
                            spec.name
                            for spec in specs
                            if spec.mode == "aligned"
                            and spec.semantic_gamma == 1.0
                        )
                        diagnostics["gamma1_ordinary_max_abs_error"] = (
                            verify_first_batch_equivalence(
                                model,
                                images,
                                descriptors[aligned_g1],
                                equivalence_tolerance,
                            )
                        )
                        if (
                            diagnostics["gamma0_zero_clip_max_abs_error"]
                            > equivalence_tolerance
                        ):
                            raise RuntimeError(
                                "gamma=0 does not reproduce the ordinary "
                                "zero-CLIP path"
                            )
                        if (
                            diagnostics["cached_direct_max_abs_error"]
                            > equivalence_tolerance
                        ):
                            raise RuntimeError(
                                "cached semantic-gamma composition does not "
                                "match the direct fusion API"
                            )
                        endpoint_diagnostics = diagnostics
                store.mark_indices(indices_np)
                for name, descriptor in descriptors.items():
                    store.write(
                        name,
                        indices_np,
                        descriptor.detach().float().cpu().numpy(),
                    )
        store.finish()
        if endpoint_diagnostics is None:
            raise RuntimeError("semantic-gamma extraction produced no batches")
        return store, endpoint_diagnostics
    except Exception:
        store.close()
        raise


def paired_counts(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    left_hits = left["hits"][1]
    right_hits = right["hits"][1]
    left_rank = left["first_positive_rank"]
    right_rank = right["first_positive_rank"]
    left_only = int(np.sum(left_hits & ~right_hits))
    right_only = int(np.sum(~left_hits & right_hits))
    return {
        "left_only_correct_r@1": left_only,
        "right_only_correct_r@1": right_only,
        "net_queries_r@1": left_only - right_only,
        "both_correct_r@1": int(np.sum(left_hits & right_hits)),
        "both_wrong_r@1": int(np.sum(~left_hits & ~right_hits)),
        "top1_reference_changed": int(np.sum(left["top1"] != right["top1"])),
        "left_better_first_positive_rank": int(np.sum(left_rank < right_rank)),
        "right_better_first_positive_rank": int(np.sum(right_rank < left_rank)),
        "rank_tied": int(np.sum(left_rank == right_rank)),
        "mean_capped_rank_advantage": float(
            np.mean(right_rank.astype(float) - left_rank.astype(float))
        ),
        "mean_margin_advantage": float(
            np.mean(
                left["positive_negative_margin"]
                - right["positive_negative_margin"]
            )
        ),
    }


def evaluate(
    store: DescriptorStore,
    dataset: MapillarySLSDataset,
    specs: tuple[SweepVariant, ...],
    k_values: tuple[int, ...],
    rank_k: int,
    minimum_net_queries: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    results = {
        spec.name: search_and_score(
            store.paths[spec.name],
            num_references=dataset.num_references,
            ground_truth=dataset.ground_truth,
            k_values=k_values,
            rank_k=rank_k,
        )
        for spec in specs
    }
    by_mode_gamma = {
        (spec.mode, spec.semantic_gamma): spec.name for spec in specs
    }
    bypass = results[by_mode_gamma[("bypass", None)]]
    zero_clip = results[by_mode_gamma[("zero_clip", 0.0)]]

    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        result = results[spec.name]
        row: dict[str, Any] = {
            **asdict(spec),
            "num_queries": dataset.num_queries,
        }
        for k in k_values:
            recall = float(result["recalls"][int(k)])
            bypass_recall = float(bypass["recalls"][int(k)])
            zero_recall = float(zero_clip["recalls"][int(k)])
            row[f"r@{k}"] = recall
            row[f"correct@{k}"] = int(result["hits"][int(k)].sum())
            row[f"delta_vs_bypass_r@{k}_pp"] = 100.0 * (
                recall - bypass_recall
            )
            row[f"delta_vs_bypass_r@{k}_queries"] = int(
                result["hits"][int(k)].sum() - bypass["hits"][int(k)].sum()
            )
            row[f"delta_vs_zero_clip_r@{k}_pp"] = 100.0 * (
                recall - zero_recall
            )
        summary_rows.append(row)

    paired_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    positive_scales = sorted(
        {
            float(spec.semantic_gamma)
            for spec in specs
            if spec.semantic_gamma is not None and spec.semantic_gamma > 0
        }
    )
    for gamma in positive_scales:
        aligned_name = by_mode_gamma[("aligned", gamma)]
        aligned = results[aligned_name]
        comparison_names = {
            "bypass": by_mode_gamma[("bypass", None)],
            "zero_clip": by_mode_gamma[("zero_clip", 0.0)],
            "global_only": by_mode_gamma[("global_only", gamma)],
            "wrong_region": by_mode_gamma[("wrong_region", gamma)],
            "wrong_image_role_derangement": by_mode_gamma[
                ("wrong_image_role_derangement", gamma)
            ],
        }
        net_queries: dict[str, int] = {}
        for comparator_mode, comparator_name in comparison_names.items():
            counts = paired_counts(aligned, results[comparator_name])
            net_queries[comparator_mode] = int(counts["net_queries_r@1"])
            paired_rows.append(
                {
                    "semantic_gamma": gamma,
                    "left": aligned_name,
                    "right": comparator_name,
                    "right_mode": comparator_mode,
                    **counts,
                    "passes_minimum_net_queries": int(
                        counts["net_queries_r@1"] >= minimum_net_queries
                    ),
                }
            )
        minimum_net = min(net_queries.values())
        selection_rows.append(
            {
                "semantic_gamma": gamma,
                "aligned_variant": aligned_name,
                "aligned_r@1": float(aligned["recalls"][1]),
                "aligned_correct@1": int(aligned["hits"][1].sum()),
                "aligned_r@5": float(aligned["recalls"].get(5, float("nan"))),
                **{
                    f"net_vs_{mode}_r@1_queries": value
                    for mode, value in net_queries.items()
                },
                "minimum_net_queries": minimum_net,
                "eligible": int(minimum_net >= minimum_net_queries),
            }
        )

    query_paths = _canonical_paths(dataset.qImages)
    reference_paths = _canonical_paths(dataset.dbImages)
    query_rows: list[dict[str, Any]] = []
    for spec in specs:
        result = results[spec.name]
        for query_index, query_path in enumerate(query_paths.tolist()):
            prediction = int(result["top1"][query_index])
            row: dict[str, Any] = {
                "query_index": query_index,
                "query_path": query_path,
                **asdict(spec),
                "top1_reference_index": prediction,
                "top1_reference_path": reference_paths[prediction],
                "top1_squared_l2": float(result["top1_distance"][query_index]),
                "first_positive_rank_capped": int(
                    result["first_positive_rank"][query_index]
                ),
                "positive_found_within_rank_k": int(
                    result["positive_found_within_rank_k"][query_index]
                ),
                "positive_negative_margin": float(
                    result["positive_negative_margin"][query_index]
                ),
            }
            for k in k_values:
                row[f"hit@{k}"] = int(result["hits"][int(k)][query_index])
            query_rows.append(row)
    return summary_rows, paired_rows, selection_rows, query_rows, results


def select_scale(
    selection_rows: list[dict[str, Any]],
) -> tuple[float | None, float]:
    if not selection_rows:
        raise ValueError("selection rows are empty")
    ranked = sorted(
        selection_rows,
        key=lambda row: (
            -int(row["minimum_net_queries"]),
            -float(row["aligned_r@1"]),
            -float(row["aligned_r@5"]),
            float(row["semantic_gamma"]),
        ),
    )
    best_observed = float(ranked[0]["semantic_gamma"])
    eligible = [row for row in ranked if int(row["eligible"])]
    selected = float(eligible[0]["semantic_gamma"]) if eligible else None
    return selected, best_observed


def print_summary(
    summary_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    verdict: dict[str, Any],
) -> None:
    print("\nCLIP-only semantic gamma sweep")
    print("=" * 108)
    print(
        f"{'gamma':>7} {'mode':<25} {'R@1':>8} {'vs bypass q':>12} "
        f"{'R@5':>8}"
    )
    for row in summary_rows:
        gamma = row["semantic_gamma"]
        gamma_text = "-" if gamma is None else f"{float(gamma):g}"
        print(
            f"{gamma_text:>7} {row['mode']:<25} "
            f"{100.0 * float(row['r@1']):>7.2f}% "
            f"{int(row['delta_vs_bypass_r@1_queries']):>+12d} "
            f"{100.0 * float(row.get('r@5', float('nan'))):>7.2f}%"
        )
    print("-" * 108)
    for row in selection_rows:
        print(
            f"gamma={float(row['semantic_gamma']):g}: "
            f"min aligned net={int(row['minimum_net_queries']):+d} queries; "
            f"eligible={bool(row['eligible'])}"
        )
    print(f"Exploratory status: {str(verdict['status']).upper()}")
    print(f"Candidate gamma: {verdict['selected_gamma']}")


def main() -> None:
    args = parse_args()
    scales = validate_args(args)
    specs = build_variant_specs(scales)
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
    donor_indices = role_preserving_derangement(
        dataset.num_references,
        dataset.num_queries,
        args.seed,
    )
    paired_dataset = PairedImageDataset(dataset, donor_indices)

    print(f"Device: {device}")
    print(f"Aligned checkpoint: {args.checkpoint}")
    print(f"Frozen RU checkpoint: {args.ru_checkpoint}")
    print(
        f"MSLS: {dataset.num_references} references, "
        f"{dataset.num_queries} queries"
    )
    print(f"Semantic gammas: {scales}")
    print(f"Descriptor variants: {len(specs)}")
    model = load_inference_model_from_ckpt(args.checkpoint, device)
    if not isinstance(model.aggregator, BoQ):
        raise TypeError("checkpoint aggregator is not repository BoQ")
    if model.semantic_region_gate is None:
        raise RuntimeError("checkpoint loader did not restore the RU gate")
    model.backbone.prepare_residual_clip_provider(device)

    descriptor_dim = int(
        model.aggregator.proj_c.out_channels
        * model.aggregator.fc.out_features
    )
    descriptor_bytes = (
        len(specs)
        * len(dataset)
        * descriptor_dim
        * np.dtype(args.descriptor_dtype).itemsize
    )
    if args.keep_descriptors:
        capacity_root = args.output.parent
    elif args.scratch_dir is not None:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
        capacity_root = args.scratch_dir
    else:
        capacity_root = Path(tempfile.gettempdir())
    capacity_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(capacity_root).free
    print(
        f"Descriptor scratch estimate: {descriptor_bytes / 2**30:.2f} GiB; "
        f"required with margin: {descriptor_bytes * 1.10 / 2**30:.2f} GiB; "
        f"free: {free_bytes / 2**30:.2f} GiB"
    )
    if free_bytes < int(descriptor_bytes * 1.10):
        raise OSError("insufficient descriptor scratch space before extraction")

    args.output.mkdir(parents=True, exist_ok=False)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    store: DescriptorStore | None = None
    try:
        if args.keep_descriptors:
            descriptor_directory = args.output / "descriptors"
        else:
            if args.scratch_dir is not None:
                args.scratch_dir.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(
                prefix="residual-clip-semantic-gamma-",
                dir=(str(args.scratch_dir) if args.scratch_dir is not None else None),
            )
            descriptor_directory = Path(temporary.name)

        store, endpoint_diagnostics = extract_descriptors(
            model=model,
            paired_dataset=paired_dataset,
            donor_indices=donor_indices,
            specs=specs,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
            equivalence_tolerance=args.equivalence_tolerance,
            directory=descriptor_directory,
            dtype=args.descriptor_dtype,
        )
        (
            summary_rows,
            paired_rows,
            selection_rows,
            query_rows,
            results,
        ) = evaluate(
            store,
            dataset,
            specs,
            args.k_values,
            args.rank_k,
            args.minimum_net_queries,
        )
        selected_gamma, best_observed_gamma = select_scale(selection_rows)
        bypass_error_pp = abs(
            100.0 * float(results["bypass"]["recalls"][1])
            - args.expected_bypass_r1
        )
        bypass_reproduced = bypass_error_pp <= args.baseline_tolerance_pp
        candidate_found = bypass_reproduced and selected_gamma is not None
        precision_eligible = args.descriptor_dtype == "float32"
        candidate_gamma = (
            selected_gamma
            if candidate_found and precision_eligible
            else None
        )
        verdict = {
            "status": (
                "candidate_found"
                if candidate_found and precision_eligible
                else "no_candidate"
            ),
            "exploratory_only": True,
            "candidate_found": candidate_found and precision_eligible,
            "precision_eligible": precision_eligible,
            "bypass_reproduced": bypass_reproduced,
            "bypass_error_pp": bypass_error_pp,
            "minimum_net_queries_required": args.minimum_net_queries,
            "selected_gamma": candidate_gamma,
            "best_eligible_gamma_before_baseline_and_precision_checks": (
                selected_gamma
            ),
            "best_observed_gamma_even_if_ineligible": best_observed_gamma,
            "decision_rule": (
                "For one explored gamma, aligned must net at least "
                "minimum_net_queries R@1 queries over bypass, zero_clip, "
                "global_only, wrong_region, and wrong_image_role_derangement "
                "at the same gamma; ties in candidates prefer larger minimum "
                "net, then R@1, R@5, then smaller gamma. Float16 descriptor "
                "storage cannot nominate a candidate."
            ),
            "scope": (
                "exploratory validation-set gain sweep; candidate_found is "
                "not a success verdict and requires matched retraining plus "
                "independent-seed or held-out confirmation"
            ),
        }

        write_csv(args.output / "gamma_retrieval_summary.csv", summary_rows)
        write_csv(args.output / "gamma_paired_comparisons.csv", paired_rows)
        write_csv(args.output / "gamma_selection.csv", selection_rows)
        write_csv(args.output / "query_outcomes.csv", query_rows)
        run_record = {
            "schema_version": 1,
            "method": "residual_clip_semantic_only_gain_sweep",
            "formula": (
                "Z = D_raw + R_zero + gamma * (R_variant - R_zero), "
                "R_zero=W(P(0)-norm(D_raw))"
            ),
            "semantic_gammas": list(scales),
            "variants": [asdict(spec) for spec in specs],
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
            "donor_mapping": {
                "type": "db_query_role_preserving_one_to_one_derangement",
                "seed": args.seed,
                "sha256": hashlib.sha256(donor_indices.tobytes()).hexdigest(),
                "unique_donors": int(len(np.unique(donor_indices))),
                "maximum_multiplicity": int(
                    np.bincount(donor_indices).max()
                ),
                "self_pairs": int(
                    np.sum(donor_indices == np.arange(len(donor_indices)))
                ),
            },
            "endpoint_diagnostics": endpoint_diagnostics,
            "runtime": {
                "device": str(device),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "image_size": list(args.image_size),
                "amp": use_amp,
                "descriptor_dtype": args.descriptor_dtype,
                "descriptor_dim": descriptor_dim,
                "descriptor_scratch_estimate_bytes": descriptor_bytes,
                "descriptors_kept": args.keep_descriptors,
            },
            "selection": selection_rows,
            "verdict": verdict,
            "limitations": [
                "All gamma values are selected and evaluated on MSLS-val; this "
                "is exploratory model selection, not independent confirmation.",
                "Gamma greater than one is an inference-time extrapolation beyond "
                "the adapter's training distribution.",
                "Inference interventions do not replace matched training controls.",
            ],
        }
        (args.output / "run.json").write_text(
            json.dumps(run_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print_summary(summary_rows, selection_rows, verdict)
        print(f"Results written to: {args.output}")
    finally:
        if store is not None:
            store.close()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
