"""Evaluate a frozen dynamic-category negative prior at BoQ attention logits.

This is a causal screening experiment, not training.  It keeps the selected
repeatability+uniqueness checkpoint frozen and compares five descriptors from
the exact same DINO feature map:

* baseline: no attention bias (the historical checkpoint path);
* zero_bias: an all-zero float mask controlling the attention-kernel path;
* aligned: ``-beta *`` the correct image's dynamic-patch area fraction;
* shuffled: a wrong image's mask from a DB/query-preserving derangement;
* random: an exact spatial permutation of each image's own mask values.

Run ``cache_dynamic_category_masks.py`` first.  The segmentation teacher is
then absent from this process and from all VPR checkpoints.
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
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_category_prior import (  # noqa: E402
    canonical_image_paths,
    file_sha256,
    load_and_validate_mask_cache,
    map_condition_query_indices,
    role_preserving_derangement,
    spatially_permute_masks,
    validate_ground_truth,
    validate_overlapping_ground_truth,
)
from scripts.eval_condition_robustness import (  # noqa: E402
    build_transform,
    choose_device,
    load_inference_model_from_ckpt,
)
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset  # noqa: E402
from src.dataloaders.valid.msls_condition import MSLSConditionDataset  # noqa: E402
from src.models.aggregators.boq import BoQ  # noqa: E402


VARIANTS = ("baseline", "zero_bias", "aligned", "shuffled", "random")
CUSTOM_CONDITION_PROTOCOL = (
    "custom condition subset of the standard MSLS query universe, searched "
    "against the standard full database; not an official condition-filtered "
    "MSLS subtask"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen a frozen dynamic-category negative prior in BoQ"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--msls-path", type=Path, default=Path("datasets/msls-val"))
    parser.add_argument("--mask-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="parent for temporary descriptor memmaps (about 5 GB float32)",
    )
    parser.add_argument(
        "--keep-descriptors",
        action="store_true",
        help="retain descriptor memmaps under output/descriptors",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        choices=("night", "season"),
        default=("night", "season"),
        help="validated condition subsets reusing the standard MSLS descriptors",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="positive logit penalty; preregistered first screen is 0.5",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(280, 280))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=(1, 5, 10))
    parser.add_argument(
        "--descriptor-dtype",
        choices=("float32", "float16"),
        default="float32",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--expected-overall-r1",
        type=float,
        default=91.22,
        help="known RU R@1 in percent; used only as a loader/checkpoint check",
    )
    parser.add_argument(
        "--baseline-tolerance-pp",
        type=float,
        default=0.15,
        help="allowed absolute baseline reproduction error in percentage points",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.conditions = tuple(sorted(set(args.conditions)))
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
    if not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError("beta must be finite and strictly positive")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid batch-size or num-workers")
    if len(args.image_size) != 2 or min(args.image_size) <= 0:
        raise ValueError("image-size must contain two positive integers")
    if any(k <= 0 for k in args.k_values):
        raise ValueError("k-values must be positive")
    args.k_values = sorted(set(int(k) for k in args.k_values))
    if 1 not in args.k_values:
        raise ValueError("k-values must include 1 for paired outcomes/verdict")
    if (
        not math.isfinite(args.expected_overall_r1)
        or not 0.0 <= args.expected_overall_r1 <= 100.0
    ):
        raise ValueError("expected-overall-r1 must be a percent in [0, 100]")
    if args.baseline_tolerance_pp < 0:
        raise ValueError("baseline-tolerance-pp must be non-negative")
    if args.scratch_dir is not None:
        args.scratch_dir = args.scratch_dir.expanduser().resolve()


def validate_ru_checkpoint_configuration(
    checkpoint_path: Path, image_size: tuple[int, int]
) -> dict[str, Any]:
    """Fail closed if the input is not the intended RU-only BoQ checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("hyper_parameters")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no hyper_parameters mapping")
    aggregator = config.get("aggregator", {}) or {}
    if aggregator.get("class") != "BoQ":
        raise ValueError("screening requires a BoQ checkpoint")
    distillation = config.get("distillation", {}) or {}
    semantic_region = distillation.get("semantic_region", {}) or {}
    if not semantic_region.get("enabled", False):
        raise ValueError("checkpoint has no enabled semantic_region gate")
    if semantic_region.get("mode") != "repeatability_uniqueness_only":
        raise ValueError(
            "checkpoint must be the repeatability_uniqueness_only control"
        )
    if float(semantic_region.get("lambda_target", 0.0)) <= 0:
        raise ValueError("checkpoint did not train an active RU gate")
    configured_size = tuple(
        int(value)
        for value in (config.get("datamodule", {}) or {}).get("val_image_size", ())
    )
    if configured_size and configured_size != tuple(image_size):
        raise ValueError(
            f"checkpoint val image size {configured_size} != requested {image_size}"
        )
    return {
        "aggregator_class": aggregator.get("class"),
        "backbone_class": (config.get("backbone", {}) or {}).get("class"),
        "val_image_size": list(configured_size),
        "semantic_region": dict(semantic_region),
    }


def extract_ru_feature_map(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model.backbone(images)
    featmap = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(featmap, torch.Tensor) or featmap.ndim != 4:
        raise TypeError("dynamic-prior screening requires a BCHW feature map")
    if model.semantic_region_gate is None:
        raise RuntimeError("RU checkpoint loader did not restore semantic_region_gate")
    featmap, _, _ = model.semantic_region_gate(featmap)
    if model.spatial_attn_head is not None:
        featmap, _ = model.spatial_attn_head(featmap)
    return featmap


def boq_descriptor(
    aggregator: BoQ,
    feature_map: torch.Tensor,
    attention_bias: torch.Tensor | None,
) -> torch.Tensor:
    output = aggregator(feature_map, attention_bias=attention_bias)
    descriptor = output[0] if isinstance(output, (tuple, list)) else output
    if descriptor.ndim != 2:
        raise ValueError(f"unexpected descriptor shape: {descriptor.shape}")
    return descriptor


class DescriptorStore:
    def __init__(self, directory: Path, image_count: int, dtype: str) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.image_count = int(image_count)
        self.dtype = np.dtype(dtype)
        self.arrays: dict[str, np.memmap] = {}
        self.paths: dict[str, Path] = {}
        self.capacity_checked = False

    def _check_capacity(self, descriptor_dim: int) -> None:
        required = (
            len(VARIANTS)
            * self.image_count
            * descriptor_dim
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
                f"{required / 2**30:.2f} GiB descriptor estimate"
            )
        self.capacity_checked = True

    def write(self, name: str, indices: np.ndarray, values: np.ndarray) -> None:
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

    def flush(self) -> None:
        for array in self.arrays.values():
            array.flush()

    def close(self) -> None:
        for array in self.arrays.values():
            array.flush()
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays.clear()


def extract_descriptors(
    *,
    model: torch.nn.Module,
    dataset: MapillarySLSDataset,
    masks: np.ndarray,
    donor_indices: np.ndarray,
    beta: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    descriptor_directory: Path,
    descriptor_dtype: str,
    use_amp: bool,
) -> DescriptorStore:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    store = DescriptorStore(descriptor_directory, len(dataset), descriptor_dtype)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    checked_grid = False

    try:
        with torch.inference_mode():
            for images, sample_indices in tqdm(
                loader, desc="Extract five matched variants"
            ):
                images = images.to(device, non_blocking=True)
                indices = torch.as_tensor(sample_indices, dtype=torch.long)
                numpy_indices = indices.numpy()
                aligned = torch.from_numpy(masks[numpy_indices]).to(
                    device=device, dtype=torch.float32
                )
                shuffled = torch.from_numpy(masks[donor_indices[numpy_indices]]).to(
                    device=device, dtype=torch.float32
                )
                random = spatially_permute_masks(aligned, indices, seed)

                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    feature_map = extract_ru_feature_map(model, images)
                    if not checked_grid:
                        expected_grid = tuple(masks.shape[-2:])
                        actual_grid = tuple(feature_map.shape[-2:])
                        if actual_grid != expected_grid:
                            raise ValueError(
                                f"DINO grid {actual_grid} != mask grid {expected_grid}"
                            )
                        checked_grid = True
                    descriptors = {
                        "baseline": boq_descriptor(
                            model.aggregator, feature_map, attention_bias=None
                        ),
                        "zero_bias": boq_descriptor(
                            model.aggregator,
                            feature_map,
                            attention_bias=torch.zeros_like(aligned),
                        ),
                        "aligned": boq_descriptor(
                            model.aggregator,
                            feature_map,
                            attention_bias=-beta * aligned,
                        ),
                        "shuffled": boq_descriptor(
                            model.aggregator,
                            feature_map,
                            attention_bias=-beta * shuffled,
                        ),
                        "random": boq_descriptor(
                            model.aggregator,
                            feature_map,
                            attention_bias=-beta * random,
                        ),
                    }
                for name, descriptor in descriptors.items():
                    store.write(
                        name,
                        numpy_indices,
                        descriptor.detach().float().cpu().numpy(),
                    )
        store.flush()
        return store
    except Exception:
        store.close()
        raise


def _search_and_score(
    descriptor_path: Path,
    *,
    num_references: int,
    query_offsets: np.ndarray,
    ground_truth: Sequence[np.ndarray],
    k_values: Sequence[int],
) -> tuple[dict[int, float], np.ndarray, dict[int, np.ndarray]]:
    descriptors = np.load(descriptor_path, mmap_mode="r")
    references = np.ascontiguousarray(
        descriptors[:num_references], dtype=np.float32
    )
    query_rows = num_references + np.asarray(query_offsets, dtype=np.int64)
    queries = np.ascontiguousarray(descriptors[query_rows], dtype=np.float32)
    index = faiss.IndexFlatL2(references.shape[1])
    index.add(references)
    _, predictions = index.search(queries, max(k_values))
    hits: dict[int, np.ndarray] = {}
    recalls: dict[int, float] = {}
    for k in k_values:
        hit = np.asarray(
            [
                np.any(np.isin(prediction[:k], ground_truth[query_index]))
                for query_index, prediction in enumerate(predictions)
            ],
            dtype=bool,
        )
        hits[int(k)] = hit
        recalls[int(k)] = float(hit.mean())
    del index, references, queries, descriptors
    return recalls, predictions[:, 0].copy(), hits


def manifest_record(path: Path) -> dict[str, Any]:
    """Record one immutable input manifest for result provenance."""

    if not path.is_file():
        raise FileNotFoundError(f"evaluation manifest not found: {path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def build_evaluation_sets(
    full_dataset: MapillarySLSDataset,
    condition_names: Sequence[str],
    transform: Any,
    msls_path: Path,
) -> list[dict[str, Any]]:
    standard_query_paths = canonical_image_paths(full_dataset.qImages)
    standard_ground_truth = validate_ground_truth(
        full_dataset.ground_truth,
        num_queries=full_dataset.num_queries,
        num_references=full_dataset.num_references,
        dataset_name="standard MSLS",
    )
    standard_query_file = msls_path / "msls_val_qImages.npy"
    standard_gt_file = msls_path / "msls_val_gt_25m.npy"
    evaluations: list[dict[str, Any]] = [
        {
            "name": full_dataset.dataset_name,
            "query_paths": standard_query_paths,
            "query_offsets": np.arange(full_dataset.num_queries, dtype=np.int64),
            "ground_truth": standard_ground_truth,
            "protocol": "standard MSLS-val protocol",
            "num_standard_query_overlap": full_dataset.num_queries,
            "num_condition_only_queries": 0,
            "manifests": {
                "queries": manifest_record(standard_query_file),
                "ground_truth": manifest_record(standard_gt_file),
            },
        }
    ]
    for condition_name in condition_names:
        try:
            condition = MSLSConditionDataset(
                condition=condition_name,
                dataset_path=str(msls_path),
                input_transform=transform,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"missing {condition_name} split; run "
                "scripts/generate_msls_condition_splits.py --msls-path "
                f"{msls_path} --force first"
            ) from exc
        if not np.array_equal(
            canonical_image_paths(condition.dbImages),
            canonical_image_paths(full_dataset.dbImages),
        ):
            raise ValueError(f"{condition_name} split does not share MSLS DB order")
        condition_query_paths = canonical_image_paths(condition.qImages)
        condition_ground_truth = validate_ground_truth(
            condition.ground_truth,
            num_queries=condition.num_queries,
            num_references=condition.num_references,
            dataset_name=condition.dataset_name,
        )
        overlap_count, condition_only_count = validate_overlapping_ground_truth(
            standard_query_paths,
            standard_ground_truth,
            condition_query_paths,
            condition_ground_truth,
            condition_name=condition.dataset_name,
        )
        query_offsets = map_condition_query_indices(
            standard_query_paths, condition_query_paths
        )
        if condition_only_count != 0 or overlap_count != condition.num_queries:
            raise AssertionError("condition subset mapping is internally inconsistent")
        query_filename, gt_filename = MSLSConditionDataset.CONDITION_FILES[
            condition_name
        ]
        evaluations.append(
            {
                "name": condition.dataset_name,
                "query_paths": condition_query_paths,
                "query_offsets": query_offsets,
                "ground_truth": condition_ground_truth,
                "protocol": CUSTOM_CONDITION_PROTOCOL,
                "num_standard_query_overlap": overlap_count,
                "num_condition_only_queries": condition_only_count,
                "manifests": {
                    "queries": manifest_record(msls_path / query_filename),
                    "ground_truth": manifest_record(msls_path / gt_filename),
                },
            }
        )
    return evaluations


def build_query_condition_strata(
    evaluations: Sequence[Mapping[str, Any]], num_queries: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode all condition memberships so shuffled donors stay matched."""

    strata = np.zeros(num_queries, dtype=np.uint64)
    for bit, evaluation in enumerate(evaluations[1:]):
        if bit >= 63:
            raise ValueError("too many condition splits for a uint64 membership mask")
        offsets = np.asarray(evaluation["query_offsets"], dtype=np.int64)
        if np.any(offsets < 0) or np.any(offsets >= num_queries):
            raise ValueError(f"invalid query offset in {evaluation['name']}")
        strata[offsets] |= np.uint64(1) << np.uint64(bit)
    counts = {
        str(int(stratum)): int(np.sum(strata == stratum))
        for stratum in np.unique(strata)
    }
    return strata, counts


def evaluate_variants(
    store: DescriptorStore,
    *,
    dataset: MapillarySLSDataset,
    evaluations: Sequence[Mapping[str, Any]],
    beta: float,
    k_values: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for evaluation in evaluations:
        variant_results: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            recalls, top1, hits = _search_and_score(
                store.paths[variant],
                num_references=dataset.num_references,
                query_offsets=evaluation["query_offsets"],
                ground_truth=evaluation["ground_truth"],
                k_values=k_values,
            )
            variant_results[variant] = {
                "recalls": recalls,
                "top1": top1,
                "hits": hits,
            }

        baseline_results = variant_results["baseline"]
        zero_bias_results = variant_results["zero_bias"]
        for variant in VARIANTS:
            result = variant_results[variant]
            recalls = result["recalls"]
            hits = result["hits"]
            row: dict[str, Any] = {
                "dataset": evaluation["name"],
                "variant": variant,
                "beta": 0.0 if variant in ("baseline", "zero_bias") else beta,
                "num_queries": len(evaluation["query_offsets"]),
            }
            for k in k_values:
                row[f"r@{k}"] = recalls[int(k)]
                row[f"delta_vs_baseline_r@{k}_pp"] = 100.0 * (
                    recalls[int(k)] - baseline_results["recalls"][int(k)]
                )
                row[f"delta_vs_baseline_r@{k}_queries"] = int(
                    hits[int(k)].sum()
                    - baseline_results["hits"][int(k)].sum()
                )
                row[f"delta_vs_zero_bias_r@{k}_pp"] = 100.0 * (
                    recalls[int(k)] - zero_bias_results["recalls"][int(k)]
                )
                row[f"delta_vs_zero_bias_r@{k}_queries"] = int(
                    hits[int(k)].sum()
                    - zero_bias_results["hits"][int(k)].sum()
                )
            summary_rows.append(row)

            for query_index, (query_path, prediction, correct) in enumerate(
                zip(
                    evaluation["query_paths"],
                    result["top1"].tolist(),
                    hits[1].tolist(),
                )
            ):
                outcome_rows.append(
                    {
                        "dataset": evaluation["name"],
                        "query_index": query_index,
                        "query_path": str(query_path),
                        "variant": variant,
                        "beta": (
                            0.0 if variant in ("baseline", "zero_bias") else beta
                        ),
                        "top1_reference_index": int(prediction),
                        "top1_reference_path": str(dataset.dbImages[prediction]).replace(
                            "\\", "/"
                        ),
                        "top1_correct": int(correct),
                    }
                )

        aligned_hits = variant_results["aligned"]["hits"][1]
        for comparator in ("baseline", "zero_bias", "shuffled", "random"):
            other_hits = variant_results[comparator]["hits"][1]
            aligned_only = int(np.sum(aligned_hits & ~other_hits))
            comparator_only = int(np.sum(~aligned_hits & other_hits))
            paired_rows.append(
                {
                    "dataset": evaluation["name"],
                    "left": "aligned",
                    "right": comparator,
                    "left_only_correct": aligned_only,
                    "right_only_correct": comparator_only,
                    "net_queries": aligned_only - comparator_only,
                    "both_correct": int(np.sum(aligned_hits & other_hits)),
                    "both_wrong": int(np.sum(~aligned_hits & ~other_hits)),
                }
            )
    return summary_rows, outcome_rows, paired_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def screening_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_overall_r1: float,
    baseline_tolerance_pp: float,
) -> dict[str, Any]:
    by_key = {(row["dataset"], row["variant"]): row for row in rows}
    overall_name = "msls-val"
    overall_baseline = by_key[(overall_name, "baseline")]
    overall_zero_bias = by_key[(overall_name, "zero_bias")]
    overall_aligned = by_key[(overall_name, "aligned")]
    baseline_error_pp = abs(
        100.0 * float(overall_baseline["r@1"]) - expected_overall_r1
    )
    baseline_reproduced = baseline_error_pp <= baseline_tolerance_pp
    overall_beats_controls = all(
        float(overall_aligned["r@1"])
        > float(by_key[(overall_name, control)]["r@1"])
        for control in ("zero_bias", "shuffled", "random")
    )
    overall_not_harmed = (
        float(overall_aligned["delta_vs_baseline_r@1_pp"]) >= -0.3
    )
    difficult_wins = []
    for dataset_name in sorted({str(row["dataset"]) for row in rows} - {overall_name}):
        aligned = by_key[(dataset_name, "aligned")]
        gain = float(aligned["delta_vs_zero_bias_r@1_pp"])
        beats_controls = all(
            float(aligned["r@1"]) > float(by_key[(dataset_name, control)]["r@1"])
            for control in ("zero_bias", "shuffled", "random")
        )
        if gain >= 1.0 and beats_controls:
            difficult_wins.append(dataset_name)
    conditions_present = any(
        str(row["dataset"]) != overall_name for row in rows
    )
    pass_screen = bool(
        baseline_reproduced
        and overall_beats_controls
        and overall_not_harmed
        and difficult_wins
    )
    return {
        "status": "pass" if pass_screen else ("fail" if conditions_present else "incomplete"),
        "baseline_reproduced": baseline_reproduced,
        "baseline_error_pp": baseline_error_pp,
        "baseline_tolerance_pp": baseline_tolerance_pp,
        "overall_aligned_beats_zero_bias_shuffled_and_random": overall_beats_controls,
        "overall_zero_bias_kernel_delta_r1_pp": float(
            overall_zero_bias["delta_vs_baseline_r@1_pp"]
        ),
        "overall_zero_bias_kernel_delta_r1_queries": int(
            overall_zero_bias["delta_vs_baseline_r@1_queries"]
        ),
        "overall_aligned_delta_r1_pp": float(
            overall_aligned["delta_vs_baseline_r@1_pp"]
        ),
        "overall_drop_limit_pp": -0.3,
        "difficult_condition_wins_ge_1pp_vs_zero_bias_and_controls": difficult_wins,
        "decision_rule": (
            "Pass only if baseline is reproduced, aligned beats zero-bias, "
            "shuffled, and random on overall, overall drops no more than 0.3 "
            "pp versus the historical baseline, and aligned gains at least 1 "
            "pp versus zero-bias while beating controls on night or season."
        ),
    }


def print_summary(rows: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any]) -> None:
    print("\nDynamic-category negative-prior screen")
    print("=" * 108)
    print(
        f"{'Dataset':<24} {'Variant':<12} {'R@1':>8} "
        f"{'vs hist pp':>11} {'vs zero pp':>11} {'vs zero q':>10}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<24} {row['variant']:<12} "
            f"{100.0 * float(row['r@1']):>7.2f}% "
            f"{float(row['delta_vs_baseline_r@1_pp']):>+10.2f} "
            f"{float(row['delta_vs_zero_bias_r@1_pp']):>+10.2f} "
            f"{int(row['delta_vs_zero_bias_r@1_queries']):>+10d}"
        )
    print("-" * 108)
    print(f"Screening verdict: {str(verdict['status']).upper()}")
    print(verdict["decision_rule"])


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
    dataset = MapillarySLSDataset(
        dataset_path=str(args.msls_path), input_transform=transform
    )
    if max(args.k_values) > dataset.num_references:
        raise ValueError("largest k-value exceeds the number of MSLS references")
    masks, cache_metadata = load_and_validate_mask_cache(
        args.mask_cache,
        expected_image_paths=dataset.image_paths,
        expected_num_references=dataset.num_references,
        expected_grid_size=grid_size,
    )
    evaluations = build_evaluation_sets(
        dataset, args.conditions, transform, args.msls_path
    )
    query_strata, query_stratum_counts = build_query_condition_strata(
        evaluations, dataset.num_queries
    )
    donor_indices = role_preserving_derangement(
        dataset.num_references,
        dataset.num_queries,
        args.seed,
        query_strata=query_strata,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mask cache: {args.mask_cache}")
    print(f"beta={args.beta}; variants={VARIANTS}")
    model = load_inference_model_from_ckpt(args.checkpoint, device)
    if not isinstance(model.aggregator, BoQ):
        raise TypeError("checkpoint aggregator is not the repository BoQ class")
    if model.semantic_region_gate is None:
        raise RuntimeError("RU semantic_region_gate was not restored")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_descriptors:
        args.output.mkdir(parents=True, exist_ok=False)
        descriptor_directory = args.output / "descriptors"
    else:
        scratch_parent = args.scratch_dir
        if scratch_parent is not None:
            scratch_parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="dynamic-category-prior-",
            dir=str(scratch_parent) if scratch_parent is not None else None,
        )
        descriptor_directory = Path(temporary.name)

    store: DescriptorStore | None = None
    try:
        store = extract_descriptors(
            model=model,
            dataset=dataset,
            masks=masks,
            donor_indices=donor_indices,
            beta=args.beta,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            descriptor_directory=descriptor_directory,
            descriptor_dtype=args.descriptor_dtype,
            use_amp=device.type == "cuda" and not args.no_amp,
        )
        summary_rows, outcome_rows, paired_rows = evaluate_variants(
            store,
            dataset=dataset,
            evaluations=evaluations,
            beta=args.beta,
            k_values=args.k_values,
        )
        verdict = screening_verdict(
            summary_rows,
            expected_overall_r1=args.expected_overall_r1,
            baseline_tolerance_pp=args.baseline_tolerance_pp,
        )
        if not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=False)
        write_csv(args.output / "summary.csv", summary_rows)
        write_csv(args.output / "query_outcomes.csv", outcome_rows)
        write_csv(args.output / "paired_comparisons.csv", paired_rows)
        run_record = {
            "schema_version": 2,
            "method": "frozen_dynamic_category_negative_attention_prior",
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
            "intervention": {
                "location": "both BoQ cross-attention blocks",
                "formula": "attention_bias = -beta * dynamic_patch_area_fraction",
                "beta": args.beta,
                "full_dynamic_patch_attention_odds_multiplier": math.exp(-args.beta),
                "feature_multiplication": False,
                "per_image_standardisation": False,
            },
            "controls": {
                "baseline": "attention_bias=None (historical checkpoint path)",
                "zero_bias": (
                    "all-zero float key_padding_mask; numerical-path control "
                    "for aligned/shuffled/random"
                ),
                "aligned": "correct image mask",
                "shuffled": (
                    "global seeded no-fixed-point donor permutation, separately "
                    "inside references and every query condition-membership stratum"
                ),
                "random": "per-image exact-value spatial permutation",
            },
            "donor_indices_sha256": hashlib.sha256(
                donor_indices.tobytes()
            ).hexdigest(),
            "query_condition_stratum_counts": query_stratum_counts,
            "query_condition_bits": {
                str(bit): evaluation["name"]
                for bit, evaluation in enumerate(evaluations[1:])
            },
            "datasets": [
                {
                    "name": evaluation["name"],
                    "num_queries": len(evaluation["query_offsets"]),
                    "protocol": evaluation["protocol"],
                    "num_standard_query_overlap": evaluation[
                        "num_standard_query_overlap"
                    ],
                    "num_condition_only_queries": evaluation[
                        "num_condition_only_queries"
                    ],
                    "manifests": evaluation["manifests"],
                }
                for evaluation in evaluations
            ],
            "descriptor_index": {
                "num_references": dataset.num_references,
                "num_standard_queries": dataset.num_queries,
                "definition": "standard MSLS DB + standard query index",
                "database_manifest": manifest_record(
                    args.msls_path / "msls_val_dbImages.npy"
                ),
            },
            "evaluation_protocol": {
                "standard": "standard MSLS-val protocol",
                "conditions": CUSTOM_CONDITION_PROTOCOL,
            },
            "image_size": list(args.image_size),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "seed": args.seed,
            "amp": device.type == "cuda" and not args.no_amp,
            "descriptor_dtype": args.descriptor_dtype,
            "descriptors_retained": bool(args.keep_descriptors),
            "results": summary_rows,
            "paired_comparisons": paired_rows,
            "verdict": verdict,
            "limitations": [
                "This is a frozen-checkpoint causal screen, not a trained model.",
                "Pascal-VOC has no rider or truck label.",
                (
                    "The prior enters after BoQ projection/encoder token mixing "
                    "and does not erase dynamic features."
                ),
                (
                    "Night/season are custom query slices over the standard full "
                    "database, not official condition-filtered MSLS subtasks."
                ),
                (
                    "Failure rejects this teacher/injection/strength route, not "
                    "all semantic VPR methods."
                ),
            ],
        }
        with (args.output / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(run_record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print_summary(summary_rows, verdict)
        print(f"Results written to: {args.output}")
    finally:
        if store is not None:
            store.close()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
