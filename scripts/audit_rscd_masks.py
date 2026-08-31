#!/usr/bin/env python3
"""Offline contract audit for matched RSCD token masks.

This script reads only the immutable SegFormer cache and GSV-Cities CSVs.  It
does not load DINO, BoQ, training images, or a checkpoint, so it should run in
seconds before the 500-step implementation preflight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rscd import RSCDMaskBuilder, load_rscd_stats


CONFIG_TO_RUNTIME_MODE = {
    "no_mask": "no_mask",
    "uniform_block": "uniform",
    "shuffled_semantic": "shuffled",
    "aligned_rscd": "aligned",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("doc/rscd_mask_audit")
    )
    parser.add_argument("--num-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-mask-fraction", type=float, default=0.15)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def eligible_rows_and_places(
    dataset_root: Path, manifest: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible global cache rows and stable integer place codes."""

    eligible_min_views = int(manifest["eligible_min_views"])
    eligible_rows: list[np.ndarray] = []
    global_place_codes = np.full(int(manifest["num_images"]), -1, dtype=np.int64)
    next_place_code = 0
    for city in manifest["cities"]:
        city_name = str(city["name"])
        csv_path = dataset_root / "Dataframes" / f"{city_name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"city CSV not found: {csv_path}")
        if file_sha256(csv_path) != city["sha256"]:
            raise ValueError(f"city CSV SHA256 mismatch: {csv_path}")
        frame = pd.read_csv(csv_path, usecols=["place_id"])
        count = int(city["count"])
        offset = int(city["offset"])
        if len(frame) != count:
            raise ValueError(
                f"city CSV row count mismatch for {city_name}: "
                f"manifest={count}, csv={len(frame)}"
            )
        local_codes, uniques = pd.factorize(frame["place_id"], sort=False)
        local_counts = np.bincount(local_codes, minlength=len(uniques))
        mapped = local_codes.astype(np.int64) + next_place_code
        global_place_codes[offset : offset + count] = mapped
        local_eligible = local_counts[local_codes] >= eligible_min_views
        eligible_rows.append(offset + np.flatnonzero(local_eligible))
        next_place_code += len(uniques)
    rows = np.concatenate(eligible_rows)
    if bool((global_place_codes < 0).any()):
        raise ValueError("manifest city ranges do not cover every cache row")
    return rows.astype(np.int64), global_place_codes


def block_integrity(mask: torch.Tensor, block_size: int = 2) -> bool:
    batch, height, width = mask.shape
    blocks = (
        mask.view(
            batch,
            height // block_size,
            block_size,
            width // block_size,
            block_size,
        )
        .sum(dim=(2, 4))
        .unique()
    )
    return set(blocks.tolist()) <= {0, block_size * block_size}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")
    if args.seed < 0 or args.global_step < 0:
        raise ValueError("--seed and --global-step must be non-negative")
    if not 0.0 < args.max_mask_fraction <= 1.0:
        raise ValueError("--max-mask-fraction must be in (0,1]")

    dataset_root = args.dataset_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    stats_path = (
        args.stats_path.expanduser().resolve()
        if args.stats_path is not None
        else cache_dir / "rscd_class_stats.json"
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    manifest = load_json(manifest_path)
    stats = load_rscd_stats(
        stats_path,
        cache_manifest=manifest_path,
        verify_array_files=True,
        expected_min_confidence=args.min_confidence,
    )
    eligible_rows, place_codes = eligible_rows_and_places(
        dataset_root, manifest
    )
    if eligible_rows.size < args.num_images:
        raise ValueError(
            f"requested {args.num_images} images but only "
            f"{eligible_rows.size} cache rows are training-eligible"
        )
    rng = np.random.default_rng(args.seed)
    sampled = np.sort(
        rng.choice(eligible_rows, size=args.num_images, replace=False)
    ).astype(np.int64)

    labels_array = np.load(cache_dir / "labels.npy", mmap_mode="r")
    confidence_array = np.load(
        cache_dir / "confidence.npy", mmap_mode="r"
    )
    shuffled = np.load(
        cache_dir / "shuffled_indices.npy", mmap_mode="r"
    )
    donors = np.asarray(shuffled[sampled], dtype=np.int64)
    donor_cross_place = place_codes[sampled] != place_codes[donors]

    labels = torch.from_numpy(np.asarray(labels_array[sampled]).copy()).long()
    confidence = torch.from_numpy(
        np.asarray(confidence_array[sampled], dtype=np.float32).copy() / 255.0
    )
    donor_labels = torch.from_numpy(
        np.asarray(labels_array[donors]).copy()
    ).long()
    donor_confidence = torch.from_numpy(
        np.asarray(confidence_array[donors], dtype=np.float32).copy() / 255.0
    )
    cache_indices = torch.from_numpy(sampled.copy()).long()

    masks: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    deterministic: dict[str, bool] = {}
    for config_mode, runtime_mode in CONFIG_TO_RUNTIME_MODE.items():
        builder = RSCDMaskBuilder(
            stats,
            mode=runtime_mode,
            confidence_threshold=args.min_confidence,
            max_coverage=args.max_mask_fraction,
            seed=args.seed,
        )
        mask, mode_stats = builder.build(
            labels,
            confidence,
            cache_indices,
            args.global_step,
            donor_labels=donor_labels,
            donor_confidence=donor_confidence,
        )
        repeated, _ = builder.build(
            labels,
            confidence,
            cache_indices,
            args.global_step,
            donor_labels=donor_labels,
            donor_confidence=donor_confidence,
        )
        masks[config_mode] = mask
        diagnostics[config_mode] = mode_stats
        deterministic[config_mode] = bool(torch.equal(mask, repeated))

    token_counts = {
        mode: mask.sum(dim=(1, 2)).cpu().numpy().astype(np.int64)
        for mode, mask in masks.items()
    }
    active_modes = ("uniform_block", "shuffled_semantic", "aligned_rscd")
    matched_quota = all(
        np.array_equal(token_counts["aligned_rscd"], token_counts[mode])
        for mode in active_modes
    )
    grid_area = math.prod(stats.grid_size)
    nuisance = torch.tensor(stats.nuisance_scores, dtype=torch.float32)
    receiver_nuisance = nuisance[labels]

    summary_rows: list[dict] = []
    per_image_rows: list[dict] = []
    for mode, mask in masks.items():
        count = token_counts[mode]
        selected_receiver_nuisance = torch.where(
            mask.any(dim=(1, 2)),
            (receiver_nuisance * mask).sum(dim=(1, 2))
            / mask.sum(dim=(1, 2)).clamp_min(1),
            torch.zeros(mask.shape[0]),
        )
        summary_rows.append(
            {
                "mode": mode,
                "num_images": args.num_images,
                "mask_fraction_mean": float(count.mean() / grid_area),
                "mask_fraction_max": float(count.max() / grid_area),
                "selected_blocks_mean": float(count.mean() / 4.0),
                "quota_blocks_mean": float(
                    diagnostics[mode]["rscd_quota_blocks"].item()
                ),
                "source_candidate_blocks_mean": float(
                    diagnostics[mode][
                        "rscd_aligned_candidate_blocks"
                    ].item()
                ),
                "donor_candidate_blocks_mean": float(
                    diagnostics[mode][
                        "rscd_shuffled_candidate_blocks"
                    ].item()
                ),
                "selection_weight_mean": float(
                    diagnostics[mode]["rscd_selected_nuisance"].item()
                ),
                "receiver_nuisance_mean": float(
                    selected_receiver_nuisance.mean().item()
                ),
                "zero_quota_fraction": float(
                    diagnostics[mode]["rscd_zero_quota_frac"].item()
                ),
            }
        )
        for row_index, cache_row in enumerate(sampled.tolist()):
            per_image_rows.append(
                {
                    "cache_index": cache_row,
                    "donor_cache_index": int(donors[row_index]),
                    "donor_cross_place": bool(donor_cross_place[row_index]),
                    "mode": mode,
                    "mask_tokens": int(count[row_index]),
                    "mask_fraction": float(count[row_index] / grid_area),
                    "receiver_nuisance": float(
                        selected_receiver_nuisance[row_index].item()
                    ),
                }
            )

    checks = {
        "stats_and_cache_hashes_verified": True,
        "all_sampled_rows_training_eligible": True,
        "all_donors_cross_place": bool(donor_cross_place.all()),
        "all_modes_bit_exact_deterministic": all(deterministic.values()),
        "no_mask_is_empty": not bool(masks["no_mask"].any()),
        "active_modes_have_exact_per_image_quota": bool(matched_quota),
        "active_masks_are_complete_nonoverlapping_2x2_blocks": all(
            block_integrity(masks[mode]) for mode in active_modes
        ),
        "active_mask_fraction_is_bounded": all(
            float(mask.float().mean(dim=(1, 2)).max())
            <= args.max_mask_fraction + 1e-8
            for mode, mask in masks.items()
            if mode != "no_mask"
        ),
        "active_mask_fraction_is_nonzero": all(
            bool(mask.any()) for mode, mask in masks.items() if mode != "no_mask"
        ),
        "aligned_differs_from_uniform": bool(
            (masks["aligned_rscd"] != masks["uniform_block"]).any()
        ),
        "aligned_differs_from_shuffled": bool(
            (masks["aligned_rscd"] != masks["shuffled_semantic"]).any()
        ),
    }
    report = {
        "schema": "openvpr_rscd_mask_audit",
        "version": 1,
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "stats_path": str(stats_path),
        "stats_sha256": stats.sha256,
        "sample_seed": args.seed,
        "global_step": args.global_step,
        "num_images": args.num_images,
        "eligible_rows": int(eligible_rows.size),
        "grid_size": list(stats.grid_size),
        "min_confidence": args.min_confidence,
        "max_mask_fraction": args.max_mask_fraction,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    (output / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output / "summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    write_csv(
        output / "per_image.csv",
        list(per_image_rows[0]),
        per_image_rows,
    )

    print("RSCD offline matched-mask audit")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"Verdict: {report['verdict']}")
    print(f"Report: {output}")
    if report["verdict"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
