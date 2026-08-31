#!/usr/bin/env python3
"""Audit the preregistered 500-step aligned RSCD implementation preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


CORE_METRICS = (
    "rscd_mask_fraction",
    "rscd_selected_blocks",
    "rscd_quota_blocks",
    "rscd_relation_loss",
    "rscd_pairwise_cos_mae",
    "rscd_descriptor_drift_rms",
    "rscd_backbone_grad_rms",
    "rscd_aggregator_grad_rms",
    "rscd_gate_grad_rms",
    "rscd_eval_clean_max_abs_error",
)
DIAGNOSTIC_METRICS = (
    "rscd_source_candidate_blocks",
    "rscd_donor_candidate_blocks",
    "rscd_masked_nuisance_mean",
    "rscd_quota_zero_fraction",
)
REQUIRED_METRICS = CORE_METRICS + DIAGNOSTIC_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read TensorBoard scalars and apply the RSCD 500-step "
            "implementation-contract checks. This audit does not evaluate "
            "retrieval efficacy."
        )
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="A Lightning version directory or a parent containing one.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--minimum-last-step", type=int, default=490)
    parser.add_argument("--tail-points", type=int, default=5)
    parser.add_argument("--zero-tolerance", type=float, default=1e-6)
    parser.add_argument("--accounting-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-mask-fraction", type=float, default=0.15)
    parser.add_argument("--grid-size", type=int, nargs=2, default=(20, 20))
    parser.add_argument("--block-size", type=int, default=2)
    return parser.parse_args()


def newest_event_run(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"log directory not found: {path}")
    events = list(path.rglob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"no TensorBoard event file below: {path}")
    return max(events, key=lambda item: item.stat().st_mtime_ns).parent


def resolve_tag(tags: Sequence[str], requested: str) -> str:
    if requested in tags:
        return requested
    matches = [
        tag
        for tag in tags
        if tag.endswith("/" + requested) or tag.endswith(requested)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"required scalar {requested!r} not found uniquely; available="
            f"{sorted(tags)}"
        )
    return matches[0]


def finite_summary(
    events: Sequence[Any], tail_points: int = 5
) -> dict[str, float | int]:
    if tail_points < 1:
        raise ValueError("tail_points must be positive")
    ordered = sorted(events, key=lambda event: int(event.step))
    values = [float(event.value) for event in ordered]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("scalar has no finite values")
    tail = values[-min(tail_points, len(values)) :]
    return {
        "count": len(values),
        "last_step": int(ordered[-1].step),
        "last": values[-1],
        "tail_mean": sum(tail) / len(tail),
        "maximum": max(values),
        "minimum": min(values),
    }


def paired_max_abs_difference(
    left_events: Sequence[Any],
    right_events: Sequence[Any],
    *,
    right_scale: float = 1.0,
) -> float:
    """Compare two scalar streams at exactly the same optimizer steps."""

    if not math.isfinite(float(right_scale)):
        raise ValueError("right_scale must be finite")

    def by_step(events: Sequence[Any]) -> dict[int, float]:
        output: dict[int, float] = {}
        for event in events:
            step = int(event.step)
            value = float(event.value)
            if not math.isfinite(value):
                raise ValueError("paired scalar contains a non-finite value")
            output[step] = value
        if not output:
            raise ValueError("paired scalar has no values")
        return output

    left = by_step(left_events)
    right = by_step(right_events)
    if set(left) != set(right):
        raise ValueError(
            "paired scalars must be logged at exactly the same optimizer steps"
        )
    return max(
        abs(left[step] - float(right_scale) * right[step]) for step in left
    )


def evaluate_checks(
    summaries: dict[str, dict[str, float | int]],
    events: dict[str, Sequence[Any]],
    *,
    minimum_last_step: int = 490,
    zero_tolerance: float = 1e-6,
    accounting_tolerance: float = 1e-5,
    max_mask_fraction: float = 0.15,
    grid_size: tuple[int, int] = (20, 20),
    block_size: int = 2,
) -> tuple[dict[str, bool], dict[str, float]]:
    """Return fail-closed RSCD implementation checks and derived errors."""

    if set(summaries) != set(REQUIRED_METRICS):
        missing = sorted(set(REQUIRED_METRICS) - set(summaries))
        extra = sorted(set(summaries) - set(REQUIRED_METRICS))
        raise ValueError(f"metric set mismatch: missing={missing}, extra={extra}")
    if set(events) != set(REQUIRED_METRICS):
        raise ValueError("event streams must match REQUIRED_METRICS exactly")
    if minimum_last_step < 0:
        raise ValueError("minimum_last_step must be non-negative")
    if zero_tolerance < 0 or accounting_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    if not 0.0 < max_mask_fraction < 1.0:
        raise ValueError("max_mask_fraction must be in (0, 1)")
    if len(grid_size) != 2 or min(int(value) for value in grid_size) < 1:
        raise ValueError("grid_size must contain two positive integers")
    if isinstance(block_size, bool) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    grid_area = int(grid_size[0]) * int(grid_size[1])
    block_area = int(block_size) ** 2
    if block_area > grid_area:
        raise ValueError("block area cannot exceed the feature-grid area")

    quota_error = paired_max_abs_difference(
        events["rscd_selected_blocks"], events["rscd_quota_blocks"]
    )
    coverage_error = paired_max_abs_difference(
        events["rscd_mask_fraction"],
        events["rscd_selected_blocks"],
        right_scale=block_area / grid_area,
    )
    mask = summaries["rscd_mask_fraction"]
    selected = summaries["rscd_selected_blocks"]
    source = summaries["rscd_source_candidate_blocks"]
    donor = summaries["rscd_donor_candidate_blocks"]
    nuisance = summaries["rscd_masked_nuisance_mean"]
    zero_fraction = summaries["rscd_quota_zero_fraction"]

    checks = {
        "preflight_reached_registered_step": all(
            int(summary["last_step"]) >= minimum_last_step
            for summary in summaries.values()
        ),
        "mask_fraction_nonzero_and_bounded": (
            float(mask["minimum"]) >= -accounting_tolerance
            and float(mask["maximum"]) > 0.0
            and float(mask["maximum"])
            <= max_mask_fraction + accounting_tolerance
        ),
        "selected_blocks_nonzero": (
            float(selected["minimum"]) >= -accounting_tolerance
            and float(selected["maximum"]) > 0.0
        ),
        "selected_blocks_match_quota": quota_error <= accounting_tolerance,
        "mask_fraction_matches_block_accounting": (
            coverage_error <= accounting_tolerance
        ),
        "relation_loss_nonzero": (
            float(summaries["rscd_relation_loss"]["maximum"]) > 0.0
        ),
        "pairwise_change_nonzero": (
            float(summaries["rscd_pairwise_cos_mae"]["maximum"]) > 0.0
        ),
        "descriptor_drift_nonzero": (
            float(summaries["rscd_descriptor_drift_rms"]["maximum"]) > 0.0
        ),
        "backbone_gradient_nonzero": (
            float(summaries["rscd_backbone_grad_rms"]["maximum"]) > 0.0
        ),
        "aggregator_gradient_nonzero": (
            float(summaries["rscd_aggregator_grad_rms"]["maximum"]) > 0.0
        ),
        "gate_gradient_nonzero": (
            float(summaries["rscd_gate_grad_rms"]["maximum"]) > 0.0
        ),
        "evaluation_path_is_clean": (
            float(summaries["rscd_eval_clean_max_abs_error"]["maximum"])
            <= zero_tolerance
        ),
        "source_candidate_count_valid": float(source["minimum"]) >= 0.0,
        "donor_candidate_count_valid": float(donor["minimum"]) >= 0.0,
        "masked_nuisance_in_unit_interval": (
            float(nuisance["minimum"]) >= -accounting_tolerance
            and float(nuisance["maximum"]) <= 1.0 + accounting_tolerance
        ),
        "quota_zero_fraction_valid": (
            float(zero_fraction["minimum"]) >= -accounting_tolerance
            and float(zero_fraction["maximum"])
            <= 1.0 + accounting_tolerance
            and float(zero_fraction["minimum"]) < 1.0
        ),
    }
    derived = {
        "selected_vs_quota_max_abs_error": quota_error,
        "mask_fraction_vs_block_accounting_max_abs_error": coverage_error,
        "block_fraction": block_area / grid_area,
    }
    return checks, derived


def main() -> None:
    args = parse_args()
    if args.tail_points < 1:
        raise ValueError("--tail-points must be positive")

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise ImportError(
            "TensorBoard is required: install the project's pinned environment"
        ) from exc

    run_dir = newest_event_run(args.logdir)
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])

    summaries: dict[str, dict[str, float | int]] = {}
    events: dict[str, Sequence[Any]] = {}
    resolved_tags: dict[str, str] = {}
    for metric in REQUIRED_METRICS:
        tag = resolve_tag(scalar_tags, metric)
        resolved_tags[metric] = tag
        metric_events = accumulator.Scalars(tag)
        events[metric] = metric_events
        summaries[metric] = finite_summary(metric_events, args.tail_points)

    checks, derived = evaluate_checks(
        summaries,
        events,
        minimum_last_step=args.minimum_last_step,
        zero_tolerance=args.zero_tolerance,
        accounting_tolerance=args.accounting_tolerance,
        max_mask_fraction=args.max_mask_fraction,
        grid_size=tuple(args.grid_size),
        block_size=args.block_size,
    )
    verdict = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": "openvpr_rscd_preflight_audit",
        "version": 1,
        "run_dir": str(run_dir),
        "thresholds": {
            "minimum_last_step": args.minimum_last_step,
            "tail_points": args.tail_points,
            "zero_tolerance": args.zero_tolerance,
            "accounting_tolerance": args.accounting_tolerance,
            "max_mask_fraction": args.max_mask_fraction,
            "grid_size": list(args.grid_size),
            "block_size": args.block_size,
        },
        "resolved_tags": resolved_tags,
        "metrics": summaries,
        "derived": derived,
        "checks": checks,
        "verdict": verdict,
        "scope": (
            "implementation contract only; retrieval efficacy requires the "
            "matched no-mask/uniform/aligned/shuffled screen"
        ),
    }

    print("RSCD-BoQ preflight")
    print(f"Run: {run_dir}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL':4s}  {name}")
    print(f"Verdict: {verdict}")

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Report: {output}")

    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
