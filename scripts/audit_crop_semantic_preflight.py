#!/usr/bin/env python3
"""Audit the preregistered Crop-CLS Semantic FiLM 500-step preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_METRICS = (
    "crop_film_zero_start_max_abs_error",
    "crop_semantic_aligned_minus_wrong_region",
    "crop_semantic_aligned_minus_wrong_place",
    "crop_film_channel_scale_grad_rms",
    "crop_semantic_projection_grad_rms",
    "crop_film_modulation_rms",
    "crop_film_descriptor_drift_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read TensorBoard scalars and apply the Crop-CLS FiLM preflight "
            "stop/go rules."
        )
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="A Lightning version directory or a parent containing one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--zero-start-tolerance", type=float, default=1e-6
    )
    parser.add_argument("--minimum-margin", type=float, default=0.05)
    parser.add_argument(
        "--minimum-last-step",
        type=int,
        default=490,
        help=(
            "Require the continuously logged causal metrics to reach at "
            "least this optimizer step (490 is the safe logging boundary "
            "for a 500-step run with log_every_n_steps=10)."
        ),
    )
    parser.add_argument(
        "--tail-points",
        type=int,
        default=5,
        help="Average this many final logged points for cosine margins.",
    )
    return parser.parse_args()


def newest_event_run(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"log directory not found: {path}")
    events = list(path.rglob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"no TensorBoard event file below: {path}")
    return max(events, key=lambda item: item.stat().st_mtime_ns).parent


def resolve_tag(tags: list[str], requested: str) -> str:
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


def finite_summary(events, tail_points: int) -> dict[str, float | int]:
    values = [float(event.value) for event in events]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("scalar has no finite values")
    tail = values[-min(tail_points, len(values)) :]
    return {
        "count": len(values),
        "last_step": int(events[-1].step),
        "last": values[-1],
        "tail_mean": sum(tail) / len(tail),
        "maximum": max(values),
        "minimum": min(values),
    }


def main() -> None:
    args = parse_args()
    if args.tail_points < 1:
        raise ValueError("--tail-points must be positive")
    if args.minimum_last_step < 0:
        raise ValueError("--minimum-last-step must be non-negative")
    if args.zero_start_tolerance < 0 or args.minimum_margin < 0:
        raise ValueError("thresholds must be non-negative")

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise ImportError(
            "TensorBoard is required: install the project's pinned environment"
        ) from exc

    run_dir = newest_event_run(args.logdir)
    accumulator = EventAccumulator(
        str(run_dir), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])

    summaries = {}
    resolved_tags = {}
    for metric in REQUIRED_METRICS:
        tag = resolve_tag(scalar_tags, metric)
        resolved_tags[metric] = tag
        summaries[metric] = finite_summary(
            accumulator.Scalars(tag), args.tail_points
        )

    checks = {
        "preflight_reached_registered_step": (
            summaries[
                "crop_semantic_aligned_minus_wrong_region"
            ]["last_step"]
            >= args.minimum_last_step
            and summaries[
                "crop_semantic_aligned_minus_wrong_place"
            ]["last_step"]
            >= args.minimum_last_step
        ),
        "zero_start_reproduces_ru": (
            summaries["crop_film_zero_start_max_abs_error"]["maximum"]
            <= args.zero_start_tolerance
        ),
        "aligned_beats_wrong_region": (
            summaries[
                "crop_semantic_aligned_minus_wrong_region"
            ]["tail_mean"]
            >= args.minimum_margin
        ),
        "aligned_beats_wrong_place": (
            summaries[
                "crop_semantic_aligned_minus_wrong_place"
            ]["tail_mean"]
            >= args.minimum_margin
        ),
        "film_gradient_nonzero": (
            summaries["crop_film_channel_scale_grad_rms"]["maximum"] > 0
        ),
        "semantic_gradient_nonzero": (
            summaries["crop_semantic_projection_grad_rms"]["maximum"] > 0
        ),
        "modulation_nonzero": (
            summaries["crop_film_modulation_rms"]["maximum"] > 0
        ),
        "descriptor_drift_nonzero": (
            summaries["crop_film_descriptor_drift_rms"]["maximum"] > 0
        ),
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": "openvpr_crop_cls_film_preflight_audit",
        "version": 1,
        "run_dir": str(run_dir),
        "thresholds": {
            "zero_start_tolerance": args.zero_start_tolerance,
            "minimum_margin": args.minimum_margin,
            "minimum_last_step": args.minimum_last_step,
            "tail_points": args.tail_points,
        },
        "resolved_tags": resolved_tags,
        "metrics": summaries,
        "checks": checks,
        "verdict": verdict,
    }

    print("Crop-CLS Local Semantic FiLM-BoQ preflight")
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
