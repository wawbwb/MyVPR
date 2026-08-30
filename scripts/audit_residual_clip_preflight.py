#!/usr/bin/env python3
"""Audit the registered 500-step residual-CLIP Phase-A preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_METRICS = (
    "residual_clip_zero_start_max_abs_error",
    "residual_clip_projection_grad_rms",
    "residual_clip_adapter_grad_rms",
    "residual_clip_residual_rms",
    "residual_clip_descriptor_drift_rms",
    "residual_clip_encoder_trainable_parameters",
    "residual_clip_encoder_training",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read TensorBoard scalars and apply the residual-CLIP preflight "
            "stop/go rules."
        )
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="A Lightning version directory or a parent containing one.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--zero-start-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--minimum-last-step",
        type=int,
        default=490,
        help=(
            "Require continuously logged residual/gradient metrics to reach "
            "this optimizer step."
        ),
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


def finite_summary(events) -> dict[str, float | int]:
    values = [float(event.value) for event in events]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("scalar has no finite values")
    return {
        "count": len(values),
        "last_step": int(events[-1].step),
        "last": values[-1],
        "maximum": max(values),
        "minimum": min(values),
    }


def main() -> None:
    args = parse_args()
    if args.minimum_last_step < 0 or args.zero_start_tolerance < 0:
        raise ValueError("audit thresholds must be non-negative")
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
        summaries[metric] = finite_summary(accumulator.Scalars(tag))

    continuously_logged = (
        "residual_clip_residual_rms",
        "residual_clip_adapter_grad_rms",
        "residual_clip_projection_grad_rms",
    )
    checks = {
        "preflight_reached_registered_step": all(
            summaries[name]["last_step"] >= args.minimum_last_step
            for name in continuously_logged
        ),
        "zero_start_reproduces_ru": (
            summaries["residual_clip_zero_start_max_abs_error"]["maximum"]
            <= args.zero_start_tolerance
        ),
        "adapter_gradient_nonzero": (
            summaries["residual_clip_adapter_grad_rms"]["maximum"] > 0
        ),
        # P_C is expected to have zero gradient on the very first backward
        # because W_zero blocks it; the maximum over 500 steps must be nonzero.
        "clip_projection_gradient_nonzero_after_zero_step": (
            summaries["residual_clip_projection_grad_rms"]["maximum"] > 0
        ),
        "residual_nonzero": (
            summaries["residual_clip_residual_rms"]["maximum"] > 0
        ),
        "descriptor_drift_nonzero": (
            summaries["residual_clip_descriptor_drift_rms"]["maximum"] > 0
        ),
        "clip_encoder_frozen": (
            summaries[
                "residual_clip_encoder_trainable_parameters"
            ]["maximum"]
            == 0
        ),
        "clip_encoder_eval": (
            summaries["residual_clip_encoder_training"]["maximum"] == 0
        ),
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": "openvpr_residual_clip_preflight_audit",
        "version": 1,
        "run_dir": str(run_dir),
        "thresholds": {
            "zero_start_tolerance": args.zero_start_tolerance,
            "minimum_last_step": args.minimum_last_step,
        },
        "resolved_tags": resolved_tags,
        "metrics": summaries,
        "checks": checks,
        "verdict": verdict,
    }

    print("DINO-anchor Residual CLIP Phase-A preflight")
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
