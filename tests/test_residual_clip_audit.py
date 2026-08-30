from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pytest

from scripts import audit_residual_clip_preflight as audit


EXPECTED_METRICS = (
    "residual_clip_zero_start_max_abs_error",
    "residual_clip_projection_grad_rms",
    "residual_clip_adapter_grad_rms",
    "residual_clip_residual_rms",
    "residual_clip_descriptor_drift_rms",
    "residual_clip_encoder_trainable_parameters",
    "residual_clip_encoder_training",
)


@dataclass(frozen=True)
class _Event:
    step: int
    value: float


def test_preflight_audit_requires_every_registered_safety_metric() -> None:
    assert audit.REQUIRED_METRICS == EXPECTED_METRICS
    assert len(set(audit.REQUIRED_METRICS)) == len(audit.REQUIRED_METRICS)


def test_finite_summary_reports_complete_scalar_extent() -> None:
    summary = audit.finite_summary(
        [_Event(step=0, value=0.0), _Event(step=490, value=0.25)]
    )

    assert summary == {
        "count": 2,
        "last_step": 490,
        "last": 0.25,
        "maximum": 0.25,
        "minimum": 0.0,
    }


@pytest.mark.parametrize(
    "events",
    (
        [],
        [_Event(step=0, value=float("nan"))],
        [_Event(step=0, value=float("inf"))],
        [_Event(step=0, value=float("-inf"))],
    ),
)
def test_finite_summary_rejects_missing_or_nonfinite_metrics(events) -> None:
    with pytest.raises(ValueError, match="no finite values"):
        audit.finite_summary(events)


def test_resolve_tag_accepts_exact_or_unique_lightning_prefix() -> None:
    tags = [
        "train/residual_clip_residual_rms",
        "residual_clip_adapter_grad_rms",
    ]

    assert audit.resolve_tag(
        tags, "residual_clip_adapter_grad_rms"
    ) == "residual_clip_adapter_grad_rms"
    assert audit.resolve_tag(
        tags, "residual_clip_residual_rms"
    ) == "train/residual_clip_residual_rms"


@pytest.mark.parametrize(
    "tags, requested",
    (
        ([], "residual_clip_residual_rms"),
        (
            [
                "train/residual_clip_residual_rms",
                "epoch/residual_clip_residual_rms",
            ],
            "residual_clip_residual_rms",
        ),
    ),
)
def test_resolve_tag_fails_closed_for_missing_or_ambiguous_metrics(
    tags, requested
) -> None:
    with pytest.raises(KeyError, match="not found uniquely"):
        audit.resolve_tag(tags, requested)


def test_newest_event_run_selects_latest_tensorboard_parent(
    tmp_path: Path,
) -> None:
    old_run = tmp_path / "version_0"
    new_run = tmp_path / "version_1"
    old_run.mkdir()
    new_run.mkdir()
    old_event = old_run / "events.out.tfevents.old"
    new_event = new_run / "events.out.tfevents.new"
    old_event.write_bytes(b"old")
    new_event.write_bytes(b"new")
    os.utime(old_event, ns=(1_000, 1_000))
    os.utime(new_event, ns=(2_000, 2_000))

    assert audit.newest_event_run(tmp_path) == new_run.resolve()


def test_newest_event_run_rejects_missing_path_or_event(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="log directory not found"):
        audit.newest_event_run(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="no TensorBoard event file"):
        audit.newest_event_run(tmp_path)
