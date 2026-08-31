from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pytest

from scripts import audit_rscd_preflight as audit


EXPECTED_CORE_METRICS = (
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
EXPECTED_DIAGNOSTIC_METRICS = (
    "rscd_source_candidate_blocks",
    "rscd_donor_candidate_blocks",
    "rscd_masked_nuisance_mean",
    "rscd_quota_zero_fraction",
)


@dataclass(frozen=True)
class _Event:
    step: int
    value: float


def _valid_events(last_step: int = 490) -> dict[str, list[_Event]]:
    values = {
        "rscd_mask_fraction": (0.08, 0.10),
        "rscd_selected_blocks": (8.0, 10.0),
        "rscd_quota_blocks": (8.0, 10.0),
        "rscd_relation_loss": (0.01, 0.02),
        "rscd_pairwise_cos_mae": (0.03, 0.04),
        "rscd_descriptor_drift_rms": (0.002, 0.003),
        "rscd_backbone_grad_rms": (1e-5, 2e-5),
        "rscd_aggregator_grad_rms": (2e-5, 3e-5),
        "rscd_gate_grad_rms": (3e-5, 4e-5),
        "rscd_eval_clean_max_abs_error": (0.0, 0.0),
        "rscd_source_candidate_blocks": (18.0, 20.0),
        "rscd_donor_candidate_blocks": (16.0, 19.0),
        "rscd_masked_nuisance_mean": (0.7, 0.8),
        "rscd_quota_zero_fraction": (0.02, 0.01),
    }
    return {
        name: [_Event(0, pair[0]), _Event(last_step, pair[1])]
        for name, pair in values.items()
    }


def _summaries(events: dict[str, list[_Event]]) -> dict:
    return {
        name: audit.finite_summary(values, tail_points=2)
        for name, values in events.items()
    }


def test_preflight_audit_pins_core_and_diagnostic_metrics() -> None:
    assert audit.CORE_METRICS == EXPECTED_CORE_METRICS
    assert audit.DIAGNOSTIC_METRICS == EXPECTED_DIAGNOSTIC_METRICS
    assert audit.REQUIRED_METRICS == (
        EXPECTED_CORE_METRICS + EXPECTED_DIAGNOSTIC_METRICS
    )
    assert len(set(audit.REQUIRED_METRICS)) == len(audit.REQUIRED_METRICS)


def test_finite_summary_reports_tail_and_full_extent() -> None:
    summary = audit.finite_summary(
        [_Event(20, 0.25), _Event(0, 0.0), _Event(10, 0.5)],
        tail_points=2,
    )
    assert summary == {
        "count": 3,
        "last_step": 20,
        "last": 0.25,
        "tail_mean": 0.375,
        "maximum": 0.5,
        "minimum": 0.0,
    }


@pytest.mark.parametrize(
    "events",
    (
        [],
        [_Event(0, float("nan"))],
        [_Event(0, float("inf"))],
        [_Event(0, float("-inf"))],
    ),
)
def test_finite_summary_rejects_missing_or_nonfinite_values(events) -> None:
    with pytest.raises(ValueError, match="no finite values"):
        audit.finite_summary(events)


def test_resolve_tag_accepts_exact_or_unique_lightning_prefix() -> None:
    tags = ["train/rscd_mask_fraction", "rscd_relation_loss"]
    assert audit.resolve_tag(tags, "rscd_relation_loss") == (
        "rscd_relation_loss"
    )
    assert audit.resolve_tag(tags, "rscd_mask_fraction") == (
        "train/rscd_mask_fraction"
    )


@pytest.mark.parametrize(
    "tags",
    (
        [],
        ["train/rscd_mask_fraction", "epoch/rscd_mask_fraction"],
    ),
)
def test_resolve_tag_fails_closed_for_missing_or_ambiguous_tags(tags) -> None:
    with pytest.raises(KeyError, match="not found uniquely"):
        audit.resolve_tag(tags, "rscd_mask_fraction")


def test_paired_difference_requires_identical_steps_and_supports_scale() -> None:
    left = [_Event(0, 0.08), _Event(10, 0.10)]
    blocks = [_Event(0, 8.0), _Event(10, 10.0)]
    assert audit.paired_max_abs_difference(
        left, blocks, right_scale=0.01
    ) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="exactly the same optimizer steps"):
        audit.paired_max_abs_difference(left, blocks[:-1])


def test_valid_preflight_metrics_pass_every_contract_check() -> None:
    events = _valid_events()
    checks, derived = audit.evaluate_checks(_summaries(events), events)

    assert checks
    assert all(checks.values())
    assert derived["selected_vs_quota_max_abs_error"] == pytest.approx(0.0)
    assert derived[
        "mask_fraction_vs_block_accounting_max_abs_error"
    ] == pytest.approx(0.0)
    assert derived["block_fraction"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "metric, replacement, failed_check",
    (
        (
            "rscd_quota_blocks",
            [_Event(0, 7.0), _Event(490, 9.0)],
            "selected_blocks_match_quota",
        ),
        (
            "rscd_mask_fraction",
            [_Event(0, 0.08), _Event(490, 0.16)],
            "mask_fraction_nonzero_and_bounded",
        ),
        (
            "rscd_relation_loss",
            [_Event(0, 0.0), _Event(490, 0.0)],
            "relation_loss_nonzero",
        ),
        (
            "rscd_backbone_grad_rms",
            [_Event(0, 0.0), _Event(490, 0.0)],
            "backbone_gradient_nonzero",
        ),
        (
            "rscd_eval_clean_max_abs_error",
            [_Event(0, 2e-6), _Event(490, 2e-6)],
            "evaluation_path_is_clean",
        ),
        (
            "rscd_masked_nuisance_mean",
            [_Event(0, -0.01), _Event(490, 1.01)],
            "masked_nuisance_in_unit_interval",
        ),
    ),
)
def test_contract_checks_fail_on_mechanical_or_gradient_errors(
    metric: str, replacement: list[_Event], failed_check: str
) -> None:
    events = _valid_events()
    events[metric] = replacement
    checks, _ = audit.evaluate_checks(_summaries(events), events)
    assert checks[failed_check] is False


def test_preflight_must_reach_the_registered_logging_boundary() -> None:
    events = _valid_events(last_step=480)
    checks, _ = audit.evaluate_checks(_summaries(events), events)
    assert checks["preflight_reached_registered_step"] is False


def test_evaluate_checks_rejects_a_missing_metric() -> None:
    events = _valid_events()
    summaries = _summaries(events)
    summaries.pop("rscd_gate_grad_rms")
    with pytest.raises(ValueError, match="metric set mismatch"):
        audit.evaluate_checks(summaries, events)


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
