#!/usr/bin/env python
"""Audit AG-SLRD teacher complementarity from precomputed descriptors.

No model or dataset loader is imported.  Each descriptor file is a 2-D NPY
array in the immutable order ``all references, then all queries``.  Positives
are supplied as the usual object NPY (one integer reference-index array per
query) or as a JSON list of lists.  The script re-normalises descriptors,
performs exact cosine retrieval in bounded query chunks, and compares frozen
RU against the aligned teacher/aligned layout, aligned teacher/shuffled
layout, and shuffled-trained teacher/aligned layout outcomes.

For the registered MSLS screen, keep the defaults: 740 queries, at least eight
semantic-only corrections, and a 5% aligned-teacher rank-advantage rate.  For
a training/held-out split with a different query count pass
``--expected-queries 0``; the same 5% diagnostic remains explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_layout_cache import file_sha256  # noqa: E402


AUDIT_SCHEMA = "openvpr_ag_slrd_complementarity_audit"
AUDIT_VERSION = 1


@dataclass(frozen=True)
class RetrievalResult:
    top1: np.ndarray
    hits_at_1: np.ndarray
    hits_at_5: np.ndarray
    hits_at_10: np.ndarray
    first_positive_rank: np.ndarray
    positive_negative_margin: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_descriptor_sidecar(
    path: Path,
    *,
    expected_kind: str,
    expected_training_mode: str | None = None,
    expected_selection: str | None = None,
) -> dict[str, Any]:
    """Bind each descriptor matrix to the registered extraction variant."""

    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"descriptor provenance sidecar not found: {sidecar}")
    with sidecar.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema") != "openvpr_ag_slrd_msls_descriptors":
        raise ValueError(f"unsupported descriptor sidecar schema: {sidecar}")
    if record.get("version") != 1 or record.get("complete") is not True:
        raise ValueError(f"descriptor sidecar is incomplete/unsupported: {sidecar}")
    if record.get("kind") != expected_kind:
        raise ValueError(
            f"descriptor kind mismatch for {path}: {record.get('kind')!r}"
        )
    if record.get("descriptor_sha256") != file_sha256(path):
        raise ValueError(f"descriptor SHA256 disagrees with sidecar: {path}")
    if (
        expected_training_mode is not None
        and record.get("teacher_training_mode") != expected_training_mode
    ):
        raise ValueError(
            f"teacher training mode mismatch for {path}: "
            f"{record.get('teacher_training_mode')!r}"
        )
    if (
        expected_selection is not None
        and record.get("layout_selection") != expected_selection
    ):
        raise ValueError(
            f"layout selection mismatch for {path}: "
            f"{record.get('layout_selection')!r}"
        )
    return record


def validate_teacher_run(path: Path, *, expected_mode: str) -> dict[str, Any]:
    """Validate the final-epoch teacher record used by descriptor extraction."""

    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema") != "openvpr_ag_slrd_semantic_teacher_run":
        raise ValueError(f"unsupported semantic teacher run schema: {path}")
    if record.get("version") != 1 or record.get("complete") is not True:
        raise ValueError(f"semantic teacher run is incomplete/unsupported: {path}")
    if record.get("mode") != expected_mode:
        raise ValueError(
            f"semantic teacher run mode mismatch: {record.get('mode')!r}"
        )
    metrics = record.get("final_metrics")
    checkpoint = record.get("checkpoint")
    if not isinstance(metrics, dict) or not isinstance(checkpoint, dict):
        raise ValueError(f"semantic teacher run lacks metrics/checkpoint: {path}")
    holdout_r1 = metrics.get("holdout_batch_r1")
    if not isinstance(holdout_r1, (int, float)) or not math.isfinite(holdout_r1):
        raise ValueError(f"semantic teacher run has invalid holdout R@1: {path}")
    if not 0.0 <= float(holdout_r1) <= 1.0:
        raise ValueError(f"semantic teacher holdout R@1 is outside [0,1]: {path}")
    checkpoint_hash = checkpoint.get("sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError(f"semantic teacher run has invalid checkpoint hash: {path}")
    return record


def load_positives(path: str | Path, *, num_queries: int, num_references: int) -> tuple[np.ndarray, ...]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    elif path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=True)
    else:
        raise ValueError("positives must be a .npy or .json file")
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != num_queries:
        raise ValueError(
            f"positives must contain {num_queries} query rows, found "
            f"{len(raw) if hasattr(raw, '__len__') else 'non-sequence'}"
        )
    result: list[np.ndarray] = []
    for query_index, values in enumerate(raw):
        positives = np.asarray(values)
        if positives.ndim != 1 or positives.dtype.kind not in "ui":
            raise ValueError(f"positives[{query_index}] must be a 1-D integer array")
        positives = positives.astype(np.int64, copy=False)
        if positives.size == 0:
            raise ValueError(f"positives[{query_index}] is empty")
        if np.any(positives < 0) or np.any(positives >= num_references):
            raise ValueError(f"positives[{query_index}] contains an invalid reference")
        if np.unique(positives).size != positives.size:
            raise ValueError(f"positives[{query_index}] contains duplicates")
        result.append(np.sort(positives))
    return tuple(result)


def load_and_normalize_descriptors(
    path: str | Path,
    *,
    expected_rows: int,
) -> np.ndarray:
    path = Path(path)
    descriptors = np.load(path, mmap_mode="r", allow_pickle=False)
    if descriptors.ndim != 2 or descriptors.shape[0] != expected_rows:
        raise ValueError(
            f"descriptor file {path} must have shape ({expected_rows}, D), "
            f"found {descriptors.shape}"
        )
    if descriptors.dtype.kind != "f":
        raise TypeError(f"descriptor file {path} must use a floating dtype")
    values = np.asarray(descriptors, dtype=np.float32)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"descriptor file {path} contains non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not bool(np.isfinite(norms).all()):
        raise ValueError(f"descriptor file {path} contains a zero/invalid row")
    return values / norms


def search_descriptors(
    descriptors: np.ndarray,
    *,
    num_references: int,
    positives: tuple[np.ndarray, ...],
    query_chunk_size: int = 64,
) -> RetrievalResult:
    """Compute exact cosine ranks/margins without a FAISS dependency."""

    descriptors = np.asarray(descriptors)
    if descriptors.ndim != 2 or descriptors.dtype.kind != "f":
        raise ValueError("descriptors must be a 2-D floating array")
    if not 1 < num_references < descriptors.shape[0]:
        raise ValueError("reference/query partitions must both be non-empty")
    num_queries = descriptors.shape[0] - num_references
    if len(positives) != num_queries:
        raise ValueError("positive rows do not match descriptor queries")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")

    references = descriptors[:num_references]
    queries = descriptors[num_references:]
    top1 = np.empty(num_queries, dtype=np.int64)
    hits1 = np.zeros(num_queries, dtype=np.bool_)
    hits5 = np.zeros(num_queries, dtype=np.bool_)
    hits10 = np.zeros(num_queries, dtype=np.bool_)
    ranks = np.empty(num_queries, dtype=np.int64)
    margins = np.empty(num_queries, dtype=np.float32)
    reference_indices = np.arange(num_references, dtype=np.int64)

    for start in range(0, num_queries, query_chunk_size):
        stop = min(start + query_chunk_size, num_queries)
        scores = queries[start:stop] @ references.T
        if not bool(np.isfinite(scores).all()):
            raise ValueError("cosine search produced non-finite scores")
        for local_index, query_index in enumerate(range(start, stop)):
            row = scores[local_index]
            positive_indices = positives[query_index]
            positive_scores = row[positive_indices]
            best_positive = float(positive_scores.max())
            best_positive_indices = positive_indices[
                positive_scores == best_positive
            ]
            best_positive_index = int(best_positive_indices.min())
            positive_mask = np.zeros(num_references, dtype=np.bool_)
            positive_mask[positive_indices] = True
            negative_scores = row[~positive_mask]
            if negative_scores.size == 0:
                raise ValueError("each query needs at least one negative reference")
            hardest_negative = float(negative_scores.max())
            prediction = int(np.argmax(row))
            top1[query_index] = prediction
            hits1[query_index] = bool(positive_mask[prediction])
            # argpartition avoids sorting the full DB.  Ties are practically
            # absent for learned descriptors; membership remains deterministic.
            for k, target in ((5, hits5), (10, hits10)):
                effective_k = min(k, num_references)
                candidates = np.argpartition(row, -effective_k)[-effective_k:]
                target[query_index] = bool(np.any(positive_mask[candidates]))
            # Match NumPy/FAISS deterministic lowest-index tie handling.  A
            # collapsed teacher must not receive rank 1 merely because every
            # database score is exactly equal.
            ranks[query_index] = 1 + int(np.count_nonzero(row > best_positive))
            ranks[query_index] += int(
                np.count_nonzero(
                    (row == best_positive)
                    & (reference_indices < best_positive_index)
                )
            )
            margins[query_index] = best_positive - hardest_negative
    return RetrievalResult(top1, hits1, hits5, hits10, ranks, margins)


def _variant_summary(result: RetrievalResult) -> dict[str, Any]:
    query_count = int(result.hits_at_1.size)
    return {
        "num_queries": query_count,
        "correct@1": int(result.hits_at_1.sum()),
        "correct@5": int(result.hits_at_5.sum()),
        "correct@10": int(result.hits_at_10.sum()),
        "r@1": float(result.hits_at_1.mean()),
        "r@5": float(result.hits_at_5.mean()),
        "r@10": float(result.hits_at_10.mean()),
        "first_positive_rank_mean": float(result.first_positive_rank.mean()),
        "first_positive_rank_p50": float(np.median(result.first_positive_rank)),
        "margin_mean": float(result.positive_negative_margin.mean()),
        "margin_positive_rate": float(
            np.mean(result.positive_negative_margin > 0.0)
        ),
    }


def paired_summary(
    left: RetrievalResult,
    right: RetrievalResult,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    if left.hits_at_1.shape != right.hits_at_1.shape:
        raise ValueError("paired retrieval results have different query counts")
    left_only = left.hits_at_1 & ~right.hits_at_1
    right_only = ~left.hits_at_1 & right.hits_at_1
    left_rank_better = left.first_positive_rank < right.first_positive_rank
    right_rank_better = right.first_positive_rank < left.first_positive_rank
    return {
        "left": left_name,
        "right": right_name,
        "both_correct@1": int(np.sum(left.hits_at_1 & right.hits_at_1)),
        "both_wrong@1": int(np.sum(~left.hits_at_1 & ~right.hits_at_1)),
        "left_only_correct@1": int(left_only.sum()),
        "right_only_correct@1": int(right_only.sum()),
        "oracle_union_correct@1": int(np.sum(left.hits_at_1 | right.hits_at_1)),
        "left_better_rank": int(left_rank_better.sum()),
        "right_better_rank": int(right_rank_better.sum()),
        "rank_tied": int(
            np.sum(left.first_positive_rank == right.first_positive_rank)
        ),
        "left_better_rank_rate": float(left_rank_better.mean()),
        "mean_rank_advantage_left": float(
            np.mean(
                right.first_positive_rank.astype(np.float64)
                - left.first_positive_rank.astype(np.float64)
            )
        ),
        "mean_margin_advantage_left": float(
            np.mean(
                left.positive_negative_margin
                - right.positive_negative_margin
            )
        ),
        "top1_reference_changed": int(np.sum(left.top1 != right.top1)),
    }


def build_audit(
    *,
    ru: RetrievalResult,
    aligned: RetrievalResult,
    wrong_layout: RetrievalResult,
    shuffled_teacher: RetrievalResult,
    min_semantic_only: int,
    min_teacher_better_rate: float,
    expected_ru_correct: int,
    aligned_holdout_batch_r1: float,
    shuffled_holdout_batch_r1: float,
) -> dict[str, Any]:
    aligned_vs_ru = paired_summary(
        aligned, ru, left_name="aligned_semantic", right_name="ru"
    )
    wrong_layout_vs_ru = paired_summary(
        wrong_layout, ru, left_name="wrong_layout", right_name="ru"
    )
    shuffled_teacher_vs_ru = paired_summary(
        shuffled_teacher, ru, left_name="shuffled_teacher", right_name="ru"
    )
    aligned_vs_wrong_layout = paired_summary(
        aligned,
        wrong_layout,
        left_name="aligned_semantic",
        right_name="wrong_layout",
    )
    aligned_vs_shuffled_teacher = paired_summary(
        aligned,
        shuffled_teacher,
        left_name="aligned_semantic",
        right_name="shuffled_teacher",
    )
    semantic_only = int(aligned_vs_ru["left_only_correct@1"])
    wrong_layout_only = int(wrong_layout_vs_ru["left_only_correct@1"])
    shuffled_teacher_only = int(shuffled_teacher_vs_ru["left_only_correct@1"])
    teacher_better_rate = float(aligned_vs_ru["left_better_rank_rate"])
    checks = {
        "ru_baseline_reproduced": {
            "pass": (
                expected_ru_correct <= 0
                or int(ru.hits_at_1.sum()) == expected_ru_correct
            ),
            "value": int(ru.hits_at_1.sum()),
            "threshold": expected_ru_correct,
            "registered_msls_value": "675/740 (91.22%)",
        },
        "semantic_only_at_least_registered_minimum": {
            "pass": semantic_only >= min_semantic_only,
            "value": semantic_only,
            "threshold": min_semantic_only,
            "registered_msls_threshold": "8/740",
        },
        "teacher_better_pair_rate_at_least_5pct": {
            "pass": teacher_better_rate >= min_teacher_better_rate,
            "value": teacher_better_rate,
            "threshold": min_teacher_better_rate,
        },
        "aligned_enriches_ru_errors_over_both_controls": {
            "pass": (
                semantic_only > wrong_layout_only
                and semantic_only > shuffled_teacher_only
            ),
            "aligned_semantic_only": semantic_only,
            "wrong_layout_only": wrong_layout_only,
            "shuffled_teacher_only": shuffled_teacher_only,
        },
        "aligned_ranks_positives_better_than_both_controls": {
            "pass": (
                aligned_vs_wrong_layout["left_better_rank"]
                > aligned_vs_wrong_layout["right_better_rank"]
                and aligned_vs_shuffled_teacher["left_better_rank"]
                > aligned_vs_shuffled_teacher["right_better_rank"]
            ),
            "vs_wrong_layout": {
                "aligned_better": aligned_vs_wrong_layout["left_better_rank"],
                "control_better": aligned_vs_wrong_layout["right_better_rank"],
            },
            "vs_shuffled_teacher": {
                "aligned_better": aligned_vs_shuffled_teacher["left_better_rank"],
                "control_better": aligned_vs_shuffled_teacher["right_better_rank"],
            },
        },
        "aligned_teacher_beats_shuffled_on_gsv_holdout": {
            "pass": aligned_holdout_batch_r1 > shuffled_holdout_batch_r1,
            "aligned_holdout_batch_r1": aligned_holdout_batch_r1,
            "shuffled_holdout_batch_r1": shuffled_holdout_batch_r1,
            "metric_definition": (
                "mean within-batch R@1 over the fixed SHA256 place holdout"
            ),
        },
    }
    passed = all(bool(check["pass"]) for check in checks.values())
    return {
        "variants": {
            "ru": _variant_summary(ru),
            "aligned_semantic": _variant_summary(aligned),
            "wrong_layout": _variant_summary(wrong_layout),
            "shuffled_teacher": _variant_summary(shuffled_teacher),
        },
        "paired": {
            "aligned_vs_ru": aligned_vs_ru,
            "wrong_layout_vs_ru": wrong_layout_vs_ru,
            "shuffled_teacher_vs_ru": shuffled_teacher_vs_ru,
            "aligned_vs_wrong_layout": aligned_vs_wrong_layout,
            "aligned_vs_shuffled_teacher": aligned_vs_shuffled_teacher,
        },
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
    }


def write_per_query_csv(
    path: Path,
    *,
    positives: tuple[np.ndarray, ...],
    results: dict[str, RetrievalResult],
) -> None:
    fields = ["query_index", "num_positives"]
    for name in results:
        fields.extend(
            (
                f"{name}_top1",
                f"{name}_correct@1",
                f"{name}_first_positive_rank",
                f"{name}_positive_negative_margin",
            )
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_index, query_positives in enumerate(positives):
            row: dict[str, Any] = {
                "query_index": query_index,
                "num_positives": len(query_positives),
            }
            for name, result in results.items():
                row[f"{name}_top1"] = int(result.top1[query_index])
                row[f"{name}_correct@1"] = int(result.hits_at_1[query_index])
                row[f"{name}_first_positive_rank"] = int(
                    result.first_positive_rank[query_index]
                )
                row[f"{name}_positive_negative_margin"] = float(
                    result.positive_negative_margin[query_index]
                )
            writer.writerow(row)


def write_summary_csv(path: Path, audit: dict[str, Any]) -> None:
    fields = (
        "variant", "num_queries", "correct@1", "r@1", "correct@5", "r@5",
        "correct@10", "r@10", "first_positive_rank_mean", "margin_mean",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in audit["variants"].items():
            writer.writerow({"variant": name, **{field: values[field] for field in fields[1:]}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ru-descriptors", type=Path, required=True)
    parser.add_argument("--aligned-descriptors", type=Path, required=True)
    parser.add_argument("--wrong-layout-descriptors", type=Path, required=True)
    parser.add_argument("--shuffled-teacher-descriptors", type=Path, required=True)
    parser.add_argument("--aligned-run", type=Path, required=True)
    parser.add_argument("--shuffled-run", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--num-references", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-name", default="msls-val")
    parser.add_argument("--expected-queries", type=int, default=740)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--min-semantic-only", type=int, default=8)
    parser.add_argument("--min-teacher-better-rate", type=float, default=0.05)
    parser.add_argument(
        "--expected-ru-correct",
        type=int,
        default=675,
        help="registered RU R@1 count; use 0 only for a non-MSLS diagnostic",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    input_paths = (
        args.ru_descriptors,
        args.aligned_descriptors,
        args.wrong_layout_descriptors,
        args.shuffled_teacher_descriptors,
        args.aligned_run,
        args.shuffled_run,
        args.positives,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"audit input not found: {path}")
    if len({path.resolve() for path in input_paths[:4]}) != 4:
        raise ValueError("all four descriptor files must be distinct")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    if args.num_references < 2 or args.query_chunk_size < 1:
        raise ValueError("num-references must be >=2 and query-chunk-size positive")
    if args.expected_queries < 0:
        raise ValueError("expected-queries must be non-negative (0 disables check)")
    if args.min_semantic_only < 0:
        raise ValueError("min-semantic-only must be non-negative")
    if args.expected_ru_correct < 0:
        raise ValueError("expected-ru-correct must be non-negative")
    if not math.isfinite(args.min_teacher_better_rate) or not (
        0.0 <= args.min_teacher_better_rate <= 1.0
    ):
        raise ValueError("min-teacher-better-rate must be finite and in [0,1]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    sidecars = {
        "ru": validate_descriptor_sidecar(
            args.ru_descriptors, expected_kind="ru"
        ),
        "aligned_semantic": validate_descriptor_sidecar(
            args.aligned_descriptors,
            expected_kind="semantic_layout",
            expected_training_mode="aligned",
            expected_selection="aligned",
        ),
        "wrong_layout": validate_descriptor_sidecar(
            args.wrong_layout_descriptors,
            expected_kind="semantic_layout",
            expected_training_mode="aligned",
            expected_selection="shuffled",
        ),
        "shuffled_teacher": validate_descriptor_sidecar(
            args.shuffled_teacher_descriptors,
            expected_kind="semantic_layout",
            expected_training_mode="shuffled",
            expected_selection="aligned",
        ),
    }
    teacher_runs = {
        "aligned": validate_teacher_run(args.aligned_run, expected_mode="aligned"),
        "shuffled": validate_teacher_run(
            args.shuffled_run, expected_mode="shuffled"
        ),
    }
    if (
        sidecars["aligned_semantic"]["checkpoint"]["sha256"]
        != teacher_runs["aligned"]["checkpoint"]["sha256"]
        or sidecars["wrong_layout"]["checkpoint"]["sha256"]
        != teacher_runs["aligned"]["checkpoint"]["sha256"]
    ):
        raise ValueError("aligned descriptor checkpoint differs from aligned run")
    if (
        sidecars["shuffled_teacher"]["checkpoint"]["sha256"]
        != teacher_runs["shuffled"]["checkpoint"]["sha256"]
    ):
        raise ValueError("shuffled descriptor checkpoint differs from shuffled run")
    ru_raw = np.load(args.ru_descriptors, mmap_mode="r", allow_pickle=False)
    if ru_raw.ndim != 2 or ru_raw.shape[0] <= args.num_references:
        raise ValueError("RU descriptors do not contain both reference/query rows")
    total_rows = int(ru_raw.shape[0])
    num_queries = total_rows - args.num_references
    del ru_raw
    if args.expected_queries and num_queries != args.expected_queries:
        raise ValueError(
            f"split has {num_queries} queries, expected {args.expected_queries}; "
            "use --expected-queries 0 only for a registered non-MSLS split"
        )
    positives = load_positives(
        args.positives,
        num_queries=num_queries,
        num_references=args.num_references,
    )
    paths = {
        "ru": args.ru_descriptors,
        "aligned_semantic": args.aligned_descriptors,
        "wrong_layout": args.wrong_layout_descriptors,
        "shuffled_teacher": args.shuffled_teacher_descriptors,
    }
    results: dict[str, RetrievalResult] = {}
    for name, path in paths.items():
        print(f"Evaluate precomputed descriptors: {name} <- {path}")
        descriptors = load_and_normalize_descriptors(path, expected_rows=total_rows)
        results[name] = search_descriptors(
            descriptors,
            num_references=args.num_references,
            positives=positives,
            query_chunk_size=args.query_chunk_size,
        )
        del descriptors
    audit = build_audit(
        ru=results["ru"],
        aligned=results["aligned_semantic"],
        wrong_layout=results["wrong_layout"],
        shuffled_teacher=results["shuffled_teacher"],
        min_semantic_only=args.min_semantic_only,
        min_teacher_better_rate=args.min_teacher_better_rate,
        expected_ru_correct=args.expected_ru_correct,
        aligned_holdout_batch_r1=float(
            teacher_runs["aligned"]["final_metrics"]["holdout_batch_r1"]
        ),
        shuffled_holdout_batch_r1=float(
            teacher_runs["shuffled"]["final_metrics"]["holdout_batch_r1"]
        ),
    )
    audit.update(
        {
            "schema": AUDIT_SCHEMA,
            "version": AUDIT_VERSION,
            "created_utc": utc_now(),
            "split_name": args.split_name,
            "num_references": args.num_references,
            "num_queries": num_queries,
            "registered_thresholds": {
                "msls_semantic_only": "at least 8/740 queries",
                "teacher_better_pair_rate": "at least 5%",
                "ru_baseline": "exactly 675/740 R@1 queries",
            },
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for name, path in paths.items()
            }
            | {
                "positives": {
                    "path": str(args.positives.resolve()),
                    "sha256": file_sha256(args.positives),
                }
            },
            "descriptor_sidecars": sidecars,
            "teacher_runs": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "record": teacher_runs[name],
                }
                for name, path in {
                    "aligned": args.aligned_run,
                    "shuffled": args.shuffled_run,
                }.items()
            },
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "summary.json", audit)
    write_summary_csv(args.output / "summary.csv", audit)
    write_per_query_csv(
        args.output / "per_query.csv", positives=positives, results=results
    )
    verdict_lines = [
        f"AG-SLRD Phase-0 complementarity audit: {audit['verdict']}",
        f"split={args.split_name}; references={args.num_references}; queries={num_queries}",
    ]
    for name, check in audit["checks"].items():
        verdict_lines.append(f"{'PASS' if check['pass'] else 'FAIL'}  {name}: {check}")
    (args.output / "verdict.txt").write_text(
        "\n".join(verdict_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(verdict_lines))
    print(f"Results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
