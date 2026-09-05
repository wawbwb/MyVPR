"""Deterministic primitives for the CC-LSA Pair-VPR Gate-A audit.

The module is deliberately NumPy-only.  Model extraction and cache I/O live in
scripts, while every causal comparison and tie rule is unit-testable here.
Nothing in this file is allowed to calibrate a threshold from MSLS labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CC_LSA_CANDIDATE_SCHEMA = "openvpr_cc_lsa_ru_candidates"
CC_LSA_FEATURE_SCHEMA = "openvpr_cc_lsa_msls_features"
CC_LSA_CALIBRATION_SCHEMA = "openvpr_cc_lsa_gsv_calibration"
CC_LSA_AUDIT_SCHEMA = "openvpr_cc_lsa_gate_a_audit"
CC_LSA_VERSION = 1


def implementation_sha256() -> dict[str, str]:
    """Bind caches to the actual implementation, including dirty worktrees."""
    root = Path(__file__).resolve().parents[1]
    names = (
        "src/cc_lsa_gate_a.py", "src/cc_lsa_config.py", "src/cc_lsa_features.py",
        "src/models/cc_lsa.py", "src/dataloaders/train/cc_lsa.py",
        "src/models/clip_teacher.py", "src/models/backbones/dinov2.py",
        "scripts/cache_gsv_cc_lsa_targets.py", "scripts/train_cc_lsa_teacher.py",
        "scripts/calibrate_cc_lsa_gate_a.py", "scripts/cache_cc_lsa_msls_features.py",
        "scripts/audit_cc_lsa_gate_a.py", "scripts/extract_ag_slrd_msls_descriptors.py",
        "scripts/eval_condition_robustness.py",
    )
    # Canonical LF permits an unchanged source checkout on Windows/Linux.
    return {
        name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in names
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def string_sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).replace("\\", "/").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _normalise_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if bool(np.any(norm <= 1e-12)):
        raise ValueError(f"{name} contains a zero-norm row")
    return value / norm


def stable_ru_topk(
    descriptors: np.ndarray,
    *,
    num_references: int,
    top_k: int = 100,
    query_chunk_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k with ties resolved by the lower database index."""

    descriptors = _normalise_rows(descriptors, name="descriptors")
    if not 0 < num_references < len(descriptors):
        raise ValueError("num_references must split references from queries")
    if not 0 < top_k <= num_references:
        raise ValueError("top_k must be in [1, num_references]")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    references = descriptors[:num_references]
    queries = descriptors[num_references:]
    indices = np.arange(num_references, dtype=np.int64)
    candidates = np.empty((len(queries), top_k), dtype=np.int32)
    candidate_scores = np.empty((len(queries), top_k), dtype=np.float32)
    for start in range(0, len(queries), query_chunk_size):
        scores = queries[start : start + query_chunk_size] @ references.T
        for offset, row in enumerate(scores):
            order = np.lexsort((indices, -row.astype(np.float64, copy=False)))
            selected = order[:top_k]
            candidates[start + offset] = selected.astype(np.int32)
            candidate_scores[start + offset] = row[selected]
    return candidates, candidate_scores


def validate_positives(
    positives: Sequence[Iterable[int]],
    *,
    num_queries: int,
    num_references: int,
) -> tuple[np.ndarray, ...]:
    if len(positives) != num_queries:
        raise ValueError("positive row count does not match num_queries")
    result: list[np.ndarray] = []
    for query_index, values in enumerate(positives):
        row = np.asarray(list(values))
        if row.ndim != 1 or row.dtype.kind not in "iu" or row.size < 1:
            raise ValueError(f"positives[{query_index}] must be non-empty integers")
        row = np.unique(row.astype(np.int64, copy=False))
        if bool(np.any(row < 0)) or bool(np.any(row >= num_references)):
            raise ValueError(f"positives[{query_index}] is outside reference range")
        result.append(row)
    return tuple(result)


def ru_candidate_diagnostics(
    candidates: np.ndarray,
    positives: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    candidates = np.asarray(candidates)
    if candidates.ndim != 2 or len(candidates) != len(positives):
        raise ValueError("candidate matrix and positives disagree")
    ru_rank = np.full(len(candidates), candidates.shape[1] + 1, dtype=np.int16)
    for query_index, (row, positive_row) in enumerate(zip(candidates, positives)):
        mask = np.isin(row, positive_row, assume_unique=False)
        positions = np.flatnonzero(mask)
        if positions.size:
            ru_rank[query_index] = int(positions[0]) + 1
    return {
        "best_positive_rank": ru_rank,
        "hit_at_1": ru_rank <= 1,
        "hit_at_10": ru_rank <= min(10, candidates.shape[1]),
        "reachable": ru_rank <= candidates.shape[1],
    }


def mutual_nearest_edges(
    query_tokens: np.ndarray,
    candidate_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic mutual nearest-neighbour token edges.

    ``np.argmax`` selects the smaller token index on a tie, which is the
    registered tie rule.
    """

    query = _normalise_rows(query_tokens, name="query_tokens")
    candidate = _normalise_rows(candidate_tokens, name="candidate_tokens")
    if query.shape[1] != candidate.shape[1]:
        raise ValueError("query/candidate token dimensions differ")
    similarity = query @ candidate.T
    query_to_candidate = similarity.argmax(axis=1)
    candidate_to_query = similarity.argmax(axis=0)
    query_indices = np.arange(len(query), dtype=np.int32)
    mutual = candidate_to_query[query_to_candidate] == query_indices
    left = query_indices[mutual]
    right = query_to_candidate[mutual].astype(np.int32, copy=False)
    values = similarity[left, right].astype(np.float32, copy=False)
    return left, right, values


def semantic_edge_weights(
    query_semantics: np.ndarray,
    candidate_semantics: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    tau_sem: float,
) -> np.ndarray:
    if not math.isfinite(float(tau_sem)) or not -1.0 < float(tau_sem) < 1.0:
        raise ValueError("tau_sem must be finite and in (-1,1)")
    query = _normalise_rows(query_semantics, name="query_semantics")
    candidate = _normalise_rows(
        candidate_semantics, name="candidate_semantics"
    )
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("edge indices must be matching 1-D arrays")
    if bool(np.any(left < 0)) or bool(np.any(left >= len(query))):
        raise ValueError("left edge index is outside the query map")
    if bool(np.any(right < 0)) or bool(np.any(right >= len(candidate))):
        raise ValueError("right edge index is outside the candidate map")
    cosine = np.einsum("nd,nd->n", query[left], candidate[right])
    weights = (cosine - float(tau_sem)) / (1.0 - float(tau_sem))
    return np.clip(weights, 0.0, 1.0).astype(np.float32, copy=False)


def top_l_zero_padded_score(values: np.ndarray, *, top_l: int = 20) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or top_l < 1:
        raise ValueError("values must be 1-D and top_l positive")
    if not bool(np.isfinite(values).all()):
        raise ValueError("pair edge values contain non-finite entries")
    if values.size == 0:
        return 0.0
    selected = np.sort(values)[-min(top_l, values.size) :]
    return float(selected.sum(dtype=np.float64) / top_l)


def dino_full_score(visual_cosine: np.ndarray, *, top_l: int = 20) -> float:
    visual_cosine = np.asarray(visual_cosine, dtype=np.float32)
    return top_l_zero_padded_score((visual_cosine + 1.0) * 0.5, top_l=top_l)


def semantic_pair_score(
    visual_cosine: np.ndarray,
    semantic_weights: np.ndarray,
    *,
    top_l: int = 20,
) -> float:
    visual = np.asarray(visual_cosine, dtype=np.float32)
    weights = np.asarray(semantic_weights, dtype=np.float32)
    if visual.shape != weights.shape or visual.ndim != 1:
        raise ValueError("visual cosine and semantic weights must match")
    if bool(np.any(weights < 0.0)) or bool(np.any(weights > 1.0)):
        raise ValueError("semantic weights must lie in [0,1]")
    return top_l_zero_padded_score(
        ((visual + 1.0) * 0.5) * weights, top_l=top_l
    )


def deterministic_permutation(
    length: int, *, image_key: str | int, seed: int = 42
) -> np.ndarray:
    if length < 1:
        raise ValueError("permutation length must be positive")
    payload = f"cc_lsa_token_permutation_v1\0{seed}\0{image_key}".encode()
    rng_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(rng_seed).permutation(length)


def count_weight_matched_scores(
    visual_cosine: np.ndarray,
    aligned_weights: np.ndarray,
    *,
    query_key: str | int,
    candidate_key: str | int,
    seeds: Sequence[int] = tuple(range(100)),
    top_l: int = 20,
) -> np.ndarray:
    """Match aligned edge count and weight multiset, but randomise placement."""

    visual = np.asarray(visual_cosine, dtype=np.float32)
    weights = np.asarray(aligned_weights, dtype=np.float32)
    if visual.ndim != 1 or weights.ndim != 1:
        raise ValueError("visual_cosine and aligned_weights must be 1-D")
    positive_weights = weights[weights > 0]
    count = len(positive_weights)
    if count > len(visual):
        raise ValueError("aligned edge count exceeds visual MNN edge count")
    output = np.zeros(len(seeds), dtype=np.float32)
    if count == 0:
        return output
    for output_index, seed in enumerate(seeds):
        payload = (
            f"cc_lsa_count_weight_v1\0{query_key}\0{candidate_key}\0{int(seed)}"
        ).encode()
        rng_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = np.random.default_rng(rng_seed)
        edge_indices = rng.choice(len(visual), size=count, replace=False)
        shuffled_weights = positive_weights[rng.permutation(count)]
        output[output_index] = semantic_pair_score(
            visual[edge_indices], shuffled_weights, top_l=top_l
        )
    return output


def conservative_percentile(values: np.ndarray, q: float = 95.0) -> float:
    values = np.asarray(values)
    if values.size == 0 or not 0.0 <= q <= 100.0:
        raise ValueError("percentile requires non-empty values and q in [0,100]")
    return float(np.percentile(values, q, method="higher"))


def _city_from_path(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 4 or parts[1] not in {"database", "query"}:
        raise ValueError(f"MSLS path has no city/role structure: {value!r}")
    return parts[0]


def _deterministic_bijection(
    receivers: Sequence[int],
    forbidden: Sequence[set[int]],
    *,
    seed: int,
    context: str,
) -> dict[int, int]:
    receiver_list = [int(value) for value in receivers]
    if len(receiver_list) < 2:
        raise ValueError(f"{context} needs at least two images")
    payload = f"cc_lsa_donor_v1\0{seed}\0{context}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
    donor_order = rng.permutation(receiver_list).tolist()
    result = dict(zip(receiver_list, donor_order))
    # Known-place exclusions are sparse. Repair a seeded permutation with
    # swaps in linear memory; never materialise N squared allowed donors.
    for receiver in receiver_list:
        donor = result[receiver]
        if donor not in forbidden[receiver]:
            continue
        for other in receiver_list:
            if (
                other != receiver
                and result[other] not in forbidden[receiver]
                and donor not in forbidden[other]
            ):
                result[receiver], result[other] = result[other], donor
                break
        else:
            # Complete iterative augmenting-path fallback for dense small
            # exclusion graphs. No recursion depth depends on city size.
            owner: dict[int, int] = {}
            result = {}
            for root in receiver_list:
                pending = deque([root])
                seen_receivers = {root}
                parent: dict[int, int] = {}
                free_donor = None
                while pending and free_donor is None:
                    left = pending.popleft()
                    for right in donor_order:
                        if right in forbidden[left] or right in parent:
                            continue
                        parent[right] = left
                        if right not in owner:
                            free_donor = right
                            break
                        next_left = owner[right]
                        if next_left not in seen_receivers:
                            seen_receivers.add(next_left)
                            pending.append(next_left)
                if free_donor is None:
                    raise ValueError(f"{context} cannot construct a valid donor bijection")
                while free_donor is not None:
                    left = parent[free_donor]
                    previous = result.get(left)
                    result[left] = free_donor
                    owner[free_donor] = left
                    free_donor = previous
            break
    if set(result) != set(receiver_list) or len(set(result.values())) != len(result):
        raise RuntimeError(f"internal error constructing {context} donor bijection")
    return result


def build_same_city_wrong_place_donors(
    paths: Sequence[str],
    *,
    num_references: int,
    positives: Sequence[np.ndarray],
    seed: int = 42,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Build same-city/role donors disjoint under the available GT relations.

    When candidates are supplied, every evaluated donor query/database pair
    is also constrained to be non-positive. Database images have no complete
    place IDs in the standard MSLS index; only known GT relations are claimed.
    """

    paths = tuple(str(value).replace("\\", "/") for value in paths)
    num_queries = len(paths) - int(num_references)
    positives = validate_positives(
        positives,
        num_queries=num_queries,
        num_references=num_references,
    )
    forbidden = [{index} for index in range(len(paths))]
    for row in positives:
        members = set(int(value) for value in row)
        for reference in members:
            forbidden[reference].update(members)
    for left in range(num_queries):
        left_set = set(int(value) for value in positives[left])
        for right in range(left + 1, num_queries):
            if left_set.isdisjoint(int(value) for value in positives[right]):
                continue
            left_global = num_references + left
            right_global = num_references + right
            forbidden[left_global].add(right_global)
            forbidden[right_global].add(left_global)

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, path in enumerate(paths):
        role = "database" if index < num_references else "query"
        groups[(role, _city_from_path(path))].append(index)
    donors = np.full(len(paths), -1, dtype=np.int32)
    group_order = sorted(groups, key=lambda item: (item[0] != "query", item[1]))
    for role, city in group_order:
        indices = groups[(role, city)]
        if role == "database" and candidates is not None:
            candidate_array = np.asarray(candidates)
            if candidate_array.ndim != 2 or len(candidate_array) != num_queries:
                raise ValueError("wrong-place candidate matrix has invalid shape")
            if candidate_array.dtype.kind not in "iu" or np.any(candidate_array < 0) or np.any(candidate_array >= num_references):
                raise ValueError("wrong-place candidates are invalid database indices")
            for query, candidate_row in enumerate(candidate_array):
                donor_query = int(donors[num_references + query]) - num_references
                if donor_query < 0:
                    raise RuntimeError("query donors must be constructed before database donors")
                excluded = positives[donor_query].tolist()
                for candidate in candidate_row.tolist():
                    forbidden[candidate].update(excluded)
        mapping = _deterministic_bijection(
            indices,
            forbidden,
            seed=seed,
            context=f"MSLS {city}/{role}",
        )
        for receiver, donor in mapping.items():
            donors[receiver] = donor
    if bool(np.any(donors < 0)) or np.unique(donors).size != len(donors):
        raise RuntimeError("wrong-place donor construction is not a bijection")
    for receiver, donor in enumerate(donors.tolist()):
        if donor in forbidden[receiver]:
            raise RuntimeError("wrong-place donor violates a known place relation")
        if _city_from_path(paths[receiver]) != _city_from_path(paths[donor]):
            raise RuntimeError("wrong-place donor crossed a city boundary")
        if (receiver < num_references) != (donor < num_references):
            raise RuntimeError("wrong-place donor crossed database/query roles")
    if candidates is not None:
        for query, candidate_row in enumerate(candidates):
            donor_query = int(donors[num_references + query]) - num_references
            if np.isin(donors[candidate_row], positives[donor_query]).any():
                raise RuntimeError("a wrong-place donor pair is GT-positive")
    return donors


def rerank_variant(
    candidates: np.ndarray,
    pair_scores: np.ndarray,
    positives: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    """Rerank fixed candidates and compute strict, all-negative corrections."""

    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(pair_scores, dtype=np.float32)
    if candidates.shape != scores.shape or candidates.ndim != 2:
        raise ValueError("candidates and pair_scores must be matching matrices")
    if len(candidates) != len(positives) or not bool(np.isfinite(scores).all()):
        raise ValueError("candidate rows/positives disagree or scores are non-finite")
    query_count, top_k = candidates.shape
    pair_rank = np.full(query_count, top_k + 1, dtype=np.int16)
    reranked_top1 = np.empty(query_count, dtype=np.int32)
    max_positive = np.full(query_count, -np.inf, dtype=np.float32)
    max_negative = np.full(query_count, -np.inf, dtype=np.float32)
    correction = np.zeros(query_count, dtype=bool)
    rank_improved = np.zeros(query_count, dtype=bool)
    original_rank = ru_candidate_diagnostics(candidates, positives)[
        "best_positive_rank"
    ]
    original_positions = np.arange(top_k, dtype=np.int64)
    for query_index, (candidate_row, score_row, positive_row) in enumerate(
        zip(candidates, scores, positives)
    ):
        order = np.lexsort((candidate_row, original_positions, -score_row))
        reranked = candidate_row[order]
        reranked_top1[query_index] = int(reranked[0])
        is_positive_original = np.isin(candidate_row, positive_row)
        is_positive_reranked = np.isin(reranked, positive_row)
        positive_positions = np.flatnonzero(is_positive_reranked)
        if positive_positions.size:
            pair_rank[query_index] = int(positive_positions[0]) + 1
            max_positive[query_index] = float(score_row[is_positive_original].max())
        if bool((~is_positive_original).any()):
            max_negative[query_index] = float(score_row[~is_positive_original].max())
        ru_wrong = int(candidate_row[0]) not in set(positive_row.tolist())
        strict_margin = max_positive[query_index] > max_negative[query_index]
        correction[query_index] = bool(
            ru_wrong and is_positive_reranked[0] and strict_margin
        )
        rank_improved[query_index] = pair_rank[query_index] < original_rank[query_index]
    return {
        "reranked_top1": reranked_top1,
        "best_positive_rank": pair_rank,
        "original_best_positive_rank": original_rank,
        "max_positive_score": max_positive,
        "max_negative_score": max_negative,
        "positive_negative_margin": max_positive - max_negative,
        "correction": correction,
        "rank_improved": rank_improved,
    }


def gate_a_verdict(
    *,
    ru_hit_at_1: np.ndarray,
    ru_reachable: np.ndarray,
    aligned: dict[str, np.ndarray],
    fixed_controls: dict[str, dict[str, np.ndarray]],
    matched_correction_counts: np.ndarray,
    matched_rank_improved_counts: np.ndarray,
    expected_ru_correct: int = 675,
    minimum_reachable_errors: int = 8,
    minimum_corrections: int = 8,
    control_gap: int = 4,
    minimum_rank_improved: int = 37,
) -> dict[str, Any]:
    ru_hit = np.asarray(ru_hit_at_1, dtype=bool)
    reachable = np.asarray(ru_reachable, dtype=bool)
    ru_correct = int(ru_hit.sum())
    reachable_errors = int((~ru_hit & reachable).sum())
    aligned_corrections = int(np.asarray(aligned["correction"], dtype=bool).sum())
    aligned_rank_improved = int(
        np.asarray(aligned["rank_improved"], dtype=bool).sum()
    )
    correction_p95 = int(conservative_percentile(matched_correction_counts))
    rank_p95 = int(conservative_percentile(matched_rank_improved_counts))
    checks: dict[str, dict[str, Any]] = {
        "ru_baseline_reproduced": {
            "value": ru_correct,
            "threshold": expected_ru_correct,
            "pass": ru_correct == expected_ru_correct,
        },
        "reachable_ru_errors": {
            "value": reachable_errors,
            "threshold": minimum_reachable_errors,
            "pass": reachable_errors >= minimum_reachable_errors,
        },
        "aligned_real_corrections": {
            "value": aligned_corrections,
            "threshold": minimum_corrections,
            "pass": aligned_corrections >= minimum_corrections,
        },
        "aligned_rank_improved": {
            "value": aligned_rank_improved,
            "threshold": minimum_rank_improved,
            "pass": aligned_rank_improved >= minimum_rank_improved,
        },
        "matched_correction_gap": {
            "aligned": aligned_corrections,
            "matched_p95": correction_p95,
            "threshold_gap": control_gap,
            "pass": aligned_corrections >= correction_p95 + control_gap,
        },
        "matched_rank_gap": {
            "aligned": aligned_rank_improved,
            "matched_p95": rank_p95,
            "threshold_gap": control_gap,
            "pass": aligned_rank_improved >= rank_p95 + control_gap,
        },
    }
    for name, result in sorted(fixed_controls.items()):
        control_corrections = int(
            np.asarray(result["correction"], dtype=bool).sum()
        )
        checks[f"correction_gap_vs_{name}"] = {
            "aligned": aligned_corrections,
            "control": control_corrections,
            "threshold_gap": control_gap,
            "pass": aligned_corrections >= control_corrections + control_gap,
        }
    passed = all(bool(record["pass"]) for record in checks.values())
    if reachable_errors < minimum_reachable_errors:
        reason = "INSUFFICIENT_CANDIDATE_RECALL"
    elif passed:
        reason = "PASS"
    else:
        reason = "REGISTERED_CHECK_FAILED"
    return {
        "verdict": "PASS" if passed else "FAIL",
        "reason": reason,
        "ru_correct": ru_correct,
        "reachable_ru_errors": reachable_errors,
        "aligned_corrections": aligned_corrections,
        "aligned_rank_improved": aligned_rank_improved,
        "matched_correction_p95": correction_p95,
        "matched_rank_improved_p95": rank_p95,
        "checks": checks,
    }
