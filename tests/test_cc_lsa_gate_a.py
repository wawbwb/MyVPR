from __future__ import annotations

import numpy as np
import pytest

from src.cc_lsa_gate_a import (
    build_same_city_wrong_place_donors,
    conservative_percentile,
    count_weight_matched_scores,
    dino_full_score,
    gate_a_verdict,
    mutual_nearest_edges,
    rerank_variant,
    ru_candidate_diagnostics,
    semantic_edge_weights,
    semantic_pair_score,
    stable_ru_topk,
    top_l_zero_padded_score,
)


def test_stable_ru_topk_uses_lower_database_index_on_tie() -> None:
    descriptors = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    candidates, scores = stable_ru_topk(
        descriptors, num_references=3, top_k=3, query_chunk_size=1
    )
    assert candidates.tolist() == [[0, 1, 2]]
    np.testing.assert_allclose(scores, [[1.0, 1.0, 0.0]])


def test_mutual_nearest_edges_are_bidirectional_and_tie_stable() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    left, right, values = mutual_nearest_edges(query, candidate)
    assert left.tolist() == [0, 1]
    assert right.tolist() == [0, 2]
    np.testing.assert_allclose(values, [1.0, 1.0])


def test_pair_score_zero_pads_and_empty_edges_score_zero() -> None:
    assert top_l_zero_padded_score(np.asarray([]), top_l=20) == 0.0
    assert top_l_zero_padded_score(np.asarray([1.0, 0.5]), top_l=4) == 0.375
    assert dino_full_score(np.asarray([1.0]), top_l=2) == 0.5
    assert semantic_pair_score(
        np.asarray([1.0, 0.0]), np.asarray([1.0, 0.5]), top_l=4
    ) == pytest.approx((1.0 + 0.25) / 4)


def test_semantic_edge_weights_apply_registered_q95_formula() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    weights = semantic_edge_weights(
        query,
        candidate,
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        tau_sem=0.5,
    )
    np.testing.assert_allclose(weights, [1.0, (2**-0.5 - 0.5) / 0.5])


def test_count_weight_control_is_deterministic_and_preserves_edge_count() -> None:
    visual = np.linspace(-0.5, 1.0, 12, dtype=np.float32)
    weights = np.asarray([1.0, 0.7, 0.2] + [0.0] * 9, dtype=np.float32)
    first = count_weight_matched_scores(
        visual,
        weights,
        query_key="q",
        candidate_key="c",
        seeds=tuple(range(10)),
        top_l=5,
    )
    second = count_weight_matched_scores(
        visual,
        weights,
        query_key="q",
        candidate_key="c",
        seeds=tuple(range(10)),
        top_l=5,
    )
    np.testing.assert_array_equal(first, second)
    assert len(first) == 10
    assert np.all(first >= 0)
    assert conservative_percentile(np.asarray([0, 1, 2, 3]), 95) == 3


def test_wrong_place_donors_are_same_city_role_and_gt_disjoint() -> None:
    paths = [
        "cph/database/images/d0.jpg",
        "cph/database/images/d1.jpg",
        "cph/database/images/d2.jpg",
        "cph/database/images/d3.jpg",
        "cph/query/images/q0.jpg",
        "cph/query/images/q1.jpg",
        "cph/query/images/q2.jpg",
        "cph/query/images/q3.jpg",
    ]
    positives = (
        np.asarray([0]),
        np.asarray([1]),
        np.asarray([2]),
        np.asarray([3]),
    )
    donors = build_same_city_wrong_place_donors(
        paths, num_references=4, positives=positives, seed=42
    )
    assert sorted(donors.tolist()) == list(range(8))
    assert all((i < 4) == (int(donors[i]) < 4) for i in range(8))
    assert all(i != int(donors[i]) for i in range(8))


def test_wrong_place_donor_fails_when_city_role_group_is_singleton() -> None:
    paths = [
        "cph/database/images/d0.jpg",
        "sf/database/images/d1.jpg",
        "cph/query/images/q0.jpg",
        "sf/query/images/q1.jpg",
    ]
    positives = (np.asarray([0]), np.asarray([1]))
    with pytest.raises(ValueError, match="at least two"):
        build_same_city_wrong_place_donors(
            paths, num_references=2, positives=positives
        )


def test_donor_pairs_cannot_accidentally_become_gt_positive():
    paths = [f"cph/database/images/d{i}.jpg" for i in range(8)]
    paths += [f"cph/query/images/q{i}.jpg" for i in range(4)]
    positives = tuple(np.asarray([i]) for i in range(4))
    candidates = np.tile(np.arange(4), (4, 1))
    donors = build_same_city_wrong_place_donors(
        paths, num_references=8, positives=positives, candidates=candidates
    )
    for q, row in enumerate(candidates):
        assert not np.isin(donors[row], positives[donors[8 + q] - 8]).any()
    np.testing.assert_array_equal(donors, build_same_city_wrong_place_donors(
        paths, num_references=8, positives=positives, candidates=candidates
    ))


def test_donor_matching_handles_large_city_without_recursive_paths():
    from src.cc_lsa_gate_a import _deterministic_bijection
    count = 2500
    result = _deterministic_bijection(
        range(count), [{i} for i in range(count)], seed=42, context="large city"
    )
    assert set(result.values()) == set(range(count))
    assert all(left != right for left, right in result.items())


def test_donor_matching_dense_exclusions_uses_complete_fallback():
    from src.cc_lsa_gate_a import _deterministic_bijection
    # Only this cycle is feasible; swapping a pair need not solve it.
    count = 7
    forbidden = [set(range(count)) - {(i + 1) % count} for i in range(count)]
    result = _deterministic_bijection(range(count), forbidden, seed=42, context="cycle")
    assert result == {i: (i + 1) % count for i in range(count)}


def test_real_correction_must_beat_every_topk_negative_strictly() -> None:
    candidates = np.asarray([[0, 1, 2]], dtype=np.int32)
    positives = (np.asarray([1]),)

    # Positive beats the original RU top-1 (candidate 0), but candidate 2 has
    # a larger pair score.  This must not be called a correction.
    result = rerank_variant(
        candidates, np.asarray([[0.2, 0.3, 0.4]]), positives
    )
    assert result["correction"].tolist() == [False]
    assert result["reranked_top1"].tolist() == [2]

    corrected = rerank_variant(
        candidates, np.asarray([[0.2, 0.5, 0.4]]), positives
    )
    assert corrected["correction"].tolist() == [True]
    assert corrected["best_positive_rank"].tolist() == [1]

    tied = rerank_variant(
        candidates, np.asarray([[0.2, 0.4, 0.4]]), positives
    )
    assert tied["reranked_top1"].tolist() == [1]
    assert tied["correction"].tolist() == [False]


def test_unreachable_query_keeps_capped_rank_in_denominator() -> None:
    candidates = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    positives = (np.asarray([2]), np.asarray([0]))
    diagnostics = ru_candidate_diagnostics(candidates, positives)
    assert diagnostics["best_positive_rank"].tolist() == [3, 3]
    result = rerank_variant(candidates, np.zeros_like(candidates, dtype=float), positives)
    assert result["best_positive_rank"].tolist() == [3, 3]
    assert result["rank_improved"].tolist() == [False, False]


def _variant(corrections: int, rank_improved: int, size: int = 740):
    correction = np.zeros(size, dtype=bool)
    correction[:corrections] = True
    rank = np.zeros(size, dtype=bool)
    rank[:rank_improved] = True
    return {"correction": correction, "rank_improved": rank}


def test_gate_verdict_enforces_every_control_and_higher_percentile() -> None:
    ru = np.zeros(740, dtype=bool)
    ru[:675] = True
    reachable = np.ones(740, dtype=bool)
    aligned = _variant(12, 45)
    controls = {
        "dino_full": _variant(8, 10),
        "raw_clip": _variant(7, 10),
        "token_permutation": _variant(6, 10),
        "wrong_place": _variant(5, 10),
    }
    matched_corrections = np.asarray([8] * 95 + [9] * 5)
    matched_ranks = np.asarray([40] * 95 + [41] * 5)
    result = gate_a_verdict(
        ru_hit_at_1=ru,
        ru_reachable=reachable,
        aligned=aligned,
        fixed_controls=controls,
        matched_correction_counts=matched_corrections,
        matched_rank_improved_counts=matched_ranks,
    )
    # Conservative 95th percentile is 9/41, requiring 13/45 respectively.
    assert result["verdict"] == "FAIL"
    assert result["matched_correction_p95"] == 9
    assert result["matched_rank_improved_p95"] == 41


def test_gate_reports_insufficient_candidate_recall() -> None:
    ru = np.ones(740, dtype=bool)
    ru[-5:] = False
    result = gate_a_verdict(
        ru_hit_at_1=ru,
        ru_reachable=np.ones(740, dtype=bool),
        aligned=_variant(20, 50),
        fixed_controls={"control": _variant(0, 0)},
        matched_correction_counts=np.zeros(100, dtype=int),
        matched_rank_improved_counts=np.zeros(100, dtype=int),
        expected_ru_correct=735,
    )
    assert result["verdict"] == "FAIL"
    assert result["reason"] == "INSUFFICIENT_CANDIDATE_RECALL"
