"""Run on training machine: difficulty/candidate/sufficiency regression tests."""
from pathlib import Path

import numpy as np
import pytest

from src.slgd_hard_protocol import (HARD_POLICY, make_plan, distance_metres,
                                   rerank_indices, diagnostic_summary)


def test_split_excludes_original_training_and_keeps_old_holdout_first():
    ids = list(range(100))
    teacher = dict(train_places=[str(x) for x in range(20, 30)], holdout_places=[str(x) for x in range(8)])
    plan = make_plan(ids, teacher, smoke=True)
    assert plan['query_places'][:8] == teacher['holdout_places']
    assert not (set(plan['query_places']) | set(plan['distractor_places'])) & set(teacher['train_places'])
    assert not set(plan['query_places']) & set(plan['distractor_places'])
    assert len(plan['query_rows']) == 48 and len(plan['gallery_rows']) == 48
    assert not set(plan['query_rows']) & set(plan['gallery_rows'])
    assert plan['query_rows'][:3] == [0, 1, 2] and plan['gallery_rows'][0] == 3
    assert plan['positive_gallery_indices'][:6] == [0, 0, 0, 1, 1, 1]


def test_split_rejects_teacher_leakage_and_incompatible_ids():
    with pytest.raises(ValueError):
        make_plan(range(100), dict(train_places=['1'], holdout_places=['1']), smoke=True)
    with pytest.raises(ValueError):
        make_plan(range(100), dict(train_places=['101'], holdout_places=['0']), smoke=True)
    with pytest.raises(ValueError):
        make_plan([0, 0], dict(train_places=[], holdout_places=[]), smoke=True)


def test_haversine_handles_nearby_and_far_places():
    distances = distance_metres([[0, 0]], [[0, 0], [0, .001], [50, 120]])
    assert distances[0, 0] == 0 and 110 < distances[0, 1] < 112 and distances[0, 2] > 1e6
    with pytest.raises(ValueError):
        distance_metres([[np.nan, 0]], [[0, 0]])


def test_rank_ties_use_frozen_ru_order_and_invalid_queries_abstain():
    scores = np.array([[.5, .5, .1], [.2, .5, .8], [.4, .1, .2]])
    valid = np.array([[True, True, True], [False, True, False], [False, False, False]])
    assert rerank_indices(scores, valid).tolist() == [0, 1, 0]


def fixture():
    groups = np.repeat(np.arange(8), 2)
    truth = groups.copy()
    candidates = np.stack((100+groups, groups), axis=1)
    raw = np.tile([.9, .1], (len(groups), 1))
    teacher = np.tile([.1, .9], (len(groups), 1))
    valid = np.ones_like(raw, dtype=bool)
    policy = dict(HARD_POLICY, min_hard_queries=4, min_reachable_error_places=2,
                  min_teacher_hard_net=1)
    return candidates, dict(raw=raw, teacher=teacher), valid, truth, groups, policy


def test_teacher_must_win_on_both_partitions():
    candidates, scores, valid, truth, groups, policy = fixture()
    summary, _ = diagnostic_summary(candidates, scores, valid, valid, truth, groups,
                                    np.ones(len(groups)), 4, policy)
    assert summary['verdict'] == 'PASS'
    scores['teacher'][groups >= 4] = scores['raw'][groups >= 4]
    summary, _ = diagnostic_summary(candidates, scores, valid, valid, truth, groups,
                                    np.ones(len(groups)), 4, policy)
    assert summary['verdict'] == 'NO_TEACHER_ADVANTAGE'


def test_saturated_raw_baseline_is_insufficient_not_algorithm_failure():
    candidates, scores, valid, truth, groups, policy = fixture()
    scores['raw'][:] = scores['teacher']
    summary, _ = diagnostic_summary(candidates, scores, valid, valid, truth, groups,
                                    np.ones(len(groups)), 4, policy)
    assert summary['verdict'] == 'INSUFFICIENT_HARD_CASES'
    assert not summary['partitions']['evaluation']['sufficiency_checks']['raw_headroom']


def test_multiple_views_do_not_count_as_independent_error_places():
    candidates, scores, valid, truth, groups, policy = fixture()
    groups[:] = np.repeat([0, 4], 8)
    summary, _ = diagnostic_summary(candidates, scores, valid, valid, truth, groups,
                                    np.ones(len(groups)), 4, policy)
    assert summary['verdict'] == 'INSUFFICIENT_HARD_CASES'
    assert summary['partitions']['evaluation']['reachable_error_places'] == 1


def test_difficulty_never_depends_on_teacher_score():
    candidates, scores, valid, truth, groups, policy = fixture()
    candidates[::2, 0] = truth[::2]
    gaps = np.tile([.1, .01], 8)
    _, first = diagnostic_summary(candidates, scores, valid, valid, truth, groups, gaps, 4, policy)
    scores['teacher'] *= -10
    _, second = diagnostic_summary(candidates, scores, valid, valid, truth, groups, gaps, 4, policy)
    assert first['hard'] == second['hard']


def test_missing_positive_is_unreachable_not_a_teacher_correction():
    candidates, scores, valid, truth, groups, policy = fixture()
    candidates[:, 1] += 1000
    summary, outcomes = diagnostic_summary(candidates, scores, valid, valid, truth, groups,
                                           np.ones(len(groups)), 4, policy)
    assert not any(outcomes['reachable']) and not any(outcomes['teacher'])
    assert summary['verdict'] == 'INSUFFICIENT_HARD_CASES'


def test_launcher_is_lf_and_teacher_stage_not_truncated():
    root = Path(__file__).resolve().parents[1]
    for name in ('run_slgd_screen.sh', 'run_slgd_hard_audit.sh'):
        assert b'\r' not in (root/'scripts'/name).read_bytes()
    assert b"gate 'teacher'" in (root/'scripts/run_slgd_screen.sh').read_bytes()


def test_cache_shard_shapes_norms_and_hashes(tmp_path):
    from scripts.audit_slgd_hard import SHAPES, DTYPES, save_npz, validate_shard
    from scripts.train_slgd import write
    path = tmp_path/'00000.npz'
    arrays = {name: np.zeros((1,)+shape, dtype=DTYPES[name]) for name, shape in SHAPES.items()}
    for name in ('global_descriptors', 'raw', 'teacher'):
        arrays[name][...] = 1/np.sqrt(SHAPES[name][-1])
    arrays['indices'] = np.array([0], dtype=np.int64)
    save_npz(path, arrays)
    result = validate_shard(path, [0])
    assert np.array_equal(result['indices'], [0])
    assert not path.with_suffix('.tmp').exists()
    write(path.with_suffix('.json'), dict(sha256='changed'))
    with pytest.raises(ValueError):
        validate_shard(path, [0])


def test_cache_shard_rows_follow_fixed_views():
    from scripts.audit_slgd_hard import shard_rows
    plan = dict(query_places=list(range(16)), dataset_indices=list(range(48)))
    assert shard_rows(plan, 0).tolist() == list(range(64))
    assert shard_rows(plan, 1).tolist() == list(range(64, 80))
    assert shard_rows(plan, 2).tolist() == list(range(80, 96))
