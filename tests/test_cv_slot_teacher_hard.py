"""CPU-only hard-pool indexing/selection and gate-regression tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.cv_slot_hard_protocol import choose_negatives, distances


def population():
    places, images = [], []
    for i in range(8):
        rows = []
        for view in range(2):
            rows.append(len(images))
            images.append(dict(coordinates=[40.+i*.001, 116.]))
        places.append(dict(place=str(i), rows=rows, split='train', city='A'))
    places[6]['city'] = 'B'
    places[7]['split'] = 'holdout'
    return dict(places=places, images=images)


def test_hard_pool_excludes_query_wrong_city_split_and_nearby():
    p = population()
    p['images'][2]['coordinates'] = p['images'][3]['coordinates'] = [40., 116.]
    selected, count = choose_negatives(p, 0, list(range(8)), np.arange(8, dtype=float), 4)
    assert selected == [5, 4, 3, 2] and count == 4


def test_both_positive_views_are_excluded_by_gps():
    p = population()
    p['images'][1]['coordinates'] = [40.003, 116.]
    selected, _ = choose_negatives(p, 0, list(range(8)), np.arange(8, dtype=float), 4)
    assert 3 not in selected


def test_score_tie_break_uses_place_id_not_pool_order():
    p = population()
    a, _ = choose_negatives(p, 0, list(range(8)), np.ones(8), 3)
    b, _ = choose_negatives(p, 0, list(range(7, -1, -1)), np.ones(8), 3)
    assert a == b == [1, 2, 3]


def test_finite_inputs_and_array_distances():
    assert distances([40., 116.], [40., 116.]) == 0
    assert np.allclose(distances([40., 116.], [[40., 116.], [40.001, 116.]]), [0, 111.19492664])
    with pytest.raises(ValueError): choose_negatives(population(), 0, list(range(8)), np.zeros(7))
    with pytest.raises(ValueError): choose_negatives(population(), 0, list(range(8)), np.full(8, np.nan))


def test_new_config_changes_pool_not_thresholds_or_temperature():
    from scripts.cv_slot_teacher_hard import configuration
    from scripts.cv_slot_teacher_audit import configuration as original
    a = configuration(ROOT/'config/cv_slot_teacher_hard.yaml')
    b = original(ROOT/'config/cv_slot_teacher_audit.yaml')
    for key in ('thresholds', 'topk', 'dino_temperature', 'clip_temperature', 'sample_per_split'):
        assert a[key] == b[key]


def test_hard_student_config_has_separate_output():
    from scripts.cv_slot_study import configuration
    a = configuration(ROOT/'config/cv_slot_boq_hard.yaml')
    b = configuration(ROOT/'config/cv_slot_boq.yaml')
    assert a['runs'] != b['runs'] and a['teacher_audit'] != b['teacher_audit']
    assert {k: v for k, v in a.items() if k not in ('runs', 'teacher_audit')} == {
        k: v for k, v in b.items() if k not in ('runs', 'teacher_audit')}


def test_verified_hard_fail_still_blocks_student(tmp_path, monkeypatch):
    from scripts import cv_slot_study as study
    from scripts import cv_slot_teacher_hard as hard
    from src.semantic_region_protocol import write, sha
    write(tmp_path/'completed.json', dict(schema=hard.SCHEMA, complete=True))
    write(tmp_path/'summary.json', dict(verdict='STOP_TEACHER_NOT_SUPPORTED',
        partitions=dict(holdout=dict(checks=dict(a=False)))))
    original = tmp_path/'original'
    original.mkdir()
    write(original/'completed.json', dict(schema='placeholder'))
    value = dict(hard_code=study.key.source_hashes(hard.SOURCES),
        frozen_helpers=study.key.source_hashes((*study.audit.SOURCES, *hard.representation.SOURCES)),
        config=dict(original_audit=str(original)), original_completed_sha256=sha(original/'completed.json'))
    monkeypatch.setattr(hard, 'verify', lambda _: value)
    monkeypatch.setattr(study.audit, 'verify', lambda _: {})
    with pytest.raises(ValueError, match='Teacher gate NOT PASS'):
        study.require_teacher(dict(teacher_audit=str(tmp_path)))
