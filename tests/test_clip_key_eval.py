"""Evaluator CPU tests intended for the training server only."""
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.eval_clip_key_study import (DualImages, VARIANTS, compare_reports,
    gt_digest, order_digest, paired_report, retrieval_report, validate_completed_dataset)
from src.semantic_region_protocol import sha


def ranking(firsts):
    return np.asarray([[int(f)]+[i for i in range(30) if i != f][:19] for f in firsts],dtype=np.int64)


def test_recall_uses_global_reference_indices_and_correct_prefix():
    p=ranking([0,4,3])
    r=retrieval_report(p,[np.asarray([0]),np.asarray([2]),np.asarray([28])],30)
    assert r['correct']==1 and r['outcomes']==[True,False,False]
    assert r['recall']==[1/3,2/3,2/3,2/3]


@pytest.mark.parametrize('kind',['short','float','duplicate','negative','outside'])
def test_invalid_rankings_rejected(kind):
    p=ranking([1])
    if kind=='short':p=p[:,:10]
    if kind=='float':p=p.astype(np.float32)
    if kind=='duplicate':p[0,1]=p[0,0]
    if kind=='negative':p[0,0]=-1
    if kind=='outside':p[0,0]=30
    with pytest.raises(ValueError):retrieval_report(p,[np.asarray([1])],30)


def test_paired_exact_corrections_and_regressions_not_aggregate_subtraction_only():
    gt=[np.asarray([0])]*4
    a=retrieval_report(ranking([0,1,1,1]),gt,30)
    b=retrieval_report(ranking([1,0,0,1]),gt,30)
    r=paired_report(a,b)
    assert r['corrections']==2 and r['regressions']==1 and r['net']==1
    assert r['correction_query_ids']==[1,2] and r['regression_query_ids']==[0]
    assert r['top1_changed']==3 and r['descriptive_mcnemar_exact_p']==1.


def test_identical_outcomes_can_still_change_top1_database_identity():
    gt=[np.asarray([0])]
    a=retrieval_report(ranking([2]),gt,30)
    b=retrieval_report(ranking([3]),gt,30)
    r=paired_report(a,b)
    assert r['net']==0 and r['top1_changed']==1


def test_comparison_requires_exact_four_arm_population():
    r=retrieval_report(ranking([0,2]),[np.asarray([0])]*2,30)
    reports={a:copy.deepcopy(r) for a in VARIANTS}
    assert compare_reports(reports)['aligned_vs']['official']['net']==0
    del reports['shuffled']
    with pytest.raises(ValueError):compare_reports(reports)


def test_order_and_gt_digests_are_order_sensitive():
    assert order_digest(['a','b']) != order_digest(['b','a'])
    assert gt_digest([[1],[2]]) != gt_digest([[2],[1]])
    assert gt_digest([np.asarray([1,2])])==gt_digest([[1,2]])


def test_dual_images_share_original_full_image_geometry_and_hash(tmp_path):
    p=tmp_path/'image.png'
    pixels=np.zeros((27,61,3),np.uint8)
    pixels[:,:30,0]=255
    Image.fromarray(pixels).save(p)
    dataset=DualImages(tmp_path,['image.png'],[sha(p)])
    student,clip,index,digest=dataset[0]
    assert student.shape==(3,322,322) and clip.shape==(3,224,224)
    assert index==0 and digest==sha(p)
    # Original left/right structure retained for both encoders; no center crop.
    assert student[0,:,:80].mean()>student[0,:,-80:].mean()
    assert clip[0,:,:50].mean()>clip[0,:,-50:].mean()
    broken=DualImages(tmp_path,['image.png'],['0'*64])
    with pytest.raises(ValueError):broken[0]


def test_path_traversal_rejected_before_image_read(tmp_path):
    dataset=DualImages(tmp_path,['../escape.png'])
    with pytest.raises(ValueError):dataset[0]


def test_finished_report_must_reproduce_rankings_gt_and_paired_counts():
    r=retrieval_report(ranking([0,2]),[np.asarray([0])]*2,30)
    variants={a:copy.deepcopy(r) for a in VARIANTS}
    d=dict(tag='msls-val',ground_truth=[np.asarray([0])]*2,references=30)
    contract=dict(datasets={'msls-val':dict(queries=2)})
    report=dict(tag='msls-val',dataset=dict(queries=2),variants=variants,comparison=compare_reports(variants))
    validate_completed_dataset(report,d,contract)
    report['variants']['aligned']['correct']=2
    with pytest.raises(ValueError):validate_completed_dataset(report,d,contract)


def test_evaluation_requires_explicit_authorization_and_has_no_train_mode():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/eval_clip_key_study.py')],capture_output=True,text=True)
    assert result.returncode != 0 and '--evaluate' in result.stderr
    source=(root/'scripts/eval_clip_key_study.py').read_text(encoding='utf-8')
    assert 'torch.optim.' not in source and '.backward(' not in source


def test_cache_precision_simulation_matches_training_storage_roundtrip():
    values=torch.randn(2,512,14,14)
    roundtrip=torch.from_numpy(values.numpy().astype(np.float16)).float()
    assert torch.equal(values.half().float(),roundtrip)
