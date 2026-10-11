from pathlib import Path

import numpy as np
import pytest

from src.slgd_decomposition import (HEADS,VARIANTS,GLOBAL_DIM,LOCAL_DIM,
    descriptor_variant,validate_shard,paired_summary)


def arrays():
    d=np.zeros((2,GLOBAL_DIM+LOCAL_DIM),np.float32)
    d[:,0]=np.sqrt(.8);d[:,GLOBAL_DIM]=np.sqrt(.2)
    return dict(**{h:d.copy() for h in HEADS},indices=np.arange(2,dtype=np.int64),
                coordinates=np.zeros((2,2),np.float64))


def test_global_only_undoes_fixed_sqrt_weight_and_preserves_norm():
    a=arrays();g=descriptor_variant(a,'retrieval_global')
    assert g.shape==(2,GLOBAL_DIM) and np.allclose(np.linalg.norm(g,axis=1),1)
    assert descriptor_variant(a,'retrieval_hybrid') is a['retrieval']


def test_counterfactual_uses_frozen_global_and_only_trained_local():
    a=arrays();a['mid10'][:,0]=0;a['mid10'][:,1]=np.sqrt(.8)
    a['mid10'][:,GLOBAL_DIM]=0;a['mid10'][:,GLOBAL_DIM+1]=np.sqrt(.2)
    mixed=descriptor_variant(a,'mid10_ru_global_trained_local')
    assert np.array_equal(mixed[:,:GLOBAL_DIM],a['initial'][:,:GLOBAL_DIM])
    assert np.array_equal(mixed[:,GLOBAL_DIM:],a['mid10'][:,GLOBAL_DIM:])
    assert np.allclose(np.linalg.norm(mixed,axis=1),1)


def test_validate_shard_checks_identity_dtype_branch_norm_and_finite():
    a=arrays();validate_shard(a,[0,1])
    with pytest.raises(ValueError):validate_shard(a,[1,0])
    a['initial'][0,0]=np.nan
    with pytest.raises(ValueError):validate_shard(a,[0,1])
    a=arrays();a['last12'][:,GLOBAL_DIM:]=0
    with pytest.raises(ValueError):validate_shard(a,[0,1])
    a=arrays();a['initial']=a['initial'].astype(np.float16)
    with pytest.raises(ValueError):validate_shard(a,[0,1])


def test_report_separates_initial_fusion_training_and_global_drift():
    outcomes={name:[True,True,True,False] for name in VARIANTS}
    outcomes['initial_hybrid']=[True,False,True,False]
    outcomes['mid10_global']=[True,True,False,False]
    outcomes['mid10_hybrid']=[True,False,False,False]
    s=paired_summary(outcomes,[0,1,2,3],[False,True,True,True],2)
    pairs=s['all']['comparisons']
    assert pairs['initial_hybrid_vs_ru_global']['net']==-1
    assert pairs['mid10_global_vs_ru_global']['net']==-1
    assert pairs['mid10_hybrid_vs_mid10_global']['net']==-1
    assert s['calibration']['queries']==2 and s['evaluation']['queries']==2


def test_no_weight_sweep_and_safe_inference_only_launcher():
    assert len(VARIANTS)==11 and len(set(VARIANTS))==11
    with pytest.raises(ValueError):descriptor_variant(arrays(),'best_alpha')
    root=Path(__file__).resolve().parents[1]
    script=(root/'scripts/run_slgd_decomposition.sh').read_bytes()
    assert b'\r' not in script and b'CUDA_VISIBLE_DEVICES=1' in script and b'flock -n' in script
    assert b'run_slgd_mid_screen.sh' not in script and b'train_slgd.py --' not in script
