import numpy as np
import pytest
import torch
from scripts.train_pmd import ordered_ids,pair_indices,restore_trainable,trainable_state,save_checkpoint,load_checkpoint


def test_pair_sampling():
    y=np.array([False,True,True,False])
    assert pair_indices(y,0)==(1,0)
    assert pair_indices(y,1)==(2,0)
    with pytest.raises(ValueError):pair_indices(np.ones(4,bool),0)


def test_order_independent_of_input_order():
    rows=[{'label':str(i)} for i in range(8)]
    a=[rows[i]['label'] for i in ordered_ids(rows,'pmd')]
    rows.reverse()
    assert a==[rows[i]['label'] for i in ordered_ids(rows,'pmd')]


def test_restore_trainable_leaves_frozen_alone():
    m=torch.nn.Linear(4,2);m.bias.requires_grad_(False)
    saved=trainable_state(m);bias=m.bias.detach().clone()
    with torch.no_grad():m.weight.add_(1)
    restore_trainable(m,saved)
    assert torch.equal(m.weight,saved['weight']) and torch.equal(m.bias,bias)
    with pytest.raises(ValueError):restore_trainable(m,{})


def test_resume_reproduces_next_optimizer_update(tmp_path):
    torch.manual_seed(42)
    m=torch.nn.Linear(4,2);opt=torch.optim.AdamW(m.parameters(),lr=.01)
    def step():
        x=torch.randn(3,4);opt.zero_grad();m(x).square().mean().backward();opt.step()
    step()
    save_checkpoint(tmp_path/'last.pt',m,opt,{'cursor':1})
    step();expected=trainable_state(m)
    assert load_checkpoint(tmp_path/'last.pt',m,opt)=={'cursor':1}
    step()
    for name,p in trainable_state(m).items():assert torch.equal(p,expected[name])
