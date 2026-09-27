import torch
from scripts.train_consensus import POLICY,MODES,save_checkpoint,load_checkpoint,trainable_state
from scripts.summarize_consensus import paired


def test_locked_policy():
    assert POLICY['dev_queries']==2048 and POLICY['topk']==20 and POLICY['epochs']==3
    assert MODES==('baseline','pointwise','spatial','shuffled')


def test_resume_next_update(tmp_path):
    torch.manual_seed(42);model=torch.nn.Linear(4,2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    def step():
        opt.zero_grad();model(torch.randn(3,4)).square().mean().backward();opt.step()
    step();save_checkpoint(tmp_path/'last.pt',model,opt,{'cursor':1})
    step();expected=trainable_state(model)
    assert load_checkpoint(tmp_path/'last.pt',model,opt)=={'cursor':1}
    step()
    assert all(torch.equal(v,expected[k]) for k,v in trainable_state(model).items())


def test_paired_regressions_not_hidden_by_equal_recall():
    a=[dict(query=1,correct=True),dict(query=2,correct=False)]
    b=[dict(query=1,correct=False),dict(query=2,correct=True)]
    assert paired(a,b)==dict(corrections=[2],regressions=[1],net=0)
