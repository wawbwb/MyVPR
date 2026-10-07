import ast
from pathlib import Path
import torch
from src.scan_mamba_diagnostics import fixed_step,gradient_comparison


def test_zero_batches_advance_all_parameter_clocks_and_momentum():
    x=torch.nn.Parameter(torch.tensor([1.]));unused=torch.nn.Parameter(torch.tensor([2.]))
    opt=torch.optim.AdamW([x,unused],lr=.01,weight_decay=0.)
    x.square().sum().backward();fixed_step(opt,[x,unused]);before=x.detach().clone()
    opt.zero_grad(set_to_none=True);(x.sum()*0).backward();fixed_step(opt,[x,unused])
    assert x.item()!=before.item(), 'Adam momentum continues on zero gradients'
    assert unused.item()==2.
    assert all(int(opt.state[p]['step'])==2 for p in (x,unused))


def test_gradient_report_does_not_accumulate_parameter_gradients():
    x=torch.nn.Parameter(torch.tensor([2.,3.]));y=torch.nn.Parameter(torch.tensor([4.]))
    loss=x.square().sum()+y.square().sum();cons=(x.sum()-y.sum()).square()
    r=gradient_comparison(loss,cons,[('base.aggregator.x',x),('mixer.y',y)])
    assert x.grad is None and y.grad is None
    assert r['mixer']['weighted_consistency_norm']>0
    assert -1.000001<=r['all']['cosine']<=1.000001


def test_trainer_unconditional_step():
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/train_scan_mamba.py').read_text()
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fixed_step']
    assert len(calls)==1
    assert not any(calls[0] in list(ast.walk(n)) for n in ast.walk(tree) if isinstance(n,ast.If))
    assert 'best.pt' not in source and 'best_epoch' not in source
