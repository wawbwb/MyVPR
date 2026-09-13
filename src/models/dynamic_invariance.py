"""Zero-initialized qkv LoRA, preserving frozen RU computation at initialization."""
from contextlib import contextmanager
import torch
from torch import nn


class LowRank(nn.Module):
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        if not isinstance(base,nn.Linear): raise TypeError('LoRA requires Linear qkv')
        self.base = base
        self.down = nn.Linear(base.in_features,rank,bias=False).to(base.weight)
        self.up = nn.Linear(rank,base.out_features,bias=False).to(base.weight)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha/rank
        self.enabled = True
        self.base.requires_grad_(False)

    def forward(self,x):
        result = self.base(x)
        return result + self.up(self.down(x))*self.scale if self.enabled else result


def install(model):
    backbone = model.backbone
    if backbone.num_unfrozen_blocks != 2 or backbone.crop_semantic_film is not None or backbone.residual_clip_fusion is not None:
        raise ValueError('Requires plain RU DINO with two gradient-enabled suffix blocks')
    model.requires_grad_(False)
    for block in backbone.dino.blocks[-2:]:
        block.attn.qkv = LowRank(block.attn.qkv)
    return [p for p in model.parameters() if p.requires_grad]


@contextmanager
def disabled(model):
    modules = [m for m in model.modules() if isinstance(m,LowRank)]
    before = [m.enabled for m in modules]
    try:
        for m in modules: m.enabled=False
        yield
    finally:
        for m,v in zip(modules,before): m.enabled=v


def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def restore(model,state):
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(params)!=set(state): raise ValueError('Adapter keys differ')
    with torch.no_grad():
        for n,p in params.items(): p.copy_(state[n].to(p))


def losses(z,teacher,labels,loss_fn,consistency=True):
    """Full-place-batch mining on clean views; two-view invariance plus clean anchor."""
    clean,one,two = z
    retrieval,_ = loss_fn(clean,labels)
    preserve = (clean-teacher).square().sum(-1).mean()
    invariant = ((one-clean).square().sum(-1).mean()
                 +(two-clean).square().sum(-1).mean()
                 +(one-two).square().sum(-1).mean())/3
    total = retrieval+preserve+(invariant if consistency else 0*invariant)
    return total, {'retrieval':float(retrieval.detach()), 'preserve':float(preserve.detach()),
                   'invariance':float(invariant.detach())}


def replay_backward(model, inputs, teacher, labels, loss_fn, microbatch, consistency):
    """Exact descriptor-gradient replay for deterministic eval-mode models."""
    if any(m.training for m in model.modules()): raise ValueError('Replay requires eval mode')
    with torch.no_grad():
        cached = [torch.cat([model(x[i:i+microbatch]) for i in range(0,len(x),microbatch)]) for x in inputs]
    leaves = [z.detach().requires_grad_(True) for z in cached]
    total,stats = losses(leaves,teacher,labels,loss_fn,consistency)
    if not torch.isfinite(total): raise ValueError('Nonfinite loss')
    total.backward()
    for x,leaf,expected in zip(inputs,leaves,cached):
        for i in range(0,len(x),microbatch):
            output = model(x[i:i+microbatch])
            if not torch.allclose(output.detach(),expected[i:i+microbatch],atol=2e-6,rtol=2e-5):
                raise ValueError('Descriptor replay mismatch')
            (output*leaf.grad[i:i+microbatch]).sum().backward()
    return stats
