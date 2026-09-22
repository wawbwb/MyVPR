import torch
from src.wppr_runtime import prefix, finish, progressive, full


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls=0
    def forward(self,x,y):
        self.calls+=1
        return x+y.mean(1,keepdim=True)*.01,y


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_embed=torch.nn.Identity()
        self.decoder_clstoken=torch.zeros(1,1,768)
        self.dec_pos_embed_cls=torch.zeros(3,768)
        self.dec_pos_embed=torch.zeros(2,768)
        self.dec_blocks=torch.nn.ModuleList([Block() for _ in range(12)])
        self.dec_norm=torch.nn.LayerNorm(768)
        self.classvprmodule=torch.nn.Linear(768,1)
    def forward(self,x,y,mode):
        return finish(self,prefix(self,x,y))[:,None]


def test_continuation_and_actual_layer_counts():
    torch.manual_seed(42)
    m=Model().eval(); head=torch.nn.Linear(1536,1).eval()
    q=torch.randn(1,2,768); d=torch.randn(44,2,768)
    with torch.inference_mode():
        reference=full(m,q,d,44)
        for block in m.dec_blocks: block.calls=0
        keep,scores,pred=progressive(m,head,q,d)
    assert torch.allclose(scores,reference[keep],atol=1e-6)
    assert len(keep.unique())==12 and pred.shape==(44,)
    assert [b.calls for b in m.dec_blocks]==[88]*2+[24]*10


def test_reject_incompatible_architecture():
    m=Model();m.dec_blocks=m.dec_blocks[:3]
    import pytest
    with pytest.raises(ValueError): prefix(m,torch.zeros(1,2,768),torch.zeros(1,2,768))
