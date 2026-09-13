"""Required server-side check; deliberately fails import if torch is missing."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_pairvpr_cls import LastCLS


class Cross(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.scale = .5
        self.projq = nn.Linear(8,8); self.projk = nn.Linear(8,8)
        self.projv = nn.Linear(8,8); self.proj = nn.Linear(8,8)
        self.proj_drop = nn.Identity()
        self.fraction = None

    def forward(self, query, key, value):
        q,k,v = [layer(x).reshape(1,-1,2,4).transpose(1,2) for layer,x in
                 [(self.projq,query),(self.projk,key),(self.projv,value)]]
        logits = (q @ k.transpose(-1,-2))*.5
        if self.fraction is not None:
            logits[:,:,:1] += torch.log1p(-.5*self.fraction)
        result = (logits.softmax(-1) @ v).transpose(1,2).reshape(1,-1,8)
        return self.proj(result)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = Cross(); self.norm3 = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8))
        self.drop_path = nn.Identity()

    def forward(self,x,y):
        x = x+self.cross_attn(x,y,y)
        return x+self.mlp(self.norm3(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_blocks = nn.ModuleList([Block()])
        self.dec_norm = nn.LayerNorm(8)
        self.classvprmodule = nn.Sequential(nn.Linear(8,4),nn.ReLU(),nn.Linear(4,1))

    def forward(self,x,y):
        return self.classvprmodule(self.dec_norm(self.dec_blocks[0](x,y))[:,0])


class InterventionTests(unittest.TestCase):
    def test_reused_prefix_matches_full_forward_and_repeated_calls(self):
        torch.manual_seed(7)
        model = Model().eval(); probe = LastCLS(model)
        x,y = torch.randn(1,6,8),torch.randn(1,5,8)
        f = torch.tensor([0.,.2,.9,1.,.4])
        with torch.inference_mode():
            original = model(x,y)
            torch.testing.assert_close(probe.score(f,0.),original)
            changed = probe.score(f,.5)
            torch.testing.assert_close(probe.score(f,.5),changed)
            torch.testing.assert_close(probe.score(f,0.),original)
            probe.close()
            model.dec_blocks[0].cross_attn.fraction = f
            torch.testing.assert_close(changed,model(x,y))
            self.assertGreater(float((changed-original).abs().max()),1e-6)


if __name__ == '__main__':
    unittest.main()
