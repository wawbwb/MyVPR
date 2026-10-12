"""Remove/replace HCEN modules without changing the sealed full implementation."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.hcen_vpr import FusionStage
from src.hcen_plain_protocol import PlainHCENVPR

VARIANTS=('linear','conv','no_sasf')
COUNTS=dict(full=891268,linear=594432,conv=892356,no_sasf=817024)


class LinearFusion(nn.Module):
    """Progressive pointwise linear mixing, no spatial or selective scan path."""
    def __init__(self,original):
        super().__init__()
        self.low_delta=original.low_delta
        self.high_delta=original.high_delta

    def forward(self,low,high):
        joint=torch.cat((low,high),1).flatten(2).transpose(1,2)
        dl=self.low_delta(joint).transpose(1,2).reshape_as(low)
        dh=self.high_delta(joint).transpose(1,2).reshape_as(high)
        return .5*(low+high)+.1*(dl+dh)


class ConvMixer(nn.Module):
    """Nearly parameter-matched local 2D convolution replaces selective scan."""
    def __init__(self,original):
        super().__init__()
        dim=original.in_proj.in_features;inner=original.out.in_features
        self.in_proj=original.in_proj
        self.depthwise=nn.Conv2d(inner,inner,3,padding=1,groups=inner)
        self.hidden=nn.Linear(inner,inner)
        self.out=original.out  # Retain the same initially zero output projection.

    def forward(self,tokens,height,width):
        u,z=self.in_proj(tokens).chunk(2,-1)
        u=self.depthwise(u.transpose(1,2).reshape(len(u),-1,height,width)).flatten(2).transpose(1,2)
        return self.out(F.silu(self.hidden(F.silu(u)))*F.silu(z))


class ConvFusion(FusionStage):
    """Keep all other MFM/SASF paths and existing weights exactly as full HCEN."""
    def __init__(self,original):
        nn.Module.__init__(self)
        self.shallow=original.shallow;self.norm=original.norm
        self.conv_mixer=ConvMixer(original.ssm)
        self.symmetric=original.symmetric
        self.low_delta=original.low_delta;self.high_delta=original.high_delta;self.spatial=original.spatial

    def forward(self,low,high):
        low=low+.1*self.shallow(low);joint=torch.cat((low,high),1)
        _,_,h,w=joint.shape;tokens=joint.flatten(2).transpose(1,2)
        update=self.conv_mixer(self.norm(tokens),h,w)
        symmetric=self.symmetric(joint).flatten(2).transpose(1,2)
        mixed=tokens+update+.1*symmetric
        dl=self.low_delta(mixed).transpose(1,2).reshape_as(low)
        dh=self.high_delta(mixed).transpose(1,2).reshape_as(high)
        return self.spatial(.5*(low+high)+.1*(dl+dh))


def build_model(visual,variant,initialization_seed,**kwargs):
    if variant not in (*VARIANTS,'full'):raise ValueError('Unknown module ablation')
    # Construct the original full neck FIRST so common parameters retain the
    # exact historical initialization. Replacement draws do not disturb RNG.
    torch.random.default_generator.manual_seed(initialization_seed)
    model=PlainHCENVPR(visual,'hierarchy',**kwargs)
    for i,stage in enumerate(model.neck.stages):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed+9100+i)
            if variant=='linear':model.neck.stages[i]=LinearFusion(stage)
            elif variant=='conv':model.neck.stages[i]=ConvFusion(stage)
            elif variant=='no_sasf':stage.spatial=nn.Identity()
    model.ablation_variant=variant
    return model


def common_parameters(model):
    return {n:p for n,p in model.named_parameters() if n.startswith(
        ('neck.norms.','neck.projections.','neck.up.','neck.aux.'))}


def module_gradient_status(model):
    variant=model.ablation_variant;status={}
    for i in range(2):
        prefix=f'neck.stages.{i}.'
        part={'linear':'low_delta','conv':'conv_mixer.hidden','no_sasf':'ssm.select','full':'ssm.select'}[variant]
        status[str(i)]=any(n.startswith(prefix+part) and p.grad is not None and bool((p.grad!=0).any())
                           for n,p in model.named_parameters())
    return status
