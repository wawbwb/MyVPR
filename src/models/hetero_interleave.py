"""ReID-inspired heterogeneous side interaction above a frozen innovation-1.

SCCU-inspired side calibration, interleaved selective recurrence, dynamic
low-rank residual, and the uploaded manuscript's LGGR equation. Not a
MambaVision/OSNet reproduction; the existing unfused reference scan is reused.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.scan_mamba import SelectiveMixer

ARMS = ('dino_proxy', 'separate_scan', 'hetero_interleaved')


def interleave(a, b):
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError('Matched BNC streams required')
    return torch.stack((a, b), dim=2).flatten(1, 2)


def uninterleave(tokens):
    if tokens.ndim != 3 or tokens.shape[1] % 2:
        raise ValueError('Expected even interleaved sequence')
    return tokens[:, 0::2], tokens[:, 1::2]


def proxy_channels(feature, channels):
    """Fixed channel-axis pooling/resampling, NO new learned proxy weights.

    This is a single-encoder source control, not a distribution-matched CNN.
    """
    b, c, h, w = feature.shape
    return F.adaptive_avg_pool1d(feature.permute(0, 2, 3, 1).reshape(-1, 1, c), channels).reshape(b, h, w, channels).permute(0, 3, 1, 2)


class MapProjection(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.project = nn.Linear(channels, dim, bias=False)

    def forward(self, x):
        b, _, h, w = x.shape
        return self.project(self.norm(x.flatten(2).transpose(1, 2))).transpose(1, 2).reshape(b, -1, h, w)


class LocalGlobalRefinement(nn.Module):
    """F + DW3(LN(F)) * [tanh(MLP(GAP(LN(F)))) * LN(F)]."""
    def __init__(self, dim, rank):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.gate = nn.Sequential(nn.Linear(dim, rank), nn.ReLU(), nn.Linear(rank, dim), nn.Tanh())

    def forward(self, feature):
        n = self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        w = self.gate(n.mean((2, 3)))[:, :, None, None]
        return feature+self.local(n)*(w*n), w


class HeterogeneousInteraction(nn.Module):
    def __init__(self, arm, channels=768, cnn_channels=(256, 512, 1024), dim=64, rank=16, rho=.025):
        super().__init__()
        if arm not in ARMS or not 0 < rho < 1 or rank < 1 or dim < rank:
            raise ValueError('Invalid arm/dimensions/residual cap')
        self.arm, self.rho, self.cnn_channels = arm, rho, cnn_channels
        self.dino = nn.ModuleList([MapProjection(channels, dim) for _ in range(3)])
        self.cnn = nn.ModuleList([MapProjection(c, dim) for c in cnn_channels])
        self.direction_scale = nn.Parameter(torch.full((3,), .1))
        self.scan_norm = nn.LayerNorm(dim)
        self.scan = SelectiveMixer(dim, inner=32, state=4, rank=4)
        # Only the final 64->768 projection is zero. Avoid a second cold scan
        # output and a zero multiplicative scalar that would delay gradients.
        nn.init.normal_(self.scan.out.weight, std=.02)
        self.down = nn.Conv2d(2*dim, rank, 1)
        self.dynamic = nn.Linear(2*dim, rank)
        self.low_rank_up = nn.Conv2d(rank, 2*dim, 1)
        self.fuse = nn.Conv2d(2*dim, dim, 1)
        self.refine = LocalGlobalRefinement(dim, rank)
        self.output = nn.Conv2d(dim, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def directional_scan(self, tokens):
        x = self.scan_norm(tokens)
        return tokens+.5*(self.scan(x)+self.scan(x.flip(1)).flip(1))

    def forward(self, dino, cnn):
        if (len(dino) != 3 or len(cnn) != 3 or any(x.shape != dino[-1].shape for x in dino)
                or any(x.shape[0] != dino[-1].shape[0] for x in cnn)):
            raise ValueError('Three aligned DINO/CNN maps required')
        if self.arm == 'dino_proxy':
            cnn = tuple(proxy_channels(x, c) for x, c in zip(dino, self.cnn_channels))
        size = dino[-1].shape[-2:]
        a = [p(x) for p, x in zip(self.dino, dino)]
        b = [p(F.interpolate(x, size=size, mode='bilinear', align_corners=False)) for p, x in zip(self.cnn, cnn)]
        # Stage-aware SIDE calibration, not feedback into frozen encoders.
        first = a[0]+self.direction_scale[0]*b[0]
        middle = a[1]+self.direction_scale[1]*b[1]
        global_map = a[2]+.1*(first+middle)/2
        local_map = b[2]+self.direction_scale[2]*a[2]
        ag, bl = global_map.flatten(2).transpose(1, 2), local_map.flatten(2).transpose(1, 2)
        if self.arm == 'separate_scan':
            ag, bl = self.directional_scan(ag), self.directional_scan(bl)
        else:
            ag, bl = uninterleave(self.directional_scan(interleave(ag, bl)))
        ag = ag.transpose(1, 2).reshape_as(global_map)
        bl = bl.transpose(1, 2).reshape_as(local_map)
        joint = torch.cat((ag, bl), dim=1)
        gate = self.dynamic(torch.cat((global_map.mean((2, 3)), local_map.mean((2, 3))), dim=1)).sigmoid()
        mixed = joint+self.low_rank_up(F.silu(self.down(joint))*gate[:, :, None, None])
        fused, refine_gate = self.refine(self.fuse(mixed))
        raw = self.output(fused)
        anchor = dino[-1]
        residual = self.rho*anchor.square().mean(1, keepdim=True).sqrt().detach()*raw.tanh()
        stats = dict(relative_residual=(residual.detach().flatten(1).norm(dim=1)/anchor.flatten(1).norm(dim=1).clamp_min(1e-8)),
            dynamic_gate=gate.detach().mean(1), refine_gate=refine_gate.detach().mean((1, 2, 3)))
        auxiliary = (F.normalize(global_map.mean((2, 3)), dim=-1), F.normalize(local_map.mean((2, 3)), dim=-1))
        return anchor+residual, auxiliary, stats


class HeterogeneousVPR(nn.Module):
    def __init__(self, anchor, resnet, arm, **kwargs):
        super().__init__()
        self.anchor = anchor.requires_grad_(False).eval()
        self.resnet = resnet.requires_grad_(False).eval()
        self.fusion = HeterogeneousInteraction(arm, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        self.resnet.eval()  # includes BatchNorm buffers
        return self

    @torch.no_grad()
    def encode(self, images):
        layers = self.anchor.features(images)
        # Native post-CLS slices -> canonical BCHW, same as innovation-1 train.
        from src.hcen_plain_protocol import canonical_features
        layers = canonical_features(layers)
        enhanced, _, _ = self.anchor.neck(layers, auxiliary=False)
        self.anchor.neck.last_raw = None
        self.anchor.neck.last_conditional_stats = None
        dino = (layers[0], layers[1], enhanced)
        maps, x = [], images
        for i, block in enumerate(self.resnet.net):
            x = block(x)
            if i in (4, 5, 6):
                maps.append(F.interpolate(x, size=enhanced.shape[-2:], mode='bilinear', align_corners=False))
        if len(maps) != 3 or tuple(x.shape[1] for x in maps) != self.fusion.cnn_channels:
            raise ValueError('Expected official cropped ResNet50 layer1/2/3')
        return (*dino, *maps)

    def describe(self, features):
        mixed, auxiliary, stats = self.fusion(features[:3], features[3:])
        return (self.anchor.readout(mixed), *auxiliary, stats)

    def forward(self, images):
        return self.describe(self.encode(images))[0]


def replay_backward(model, features, gradients, microbatch):
    for start in range(0, len(features[0]), microbatch):
        outputs = model.describe(tuple(f[start:start+microbatch] for f in features))[:3]
        torch.autograd.backward(outputs, tuple(g[start:start+microbatch] for g in gradients))
