"""Official ResNet50-BoQ plus the existing progressive/conditional necks.

Only the backbone interface is new. No changes to sealed DINO experiments.
Native layer1/2/3 maps have 256/512/1024 channels. Area pooling aligns grids;
deterministic channel repetition preserves RMS and makes all input-routing
controls have exactly the same learned parameter inventory.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models.official_boq import COMMIT, SOURCE_SHA, OfficialBoQ, file_sha, load_source
from src.models.hcen_vpr import HierarchicalNeck
from src.models.hcen_ablation import ConvFusion
from src.models.conv_residual_ablation import ResidualRuleNeck
from src.models.conditional_complement import AnchoredConditionalNeck

WEIGHT_URL = ('https://github.com/amaralibey/Bag-of-Queries/releases/download/'
              'v1.0/resnet50_16384.pth')
ARMS = ('conv_last', 'conv_cross', 'conditional_last', 'conditional_cross')


def verify_resnet_download(cache):
    cache = Path(cache)
    m = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    if (m['commit'] != COMMIT or m['weight_url'] != WEIGHT_URL
            or m['source_sha256'] != SOURCE_SHA):
        raise ValueError('Official ResNet provenance differs')
    if file_sha(cache / 'resnet50_16384.pth') != m['weight_sha256']:
        raise ValueError('Official ResNet weight SHA256 differs')
    for name, digest in SOURCE_SHA.items():
        if file_sha(cache / 'source' / name) != digest:
            raise ValueError('Pinned author source differs: ' + name)
    return m


def load_official_resnet(cache, device='cpu'):
    cache = Path(cache)
    verify_resnet_download(cache)
    backbones = load_source(cache / 'source/backbones.py', 'author_resnet_transfer_backbones')
    aggregation = load_source(cache / 'source/boq.py', 'author_resnet_transfer_boq')
    # No ImageNet download: the full VPR checkpoint replaces EVERY tensor,
    # including BatchNorm buffers, after offline construction.
    backbone = backbones.ResNet(backbone_name='resnet50', pretrained=False,
                               unfreeze_n_blocks=0, crop_last_block=True)
    model = OfficialBoQ(backbone, aggregation.BoQ(1024, 512, 64, 2, 32))
    state = torch.load(cache / 'resnet50_16384.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if any(not torch.equal(v, state[k]) for k, v in model.state_dict().items()):
        raise ValueError('Strict ResNet checkpoint tensor reproduction failed')
    return model.requires_grad_(False).eval().to(device)


def align_native_maps(maps):
    """Parameter-free interface; not a learned improvement or a fourth stage."""
    if len(maps) != 3 or any(x.ndim != 4 for x in maps):
        raise ValueError('Need three native BCHW stage maps')
    last = maps[-1]
    if tuple(x.shape[1] for x in maps) != (256, 512, 1024):
        raise ValueError('Expected cropped ResNet50 layer1/layer2/layer3')
    if any(x.shape[0] != last.shape[0] or x.shape[-2] < last.shape[-2]
           or x.shape[-1] < last.shape[-1] for x in maps):
        raise ValueError('Incompatible ResNet stage grids')
    aligned = []
    for x in maps[:-1]:
        x = F.adaptive_avg_pool2d(x, last.shape[-2:])
        aligned.append(x.repeat(1, 1024 // x.shape[1], 1, 1))
    return (*aligned, last)


class ScaledConvNeck(ResidualRuleNeck):
    def __init__(self, original):
        super().__init__(original, 'rms_tanh')
        self.output_scale = .25
        self.aux.requires_grad_(False)

    def forward(self, layers, auxiliary=False):
        _, _, residual = super().forward(layers, auxiliary=False)
        residual = self.output_scale * residual
        return layers[-1] + residual, (), residual


def make_neck(arm, seed):
    if arm not in ARMS:
        raise ValueError('Unknown ResNet transfer arm')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        original = HierarchicalNeck(channels=1024, dim=128, inner=32, state=4)
        for i, stage in enumerate(original.stages):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 9100 + i)
                original.stages[i] = ConvFusion(stage)
                original.stages[i].spatial = nn.Identity()
        neck = ScaledConvNeck(original)
        if arm.startswith('conditional_'):
            # Fresh ResNet anchor in both conditional arms, NOT DINO Warm.
            # Both up projections are zero, not their multiplicative gates.
            torch.random.default_generator.manual_seed(seed + 20000)
            route = ('b_last_condition' if arm.endswith('_last') else 'c_cross_condition')
            neck = AnchoredConditionalNeck(neck, route, channels=1024, dim=128, rank=32)
    return neck


class ResNetModuleTransfer(nn.Module):
    def __init__(self, visual, arm, seed=42):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown ResNet transfer arm')
        self.visual = visual.requires_grad_(False).eval()
        self.arm = arm
        self.neck = make_neck(arm, seed)

    def train(self, mode=True):
        super().train(mode)
        # Freeze running_mean/variance and batch counters, not just weights.
        self.visual.eval()
        return self

    @torch.no_grad()
    def features(self, images):
        captured = {}
        hooks = []
        try:
            for index in (4, 5):
                def capture(module, inputs, value, key=index):
                    captured[key] = value
                hooks.append(self.visual.backbone.net[index].register_forward_hook(capture))
            last = self.visual.backbone(images)
        finally:
            for hook in hooks:
                hook.remove()
        return align_native_maps((captured[4], captured[5], last))

    def readout(self, features):
        # ResNet produces native contiguous NCHW, not DINO's post-CLS view.
        return self.visual.aggregator(features.contiguous())[0]

    def aggregate(self, a, b, last):
        layers = (last, last, last) if self.arm == 'conv_last' else (a, b, last)
        mixed, _, residual = self.neck(layers)
        return self.readout(mixed), residual

    def forward(self, images):
        return self.aggregate(*self.features(images))[0]


def gradient_groups(model):
    groups = {}
    parts = ('conv_mixer.hidden', 'shallow', 'symmetric', 'low_delta', 'high_delta')
    for i in range(2):
        for part in parts:
            prefix = f'neck.stages.{i}.{part}.'
            groups[prefix] = any(p.grad is not None and bool((p.grad != 0).any())
                                 for n, p in model.named_parameters() if n.startswith(prefix))
    for prefix in ('neck.up.', 'neck.projections.', 'neck.norms.'):
        groups[prefix] = any(p.grad is not None and bool((p.grad != 0).any())
                             for n, p in model.named_parameters() if n.startswith(prefix))
    if model.arm.startswith('conditional_'):
        from src.models.conditional_complement import conditional_gradient_status
        groups.update({'conditional_' + k: v for k, v in conditional_gradient_status(model.neck).items()})
    return groups
