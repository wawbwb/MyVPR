"""Depth-only convolution ablation and SciceVPR-inspired fusion in the same BoQ."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.hcen_followup import build_model as original_model
from src.models.hcen_followup import module_gradient_status as original_gradients

ARMS = ('conv_last', 'concat_token')
COUNTS = dict(conv=818112, conv_last=818112, concat_token=1105600)


class TokenMixer(nn.Module):
    """SciceVPR-style channel LN followed by a shared spatial-token MLP."""
    def __init__(self, dim, tokens):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mix = nn.Sequential(nn.Linear(tokens, tokens), nn.ReLU(), nn.Linear(tokens, tokens))
        for layer in self.mix.modules():
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return x + self.mix(self.norm(x.transpose(1, 2)).transpose(1, 2))


class ConcatTokenNeck(nn.Module):
    """Raw multi-depth LN/concat -> 1x1/ReLU -> two token mixers -> bounded residual.

    Same input depths, output width, zero up-projection and 10% cap as the local
    fusion arm. NOT official SciceVPR's last-four-layer/GeM/distillation pipeline.
    Original per-depth projections/stages are removed, not unused fillers.
    """
    def __init__(self, original, grid):
        super().__init__()
        channels = original.up.out_channels; dim = original.up.in_channels
        self.grid = tuple(grid)
        self.norms = original.norms
        self.channel = nn.Conv2d(3*channels, dim, 1)
        self.tokenmix = nn.ModuleList([TokenMixer(dim, grid[0]*grid[1]) for _ in range(2)])
        self.aux = original.aux  # Same legacy zero-weight heads; never effectively updated.
        self.up = original.up

    def forward(self, layers, auxiliary=False):
        if any(tuple(f.shape[-2:]) != self.grid for f in layers):
            raise ValueError('Token mixer requires the preregistered native grid')
        normalized = [n(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(f)
                      for f, n in zip(layers, self.norms)]
        g = F.relu(self.channel(torch.cat(normalized, 1))).flatten(2)
        for mixer in self.tokenmix:
            g = mixer(g)
        g = g.reshape(len(g), -1, *self.grid)
        last = layers[-1]; rms = last.square().mean(1, keepdim=True).sqrt().detach()
        residual = .1*rms*self.up(g).tanh()
        auxiliaries = tuple(F.normalize(head(g.mean((2, 3))), dim=-1) for head in self.aux) if auxiliary else ()
        return last+residual, auxiliaries, residual


def build_model(visual, arm, initialization_seed, grid=(20, 20), **kwargs):
    if arm not in (*ARMS, 'conv'):
        raise ValueError('Unknown convolution control')
    model = original_model(visual, 'conv', initialization_seed, **kwargs)
    if arm == 'conv_last':
        model.mode = 'last_only'  # Exact same neck; only input depth changes.
    elif arm == 'concat_token':
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed+17000)
            model.neck = ConcatTokenNeck(model.neck, grid)
    model.control_arm = arm
    return model


def interface_parameters(model):
    # Shared normalization, residual output and disabled heads across mechanisms.
    return {n: p for n, p in model.named_parameters() if n.startswith(
        ('neck.norms.', 'neck.up.', 'neck.aux.'))}


def gradient_inventory(arm):
    return ('channel', 'token0', 'token1') if arm == 'concat_token' else ('0', '1')


def module_gradient_status(model):
    if model.control_arm != 'concat_token':
        return original_gradients(model)
    result = {}
    for key, prefix in [('channel', 'neck.channel.'), ('token0', 'neck.tokenmix.0.mix.'), ('token1', 'neck.tokenmix.1.mix.')]:
        result[key] = any(n.startswith(prefix) and p.grad is not None and bool((p.grad != 0).any())
                          for n, p in model.named_parameters())
    return result
