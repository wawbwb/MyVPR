"""RGB-only student; CLIP is used by external, stop-gradient training graphs.

Independent wrapper: none of the archived adapter/innovation-1 files change.
The small visual router is an experimental component, not a novelty claim.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.clip_key_boq import KeyRouter
from src.models.official_boq import native_feature_layout

ARMS = ('no_correspondence', 'dino', 'clip', 'shuffled')


class ChannelAdapter(nn.Module):
    def __init__(self, channels, hidden=64, gain=.1):
        super().__init__()
        if min(channels, hidden) < 1 or not 0 < gain <= 1:
            raise ValueError('Invalid adapter dimensions/gain')
        self.down = nn.Conv2d(channels, hidden, 1)
        self.middle = nn.Conv2d(hidden, hidden, 1, groups=hidden)
        self.up = nn.Conv2d(hidden, channels, 1)
        self.gain = float(gain)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return features + self.gain*self.up(F.gelu(self.middle(F.gelu(self.down(features)))))


class CVSlotBoQ(nn.Module):
    def __init__(self, official, hidden=64, gain=.1, bias_cap=2., grid=14):
        super().__init__()
        if grid < 1:
            raise ValueError('Invalid slot grid')
        self.visual = official.requires_grad_(False).eval()
        channels = official.aggregator.proj_c.in_channels
        self.adapter = ChannelAdapter(channels, hidden, gain)
        self.routers = nn.ModuleList([KeyRouter(block.queries.shape[-1], block.queries.shape[1],
            semantic_channels=channels, hidden=hidden, cap=bias_cap, temperature=5.)
            for block in official.aggregator.boqs])
        self.grid = grid

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        return self.visual.backbone(images)

    def describe(self, features, baseline=False):
        if features.ndim != 4:
            raise ValueError('Expected BCHW frozen DINO map')
        adapted = features if baseline else self.adapter(features)
        b, _, h, w = adapted.shape
        visual_tokens = adapted.flatten(2).transpose(1, 2)
        agg = self.visual.aggregator
        # Frozen weights still transmit gradients to the input adapter. Do not
        # wrap this path in no_grad during student replay.
        x = agg.norm_input(agg.proj_c(native_feature_layout(adapted)).flatten(2).transpose(1, 2))
        outputs, assignments = [], []
        for block, router in zip(agg.boqs, self.routers):
            x = block.encoder(x)
            with torch.no_grad():
                q = block.queries.repeat(b, 1, 1)
                q = block.norm_q(q + block.self_attn(q, q, q)[0])
            is_zero = baseline or not bool(torch.count_nonzero(router.beta.detach()))
            if baseline or (is_zero and not torch.is_grad_enabled()):
                out, weights = block.cross_attn(q, x, x)
            else:
                bias = router(q, visual_tokens)
                mask = bias[:, None].expand(b, block.cross_attn.num_heads, q.size(1), x.size(1))
                mask = mask.reshape(b*block.cross_attn.num_heads, q.size(1), x.size(1))
                guided, guided_weights = block.cross_attn(q, x, x, attn_mask=mask)
                if is_zero:
                    # Exact unmasked CUDA forward; ONE masked mathematical
                    # derivative, not a doubled input-adapter gradient.
                    with torch.no_grad():
                        base, base_weights = block.cross_attn(q, x, x)
                    out = base + (guided-guided.detach())
                    weights = base_weights + (guided_weights-guided_weights.detach())
                else:
                    out, weights = guided, guided_weights
            outputs.append(block.norm_out(out))
            slots = F.adaptive_avg_pool2d(weights.reshape(b, q.size(1), h, w), (self.grid, self.grid)).flatten(2)
            assignments.append(slots / slots.sum(-1, keepdim=True).clamp_min(1e-12))
        descriptor = agg.fc(torch.cat(outputs, 1).transpose(1, 2)).flatten(1)
        return F.normalize(descriptor, dim=-1), torch.stack(assignments, dim=1)

    def forward(self, images):
        # No CLIP, labels, paired views or cached semantics at inference.
        return self.describe(self.encode(images))[0]

    def student_state(self):
        return dict(adapter=self.adapter.state_dict(), routers=self.routers.state_dict())

    def load_student(self, state):
        if set(state) != {'adapter', 'routers'}:
            raise ValueError('Student checkpoint inventory differs')
        self.adapter.load_state_dict(state['adapter'], strict=True)
        self.routers.load_state_dict(state['routers'], strict=True)


@torch.no_grad()
def transport_matrix(graph, arm, row_peak_floor=.25):
    if arm not in ('dino', 'clip', 'shuffled') or not 0 < row_peak_floor <= 1:
        raise ValueError('Invalid teacher arm/peak threshold')
    destinations, support = graph['destinations'], graph['support']
    n = len(destinations)
    if destinations.max() >= n:
        raise ValueError('Only equal-grid view correspondence is supported')
    # The reliability/overlap selector is DINO-only and identical in all arms.
    valid = support.any(-1) & (graph['distributions']['dino'].max(-1).values >= row_peak_floor)
    matrix = graph['distributions'][arm].new_zeros(n, n)
    matrix.scatter_add_(1, destinations, graph['distributions'][arm]*valid[:, None])
    return matrix.detach()


def transported_js(source, destination, matrix):
    """Conditional-overlap JS, averaged over valid block/query slots.

    Both student views receive gradients. Empty overlap abstains with a finite
    connected zero; no fake uniform target is inserted.
    """
    if source.shape != destination.shape or source.shape[-1] != len(matrix):
        raise ValueError('Assignment/transport geometry differs')
    reachable = matrix.sum(0) > 0
    p = source @ matrix.detach()
    q = destination * reachable
    mass_p, mass_q = p.sum(-1, keepdim=True), q.sum(-1, keepdim=True)
    valid = (mass_p[..., 0] > 1e-8) & (mass_q[..., 0] > 1e-8)
    p = p / mass_p.clamp_min(1e-8)
    q = q / mass_q.clamp_min(1e-8)
    midpoint = (p+q)/2
    js = .5*((p*(p.clamp_min(1e-8).log()-midpoint.clamp_min(1e-8).log())).sum(-1)
             +(q*(q.clamp_min(1e-8).log()-midpoint.clamp_min(1e-8).log())).sum(-1))
    if not bool(valid.any()):
        return (source.sum()+destination.sum())*0
    return js[valid].mean()


def slot_gram_loss(slots, reference):
    if slots.shape != reference.shape:
        raise ValueError('Baseline/student slot geometry differs')
    a = F.normalize(slots, dim=-1)
    b = F.normalize(reference.detach(), dim=-1)
    return F.mse_loss(a @ a.transpose(-1, -2), b @ b.transpose(-1, -2))


def correspondence_loss(slots, graphs, arm, row_peak_floor=.25):
    if arm not in ARMS:
        raise ValueError('Unknown student arm')
    if arm == 'no_correspondence':
        return slots.sum()*0
    losses = []
    for a, b, ab, ba in graphs:
        losses.append(transported_js(slots[a], slots[b], transport_matrix(ab, arm, row_peak_floor)))
        losses.append(transported_js(slots[b], slots[a], transport_matrix(ba, arm, row_peak_floor)))
    if not losses:
        return slots.sum()*0
    return torch.stack(losses).mean()


def replay_backward(model, features, descriptor_gradient, slot_gradient, microbatch):
    if len(features) != len(descriptor_gradient) or len(features) != len(slot_gradient) or microbatch < 1:
        raise ValueError('Invalid gradient-cache populations/microbatch')
    for start in range(0, len(features), microbatch):
        stop = start+microbatch
        z, slots = model.describe(features[start:stop])
        torch.autograd.backward((z, slots), (descriptor_gradient[start:stop], slot_gradient[start:stop]))
