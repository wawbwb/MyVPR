"""Train-only cross-view supervision for image-only, centered scale allocation.

FC output groups are algebraic descriptor groups, not spatial regions.
"""
import torch
from torch.nn import functional as F
from src.models.scale_agreement_fusion import ScaleAgreementFusion, agreement_features, mix
from src.models.complementary_covariance import permute_positions

ARMS = ('quality_group', 'quality_static', 'quality_global', 'quality_shuffled', 'ms_group')


def centered(value):
    # tanh values in [-1,1]; centering then halving keeps this in [-1,1].
    return .5*(value-value.mean(1, keepdim=True))


class ScaleReliabilityGate(ScaleAgreementFusion):
    def __init__(self, arm, base, channels=384, groups=32):
        if arm not in ARMS or not -.5 < base < .5:
            raise ValueError('Invalid reliability gate')
        super().__init__('group_gate', channels, groups)
        self.reliability_arm = arm
        self.register_buffer('base', torch.tensor(float(base)))
        self.register_buffer('radius', torch.tensor(min(.35, .5-abs(float(base)))))

    def scores(self, low, high, seeds=None):
        if low.shape != high.shape or low.ndim != 2 or low.shape[1] != self.channels*self.groups:
            raise ValueError('Matched flattened descriptors required')
        feature = agreement_features(low.reshape(-1, self.channels, self.groups), high.reshape(-1, self.channels, self.groups))
        if self.reliability_arm == 'quality_static':
            feature = self.static_code[None].expand(len(low), -1, -1)
        elif self.reliability_arm == 'quality_global':
            feature = feature.mean(1, keepdim=True).expand(-1, self.groups, -1)
        value = self.up(F.gelu(self.down(feature))).squeeze(-1).tanh()
        if self.reliability_arm == 'quality_shuffled':
            if seeds is None:
                raise ValueError('Content seeds required')
            value = permute_positions(value[:, :, None], seeds).squeeze(-1)
        return value

    def gates(self, low, high, seeds=None):
        score = self.scores(low, high, seeds)
        residual = score if self.reliability_arm == 'quality_global' else centered(score)
        return self.base+self.radius*residual

    def forward(self, low, high, seeds=None):
        gate = self.gates(low, high, seeds)
        return mix(low, high, gate, self.channels, self.groups), gate


@torch.no_grad()
def cross_view_teacher(low, high, labels, coordinates, channels=384, groups=32, temperature=.1):
    """Leave-self-out positive support minus a COMMON GPS-safe hard negative.

    The negative is selected with the fixed equal-scale descriptor, then used
    for both branches and all groups. Labels/GPS exist only in training.
    """
    n = len(low)
    if low.shape != high.shape or low.shape != (n, channels*groups) or labels.shape != (n,) or coordinates.shape != (n, 2):
        raise ValueError('Invalid teacher batch')
    if temperature <= 0 or not torch.isfinite(low).all() or not torch.isfinite(high).all() or not torch.isfinite(coordinates).all():
        raise ValueError('Invalid teacher values')
    labels = labels.to(low.device)
    positive = labels[:, None].eq(labels[None, :]) & ~torch.eye(n, dtype=torch.bool, device=low.device)
    xy = torch.deg2rad(coordinates.to(low.device).double())
    lat, lon = xy[:, 0], xy[:, 1]
    distance = torch.sin((lat[:, None]-lat[None, :])/2).square()
    distance += torch.cos(lat[:, None])*torch.cos(lat[None, :])*torch.sin((lon[:, None]-lon[None, :])/2).square()
    safe = (6371000.*2*torch.asin(distance.clamp(0, 1).sqrt()) >= 25.) & labels[:, None].ne(labels[None, :])
    valid = positive.any(1) & safe.any(1)
    if not bool(valid.any()):
        raise ValueError('No valid cross-view teacher rows')
    average = F.normalize(low+high, dim=1)
    negative = (average@average.T).masked_fill(~safe, -torch.inf).argmax(1)
    margins = []
    for branch in (low, high):
        token = F.normalize(branch.reshape(n, channels, groups), dim=1)
        weights = positive.to(token.dtype)/positive.sum(1, keepdim=True).clamp_min(1)
        support = torch.einsum('ij,jcg->icg', weights, token)
        margins.append((token*(support-token[negative])).sum(1))
    target = ((margins[1]-margins[0])/temperature).tanh()
    return target, valid, dict(valid_rows=int(valid.sum()), positive_pairs=int(positive.sum()),
        safe_negative_pairs=int(safe.sum()), selected_negative_rows=negative.cpu().tolist(),
        target_mean=float(target[valid].mean()), target_within_image_std=float(target[valid].std(1, unbiased=False).mean()))
