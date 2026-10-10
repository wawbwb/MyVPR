"""Single-image local-to-global student; pair matching is training-only.

OT implementation is reused from the repository's SALAD implementation.
This is an experimental adaptation, not a reproduction of SelaVPR/SALAD.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.aggregators.salad import log_optimal_transport


class LocalProjector(nn.Module):
    def __init__(self, channels=768, dim=128):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 256),
                                    nn.GELU(), nn.Linear(256, dim))

    def forward(self, tokens):
        return F.normalize(self.layers(tokens), dim=-1)


def local_tokens(features, side=10):
    """100 native pooled tokens, not dense ground-truth correspondences."""
    return F.adaptive_avg_pool2d(features, (side, side)).flatten(2).transpose(1, 2)


def mutual_score(left, right):
    """Differentiable mean cosine of detached mutual-NN indices, per pair.

    Both directions describe the same mutual pairs; count each pair once.
    No match gives zero rather than a NaN. Counts are diagnostics, not losses.
    """
    if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]:
        raise ValueError('Expected paired B,N,D tensors')
    if left.shape[-1] != right.shape[-1] or not left.shape[1] or not right.shape[1]:
        raise ValueError('Invalid token dimensions')
    similarity = left @ right.transpose(1, 2)
    forward = similarity.detach().argmax(2)
    backward = similarity.detach().argmax(1)
    indices = torch.arange(left.shape[1], device=left.device)[None]
    mutual = backward.gather(1, forward) == indices
    values = similarity.gather(2, forward[..., None]).squeeze(2)
    count = mutual.sum(1)
    score = (values * mutual).sum(1) / count.clamp_min(1)
    return score, count


def mine_pairs(global_descriptors, labels):
    """Fixed RU difficulty: hardest positive and hardest negative per anchor."""
    if len(global_descriptors) != len(labels):
        raise ValueError('Descriptor/label count differs')
    scores = global_descriptors.detach() @ global_descriptors.detach().T
    same = labels[:, None] == labels[None]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    if not positive.any(1).all() or not (~same).any(1).all():
        raise ValueError('Each anchor needs a positive and a different-place negative')
    pos = scores.masked_fill(~positive, torch.inf).argmin(1)
    neg = scores.masked_fill(same, -torch.inf).argmax(1)
    return pos, neg


def local_objective(tokens, pos, neg, margin=.05):
    ps, pc = mutual_score(tokens, tokens[pos])
    ns, nc = mutual_score(tokens, tokens[neg])
    valid = (pc >= 3) & (nc >= 3)
    losses = F.relu(margin - ps + ns)
    loss = losses[valid].mean() if valid.any() else tokens.sum() * 0
    return loss, ps - ns, valid


def compression_loss(descriptors, pos, neg, teacher_margin, reliable,
                     temperature=.07):
    """Bernoulli relational KD; target margins are detached from the teacher.

    Only positive teacher margins (and >=3 mutual matches each side) supervise.
    Both KL target and student use the same temperature. Zero/incorrect targets
    abstain, with coverage reported separately by the trainer.
    """
    if not reliable.any():
        return descriptors.sum() * 0
    student_margin = ((descriptors * descriptors[pos]).sum(1)
                      - (descriptors * descriptors[neg]).sum(1))
    target = torch.sigmoid(teacher_margin.detach() / temperature)[reliable]
    logits = student_margin[reliable] / temperature
    # Cross entropy minus target entropy = KL; its constant does not affect gradients.
    entropy = -(target * target.clamp_min(1e-7).log()
                + (1-target) * (1-target).clamp_min(1e-7).log())
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            - entropy).mean() * temperature**2


class SLGDStudent(nn.Module):
    """RU global branch + shared OT local slots; one fixed vector/dot product.

    Keep the trained RU global dimension instead of a randomly initialized
    256-D compression, so this first screen does not conflate two compressions.
    A and B have exactly the same architecture and initialization.
    """
    def __init__(self, visual, teacher, slots=16, local_dim=128, beta=.2):
        super().__init__()
        if not 0 < beta < 1 or slots < 1:
            raise ValueError('Require 0 < beta < 1 and positive slot count')
        if visual.backbone.crop_semantic_film is not None or visual.backbone.residual_clip_fusion is not None:
            raise ValueError('Expected original RU backbone')
        self.visual = visual.requires_grad_(False)
        self.visual.aggregator.requires_grad_(True)
        self.local = copy.deepcopy(teacher)
        self.local.requires_grad_(True)
        self.slot_score = nn.Linear(local_dim, slots)
        self.dustbin = nn.Parameter(torch.tensor(1.))
        self.beta = beta
        self.slots = slots

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        # The pretrained BoQ branch is trainable but deterministic (no dropout).
        return self

    def aggregate(self, features, return_training_tokens=True):
        gated = features
        if self.visual.semantic_region_gate is not None:
            gated = self.visual.semantic_region_gate(gated)[0]
        global_result = self.visual.aggregator(gated)
        global_descriptor = global_result[0] if isinstance(global_result, tuple) else global_result
        tokens = self.local(features.flatten(2).transpose(1, 2))
        if tokens.shape[1] <= self.slots:
            raise ValueError('OT requires more tokens than slots')
        assignments = log_optimal_transport(self.slot_score(tokens).transpose(1, 2),
                                            self.dustbin, 5).exp()[:, :-1]
        slot_features = assignments @ tokens
        local_descriptor = F.normalize(F.normalize(slot_features, dim=-1).flatten(1), dim=-1)
        global_descriptor = F.normalize(global_descriptor, dim=-1)
        descriptor = torch.cat((math.sqrt(1-self.beta) * global_descriptor,
                                math.sqrt(self.beta) * local_descriptor), dim=1)
        training_tokens = self.local(local_tokens(features)) if return_training_tokens else None
        return descriptor, training_tokens

    def forward(self, images):
        with torch.no_grad():
            features = self.visual.backbone(images)
            if isinstance(features, tuple):
                features = features[0]
        return self.aggregate(features, return_training_tokens=False)[0]
