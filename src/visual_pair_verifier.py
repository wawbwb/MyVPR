"""Small visual-only set verifier. Not a reproduction of Pair-VPR."""
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def pair_edges(query, database, count=64):
    """Native 20x20 tokens -> top mutual-NN edges, padded with validity flag.

    Channels: cosine, qx, qy, dx, dy, x displacement, y displacement, valid.
    No semantics, attention proxy, CLIP grid pooling or ground truth is used.
    """
    if query.shape[-2] != 400 or database.shape[-2] != 400:
        raise ValueError('Expected native 20x20 DINO tokens')
    if not 1 <= count <= 400:
        raise ValueError('Invalid edge count')
    sim = F.normalize(query.float(), dim=-1) @ F.normalize(database.float(), dim=-1).transpose(-1, -2)
    values, right = sim.max(-1)
    reverse = sim.argmax(-2)
    left = torch.arange(400, device=sim.device).expand_as(right)
    valid = reverse.gather(-1, right) == left
    chosen = values.masked_fill(~valid, -2).topk(count, dim=-1).indices
    dst = right.gather(-1, chosen)
    good = valid.gather(-1, chosen)
    qx, qy = (chosen % 20).float() / 19, (chosen // 20).float() / 19
    dx, dy = (dst % 20).float() / 19, (dst // 20).float() / 19
    edges = torch.stack([values.gather(-1, chosen), qx, qy, dx, dy,
                         dx-qx, dy-qy, good.float()], -1)
    return edges * good.unsqueeze(-1)


class VisualPairVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(130, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, edges, ru):
        valid = edges[..., 7:8]
        h = self.edge(edges[..., :7])
        mean = (h * valid).sum(-2) / valid.sum(-2).clamp_min(1)
        maximum = h.masked_fill(valid == 0, -1e4).amax(-2)
        maximum = torch.where(valid.sum(-2) > 0, maximum, torch.zeros_like(maximum))
        return self.head(torch.cat([mean, maximum, valid.mean(-2), ru.unsqueeze(-1)], -1)).squeeze(-1)


def anchored_score(ru, residual, alpha):
    return ru + alpha * residual.tanh()
