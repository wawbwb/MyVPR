"""Frozen Pair-VPR decoder prefix/continuation; no model or score changes."""
import torch
from torch.nn import functional as F


def prefix(model, first, second):
    if len(model.dec_blocks) != 12 or model.decoder_clstoken is None:
        raise ValueError('Expected stage-two 12-layer Pair-VPR')
    x = model.decoder_embed(first)
    y = model.decoder_embed(second)
    x = torch.cat((model.decoder_clstoken.expand(len(x), -1, -1), x), 1)
    x = x + model.dec_pos_embed_cls
    y = y + model.dec_pos_embed
    for block in model.dec_blocks[:2]:
        x, y = block(x, y)
    return x, y


def finish(model, state):
    x, y = state
    for block in model.dec_blocks[2:]:
        x, y = block(x, y)
    return model.classvprmodule(model.dec_norm(x)[:, 0]).flatten()


def progressive(model, head, query, database):
    """Batch=1 per direction, matching full baseline; keep states on GPU."""
    if len(database) != 44:
        raise ValueError('Expected 44 candidates')
    states = []
    features = []
    for i in range(44):
        forward = prefix(model, query, database[i:i+1])
        backward = prefix(model, database[i:i+1], query)
        states.append((forward, backward))
        features.append(torch.cat([F.layer_norm(s[0][:, 0].float(), (768,))
                                   for s in (forward, backward)], -1))
    prediction = head(torch.cat(features)).flatten()
    keep = torch.argsort(prediction, descending=True, stable=True)[:12]
    # Release rejected activations before deep continuation, without recomputation.
    indices = keep.tolist()
    survivors = [states[i] for i in indices]
    del states, forward, backward, features
    scores = torch.cat([(finish(model, f) + finish(model, b)) for f, b in survivors])
    return keep, scores, prediction


def full(model, query, database, count):
    return torch.cat([(model(query, database[i:i+1], 'pairvpr') +
                       model(database[i:i+1], query, 'pairvpr')).flatten()
                      for i in range(count)])
