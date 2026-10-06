"""Paired objectives on identical descriptors/labels; no optimization."""
import torch


@torch.no_grad()
def paired_losses(objective, baseline, trained, labels):
    if baseline.shape != trained.shape or baseline.ndim != 2 or len(labels) != len(baseline):
        raise ValueError('Unmatched paired batch')
    if not torch.isfinite(baseline).all() or not torch.isfinite(trained).all():
        raise ValueError('Nonfinite descriptors')
    original = objective.miner(baseline, labels)
    current = objective.miner(trained, labels)
    result = dict(ru_loss=float(objective.loss_fn(baseline, labels, original)),
                  trained_loss=float(objective.loss_fn(trained, labels, current)),
                  trained_ru_pairs_loss=float(objective.loss_fn(trained, labels, original)),
                  descriptor_l2=float((trained-baseline).norm(dim=1).mean()))
    for name, indices in (('ru', original), ('trained', current)):
        if len(indices) != 4: raise ValueError('Expected pair miner')
        result[name+'_positive_pairs'] = len(indices[0])
        result[name+'_negative_pairs'] = len(indices[2])
        result[name+'_positive_anchors'] = len(indices[0].unique())
    if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
        raise ValueError('Nonfinite paired objective')
    return result
