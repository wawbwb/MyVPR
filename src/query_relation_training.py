"""Fixed expanded-gallery diagnostic. Never used for checkpoint selection."""
import torch
import numpy as np


def protocol_schedule(plan, sampling):
    """Same 256 broad batches; replace the 64 hard slots with random place batches.

    Extra broad draws are without replacement within the 1024-place epoch draw.
    They can repeat places in the base 4096 coverage, as hard slots also do.
    No development identity or benchmark result is consulted.
    """
    from src.dsa_training import schedules
    if sampling not in ('mixed', 'broad_matched'):
        raise ValueError('Unknown sampling policy')
    result = schedules(plan)
    if sampling == 'broad_matched':
        for epoch, batches in enumerate(result):
            extra = np.random.default_rng(52031 + epoch).permutation(4096)[:1024]
            for block in range(64):
                batches[block * 5 + 4] = [plan['train_places'][int(i)]
                                         for i in extra[block*16:(block+1)*16]]
    return result


@torch.no_grad()
def score_development(features,query_start,chunk=32):
    if features.ndim!=2 or len(features)%4 or query_start%4 or not 0<=query_start<len(features):
        raise ValueError('Expected four views/place and a nonempty query suffix')
    if not torch.isfinite(features).all():raise ValueError('Nonfinite descriptors')
    labels=torch.arange(len(features),device=features.device)//4
    predictions,losses=[],[]
    for start in range(query_start,len(features),chunk):
        scores=features[start:start+chunk]@features.T
        rows=torch.arange(len(scores),device=features.device)
        scores[rows,start+rows]=-torch.inf
        pos=labels[start:start+len(scores),None]==labels[None]
        positive=scores.masked_fill(~pos,-torch.inf).amax(1)
        negative=scores.masked_fill(pos,-torch.inf).amax(1)
        losses.extend(torch.nn.functional.softplus((negative-positive)/.05).cpu().tolist())
        predictions.extend(scores.argmax(1).cpu().tolist())
    hits=np.asarray(predictions)//4==np.arange(query_start,len(features))//4
    return dict(correct=int(hits.sum()),queries=len(hits),gallery=len(features),
                margin_loss=float(np.mean(losses)),predictions=predictions,outcomes=hits.tolist())
