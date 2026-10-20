"""Fixed-pool hard negatives. No CLIP-dependent candidate selection."""
import numpy as np


def distances(left, right):
    left, right = np.deg2rad(np.asarray(left, float)), np.deg2rad(np.asarray(right, float))
    a = np.sin((left[..., 0]-right[..., 0])/2)**2
    a += np.cos(left[..., 0])*np.cos(right[..., 0])*np.sin((left[..., 1]-right[..., 1])/2)**2
    return 6371000*2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def choose_negatives(plan, query_index, pool, similarities, count=4, minimum_metres=25.):
    query = plan['places'][query_index]
    if len(pool) != len(similarities) or not np.isfinite(similarities).all() or count < 1:
        raise ValueError('Invalid candidate scores/count')
    rows = [plan['images'][plan['places'][j]['rows'][1]] for j in pool]
    xy = np.asarray([row['coordinates'] for row in rows], float)
    eligible = np.asarray([j != query_index and plan['places'][j]['split'] == query['split']
        and plan['places'][j]['city'] == query['city'] for j in pool])
    for image_index in query['rows'][:2]:
        eligible &= distances(plan['images'][image_index]['coordinates'], xy) >= minimum_metres
    allowed = np.flatnonzero(eligible)
    # Stable global-place-ID tie break, independent of CLIP and cache order.
    order = sorted(allowed, key=lambda k: (-float(similarities[k]), pool[k]))[:count]
    return [pool[k] for k in order], int(eligible.sum())
