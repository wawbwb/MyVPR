"""SegVLAD-inspired region retrieval primitives (not an official reproduction).

Reference: Revisit Anything, ECCV 2024, AnyLoc/Revisit-Anything/func_vpr.py.
Independent implementation: cosine assignment, union of neighbouring masks,
intra-cluster and descriptor normalization, similarity-weighted image voting.
"""
import numpy as np
import torch
from torch.nn import functional as F


def grid_masks(count, size=280):
    """Exactly count non-overlapping rectangles, with full image coverage."""
    if not 1 <= count <= size * size:
        raise ValueError('Invalid grid count')
    boxes = [(0, size, 0, size)]
    while len(boxes) < count:
        i = max(range(len(boxes)), key=lambda k: (boxes[k][1]-boxes[k][0]) * (boxes[k][3]-boxes[k][2]))
        y0, y1, x0, x1 = boxes.pop(i)
        if y1-y0 >= x1-x0:
            mid = (y0+y1)//2
            boxes.extend([(y0, mid, x0, x1), (mid, y1, x0, x1)])
        else:
            mid = (x0+x1)//2
            boxes.extend([(y0, y1, x0, mid), (y0, y1, mid, x1)])
    masks = np.zeros((count, size, size), dtype=bool)
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        masks[i, y0:y1, x0:x1] = True
    return masks


def select_masks(masks, count):
    """Deterministic area-first subset; no GT or feature-score selection."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or len(masks) < count or not masks.reshape(len(masks), -1).any(1).all():
        raise ValueError('Not enough nonempty masks')
    order = np.argsort(-masks.sum((1, 2)), kind='stable')
    return masks[order[:count]]


def neighbour_union(masks, order=1):
    """Centroid Delaunay adjacency. Degenerate cases use identity, explicitly.

    Unlike the official small-mask special case, never drops the seed region.
    Returns pixel masks and whether the conservative fallback was required.
    """
    from scipy.spatial import Delaunay, QhullError
    masks = np.asarray(masks, dtype=bool)
    n = len(masks)
    if masks.ndim != 3 or n == 0 or order < 0 or not masks.reshape(n, -1).any(1).all():
        raise ValueError('Invalid masks/order')
    adjacency = np.eye(n, dtype=bool)
    fallback = False
    if order:
        points = np.array([np.argwhere(m).mean(0) for m in masks])
        if n < 4 or len(np.unique(points, axis=0)) != n:
            fallback = True
        else:
            try:
                for simplex in Delaunay(points).simplices:
                    adjacency[np.ix_(simplex, simplex)] = True
            except QhullError:
                fallback = True
        reach = adjacency.copy()
        for _ in range(order-1):
            reach = (reach.astype(np.int32) @ adjacency.astype(np.int32)) > 0
        adjacency = reach
    merged = np.stack([masks[row].any(0) for row in adjacency])
    return merged, fallback


def patch_membership(masks, grid=20):
    """Official-style ANY-pixel membership, not a majority-area threshold."""
    m, h, w = masks.shape
    if h % grid or w % grid:
        raise ValueError('Mask resolution must be divisible by token grid')
    return masks.reshape(m, grid, h//grid, grid, w//grid).any((2, 4)).reshape(m, -1)


def region_vlad(tokens, centers, membership):
    if tokens.ndim != 2 or centers.ndim != 2 or tokens.shape[1] != centers.shape[1]:
        raise ValueError('Incompatible feature/vocabulary shapes')
    if membership.ndim != 2 or membership.shape[1] != len(tokens) or not membership.any(1).all():
        raise ValueError('Invalid region membership')
    x = F.normalize(tokens.float(), dim=1)
    c = centers.to(x).float()
    labels = (x @ F.normalize(c, dim=1).T).argmax(1)
    residual = x-c[labels]
    masks = membership.to(device=x.device, dtype=x.dtype)
    parts = []
    for k in range(len(c)):
        selected = labels == k
        parts.append(F.normalize(masks[:, selected] @ residual[selected], dim=1))
    result = F.normalize(torch.stack(parts, dim=1).flatten(1), dim=1)
    if not torch.isfinite(result).all() or (result.norm(dim=1) < 1e-8).any():
        raise ValueError('Nonfinite/zero region VLAD')
    return result


def image_vote(region_ids, similarities, owners, minimum, maximum, top_k=20):
    """Sum min-max normalized region similarities by database image ID.

    Bounds are shared over the complete query search (no GT used). No RU
    fallback is inserted into this standalone regional ranking.
    """
    if maximum <= minimum:
        raise ValueError('Degenerate region similarities')
    ids = np.asarray(owners)[np.asarray(region_ids).reshape(-1)]
    weights = np.clip((np.asarray(similarities).reshape(-1)-minimum)/(maximum-minimum), 0, 1)
    scores = np.bincount(ids, weights=weights)
    present = np.unique(ids)
    order = np.lexsort((present, -scores[present]))
    result = np.full(top_k, -1, dtype=np.int64)
    selected = present[order[:top_k]]
    result[:len(selected)] = selected
    return result


def equal_budget_union(ru, regional, top_k=20):
    """RU10 + region10, then RU20 fill; never consults ground truth."""
    half = top_k//2
    ordered = list(dict.fromkeys(int(i) for i in [*ru[:half], *regional[:half], *ru[:top_k]] if i >= 0))
    if len(ordered) < top_k:
        raise ValueError('Insufficient unique candidates')
    return np.asarray(ordered[:top_k], dtype=np.int64)
