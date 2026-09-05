"""Shared feature extraction helpers for CC-LSA calibration and MSLS cache."""

from __future__ import annotations

import torch
import numpy as np
from torch.nn import functional as F


CC_LSA_LOCAL_GRID = (14, 14)
CC_LSA_DINO_FEATURE_STAGE = (
    "ru_trained_dinov2_final_patch_map_pre_semantic_region_gate_pre_boq"
)


def mutual_nearest_edges_batch(
    query: np.ndarray, candidates: np.ndarray, *, device: torch.device
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Shared FP32 matcher for GSV calibration and MSLS audit.

    Disable TF32 explicitly and copy whole batched reductions once, rather
    than synchronising the GPU for every candidate's individual edge list.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    query_t = normalise_local_tokens(
        torch.tensor(np.asarray(query), device=device, dtype=torch.float32)[None]
    )[0]
    candidate_t = normalise_local_tokens(
        torch.tensor(np.asarray(candidates), device=device, dtype=torch.float32)
    )
    with torch.inference_mode():
        similarity = torch.einsum("id,kjd->kij", query_t, candidate_t)
        q_to_c = similarity.argmax(dim=2)
        c_to_q = similarity.argmax(dim=1)
        scores = similarity.gather(2, q_to_c.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        right = q_to_c.cpu().numpy()
        reverse = c_to_q.cpu().numpy()
    left = np.arange(len(query), dtype=np.int32)
    return [
        (left[reverse[k, row] == left], row[reverse[k, row] == left].astype(np.int32),
         scores[k, reverse[k, row] == left].astype(np.float32))
        for k, row in enumerate(right)
    ]


def normalise_local_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or not tokens.is_floating_point():
        raise ValueError("local tokens must have shape (B,N,D)")
    tokens = tokens.float()
    norms = tokens.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(tokens).all()) or bool((norms <= 1e-12).any()):
        raise RuntimeError("local tokens are non-finite or zero norm")
    return F.normalize(tokens, dim=-1)


def extract_ru_descriptor_and_local(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    output_grid: tuple[int, int] = CC_LSA_LOCAL_GRID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one RU forward and return descriptor plus pre-gate DINO tokens."""

    backbone_output = model.backbone(images)
    if hasattr(model, "_split_backbone_output"):
        featmap, restore = model._split_backbone_output(backbone_output)
    elif isinstance(backbone_output, (tuple, list)):
        featmap = backbone_output[0]
        tail = tuple(backbone_output[1:])
        restore = lambda local: (local, *tail)
    else:
        featmap = backbone_output
        restore = lambda local: local
    if featmap.ndim != 4:
        raise ValueError("RU backbone must return a 4-D patch feature map")
    pooled = F.adaptive_avg_pool2d(featmap.float(), output_grid)
    local = normalise_local_tokens(pooled.flatten(2).transpose(1, 2))
    routed = featmap
    semantic_gate = getattr(model, "semantic_region_gate", None)
    if semantic_gate is not None:
        routed, _, _ = semantic_gate(routed)
    spatial_head = getattr(model, "spatial_attn_head", None)
    if spatial_head is not None:
        routed, _ = spatial_head(routed)
    descriptor = model.aggregator(restore(routed))
    if isinstance(descriptor, (tuple, list)):
        descriptor = descriptor[0]
    if descriptor.ndim != 2 or not bool(torch.isfinite(descriptor).all()):
        raise RuntimeError("RU aggregator returned invalid descriptors")
    descriptor = F.normalize(descriptor.float(), dim=-1)
    return descriptor, local
