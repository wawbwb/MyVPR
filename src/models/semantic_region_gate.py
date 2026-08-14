"""Local semantic-region targets and bounded DINO feature gating.

Place supervision is produced only from DINO tokens and the existing
place-grouped batch.  A precomputed, sparse CLIP affinity may propagate that
reliability inside local semantic regions; it never selects positive or
negative places.  The cache is training-only.  Inference retains just the
small residual gate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticRegionGate(nn.Module):
    """Predict a bounded residual modulation for local backbone features."""

    def __init__(self, in_channels: int, alpha: float = 0.2) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if not 0.0 <= alpha <= 0.2:
            raise ValueError("alpha must be in [0, 0.2]")
        self.alpha = float(alpha)
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1)
        # The first forward is exactly the visual baseline.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def predict(self, featmap: torch.Tensor) -> torch.Tensor:
        if featmap.ndim != 4:
            raise ValueError("featmap must have shape (B,C,H,W)")
        return torch.tanh(self.proj(featmap).float())

    def forward(
        self, featmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = self.predict(featmap)
        gate = 1.0 + self.alpha * score
        return featmap * gate.to(featmap.dtype), score, gate


class SemanticRegionReliabilityTarget(nn.Module):
    """Build detached local targets from DINO and cached CLIP regions.

    Modes:
      * ``repeatability_only``: same-place DINO repeatability only.
      * ``repeatability_uniqueness_only``: DINO repeatability + uniqueness,
        without semantic propagation (the key no-semantics control).
      * ``semantic_only``: CLIP region confidence only (control).
      * ``full``: CLIP regions propagate DINO repeatability and uniqueness.
      * ``shuffled``: as ``full``, but the CLIP cache is rolled by place.
    """

    MODES = {
        "repeatability_only",
        "repeatability_uniqueness_only",
        "semantic_only",
        "full",
        "shuffled",
    }

    def __init__(
        self,
        mode: str = "full",
        match_grid: int = 10,
        target_scale: float = 2.0,
        place_chunk_size: int = 8,
        min_spatial_std: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if match_grid < 1:
            raise ValueError("match_grid must be positive")
        if place_chunk_size < 1:
            raise ValueError("place_chunk_size must be positive")
        if target_scale <= 0 or min_spatial_std < 0 or eps <= 0:
            raise ValueError(
                "target_scale/eps must be positive and min_spatial_std non-negative"
            )
        self.mode = mode
        self.match_grid = int(match_grid)
        self.target_scale = float(target_scale)
        self.place_chunk_size = int(place_chunk_size)
        self.min_spatial_std = float(min_spatial_std)
        self.eps = float(eps)

    def _standardize(self, values: torch.Tensor) -> torch.Tensor:
        centre = values.mean(dim=-1, keepdim=True)
        scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
            self.eps
        )
        return (values - centre) / scale

    def _vpr_components(
        self, featmap: torch.Tensor, place_count: int, views_per_place: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = featmap.shape[0]
        if batch_size != place_count * views_per_place:
            raise ValueError("batch size must equal place_count * views_per_place")
        if place_count < 2 or views_per_place < 2:
            raise ValueError("local reliability requires at least 2 places and 2 views")

        # Pool before promoting to fp32: at P=40,K=4 this avoids a temporary
        # fp32 copy of the full 160x768x20x20 DINO tensor.
        pooled = F.adaptive_avg_pool2d(
            featmap.detach(), (self.match_grid, self.match_grid)
        ).flatten(2).transpose(1, 2).float()
        tokens = F.normalize(pooled, dim=-1, eps=self.eps)
        token_count = tokens.shape[1]
        tokens_pk = tokens.view(
            place_count, views_per_place, token_count, tokens.shape[-1]
        )

        # Position-independent nearest-neighbour matching across the K-1
        # other views of the same place.  Chunking places avoids hundreds of
        # tiny CUDA launches without creating a P-wide 5-D similarity tensor.
        repeatability = torch.empty(
            (place_count, views_per_place, token_count),
            device=tokens.device,
            dtype=torch.float32,
        )
        view_diagonal = torch.eye(
            views_per_place, device=tokens.device, dtype=torch.bool
        ).view(1, views_per_place, views_per_place, 1)
        for start in range(0, place_count, self.place_chunk_size):
            stop = min(start + self.place_chunk_size, place_count)
            chunk = tokens_pk[start:stop]
            chunk_size = stop - start
            flat_chunk = chunk.flatten(0, 1)
            pair_similarity = torch.bmm(
                flat_chunk,
                chunk[:, None]
                .expand(-1, views_per_place, -1, -1, -1)
                .reshape(
                    chunk_size * views_per_place,
                    views_per_place * token_count,
                    tokens.shape[-1],
                )
                .transpose(1, 2),
            )
            pair_similarity = pair_similarity.view(
                chunk_size,
                views_per_place,
                token_count,
                views_per_place,
                token_count,
            )
            best_match = pair_similarity.amax(dim=-1).permute(0, 1, 3, 2)
            best_match = best_match.masked_fill(view_diagonal, 0.0)
            repeatability[start:stop] = best_match.sum(dim=2) / (
                views_per_place - 1
            )

        # Pick the most visually confusable *different* place from a detached
        # DINO prototype, then estimate how common each local token is there.
        place_proto = F.normalize(
            tokens_pk.mean(dim=(1, 2)), dim=-1, eps=self.eps
        )
        place_similarity = place_proto @ place_proto.transpose(0, 1)
        place_similarity.fill_diagonal_(-torch.inf)
        hard_places = place_similarity.argmax(dim=1)
        negative_tokens = tokens_pk[hard_places].flatten(1, 2)

        commonness = torch.empty_like(repeatability)
        for start in range(0, place_count, self.place_chunk_size):
            stop = min(start + self.place_chunk_size, place_count)
            chunk_size = stop - start
            anchors = tokens_pk[start:stop].flatten(0, 1)
            expanded_negatives = (
                negative_tokens[start:stop, None]
                .expand(-1, views_per_place, -1, -1)
                .reshape(
                    chunk_size * views_per_place,
                    negative_tokens.shape[1],
                    negative_tokens.shape[2],
                )
            )
            negative_similarity = torch.bmm(
                anchors, expanded_negatives.transpose(1, 2)
            )
            commonness[start:stop] = negative_similarity.amax(dim=-1).view(
                chunk_size, views_per_place, token_count
            )

        repeatability = repeatability.flatten(0, 1)
        uniqueness = (1.0 - commonness).flatten(0, 1)
        stats = {
            "region_repeatability": repeatability.mean(),
            "region_uniqueness": uniqueness.mean(),
            "region_commonness": commonness.mean(),
            "region_hard_place_similarity": place_similarity.amax(dim=1).mean(),
        }
        return repeatability, uniqueness, stats

    @staticmethod
    def _square_side(token_count: int, name: str) -> int:
        side = math.isqrt(token_count)
        if side * side != token_count:
            raise ValueError(f"{name} token count must form a square grid")
        return side

    def _sparse_semantic_smooth(
        self,
        reliability: torch.Tensor,
        semantic_indices: torch.Tensor,
        semantic_weights: torch.Tensor,
        semantic_confidence: torch.Tensor,
        place_count: int,
        views_per_place: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = reliability.shape[0]
        if semantic_indices.ndim != 3:
            raise ValueError("semantic_indices must have shape (B,N,topk)")
        if semantic_indices.shape[2] < 1:
            raise ValueError("semantic cache topk dimension must be non-empty")
        if semantic_weights.shape != semantic_indices.shape:
            raise ValueError("semantic_weights must match semantic_indices")
        if semantic_confidence.shape != semantic_indices.shape[:2]:
            raise ValueError("semantic_confidence must have shape (B,N)")
        if semantic_indices.shape[0] != batch_size:
            raise ValueError("semantic cache batch size does not match features")

        if self.mode == "shuffled":
            cache_shape = (
                place_count,
                views_per_place,
                *semantic_indices.shape[1:],
            )
            semantic_indices = semantic_indices.view(cache_shape).roll(1, 0).flatten(0, 1)
            semantic_weights = semantic_weights.view(cache_shape).roll(1, 0).flatten(0, 1)
            confidence_shape = (
                place_count,
                views_per_place,
                semantic_confidence.shape[1],
            )
            semantic_confidence = (
                semantic_confidence.view(confidence_shape).roll(1, 0).flatten(0, 1)
            )

        patch_count = semantic_indices.shape[1]
        patch_side = self._square_side(patch_count, "semantic cache")
        reliability_side = self._square_side(
            reliability.shape[1], "reliability"
        )
        reliability = F.interpolate(
            reliability.view(-1, 1, reliability_side, reliability_side),
            size=(patch_side, patch_side),
            mode="bilinear",
            align_corners=False,
        ).flatten(1)

        # Cache generation and Dataset manifest validation guarantee uint8
        # indices in [0, patch_count). Avoid per-step GPU .item() syncs here.
        if semantic_indices.dtype != torch.uint8 or patch_count > 256:
            raise ValueError(
                "semantic indices must be uint8 for a cache with <=256 patches"
            )
        indices = semantic_indices.long()
        weights = semantic_weights.float().clamp_min(0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        gathered = reliability.gather(1, indices.flatten(1)).view_as(weights)
        smoothed = (gathered * weights).sum(dim=-1)
        confidence = semantic_confidence.float().clamp(0.0, 1.0)
        output = reliability + confidence * (smoothed - reliability)
        # Bring every ablation back to the same match_grid before the final
        # resize to DINO resolution.  This prevents full-vs-repeatability
        # comparisons from being confounded by an extra interpolation stage.
        output = F.interpolate(
            output.view(-1, 1, patch_side, patch_side),
            size=(reliability_side, reliability_side),
            mode="bilinear",
            align_corners=False,
        ).flatten(1)
        return output, confidence

    def forward(
        self,
        featmap: torch.Tensor,
        place_count: int,
        views_per_place: int,
        semantic_indices: torch.Tensor | None = None,
        semantic_weights: torch.Tensor | None = None,
        semantic_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if featmap.ndim != 4:
            raise ValueError("featmap must have shape (B,C,H,W)")

        with torch.no_grad(), torch.autocast(
            device_type=featmap.device.type, enabled=False
        ):
            stats: dict[str, torch.Tensor] = {}
            confidence_mean = featmap.new_zeros((), dtype=torch.float32)

            if self.mode == "semantic_only":
                if semantic_confidence is None:
                    raise ValueError("semantic_confidence is required in semantic_only mode")
                confidence_raw = semantic_confidence.detach().float().clamp(0.0, 1.0)
                raw_spatial_std = confidence_raw.std(
                    dim=-1, unbiased=False
                )
                informative = raw_spatial_std >= self.min_spatial_std
                reliability = confidence_raw
                semantic_side = self._square_side(
                    reliability.shape[1], "semantic confidence"
                )
                reliability = F.interpolate(
                    reliability.view(-1, 1, semantic_side, semantic_side),
                    size=(self.match_grid, self.match_grid),
                    mode="bilinear",
                    align_corners=False,
                ).flatten(1)
                confidence_mean = reliability.mean()
            else:
                repeatability, uniqueness, stats = self._vpr_components(
                    featmap.detach(), place_count, views_per_place
                )
                if self.mode == "repeatability_only":
                    reliability = repeatability
                else:
                    # Equalised scales prevent either repeatability or
                    # uniqueness from dominating solely due to cosine range.
                    reliability = 0.5 * (
                        self._standardize(repeatability)
                        + self._standardize(uniqueness)
                    )
                    if self.mode != "repeatability_uniqueness_only":
                        if any(
                            value is None
                            for value in (
                                semantic_indices,
                                semantic_weights,
                                semantic_confidence,
                            )
                        ):
                            raise ValueError(
                                "semantic cache tensors are required in "
                                f"{self.mode} mode"
                            )
                        reliability, confidence_map = self._sparse_semantic_smooth(
                            reliability,
                            semantic_indices.detach(),
                            semantic_weights.detach(),
                            semantic_confidence.detach(),
                            place_count,
                            views_per_place,
                        )
                        confidence_mean = confidence_map.mean()

            target_side = self._square_side(reliability.shape[1], "target")
            target = F.interpolate(
                reliability.view(-1, 1, target_side, target_side),
                size=featmap.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            flat_target = target.flatten(1)
            flat_target = self._standardize(flat_target)
            target = torch.tanh(self.target_scale * flat_target).view_as(target)
            if self.mode == "semantic_only":
                target = target * informative.view(-1, 1, 1, 1)
                stats["region_semantic_informative_frac"] = informative.float().mean()
                stats["region_semantic_spatial_std"] = raw_spatial_std.mean()

            stats["region_semantic_confidence"] = confidence_mean
            stats["region_target_std"] = target.std(unbiased=False)
            stats["region_target_abs_mean"] = target.abs().mean()
        return target, stats
