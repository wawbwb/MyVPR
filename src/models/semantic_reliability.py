"""VPR-conditioned semantic reliability targets built from frozen CLIP.

The target is deliberately relational instead of copying a single-image CLIP
saliency map.  A patch is reliable when it can be matched in other views of
the same place, but cannot be matched in a semantically similar different
place.  Patch matching is position independent, which is important for VPR
viewpoint changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VPRSemanticReliabilityTarget(nn.Module):
    """Build a probability distribution over CLIP patches for every image.

    Positive reliability is the mean, across other views of the same place,
    of the best semantic match for every anchor patch.  Semantic hard-negative
    places are selected by frozen CLIP place prototypes.  Negative commonness
    is the best patch match in those different places.

    All computations are target construction only and therefore run without
    gradients.  ``pair_chunk_size`` bounds the temporary ``N x N`` patch
    correlation tensors for large P x K batches.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        negative_topk: int = 1,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        pair_chunk_size: int = 32,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if negative_topk < 1:
            raise ValueError("negative_topk must be at least 1")
        if positive_weight < 0 or negative_weight < 0:
            raise ValueError("positive_weight and negative_weight must be non-negative")
        if positive_weight == 0 and negative_weight == 0:
            raise ValueError("at least one reliability weight must be positive")
        if pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be at least 1")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.temperature = float(temperature)
        self.negative_topk = int(negative_topk)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.pair_chunk_size = int(pair_chunk_size)
        self.eps = float(eps)

    @staticmethod
    def _validate_inputs(
        patch_embeddings: torch.Tensor,
        global_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if patch_embeddings.ndim != 3:
            raise ValueError(
                "patch_embeddings must have shape (B,N,D), "
                f"got {tuple(patch_embeddings.shape)}"
            )
        if global_embeddings.ndim != 2:
            raise ValueError(
                "global_embeddings must have shape (B,D), "
                f"got {tuple(global_embeddings.shape)}"
            )
        labels = labels.reshape(-1)
        batch_size = patch_embeddings.shape[0]
        if global_embeddings.shape[0] != batch_size or labels.numel() != batch_size:
            raise ValueError(
                "patch embeddings, global embeddings and labels must have "
                "the same batch size"
            )
        if patch_embeddings.shape[1] < 1 or patch_embeddings.shape[2] < 1:
            raise ValueError("patch_embeddings must contain at least one patch and channel")
        return labels

    def _best_patch_matches(
        self,
        patch_embeddings: torch.Tensor,
        anchor_indices: torch.Tensor,
        other_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return each anchor patch's best match for a list of image pairs."""
        if anchor_indices.numel() == 0:
            return torch.empty(
                (0, patch_embeddings.shape[1]),
                device=patch_embeddings.device,
                dtype=torch.float32,
            )

        chunks: list[torch.Tensor] = []
        for start in range(0, anchor_indices.numel(), self.pair_chunk_size):
            end = min(start + self.pair_chunk_size, anchor_indices.numel())
            # Lightning's outer 16-mixed autocast would otherwise cast this
            # bmm back to fp16 even after ``.float()``. Reliability is a small
            # positive-minus-negative margin, so construct it explicitly in
            # fp32 and keep index_add dtypes consistent.
            with torch.autocast(
                device_type=patch_embeddings.device.type, enabled=False
            ):
                anchor = F.normalize(
                    patch_embeddings[anchor_indices[start:end]].float(), dim=-1
                )
                other = F.normalize(
                    patch_embeddings[other_indices[start:end]].float(), dim=-1
                )
                pairwise = torch.bmm(anchor, other.transpose(1, 2))
                chunks.append(pairwise.amax(dim=-1))
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _valid_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean over patches and valid anchors, returning zero for no anchors."""
        if values.ndim == 1:
            per_anchor = values
        else:
            per_anchor = values.flatten(1).mean(dim=1)
        weights = valid.to(per_anchor.dtype)
        return (per_anchor * weights).sum() / weights.sum().clamp_min(1.0)

    @torch.no_grad()
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        global_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(target, valid_anchor_mask, scalar_statistics)``.

        Invalid anchors (no positive view or no different place in a short
        final batch) receive a uniform target and are excluded from the KL.
        """
        labels = self._validate_inputs(patch_embeddings, global_embeddings, labels)
        device = patch_embeddings.device
        batch_size, num_patches, _ = patch_embeddings.shape

        unique_labels, inverse, place_counts = torch.unique(
            labels, sorted=True, return_inverse=True, return_counts=True
        )
        num_places = unique_labels.numel()

        # Same-place matching is directed: every image is an anchor and all
        # other views of its place contribute equally to positive stability.
        same_place = labels[:, None].eq(labels[None, :])
        same_place.fill_diagonal_(False)
        positive_anchor, positive_other = same_place.nonzero(as_tuple=True)
        positive_pair_scores = self._best_patch_matches(
            patch_embeddings, positive_anchor, positive_other
        )

        positive_scores = torch.zeros(
            (batch_size, num_patches), device=device, dtype=torch.float32
        )
        positive_counts = torch.zeros(batch_size, device=device, dtype=torch.float32)
        if positive_anchor.numel() > 0:
            positive_scores.index_add_(0, positive_anchor, positive_pair_scores)
            positive_counts.index_add_(
                0,
                positive_anchor,
                torch.ones_like(positive_anchor, dtype=torch.float32),
            )
        positive_scores = positive_scores / positive_counts[:, None].clamp_min(1.0)
        has_positive = place_counts[inverse] > 1

        negative_scores = torch.zeros_like(positive_scores)
        has_negative = torch.zeros(batch_size, device=device, dtype=torch.bool)
        hard_negative_place_similarity = torch.zeros(
            batch_size, device=device, dtype=torch.float32
        )

        if self.negative_weight > 0 and num_places > 1:
            # CLIP place prototypes make the negative explicitly semantic and
            # reduce selection noise from a single image/view.
            with torch.autocast(device_type=device.type, enabled=False):
                global_fp32 = F.normalize(global_embeddings.float(), dim=-1)
                place_sums = torch.zeros(
                    (num_places, global_fp32.shape[-1]),
                    device=device,
                    dtype=torch.float32,
                )
                place_sums.index_add_(0, inverse, global_fp32)
                place_prototypes = F.normalize(
                    place_sums / place_counts[:, None].float().clamp_min(1.0),
                    dim=-1,
                )
                place_similarity = place_prototypes @ place_prototypes.t()
            place_similarity.fill_diagonal_(-torch.inf)

            topk = min(self.negative_topk, num_places - 1)
            top_values, top_places = place_similarity.topk(topk, dim=1)
            selected_places = top_places[inverse]  # (B, topk)

            # Within each selected negative place, use the view most similar
            # to the anchor image in CLIP global space.  This keeps patch
            # comparisons bounded to B*topk pairs instead of all B^2 pairs.
            with torch.autocast(device_type=device.type, enabled=False):
                image_similarity = global_fp32 @ global_fp32.t()
            candidate_mask = inverse.view(1, 1, batch_size).eq(
                selected_places.unsqueeze(-1)
            )
            candidate_similarity = image_similarity.unsqueeze(1).masked_fill(
                ~candidate_mask, -torch.inf
            )
            negative_images = candidate_similarity.argmax(dim=-1)

            negative_anchor = torch.arange(device=device, end=batch_size).repeat_interleave(topk)
            negative_other = negative_images.reshape(-1)
            negative_pair_scores = self._best_patch_matches(
                patch_embeddings, negative_anchor, negative_other
            ).view(batch_size, topk, num_patches)
            negative_scores = negative_pair_scores.amax(dim=1)
            hard_negative_place_similarity = top_values[inverse].mean(dim=1)
            has_negative.fill_(True)

        valid_positive = has_positive if self.positive_weight > 0 else torch.ones_like(has_positive)
        valid_negative = has_negative if self.negative_weight > 0 else torch.ones_like(has_negative)
        valid = valid_positive & valid_negative
        with torch.autocast(device_type=device.type, enabled=False):
            reliability = (
                self.positive_weight * positive_scores
                - self.negative_weight * negative_scores
            )
            target = F.softmax(reliability / self.temperature, dim=-1)
        uniform = torch.full_like(target, 1.0 / num_patches)
        target = torch.where(valid[:, None], target, uniform)

        target_safe = target.clamp_min(self.eps)
        target_entropy = -(target_safe * target_safe.log()).sum(dim=-1)
        entropy_normalizer = max(math.log(num_patches), self.eps)
        top20_count = max(1, math.ceil(0.2 * num_patches))
        target_top20_mass = target.topk(top20_count, dim=-1).values.sum(dim=-1)
        margin = positive_scores - negative_scores

        stats = {
            "reliability_pos_sim": self._valid_mean(positive_scores, valid),
            "reliability_neg_sim": self._valid_mean(negative_scores, valid),
            "reliability_margin": self._valid_mean(margin, valid),
            "reliability_positive_margin_frac": self._valid_mean(
                (margin > 0).float(), valid
            ),
            "reliability_target_entropy_norm": self._valid_mean(
                target_entropy / entropy_normalizer, valid
            ),
            "reliability_target_peak": self._valid_mean(
                target.amax(dim=-1), valid
            ),
            "reliability_target_top20_mass": self._valid_mean(
                target_top20_mass, valid
            ),
            "reliability_hard_negative_place_sim": self._valid_mean(
                hard_negative_place_similarity, valid
            ),
            "reliability_valid_anchor_frac": valid.float().mean(),
        }
        return target, valid, stats
