"""CLIP-guided semantic-alias negatives for VPR descriptor training.

CLIP is used only as a frozen miner: it identifies geographically distinct
places that are semantically easy to confuse.  The optimized quantity is a
student-to-student descriptor margin, so this objective does not ask the VPR
model to reproduce CLIP's embedding geometry.
"""

from __future__ import annotations

import math
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPSemanticAliasLoss(nn.Module):
    """Penalize high student similarity to CLIP-mined negative places.

    Place prototypes are formed by averaging normalized per-image CLIP global
    embeddings.  For each anchor place, ``negative_topk`` different places are
    selected after rejecting GPS distances below ``min_geo_distance_m``.  The
    ``random`` ranks unordered label pairs with a deterministic hash. The
    stricter ``shuffled`` control jointly permutes CLIP's rows and columns, so
    it preserves the score distribution and graph symmetry while breaking the
    correspondence between semantics and the real place labels.

    CLIP mining is fully detached.  For every selected directed place pair the
    loss uses the most similar pair of normalized student view descriptors and
    applies a smooth upper-margin penalty.
    """

    EARTH_RADIUS_M = 6_371_008.8
    _RANDOM_CONTROL_SEED = 0
    _HASH_MODULUS = 2_147_483_647

    def __init__(
        self,
        selection: str = "clip",
        negative_topk: int = 1,
        min_geo_distance_m: float = 50.0,
        student_margin: float = 0.2,
        loss_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if selection not in {"clip", "random", "shuffled"}:
            raise ValueError(
                "selection must be one of: 'clip', 'random', 'shuffled'"
            )
        try:
            negative_topk = operator.index(negative_topk)
        except TypeError as exc:
            raise TypeError("negative_topk must be an integer") from exc
        if negative_topk < 1:
            raise ValueError("negative_topk must be at least 1")

        min_geo_distance_m = float(min_geo_distance_m)
        student_margin = float(student_margin)
        loss_temperature = float(loss_temperature)
        if not math.isfinite(min_geo_distance_m) or min_geo_distance_m < 0:
            raise ValueError("min_geo_distance_m must be finite and non-negative")
        if not math.isfinite(student_margin) or not -1.0 <= student_margin <= 1.0:
            raise ValueError("student_margin must be finite and in [-1, 1]")
        if not math.isfinite(loss_temperature) or loss_temperature <= 0:
            raise ValueError("loss_temperature must be finite and positive")

        self.selection = selection
        self.negative_topk = negative_topk
        self.min_geo_distance_m = min_geo_distance_m
        self.student_margin = student_margin
        self.loss_temperature = loss_temperature

    @staticmethod
    def _validate_inputs(
        student_descriptors: torch.Tensor,
        clip_embeddings: torch.Tensor,
        labels: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if student_descriptors.ndim != 2:
            raise ValueError(
                "student_descriptors must have shape (B,D), "
                f"got {tuple(student_descriptors.shape)}"
            )
        if clip_embeddings.ndim != 2:
            raise ValueError(
                "clip_embeddings must have shape (B,D), "
                f"got {tuple(clip_embeddings.shape)}"
            )
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "coordinates must have shape (B,2) in latitude/longitude "
                f"order, got {tuple(coordinates.shape)}"
            )
        if not student_descriptors.is_floating_point():
            raise TypeError("student_descriptors must be floating point")
        if not clip_embeddings.is_floating_point():
            raise TypeError("clip_embeddings must be floating point")
        if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
            raise TypeError("labels must have an integer dtype")

        labels = labels.reshape(-1)
        batch_size = student_descriptors.shape[0]
        if batch_size < 1:
            raise ValueError("the batch must contain at least one descriptor")
        if student_descriptors.shape[1] < 1 or clip_embeddings.shape[1] < 1:
            raise ValueError("student and CLIP embeddings must have a feature dimension")
        if (
            clip_embeddings.shape[0] != batch_size
            or labels.numel() != batch_size
            or coordinates.shape[0] != batch_size
        ):
            raise ValueError(
                "student descriptors, CLIP embeddings, labels and coordinates "
                "must have the same batch size"
            )
        device = student_descriptors.device
        if (
            clip_embeddings.device != device
            or labels.device != device
            or coordinates.device != device
        ):
            raise ValueError("all semantic-alias inputs must be on the same device")
        # Non-finite GPS is intentionally not rejected here.  It makes every
        # pair involving that place invalid, which safely handles incomplete
        # metadata without aborting an otherwise usable training batch.
        return labels

    @staticmethod
    def _place_prototypes(
        image_embeddings: torch.Tensor,
        inverse: torch.Tensor,
        place_counts: torch.Tensor,
    ) -> torch.Tensor:
        num_places = place_counts.numel()
        normalized = F.normalize(image_embeddings.float(), dim=-1)
        sums = torch.zeros(
            (num_places, normalized.shape[1]),
            device=normalized.device,
            dtype=torch.float32,
        )
        sums.index_add_(0, inverse, normalized)
        means = sums / place_counts[:, None].float().clamp_min(1.0)
        return F.normalize(means, dim=-1)

    @classmethod
    def _place_distances(
        cls,
        coordinates: torch.Tensor,
        inverse: torch.Tensor,
        place_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return minimum cross-view distances and finite-place validity.

        Using the minimum image-to-image distance is conservative: a place
        pair is eligible only when *every* sampled cross-view pair is at least
        ``min_geo_distance_m`` apart.
        """
        num_places = place_counts.numel()
        coordinates = coordinates.detach().double()
        image_valid = (
            torch.isfinite(coordinates).all(dim=1)
            & coordinates[:, 0].abs().le(90.0)
            & coordinates[:, 1].abs().le(180.0)
        )

        # Sanitize before trigonometry and retain a separate conservative
        # mask: one bad coordinate invalidates all pairs involving its place.
        safe_coordinates = torch.where(
            image_valid[:, None], coordinates, torch.zeros_like(coordinates)
        )
        latitude = torch.deg2rad(safe_coordinates[:, 0])
        longitude = torch.deg2rad(safe_coordinates[:, 1])
        cos_latitude = latitude.cos()
        image_xyz = torch.stack(
            (
                cos_latitude * longitude.cos(),
                cos_latitude * longitude.sin(),
                latitude.sin(),
            ),
            dim=1,
        )

        valid_counts = torch.zeros(
            num_places, device=coordinates.device, dtype=torch.long
        )
        valid_counts.index_add_(0, inverse, image_valid.long())
        place_valid = valid_counts.eq(place_counts)

        # Chord-to-angle is stable for the small distances relevant to VPR;
        # acos(dot) loses too much precision near zero even in some fp64 paths.
        chord = torch.linalg.vector_norm(
            image_xyz[:, None, :] - image_xyz[None, :, :], dim=-1
        )
        angle = 2.0 * torch.asin((0.5 * chord).clamp(min=0.0, max=1.0))
        image_distance = angle * cls.EARTH_RADIUS_M
        image_pair_valid = image_valid[:, None] & image_valid[None, :]
        image_distance = image_distance.masked_fill(~image_pair_valid, torch.inf)

        place_pair_index = (
            inverse[:, None] * num_places + inverse[None, :]
        ).reshape(-1)
        distance = image_distance.new_full(
            (num_places * num_places,), torch.inf
        ).scatter_reduce(
            dim=0,
            index=place_pair_index,
            src=image_distance.reshape(-1),
            reduce="amin",
            include_self=True,
        ).view(num_places, num_places)
        pair_valid = place_valid[:, None] & place_valid[None, :]
        distance = distance.masked_fill(~pair_valid, torch.nan)
        return distance, place_valid

    @classmethod
    def _random_control_priority(
        cls, unique_labels: torch.Tensor
    ) -> torch.Tensor:
        """Return a stable symmetric pseudo-random score per label pair.

        Scores depend on the actual unordered labels rather than their rank in
        a particular batch. Integer modular mixing is device-local and does
        not read or modify PyTorch's global random-number generators.
        """
        first = unique_labels[:, None]
        second = unique_labels[None, :]
        low = torch.remainder(torch.minimum(first, second), cls._HASH_MODULUS)
        high = torch.remainder(torch.maximum(first, second), cls._HASH_MODULUS)
        state = torch.remainder(
            low * 1_103_515_245 + 12_345 + cls._RANDOM_CONTROL_SEED,
            cls._HASH_MODULUS,
        )
        state = torch.remainder(
            state + high * 48_271,
            cls._HASH_MODULUS,
        )
        state = torch.remainder(
            state * 1_103_515_245 + 12_345,
            cls._HASH_MODULUS,
        )
        return state.double() / float(cls._HASH_MODULUS)

    @torch.no_grad()
    def _mine_place_pairs(
        self,
        clip_embeddings: torch.Tensor,
        labels: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return selected place indices plus detached mining diagnostics."""
        unique_labels, inverse, place_counts = torch.unique(
            labels, sorted=True, return_inverse=True, return_counts=True
        )
        num_places = place_counts.numel()

        with torch.autocast(device_type=clip_embeddings.device.type, enabled=False):
            detached_clip = clip_embeddings.detach()
            clip_image_valid = torch.isfinite(detached_clip).all(dim=1)
            safe_clip = torch.where(
                clip_image_valid[:, None],
                detached_clip,
                torch.zeros_like(detached_clip),
            )
            clip_valid_counts = torch.zeros_like(place_counts)
            clip_valid_counts.index_add_(0, inverse, clip_image_valid.long())
            clip_place_valid = clip_valid_counts.eq(place_counts)
            prototypes = self._place_prototypes(
                safe_clip, inverse, place_counts
            )
            clip_similarity = prototypes @ prototypes.t()
            geo_distance, _ = self._place_distances(
                coordinates, inverse, place_counts
            )

        candidate_mask = (
            torch.isfinite(geo_distance)
            & geo_distance.ge(self.min_geo_distance_m)
            & clip_place_valid[:, None]
            & clip_place_valid[None, :]
        )
        candidate_mask.fill_diagonal_(False)
        candidate_counts = candidate_mask.sum(dim=1)

        if num_places < 2:
            empty = torch.empty(0, device=labels.device, dtype=torch.long)
            return (
                inverse,
                place_counts,
                empty,
                empty,
                clip_similarity,
                geo_distance,
                candidate_counts,
            )

        if self.selection == "clip":
            priority = clip_similarity
        elif self.selection == "random":
            priority = self._random_control_priority(unique_labels)
        else:
            # Joint row/column permutation keeps the CLIP similarity matrix
            # symmetric and preserves its complete score/graph structure, but
            # assigns that structure to the wrong place identities.
            label_priority = self._random_control_priority(
                unique_labels
            ).diagonal()
            permutation = torch.argsort(label_priority)
            priority = clip_similarity[permutation][:, permutation]

        topk = min(self.negative_topk, num_places - 1)
        masked_priority = priority.masked_fill(~candidate_mask, -torch.inf)
        top_values, selected_negative = masked_priority.topk(topk, dim=1)
        selected_valid = torch.isfinite(top_values)
        selected_anchor = torch.arange(
            num_places, device=labels.device, dtype=torch.long
        )[:, None].expand(-1, topk)
        return (
            inverse,
            place_counts,
            selected_anchor[selected_valid],
            selected_negative[selected_valid],
            clip_similarity,
            geo_distance,
            candidate_counts,
        )

    @staticmethod
    def _zero_stat(reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=reference.device, dtype=torch.float32)

    @staticmethod
    def _hard_student_similarities(
        student_descriptors: torch.Tensor,
        inverse: torch.Tensor,
        place_counts: torch.Tensor,
        selected_anchor: torch.Tensor,
        selected_negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hardest cross-view cosine only for selected place pairs."""
        normalized = F.normalize(student_descriptors.float(), dim=-1)
        num_images, descriptor_dim = normalized.shape
        num_places = place_counts.numel()
        max_views = int(place_counts.max().item())

        # Pack variable-size label groups into (G,max_K,D) without assuming
        # that images from the same place are contiguous in the input batch.
        order = torch.argsort(inverse)
        group_starts = place_counts.cumsum(dim=0) - place_counts
        sorted_positions = torch.arange(
            num_images, device=inverse.device
        ) - torch.repeat_interleave(group_starts, place_counts)
        positions = torch.empty_like(sorted_positions)
        positions[order] = sorted_positions
        flat_index = inverse * max_views + positions
        packed = normalized.new_zeros(
            (num_places * max_views, descriptor_dim)
        ).index_copy(0, flat_index, normalized)
        packed = packed.view(num_places, max_views, descriptor_dim)
        valid_views = (
            torch.arange(max_views, device=inverse.device)[None, :]
            < place_counts[:, None]
        )

        anchor_views = packed[selected_anchor]
        negative_views = packed[selected_negative]
        pair_similarity = torch.bmm(
            anchor_views, negative_views.transpose(1, 2)
        )
        pair_valid = (
            valid_views[selected_anchor, :, None]
            & valid_views[selected_negative, None, :]
        )
        return pair_similarity.masked_fill(~pair_valid, -torch.inf).amax(
            dim=(1, 2)
        )

    def forward(
        self,
        student_descriptors: torch.Tensor,
        clip_embeddings: torch.Tensor,
        labels: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(smooth_margin_loss, detached_scalar_statistics)``."""
        labels = self._validate_inputs(
            student_descriptors, clip_embeddings, labels, coordinates
        )
        (
            inverse,
            place_counts,
            selected_anchor,
            selected_negative,
            clip_similarity,
            geo_distance,
            candidate_counts,
        ) = self._mine_place_pairs(clip_embeddings, labels, coordinates)

        num_places = candidate_counts.numel()
        candidate_pair_count = candidate_counts.sum()
        distinct_pair_count = num_places * max(num_places - 1, 0)
        zero_stat = self._zero_stat(student_descriptors)

        stats: dict[str, torch.Tensor] = {
            "semantic_alias_valid_place_frac": (
                candidate_counts.gt(0).float().mean()
                if num_places > 0
                else zero_stat
            ),
            "semantic_alias_valid_pair_count": torch.tensor(
                float(selected_anchor.numel()),
                device=student_descriptors.device,
                dtype=torch.float32,
            ),
            "semantic_alias_candidate_pair_count": candidate_pair_count.float(),
            "semantic_alias_candidate_pair_frac": (
                candidate_pair_count.float() / float(distinct_pair_count)
                if distinct_pair_count > 0
                else zero_stat
            ),
            "semantic_alias_candidates_per_place": (
                candidate_counts.float().mean()
                if num_places > 0
                else zero_stat
            ),
            "semantic_alias_random_control": torch.tensor(
                float(self.selection == "random"),
                device=student_descriptors.device,
                dtype=torch.float32,
            ),
            "semantic_alias_shuffled_control": torch.tensor(
                float(self.selection == "shuffled"),
                device=student_descriptors.device,
                dtype=torch.float32,
            ),
        }

        if selected_anchor.numel() > 0:
            directed_index = selected_anchor * num_places + selected_negative
            selected_lookup = torch.zeros(
                num_places * num_places,
                device=student_descriptors.device,
                dtype=torch.bool,
            )
            selected_lookup[directed_index] = True
            reverse_index = selected_negative * num_places + selected_anchor
            unordered_index = (
                torch.minimum(selected_anchor, selected_negative) * num_places
                + torch.maximum(selected_anchor, selected_negative)
            )
            stats.update(
                {
                    "semantic_alias_unique_pair_count": torch.unique(
                        unordered_index
                    ).numel()
                    * torch.ones_like(zero_stat),
                    "semantic_alias_reciprocal_frac": selected_lookup[
                        reverse_index
                    ].float().mean(),
                }
            )
        else:
            stats.update(
                {
                    "semantic_alias_unique_pair_count": zero_stat,
                    "semantic_alias_reciprocal_frac": zero_stat,
                }
            )

        if selected_anchor.numel() == 0:
            loss = student_descriptors.sum() * 0.0
            stats.update(
                {
                    "semantic_alias_selected_clip_sim": zero_stat,
                    "semantic_alias_selected_geo_distance_m": zero_stat,
                    "semantic_alias_hard_student_sim": zero_stat,
                    "semantic_alias_margin_violation_frac": zero_stat,
                }
            )
            return loss, {key: value.detach() for key, value in stats.items()}

        with torch.autocast(
            device_type=student_descriptors.device.type, enabled=False
        ):
            hard_student_similarity = self._hard_student_similarities(
                student_descriptors,
                inverse,
                place_counts,
                selected_anchor,
                selected_negative,
            )
            per_pair_loss = self.loss_temperature * F.softplus(
                (hard_student_similarity - self.student_margin)
                / self.loss_temperature
            )
            loss = per_pair_loss.mean()

        stats.update(
            {
                "semantic_alias_selected_clip_sim": clip_similarity[
                    selected_anchor, selected_negative
                ].float().mean(),
                "semantic_alias_selected_geo_distance_m": geo_distance[
                    selected_anchor, selected_negative
                ].float().mean(),
                "semantic_alias_hard_student_sim": hard_student_similarity.mean(),
                "semantic_alias_margin_violation_frac": hard_student_similarity.gt(
                    self.student_margin
                ).float().mean(),
            }
        )
        return loss, {key: value.detach() for key, value in stats.items()}
