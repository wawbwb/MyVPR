"""CLIP-disagreement positive pairs for VPR descriptor training.

The frozen CLIP teacher is used only as a pair selector.  Within every known
place, views with the lowest CLIP cosine similarity are treated as condition-
changing hard positives and are pulled together in the native VPR descriptor
space.  CLIP is never an inference-time dependency and never receives a
gradient from this objective.
"""

from __future__ import annotations

import operator

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPSemanticPositiveLoss(nn.Module):
    """Pull together same-place views selected by a detached pair miner.

    ``clip`` selects the lowest-CLIP-similarity pair in each place.  The three
    controls use exactly the same candidate pool and descriptor loss:

    * ``random`` ranks pairs with a deterministic label/view-position hash;
    * ``shuffled`` permutes the complete batch's CLIP pair-score multiset;
    * ``student`` selects the lowest detached student cosine similarity.

    The optimized loss is simply ``1 - cosine`` on the selected, normalized
    student descriptors.  Keeping the first experiment margin-free avoids
    introducing a second explanation for any observed gain.
    """

    VALID_SELECTIONS = frozenset({"clip", "random", "shuffled", "student"})
    _HASH_MODULUS = 2_147_483_647

    def __init__(self, selection: str = "clip", positive_topk: int = 1) -> None:
        super().__init__()
        selection = str(selection).lower()
        if selection not in self.VALID_SELECTIONS:
            raise ValueError(
                "selection must be one of: 'clip', 'random', 'shuffled', "
                "or 'student'"
            )
        if isinstance(positive_topk, bool):
            raise TypeError("positive_topk must be an integer")
        try:
            positive_topk = operator.index(positive_topk)
        except TypeError as exc:
            raise TypeError("positive_topk must be an integer") from exc
        if positive_topk < 1:
            raise ValueError("positive_topk must be at least 1")

        self.selection = selection
        self.positive_topk = positive_topk

    @staticmethod
    def _validate_inputs(
        student_descriptors: torch.Tensor,
        clip_embeddings: torch.Tensor,
        labels: torch.Tensor,
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
            raise ValueError("student and CLIP tensors need a feature dimension")
        if clip_embeddings.shape[0] != batch_size or labels.numel() != batch_size:
            raise ValueError(
                "student descriptors, CLIP embeddings and labels must have "
                "the same batch size"
            )
        if (
            clip_embeddings.device != student_descriptors.device
            or labels.device != student_descriptors.device
        ):
            raise ValueError("all semantic-positive inputs must be on the same device")
        return labels

    @staticmethod
    def _packing_layout(
        inverse: torch.Tensor, place_counts: torch.Tensor
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Return max views, packed positions and packed original indices."""
        num_images = inverse.numel()
        num_places = place_counts.numel()
        max_views = int(place_counts.max().item())

        order = torch.argsort(inverse, stable=True)
        group_starts = place_counts.cumsum(dim=0) - place_counts
        sorted_positions = torch.arange(
            num_images, device=inverse.device
        ) - torch.repeat_interleave(group_starts, place_counts)
        positions = torch.empty_like(sorted_positions)
        positions[order] = sorted_positions
        flat_positions = inverse * max_views + positions

        packed_indices = torch.full(
            (num_places * max_views,),
            -1,
            device=inverse.device,
            dtype=torch.long,
        )
        packed_indices.index_copy_(
            0,
            flat_positions,
            torch.arange(num_images, device=inverse.device),
        )
        return max_views, flat_positions, packed_indices.view(num_places, max_views)

    @classmethod
    def _pair_hash(
        cls, unique_labels: torch.Tensor, max_views: int
    ) -> torch.Tensor:
        """Stable per-place/view-pair scores without consuming global RNG."""
        device = unique_labels.device
        first = torch.arange(max_views, device=device, dtype=torch.long)
        second = torch.arange(max_views, device=device, dtype=torch.long)
        state = torch.remainder(
            unique_labels.long()[:, None, None], cls._HASH_MODULUS
        )
        state = torch.remainder(
            state * 1_103_515_245 + 12_345,
            cls._HASH_MODULUS,
        )
        state = torch.remainder(
            state
            + (first[None, :, None] + 1) * 48_271
            + (second[None, None, :] + 1) * 69_621,
            cls._HASH_MODULUS,
        )
        state = torch.remainder(
            state * 1_103_515_245 + 12_345,
            cls._HASH_MODULUS,
        )
        return state.double() / float(cls._HASH_MODULUS)

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros((), dtype=torch.float32)

    @staticmethod
    def _safe_masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = values.float()
        valid_float = valid.to(values.dtype)
        safe_values = torch.where(valid, values, torch.zeros_like(values))
        return safe_values.sum() / valid_float.sum().clamp_min(1.0)

    def _base_stats(
        self,
        reference: torch.Tensor,
        candidate_counts: torch.Tensor,
        valid_view_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = self._zero(reference)
        num_places = candidate_counts.numel()
        total_views = reference.shape[0]
        return {
            "semantic_positive_valid_place_frac": (
                candidate_counts.gt(0).float().mean() if num_places else zero
            ),
            "semantic_positive_valid_view_frac": (
                valid_view_count.float() / float(total_views)
                if total_views > 0
                else zero
            ),
            "semantic_positive_candidate_pair_count": candidate_counts.sum().float(),
            "semantic_positive_candidates_per_place": (
                candidate_counts.float().mean() if num_places else zero
            ),
            "semantic_positive_random_control": reference.new_tensor(
                float(self.selection == "random"), dtype=torch.float32
            ),
            "semantic_positive_shuffled_control": reference.new_tensor(
                float(self.selection == "shuffled"), dtype=torch.float32
            ),
            "semantic_positive_student_control": reference.new_tensor(
                float(self.selection == "student"), dtype=torch.float32
            ),
        }

    def _add_metadata_stats(
        self,
        stats: dict[str, torch.Tensor],
        first_indices: torch.Tensor,
        second_indices: torch.Tensor,
        years: torch.Tensor | None,
        months: torch.Tensor | None,
        headings: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> None:
        """Add detached condition-gap diagnostics when metadata is available."""
        zero = self._zero(reference)
        empty_valid = torch.zeros(
            first_indices.numel(), device=reference.device, dtype=torch.bool
        )

        def prepare(values: torch.Tensor | None, name: str) -> torch.Tensor | None:
            if values is None:
                return None
            values = values.reshape(-1)
            if values.numel() != reference.shape[0]:
                raise ValueError(f"{name} must have one value per batch image")
            if values.device != reference.device:
                raise ValueError(f"{name} must be on the descriptor device")
            return values.float()

        years = prepare(years, "years")
        months = prepare(months, "months")
        headings = prepare(headings, "headings")

        if years is not None:
            first = years[first_indices]
            second = years[second_indices]
            valid = torch.isfinite(first) & torch.isfinite(second)
            year_valid = valid
            stats["semantic_positive_selected_year_gap"] = self._safe_masked_mean(
                (first - second).abs(), valid
            )
        else:
            stats["semantic_positive_selected_year_gap"] = zero
            year_valid = empty_valid

        if months is not None:
            first = months[first_indices]
            second = months[second_indices]
            valid = (
                torch.isfinite(first)
                & torch.isfinite(second)
                & first.ge(1)
                & first.le(12)
                & second.ge(1)
                & second.le(12)
            )
            month_valid = valid
            difference = (first - second).abs()
            circular = torch.minimum(difference, 12.0 - difference)
            stats["semantic_positive_selected_month_gap"] = self._safe_masked_mean(
                circular, valid
            )
        else:
            stats["semantic_positive_selected_month_gap"] = zero
            month_valid = empty_valid

        if headings is not None:
            first = torch.remainder(headings[first_indices], 360.0)
            second = torch.remainder(headings[second_indices], 360.0)
            valid = torch.isfinite(first) & torch.isfinite(second)
            heading_valid = valid
            difference = (first - second).abs()
            circular = torch.minimum(difference, 360.0 - difference)
            stats["semantic_positive_selected_heading_gap_deg"] = (
                self._safe_masked_mean(circular, valid)
            )
        else:
            stats["semantic_positive_selected_heading_gap_deg"] = zero
            heading_valid = empty_valid

        def valid_fraction(valid: torch.Tensor) -> torch.Tensor:
            return valid.float().mean() if valid.numel() > 0 else zero

        stats["semantic_positive_year_pair_frac"] = valid_fraction(year_valid)
        stats["semantic_positive_month_pair_frac"] = valid_fraction(month_valid)
        stats["semantic_positive_heading_pair_frac"] = valid_fraction(
            heading_valid
        )

        metadata_valid = year_valid & month_valid & heading_valid
        stats["semantic_positive_metadata_pair_frac"] = (
            valid_fraction(metadata_valid)
        )

    def forward(
        self,
        student_descriptors: torch.Tensor,
        clip_embeddings: torch.Tensor,
        labels: torch.Tensor,
        years: torch.Tensor | None = None,
        months: torch.Tensor | None = None,
        headings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(same-place positive loss, detached scalar statistics)``."""
        labels = self._validate_inputs(
            student_descriptors, clip_embeddings, labels
        )
        unique_labels, inverse, place_counts = torch.unique(
            labels, sorted=True, return_inverse=True, return_counts=True
        )
        num_places = place_counts.numel()
        max_views, flat_positions, packed_indices = self._packing_layout(
            inverse, place_counts
        )

        with torch.autocast(
            device_type=student_descriptors.device.type, enabled=False
        ):
            normalized_student = F.normalize(
                student_descriptors.float(), dim=-1
            )
            packed_student = normalized_student.new_zeros(
                num_places * max_views, normalized_student.shape[1]
            ).index_copy(0, flat_positions, normalized_student)
            packed_student = packed_student.view(
                num_places, max_views, normalized_student.shape[1]
            )
            student_similarity = torch.bmm(
                packed_student, packed_student.transpose(1, 2)
            )

        with torch.no_grad(), torch.autocast(
            device_type=clip_embeddings.device.type, enabled=False
        ):
            finite_clip = torch.isfinite(clip_embeddings).all(dim=1)
            safe_clip = torch.where(
                finite_clip[:, None],
                clip_embeddings.detach(),
                torch.zeros_like(clip_embeddings),
            )
            normalized_clip = F.normalize(safe_clip.float(), dim=-1)
            packed_clip = normalized_clip.new_zeros(
                num_places * max_views, normalized_clip.shape[1]
            ).index_copy(0, flat_positions, normalized_clip)
            packed_clip = packed_clip.view(
                num_places, max_views, normalized_clip.shape[1]
            )
            clip_similarity = torch.bmm(
                packed_clip, packed_clip.transpose(1, 2)
            )

            packed_clip_valid = torch.zeros(
                num_places * max_views,
                device=student_descriptors.device,
                dtype=torch.bool,
            ).index_copy(0, flat_positions, finite_clip)
            packed_clip_valid = packed_clip_valid.view(num_places, max_views)
            packed_view_valid = packed_indices.ge(0)
            valid_views = packed_view_valid & packed_clip_valid
            upper_triangle = torch.triu(
                torch.ones(
                    max_views,
                    max_views,
                    device=student_descriptors.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            pair_valid = (
                valid_views[:, :, None]
                & valid_views[:, None, :]
                & upper_triangle[None, :, :]
            )
            candidate_counts = pair_valid.flatten(1).sum(dim=1)

            if self.selection == "clip":
                priority = clip_similarity
            elif self.selection == "student":
                priority = student_similarity.detach()
            else:
                pair_hash = self._pair_hash(unique_labels, max_views)
                if self.selection == "random":
                    priority = pair_hash
                else:
                    # Assign every valid score to a different candidate while
                    # preserving the exact batch-level score multiset.
                    flat_valid = pair_valid.reshape(-1)
                    valid_scores = clip_similarity.reshape(-1)[flat_valid]
                    valid_hashes = pair_hash.reshape(-1)[flat_valid]
                    order = torch.argsort(valid_hashes)
                    source_order = order.roll(1)
                    shuffled_scores = torch.empty_like(valid_scores)
                    shuffled_scores[order] = valid_scores[source_order]
                    flat_priority = torch.full_like(
                        clip_similarity.reshape(-1), torch.inf
                    )
                    flat_priority[flat_valid] = shuffled_scores
                    priority = flat_priority.view_as(clip_similarity)

            flat_priority = priority.masked_fill(
                ~pair_valid, torch.inf
            ).flatten(1)
            select_k = min(self.positive_topk, flat_priority.shape[1])
            selected_values, selected_flat = flat_priority.topk(
                select_k, dim=1, largest=False
            )
            selected_valid = torch.isfinite(selected_values)
            selected_places = torch.arange(
                num_places, device=student_descriptors.device
            )[:, None].expand(-1, select_k)

            selected_places = selected_places[selected_valid]
            selected_flat = selected_flat[selected_valid]
            selected_first_local = torch.div(
                selected_flat, max_views, rounding_mode="floor"
            )
            selected_second_local = torch.remainder(selected_flat, max_views)
            first_indices = packed_indices[
                selected_places, selected_first_local
            ]
            second_indices = packed_indices[
                selected_places, selected_second_local
            ]

        stats = self._base_stats(
            student_descriptors,
            candidate_counts,
            finite_clip.sum(),
        )
        zero = self._zero(student_descriptors)

        if first_indices.numel() == 0:
            loss = student_descriptors.sum() * 0.0
            stats.update(
                {
                    "semantic_positive_selected_pair_count": zero,
                    "semantic_positive_selected_clip_sim": zero,
                    "semantic_positive_selected_clip_disagreement": zero,
                    "semantic_positive_all_clip_sim": zero,
                    "semantic_positive_selected_student_sim": zero,
                    "semantic_positive_selected_student_disagreement": zero,
                    "semantic_positive_all_student_sim": zero,
                    "semantic_positive_student_hardness_gap": zero,
                    "semantic_positive_clip_hard_overlap_frac": zero,
                    "semantic_positive_student_hard_overlap_frac": zero,
                    "semantic_positive_view_coverage_frac": zero,
                }
            )
            self._add_metadata_stats(
                stats,
                first_indices,
                second_indices,
                years,
                months,
                headings,
                student_descriptors,
            )
            return loss, {key: value.detach() for key, value in stats.items()}

        selected_student_similarity = (
            normalized_student[first_indices]
            * normalized_student[second_indices]
        ).sum(dim=-1).clamp(min=-1.0, max=1.0)
        loss = (1.0 - selected_student_similarity).mean()

        with torch.no_grad():
            selected_clip_similarity = (
                normalized_clip[first_indices] * normalized_clip[second_indices]
            ).sum(dim=-1)
            all_clip_similarity = clip_similarity[pair_valid]
            all_student_similarity = student_similarity.detach()[pair_valid]

            student_priority = student_similarity.detach().masked_fill(
                ~pair_valid, torch.inf
            ).flatten(1)
            hard_values, hard_flat = student_priority.min(dim=1)
            hard_valid = torch.isfinite(hard_values)
            clip_priority = clip_similarity.masked_fill(
                ~pair_valid, torch.inf
            ).flatten(1)
            clip_hard_values, clip_hard_flat = clip_priority.min(dim=1)
            clip_hard_valid = torch.isfinite(clip_hard_values)
            selected_matrix = torch.full(
                (num_places, select_k),
                -1,
                device=student_descriptors.device,
                dtype=torch.long,
            )
            selected_matrix[selected_valid] = selected_flat
            hard_overlap = selected_matrix.eq(hard_flat[:, None]).any(dim=1)
            clip_hard_overlap = selected_matrix.eq(
                clip_hard_flat[:, None]
            ).any(dim=1)

            selected_views = torch.unique(
                torch.cat((first_indices, second_indices), dim=0)
            )
            stats.update(
                {
                    "semantic_positive_selected_pair_count": first_indices.new_tensor(
                        float(first_indices.numel()), dtype=torch.float32
                    ),
                    "semantic_positive_selected_clip_sim": selected_clip_similarity.mean(),
                    "semantic_positive_selected_clip_disagreement": (
                        1.0 - selected_clip_similarity
                    ).mean(),
                    "semantic_positive_all_clip_sim": all_clip_similarity.mean(),
                    "semantic_positive_selected_student_sim": (
                        selected_student_similarity.detach().mean()
                    ),
                    "semantic_positive_selected_student_disagreement": (
                        1.0 - selected_student_similarity.detach()
                    ).mean(),
                    "semantic_positive_all_student_sim": all_student_similarity.mean(),
                    "semantic_positive_student_hardness_gap": (
                        all_student_similarity.mean()
                        - selected_student_similarity.detach().mean()
                    ),
                    "semantic_positive_clip_hard_overlap_frac": (
                        self._safe_masked_mean(
                            clip_hard_overlap.float(), clip_hard_valid
                        )
                    ),
                    "semantic_positive_student_hard_overlap_frac": (
                        self._safe_masked_mean(hard_overlap.float(), hard_valid)
                    ),
                    "semantic_positive_view_coverage_frac": (
                        selected_views.numel()
                        * torch.ones_like(zero)
                        / finite_clip.sum().float().clamp_min(1.0)
                    ),
                }
            )
            self._add_metadata_stats(
                stats,
                first_indices,
                second_indices,
                years,
                months,
                headings,
                student_descriptors,
            )

        if not torch.isfinite(loss):
            raise RuntimeError("semantic-positive loss became non-finite")
        return loss, {key: value.detach() for key, value in stats.items()}
