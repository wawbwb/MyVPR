"""Shared protocol helpers for the Query-conditioned Semantic BoQ cache."""

from __future__ import annotations

import numpy as np
import pandas as pd


QUERY_SEMANTIC_CACHE_SCHEMA = "openvpr_ade20k_patch_labels"
QUERY_SEMANTIC_CACHE_VERSION = 2
QUERY_SEMANTIC_SHUFFLE_ALGORITHM = "stable_place_group_rotation_v1"


def build_cross_place_bijection(
    place_ids: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, int]:
    """Return a deterministic donor permutation with no same-place pairs.

    Rows are stably grouped by place (place order follows first appearance and
    row order is preserved inside a place), then the grouped row list is
    rotated by the largest place size.  A valid cross-place bijection exists
    exactly when no place owns more than half of all rows.  Under that
    condition, both directions around the grouped circle span at least the
    largest group, so the rotation cannot remain inside a place block.

    The returned indices are local to ``place_ids``.  The integer is the
    grouped-list rotation recorded in the cache manifest for auditing.
    """

    place_ids = np.asarray(place_ids)
    if place_ids.ndim != 1:
        raise ValueError(f"{context} place_ids must be one-dimensional")
    row_count = int(place_ids.size)
    if row_count < 2:
        raise ValueError(f"{context} needs at least two rows for shuffling")
    if bool(pd.isna(place_ids).any()):
        raise ValueError(f"{context} contains a missing place_id")

    codes, unique_places = pd.factorize(place_ids, sort=False)
    if len(unique_places) < 2:
        raise ValueError(
            f"{context} has only one place_id; no cross-place bijection exists"
        )
    counts = np.bincount(codes, minlength=len(unique_places))
    largest_place_count = int(counts.max())
    if largest_place_count * 2 > row_count:
        raise ValueError(
            f"{context} cannot form a cross-place bijection: its largest "
            f"place has {largest_place_count} of {row_count} eligible rows"
        )

    grouped_positions = np.argsort(codes, kind="stable")
    rotated_positions = np.roll(grouped_positions, -largest_place_count)
    donors = np.empty(row_count, dtype=np.int64)
    donors[grouped_positions] = rotated_positions

    if (
        np.unique(donors).size != row_count
        or bool(np.any(donors < 0))
        or bool(np.any(donors >= row_count))
    ):
        raise RuntimeError(f"internal error: invalid donor bijection for {context}")
    if bool(np.any(place_ids == place_ids[donors])):
        raise RuntimeError(
            f"internal error: same-place donor remained for {context}"
        )
    return donors, largest_place_count
