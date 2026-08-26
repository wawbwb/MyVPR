"""Pure manifest constants for the custom full-DB MSLS condition screen."""

from __future__ import annotations

from collections.abc import Mapping


STANDARD_DB_FILE = "msls_val_dbImages.npy"
STANDARD_QUERY_FILE = "msls_val_qImages.npy"
STANDARD_GT_FILE = "msls_val_gt_25m.npy"
CONDITION_UNION_QUERY_FILE = "msls_val_condition_union_qImages.npy"

# This order is part of the query-union/cache protocol.
CONDITION_ORDER = (
    "night",
    "winter2summer",
    "summer2winter",
)

CONDITION_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "night": ("n2d",),
    "winter2summer": ("w2s",),
    "summer2winter": ("s2w",),
}

CONDITION_FILES: Mapping[str, tuple[str, str]] = {
    **{
        condition: (
            f"msls_val_{condition}_full_db_qImages.npy",
            f"msls_val_{condition}_full_db_gt_25m.npy",
        )
        for condition in CONDITION_ORDER
    },
    # Compatibility aggregate used by existing validation entry points.  The
    # dynamic-prior screen itself reports the two directions separately.
    "season": (
        "msls_val_season_full_db_qImages.npy",
        "msls_val_season_full_db_gt_25m.npy",
    ),
}
