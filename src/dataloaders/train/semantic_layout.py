"""Place-balanced GSV-Cities loader for the AG-SLRD layout teacher.

Unlike :mod:`gsv_cities`, this dataset never opens an RGB image.  It samples
K cached semantic layouts from a place and preserves the exact receiver rows
for both the aligned and fixed cross-place shuffled controls.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.semantic_layout_cache import (
    SEMANTIC_LAYOUT_CLASSES,
    SEMANTIC_LAYOUT_IGNORE_INDEX,
    file_sha256,
    validate_semantic_layout_cache,
)


AG_SLRD_SPLIT_ALGORITHM = "sha256_place_v1"


def place_split_remainder(
    city: str,
    place_id: int,
    *,
    seed: int,
    modulus: int,
) -> int:
    """Map a place identity to a stable split bucket."""

    if not isinstance(city, str) or not city:
        raise ValueError("city must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("split seed must be an integer")
    if isinstance(modulus, bool) or not isinstance(modulus, int) or modulus < 2:
        raise ValueError("holdout_modulus must be an integer of at least two")
    payload = (
        f"{AG_SLRD_SPLIT_ALGORITHM}\0{seed}\0{city}\0{int(place_id)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % modulus


def _normalise_cities(
    requested: str | Sequence[str] | None,
    available: Sequence[str],
) -> tuple[str, ...]:
    if requested is None or requested == "all":
        return tuple(available)
    if isinstance(requested, str):
        requested = (requested,)
    cities = tuple(str(value) for value in requested)
    if not cities or len(set(cities)) != len(cities):
        raise ValueError("cities must be a non-empty sequence without duplicates")
    missing = sorted(set(cities) - set(available))
    if missing:
        raise ValueError(f"cities are absent from semantic cache: {missing}")
    return cities


class SemanticLayoutPlaceDataset(Dataset):
    """Return one place containing K cached semantic label grids.

    A fixed hash split is applied at the *place* level.  ``mode='shuffled'``
    retains the receiver place and batch labels but reads each grid from the
    immutable cross-place donor index.  This is the sole intended difference
    between the two Phase-0 controls.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        cache_dir: str | Path,
        *,
        cities: str | Sequence[str] | None = "all",
        views_per_place: int = 4,
        mode: str = "aligned",
        split: str = "train",
        split_algorithm: str = AG_SLRD_SPLIT_ALGORITHM,
        split_seed: int = 42,
        holdout_modulus: int = 10,
        holdout_remainder: int = 0,
        random_sample: bool = True,
        verify_cache_hashes: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"GSV-Cities root not found: {self.dataset_root}"
            )
        if (
            isinstance(views_per_place, bool)
            or not isinstance(views_per_place, int)
            or views_per_place < 2
        ):
            raise ValueError("views_per_place must be an integer of at least two")
        self.views_per_place = views_per_place
        self.mode = str(mode).lower()
        if self.mode not in {"aligned", "shuffled"}:
            raise ValueError("mode must be aligned or shuffled")
        self.split = str(split).lower()
        if self.split not in {"train", "holdout", "all"}:
            raise ValueError("split must be train, holdout, or all")
        if split_algorithm != AG_SLRD_SPLIT_ALGORITHM:
            raise ValueError(
                f"split_algorithm must be {AG_SLRD_SPLIT_ALGORITHM!r}"
            )
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise TypeError("split_seed must be an integer")
        if (
            isinstance(holdout_modulus, bool)
            or not isinstance(holdout_modulus, int)
            or holdout_modulus < 2
        ):
            raise ValueError("holdout_modulus must be at least two")
        if (
            isinstance(holdout_remainder, bool)
            or not isinstance(holdout_remainder, int)
            or not 0 <= holdout_remainder < holdout_modulus
        ):
            raise ValueError(
                "holdout_remainder must lie in [0, holdout_modulus)"
            )
        self.split_seed = split_seed
        self.holdout_modulus = holdout_modulus
        self.holdout_remainder = holdout_remainder
        self.random_sample = bool(random_sample)

        manifest, arrays, hashes = validate_semantic_layout_cache(
            self.cache_dir,
            verify_array_files=verify_cache_hashes,
            expected_index_type="gsv_city_csv_row",
        )
        self.manifest = manifest
        self.array_hashes = hashes
        index_record = manifest["index"]
        if index_record.get("eligible_min_views") != self.views_per_place:
            raise ValueError(
                "semantic-layout cache eligible_min_views does not match "
                f"views_per_place={self.views_per_place}"
            )
        city_entries = manifest.get("cities")
        if not isinstance(city_entries, list) or not city_entries:
            raise ValueError("GSV semantic-layout cache requires city entries")
        entry_by_name: dict[str, dict[str, Any]] = {}
        expected_offset = 0
        for raw_entry in city_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError("semantic-layout city entry must be a mapping")
            entry = dict(raw_entry)
            name = entry.get("name")
            offset = entry.get("offset")
            count = entry.get("count")
            csv_hash = entry.get("sha256")
            if not isinstance(name, str) or not name or name in entry_by_name:
                raise ValueError("semantic-layout city names must be unique")
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset != expected_offset
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise ValueError(
                    "semantic-layout city offsets/counts must form a "
                    "contiguous positive index"
                )
            if not isinstance(csv_hash, str) or len(csv_hash) != 64:
                raise ValueError(f"city {name!r} has no valid CSV SHA256")
            entry_by_name[name] = entry
            expected_offset += count
        if expected_offset != manifest["num_images"]:
            raise ValueError("semantic-layout city counts do not match num_images")
        self.cities = _normalise_cities(cities, tuple(entry_by_name))

        labels = arrays["labels"]
        shuffled = arrays["shuffled_indices"]
        place_rows: list[np.ndarray] = []
        place_keys: list[tuple[str, int]] = []
        heldout_flags: list[bool] = []
        for city in self.cities:
            entry = entry_by_name[city]
            csv_path = self.dataset_root / "Dataframes" / f"{city}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"GSV city CSV not found: {csv_path}")
            if file_sha256(csv_path) != entry["sha256"]:
                raise ValueError(
                    f"GSV city CSV SHA256 differs from cache for {city!r}"
                )
            frame = pd.read_csv(csv_path, usecols=["place_id"])
            if len(frame) != entry["count"]:
                raise ValueError(
                    f"GSV city row count differs from cache for {city!r}"
                )
            offset = int(entry["offset"])
            raw_place_ids = frame["place_id"].to_numpy()
            for place_id, local_rows in frame.groupby(
                "place_id", sort=True
            ).indices.items():
                local_rows = np.asarray(local_rows, dtype=np.int64)
                if len(local_rows) < self.views_per_place:
                    continue
                global_rows = offset + local_rows
                donors = np.asarray(shuffled[global_rows], dtype=np.int64)
                if bool(np.any(donors == global_rows)):
                    raise ValueError(
                        f"eligible place {city}/{place_id} contains a self donor"
                    )
                if bool(np.any(donors < offset)) or bool(
                    np.any(donors >= offset + int(entry["count"]))
                ):
                    raise ValueError(
                        f"eligible place {city}/{place_id} has a cross-city donor"
                    )
                donor_local = donors - offset
                if bool(
                    np.any(raw_place_ids[donor_local] == int(place_id))
                ):
                    raise ValueError(
                        f"eligible place {city}/{place_id} has a same-place donor"
                    )
                is_holdout = (
                    place_split_remainder(
                        city,
                        int(place_id),
                        seed=self.split_seed,
                        modulus=self.holdout_modulus,
                    )
                    == self.holdout_remainder
                )
                if self.split == "train" and is_holdout:
                    continue
                if self.split == "holdout" and not is_holdout:
                    continue
                place_rows.append(global_rows.astype(np.int64, copy=False))
                place_keys.append((city, int(place_id)))
                heldout_flags.append(is_holdout)
        if not place_rows:
            raise ValueError(
                f"semantic-layout {self.split} split contains no eligible places"
            )
        if self.split == "train" and all(heldout_flags):
            raise ValueError("semantic-layout training split contains no train places")
        if self.split == "holdout" and not all(heldout_flags):
            raise RuntimeError("internal holdout split construction error")

        # The initial validator's memmaps must not be inherited across worker
        # processes.  Each worker opens its own read-only maps lazily.
        del labels, shuffled, arrays
        self._place_rows = tuple(place_rows)
        self.place_keys = tuple(place_keys)
        self._arrays: dict[str, np.ndarray] | None = None

    @property
    def num_classes(self) -> int:
        return len(SEMANTIC_LAYOUT_CLASSES)

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                "labels": np.load(
                    self.cache_dir / "labels.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                "shuffled_indices": np.load(
                    self.cache_dir / "shuffled_indices.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            }
        return self._arrays

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def close(self) -> None:
        arrays = self._arrays
        self._arrays = None
        if arrays is None:
            return
        for array in arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._place_rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        rows = self._place_rows[int(index)]
        if self.random_sample:
            selected = torch.randperm(len(rows))[: self.views_per_place].numpy()
            receiver_rows = rows[selected]
        else:
            receiver_rows = rows[: self.views_per_place]
        arrays = self._open_arrays()
        if self.mode == "aligned":
            source_rows = receiver_rows
        else:
            source_rows = np.asarray(
                arrays["shuffled_indices"][receiver_rows], dtype=np.int64
            )
        layouts = np.asarray(arrays["labels"][source_rows]).copy()
        if layouts.size:
            valid = layouts != SEMANTIC_LAYOUT_IGNORE_INDEX
            if bool(np.any(layouts[valid] >= self.num_classes)):
                raise ValueError("cache contains a semantic-layout class out of range")
        place_label = torch.full(
            (self.views_per_place,), int(index), dtype=torch.long
        )
        metadata = {
            "receiver_cache_indices": torch.from_numpy(
                receiver_rows.astype(np.int64, copy=True)
            ),
            "source_cache_indices": torch.from_numpy(
                np.asarray(source_rows, dtype=np.int64).copy()
            ),
        }
        return torch.from_numpy(layouts), place_label, metadata

    def split_cache_index_and_place_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Return every split row once, in stable receiver-index order.

        This is used by the offline descriptor/audit path; it never applies the
        shuffled donor because audit code needs receiver identity separately.
        """

        pairs = [
            (int(row), place_index)
            for place_index, rows in enumerate(self._place_rows)
            for row in rows
        ]
        pairs.sort(key=lambda value: value[0])
        return (
            np.asarray([value[0] for value in pairs], dtype=np.int64),
            np.asarray([value[1] for value in pairs], dtype=np.int64),
        )


class SemanticLayoutIndexDataset(Dataset):
    """Read semantic layouts for a fixed sequence of receiver cache rows."""

    def __init__(
        self,
        cache_dir: str | Path,
        indices: Sequence[int] | np.ndarray,
        *,
        mode: str = "aligned",
        verify_cache_hashes: bool = True,
        expected_index_type: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.mode = str(mode).lower()
        if self.mode not in {"aligned", "shuffled"}:
            raise ValueError("mode must be aligned or shuffled")
        manifest, arrays, hashes = validate_semantic_layout_cache(
            self.cache_dir,
            verify_array_files=verify_cache_hashes,
            expected_index_type=expected_index_type,
        )
        self.manifest = manifest
        self.array_hashes = hashes
        indices_array = np.asarray(indices)
        if indices_array.ndim != 1 or indices_array.dtype.kind not in "ui":
            raise ValueError("semantic-layout indices must be a 1D integer array")
        self.indices = indices_array.astype(np.int64, copy=True)
        if self.indices.size and (
            int(self.indices.min()) < 0
            or int(self.indices.max()) >= int(manifest["num_images"])
        ):
            raise ValueError("semantic-layout index is outside cache bounds")
        del arrays
        self._arrays: dict[str, np.ndarray] | None = None

    def _open_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                "labels": np.load(
                    self.cache_dir / "labels.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                "shuffled_indices": np.load(
                    self.cache_dir / "shuffled_indices.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            }
        return self._arrays

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def close(self) -> None:
        arrays = self._arrays
        self._arrays = None
        if arrays is None:
            return
        for array in arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        receiver = int(self.indices[int(index)])
        arrays = self._open_arrays()
        source = (
            receiver
            if self.mode == "aligned"
            else int(arrays["shuffled_indices"][receiver])
        )
        labels = np.asarray(arrays["labels"][source]).copy()
        return torch.from_numpy(labels), receiver


__all__ = [
    "AG_SLRD_SPLIT_ALGORITHM",
    "SemanticLayoutIndexDataset",
    "SemanticLayoutPlaceDataset",
    "place_split_remainder",
]
