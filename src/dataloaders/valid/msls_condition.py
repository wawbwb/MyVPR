"""Datasets and shared manifests for custom MSLS condition screening.

The condition queries are searched against the unchanged standard MSLS-val
database. This is intentionally a custom full-database protocol, not the
official MSLS condition protocol (which filters the database as well).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.dataloaders.valid.msls_condition_protocol import (
    CONDITION_COLUMNS,
    CONDITION_FILES,
    CONDITION_ORDER,
    CONDITION_UNION_QUERY_FILE,
    STANDARD_DB_FILE,
    STANDARD_GT_FILE,
    STANDARD_QUERY_FILE,
)

__all__ = [
    "CONDITION_COLUMNS",
    "CONDITION_FILES",
    "CONDITION_ORDER",
    "CONDITION_UNION_QUERY_FILE",
    "STANDARD_DB_FILE",
    "STANDARD_GT_FILE",
    "STANDARD_QUERY_FILE",
    "MSLSConditionDataset",
    "MSLSConditionUnionDataset",
]


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        # Keep manifest-only tools/tests independent of optional FAISS imports
        # pulled in by ``src.utils`` when an explicit path is already supplied.
        from src.utils import config_manager

        dataset_path = config_manager.get_dataset_path(
            dataset_name="msls-val", dataset_type="val"
        )
    resolved = Path(dataset_path)
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"The MSLS validation directory {resolved} does not exist"
        )
    return resolved


def _load_path_manifest(dataset_path: Path, filename: str) -> np.ndarray:
    path = dataset_path / filename
    if not path.is_file():
        raise FileNotFoundError(f"MSLS manifest not found: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1:
        raise ValueError(f"MSLS manifest must be one-dimensional: {path}")
    canonical = np.asarray(
        [str(value).replace("\\", "/") for value in values.tolist()],
        dtype=np.str_,
    )
    if len(set(canonical.tolist())) != len(canonical):
        raise ValueError(f"MSLS manifest contains duplicate paths: {path}")
    return canonical


def _load_ground_truth(
    dataset_path: Path,
    filename: str,
    *,
    num_queries: int,
    num_references: int,
) -> np.ndarray:
    path = dataset_path / filename
    if not path.is_file():
        raise FileNotFoundError(f"MSLS ground-truth manifest not found: {path}")
    ground_truth = np.load(path, allow_pickle=True)
    if ground_truth.ndim != 1 or len(ground_truth) != num_queries:
        raise ValueError(
            f"MSLS ground truth/query count mismatch in {path}: "
            f"{len(ground_truth)} vs {num_queries}"
        )
    for query_index, positives in enumerate(ground_truth):
        indices = np.asarray(positives)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(
                f"MSLS query {query_index} in {path} has no one-dimensional "
                "positive set"
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                f"MSLS query {query_index} in {path} has non-integer positives"
            )
        indices = indices.astype(np.int64, copy=False)
        if (
            len(np.unique(indices)) != len(indices)
            or int(indices.min()) < 0
            or int(indices.max()) >= num_references
        ):
            raise ValueError(
                f"MSLS query {query_index} in {path} has invalid positives"
            )
    return ground_truth


class _MSLSImageDataset(Dataset):
    """Common image loading for fixed MSLS reference/query manifests."""

    dataset_name: str
    dataset_path: Path
    dbImages: np.ndarray
    qImages: np.ndarray
    image_paths: np.ndarray
    num_references: int
    num_queries: int

    def __init__(self, input_transform: Callable | None) -> None:
        self.input_transform = input_transform

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image_path = self.dataset_path / self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.input_transform is not None:
                image = self.input_transform(image)
            else:
                image = image.copy()
        return image, index

    def __len__(self) -> int:
        return len(self.image_paths)


class MSLSConditionDataset(_MSLSImageDataset):
    """One custom condition-query set searched against the standard full DB."""

    CONDITION_ORDER = CONDITION_ORDER
    CONDITION_COLUMNS = CONDITION_COLUMNS
    CONDITION_FILES = CONDITION_FILES

    def __init__(
        self,
        condition: str,
        dataset_path: str | Path | None = None,
        input_transform: Callable | None = None,
    ) -> None:
        super().__init__(input_transform)
        if condition not in CONDITION_FILES:
            raise ValueError(
                f"Unknown condition {condition!r}. Choose from: "
                f"{list(CONDITION_FILES)}"
            )

        self.condition = condition
        self.dataset_name = f"msls-val-{condition}-full-db"
        self.dataset_path = _resolve_dataset_path(dataset_path)
        query_filename, gt_filename = CONDITION_FILES[condition]

        self.dbImages = _load_path_manifest(
            self.dataset_path, STANDARD_DB_FILE
        )
        self.qImages = _load_path_manifest(self.dataset_path, query_filename)
        if set(self.dbImages.tolist()) & set(self.qImages.tolist()):
            raise ValueError(
                f"{self.dataset_name} database and query manifests overlap"
            )
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        self.ground_truth = _load_ground_truth(
            self.dataset_path,
            gt_filename,
            num_queries=self.num_queries,
            num_references=self.num_references,
        )
        self.image_paths = np.concatenate((self.dbImages, self.qImages))


class MSLSConditionUnionDataset(_MSLSImageDataset):
    """Standard DB plus the deterministic union of all evaluation queries.

    The union manifest must begin with the standard query manifest in exactly
    the same order. Condition-only queries are appended by the split
    generator. Consequently standard query descriptor offsets remain the first
    num_standard_queries rows, while every condition query can be mapped by
    exact path into this one shared descriptor/cache index.
    """

    QUERY_UNION_FILE = CONDITION_UNION_QUERY_FILE

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        input_transform: Callable | None = None,
    ) -> None:
        super().__init__(input_transform)
        self.dataset_name = "msls-val-condition-union-full-db"
        self.dataset_path = _resolve_dataset_path(dataset_path)
        self.dbImages = _load_path_manifest(
            self.dataset_path, STANDARD_DB_FILE
        )
        self.standardQImages = _load_path_manifest(
            self.dataset_path, STANDARD_QUERY_FILE
        )
        self.qImages = _load_path_manifest(
            self.dataset_path, CONDITION_UNION_QUERY_FILE
        )

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        self.num_standard_queries = len(self.standardQImages)
        if self.num_queries < self.num_standard_queries or not np.array_equal(
            self.qImages[: self.num_standard_queries], self.standardQImages
        ):
            raise ValueError(
                "condition query union must begin with the complete standard "
                "MSLS query manifest in its original order"
            )
        if set(self.dbImages.tolist()) & set(self.qImages.tolist()):
            raise ValueError(
                "condition-union database and query manifests overlap"
            )
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
