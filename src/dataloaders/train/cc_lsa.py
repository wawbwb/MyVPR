"""Stable one-view-per-place GSV dataset for CC-LSA pre-training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T2

from src.cc_lsa_gate_a import file_sha256, implementation_sha256, string_sequence_sha256
from src.dataloaders.train.semantic_layout import place_split_remainder


CC_LSA_TARGET_SCHEMA = "openvpr_cc_lsa_gsv_crop_cls"
CC_LSA_TARGET_VERSION = 1
CC_LSA_IMAGE_SIZE = (280, 280)
CC_LSA_REGION_GRID = (2, 2)
CC_LSA_MODEL_NAME = "ViT-B-16"
CC_LSA_PRETRAINED = "openai"
CC_LSA_SOURCE_TRANSFORM = {
    "input": "clean_rgb",
    "resize": [280, 280],
    "interpolation": "bicubic",
    "antialias": True,
    "tensor_dtype": "float32_scaled_0_1",
    "normalization": "imagenet",
    "operation_order": ["to_image", "resize", "to_float", "normalize"],
}
CC_LSA_TARGET_IMPLEMENTATION_FILES = {
    "src/models/cc_lsa.py",
    "src/dataloaders/train/cc_lsa.py",
    "src/models/clip_teacher.py",
    "scripts/cache_gsv_cc_lsa_targets.py",
}


def build_cc_lsa_image_transform(
    image_size: tuple[int, int] = CC_LSA_IMAGE_SIZE,
) -> T2.Compose:
    """Match the canonical clean RU validation/teacher-view geometry exactly."""

    size = tuple(int(value) for value in image_size)
    if size != CC_LSA_IMAGE_SIZE:
        raise ValueError("registered CC-LSA source geometry is exactly 280x280")
    return T2.Compose(
        [
            T2.ToImage(),
            T2.Resize(
                size=size,
                interpolation=T2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@dataclass(frozen=True)
class GSVPlaceView:
    city: str
    place_id: int
    relative_path: str
    split: int  # 0=train, 1=holdout


def gsv_image_name(row: pd.Series) -> str:
    city = str(row["city_id"])
    place_id = str(int(row["place_id"]) % 10**5).zfill(7)
    # Keep this byte-for-byte compatible with GSVCitiesDataset.get_img_name
    # and cache_gsv_patch_semantics.image_name.
    year = str(row["year"]).zfill(4)
    month = str(row["month"]).zfill(2)
    heading = str(row["northdeg"]).zfill(3)
    return (
        f"{city}_{place_id}_{year}_{month}_{heading}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
    )


def discover_gsv_place_views(
    dataset_root: str | Path,
    *,
    cities: str | Sequence[str] = "all",
    seed: int = 42,
    minimum_views: int = 4,
    holdout_modulus: int = 10,
) -> tuple[tuple[GSVPlaceView, ...], list[dict[str, object]]]:
    root = Path(dataset_root).expanduser().resolve()
    dataframe_dir = root / "Dataframes"
    if cities == "all":
        selected_cities = tuple(
            path.stem for path in sorted(dataframe_dir.glob("*.csv"))
        )
    else:
        selected_cities = tuple(str(value) for value in cities)
    if not selected_cities or len(set(selected_cities)) != len(selected_cities):
        raise ValueError("cities must be non-empty and unique")
    if minimum_views < 1:
        raise ValueError("minimum_views must be positive")
    records: list[GSVPlaceView] = []
    csv_records: list[dict[str, object]] = []
    for city in selected_cities:
        csv_path = dataframe_dir / f"{city}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"GSV city CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        required = {
            "place_id",
            "city_id",
            "panoid",
            "year",
            "month",
            "northdeg",
            "lat",
            "lon",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"GSV CSV {csv_path} lacks required columns")
        if bool((frame["city_id"].astype(str) != city).any()):
            raise ValueError(f"GSV CSV {csv_path} contains a different city_id")
        csv_records.append(
            {"city": city, "rows": len(frame), "sha256": file_sha256(csv_path)}
        )
        for place_id, group in frame.groupby("place_id", sort=True):
            if len(group) < minimum_views:
                continue
            choices: list[tuple[bytes, str]] = []
            for _, row in group.iterrows():
                filename = gsv_image_name(row)
                relative = f"Images/{city}/{filename}"
                digest = hashlib.sha256(
                    f"cc_lsa_view_v1\0{seed}\0{city}\0{int(place_id)}\0{relative}".encode()
                ).digest()
                choices.append((digest, relative))
            _, relative_path = min(choices)
            if not (root / relative_path).is_file():
                raise FileNotFoundError(f"GSV image not found: {root / relative_path}")
            split = int(
                place_split_remainder(
                    city,
                    int(place_id),
                    seed=seed,
                    modulus=holdout_modulus,
                )
                == 0
            )
            records.append(
                GSVPlaceView(city, int(place_id), relative_path, split)
            )
    records.sort(key=lambda item: (item.city, item.place_id))
    if not records or not any(record.split == 0 for record in records):
        raise ValueError("GSV selection produced no train places")
    if not any(record.split == 1 for record in records):
        raise ValueError("GSV selection produced no holdout places")
    return tuple(records), csv_records


def image_to_imagenet_tensor(
    image: Image.Image, image_size: tuple[int, int] = CC_LSA_IMAGE_SIZE
) -> torch.Tensor:
    return build_cc_lsa_image_transform(image_size)(image.convert("RGB"))


class GSVPlaceViewDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        records: Sequence[GSVPlaceView],
        *,
        image_size: tuple[int, int] = CC_LSA_IMAGE_SIZE,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.records = tuple(records)
        self.image_size = tuple(int(value) for value in image_size)
        self.transform = build_cc_lsa_image_transform(self.image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.dataset_root / record.relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(index)


def load_target_manifest(
    cache_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> dict:
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CC-LSA target manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != CC_LSA_TARGET_SCHEMA:
        raise ValueError("unsupported CC-LSA target cache schema")
    if manifest.get("version") != CC_LSA_TARGET_VERSION:
        raise ValueError("unsupported CC-LSA target cache version")
    if manifest.get("complete") is not True:
        raise ValueError("CC-LSA target cache is incomplete")
    count = int(manifest.get("num_places", -1))
    semantic_dim = int(manifest.get("semantic_dim", -1))
    if count < 1 or semantic_dim != 512:
        raise ValueError("CC-LSA target count/dimension is invalid")
    if manifest.get("image_size") != list(CC_LSA_IMAGE_SIZE):
        raise ValueError("CC-LSA target image geometry differs")
    if manifest.get("region_grid") != list(CC_LSA_REGION_GRID):
        raise ValueError("CC-LSA target region geometry differs")
    if manifest.get("source_transform") != CC_LSA_SOURCE_TRANSFORM:
        raise ValueError("CC-LSA target source transform differs")
    implementation = manifest.get("implementation_sha256")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != CC_LSA_TARGET_IMPLEMENTATION_FILES
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in implementation.values()
        )
    ):
        raise ValueError("CC-LSA target implementation provenance is invalid")
    current_implementation = implementation_sha256()
    if any(
        current_implementation.get(name) != fingerprint
        for name, fingerprint in implementation.items()
    ):
        raise ValueError("CC-LSA target was built by a different implementation")
    selection = manifest.get("selection")
    expected_selection = {
        "algorithm": "sha256_one_view_per_place_v1",
        "seed": 42,
        "minimum_views": 4,
        "split_algorithm": "sha256_place_v1",
        "holdout_modulus": 10,
    }
    if selection != expected_selection:
        raise ValueError("CC-LSA target place-selection contract differs")
    teacher = manifest.get("teacher")
    if (
        not isinstance(teacher, dict)
        or teacher.get("model_name") != CC_LSA_MODEL_NAME
        or teacher.get("pretrained") != CC_LSA_PRETRAINED
        or not isinstance(teacher.get("base_model_state_sha256"), str)
        or len(teacher["base_model_state_sha256"]) != 64
    ):
        raise ValueError("CC-LSA target teacher identity differs")
    specs = {
        "paths.npy": ((count,), None),
        "city.npy": ((count,), None),
        "place_ids.npy": ((count,), np.dtype("int64")),
        "split.npy": ((count,), np.dtype("uint8")),
        "crop_cls.npy": ((count, 4, semantic_dim), np.dtype("float16")),
    }
    declared = manifest.get("array_sha256")
    if not isinstance(declared, dict):
        raise ValueError("CC-LSA target manifest has no array hashes")
    for filename, (shape, dtype) in specs.items():
        path = cache_dir / filename
        if not path.is_file() or declared.get(filename) != file_sha256(path):
            raise ValueError(f"CC-LSA target array missing or SHA mismatch: {filename}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or (dtype is not None and array.dtype != dtype):
            raise ValueError(f"CC-LSA target array shape/dtype mismatch: {filename}")
    paths = np.load(cache_dir / "paths.npy", allow_pickle=False)
    if manifest.get("path_sequence_sha256") != string_sequence_sha256(paths.tolist()):
        raise ValueError("CC-LSA target path sequence hash differs")
    cities = np.load(cache_dir / "city.npy", allow_pickle=False)
    place_ids = np.load(cache_dir / "place_ids.npy", allow_pickle=False)
    split_values = np.load(cache_dir / "split.npy", allow_pickle=False)
    if not bool(np.isin(split_values, (0, 1)).all()):
        raise ValueError("CC-LSA target split array contains an invalid value")
    if int((split_values == 0).sum()) != int(manifest.get("num_train_places", -1)):
        raise ValueError("CC-LSA target train-place count differs")
    if int((split_values == 1).sum()) != int(manifest.get("num_holdout_places", -1)):
        raise ValueError("CC-LSA target holdout-place count differs")
    identities = {(str(city), int(place)) for city, place in zip(cities, place_ids)}
    if len(identities) != count:
        raise ValueError("CC-LSA target contains duplicate place identities")
    for relative, city in zip(paths.tolist(), cities.tolist()):
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) < 3
            or relative_path.parts[0] != "Images"
            or relative_path.parts[1] != str(city)
        ):
            raise ValueError("CC-LSA target contains a non-canonical image path")
    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        if Path(str(manifest.get("dataset_root", ""))).resolve() != root:
            raise ValueError("CC-LSA target dataset root differs")
        csv_records = manifest.get("csv_files")
        if not isinstance(csv_records, list) or not csv_records:
            raise ValueError("CC-LSA target manifest has no CSV provenance")
        for record in csv_records:
            if not isinstance(record, dict) or set(record) != {"city", "rows", "sha256"}:
                raise ValueError("CC-LSA target CSV provenance is malformed")
            csv_path = root / "Dataframes" / f"{record['city']}.csv"
            if not csv_path.is_file() or file_sha256(csv_path) != record["sha256"]:
                raise ValueError(f"CC-LSA source CSV differs: {csv_path}")
    return manifest


class CCLSATargetDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        cache_dir: str | Path,
        *,
        split: str,
    ) -> None:
        if split not in {"train", "holdout", "all"}:
            raise ValueError("split must be train, holdout, or all")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest = load_target_manifest(
            self.cache_dir, dataset_root=self.dataset_root
        )
        self.paths = np.load(
            self.cache_dir / "paths.npy", mmap_mode="r", allow_pickle=False
        )
        self.targets = np.load(
            self.cache_dir / "crop_cls.npy", mmap_mode="r", allow_pickle=False
        )
        split_values = np.load(
            self.cache_dir / "split.npy", mmap_mode="r", allow_pickle=False
        )
        if split == "all":
            self.indices = np.arange(len(self.paths), dtype=np.int64)
        else:
            wanted = 0 if split == "train" else 1
            self.indices = np.flatnonzero(split_values == wanted).astype(np.int64)
        self.image_size = tuple(int(value) for value in self.manifest["image_size"])
        self.transform = build_cc_lsa_image_transform(self.image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, offset: int):
        index = int(self.indices[offset])
        with Image.open(self.dataset_root / str(self.paths[index])) as image:
            tensor = self.transform(image.convert("RGB"))
        target = torch.from_numpy(np.asarray(self.targets[index], dtype=np.float32))
        return tensor, target, index
