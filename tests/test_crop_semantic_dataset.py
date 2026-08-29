"""Offline contract tests for Crop-CLS teacher metadata.

The online CLIP teacher must receive one clean full image per place, while the
student still receives all K independently transformed views.  These tests use
tiny synthetic GSV-Cities files and never instantiate a network or access the
internet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from src.core.vpr_datamodule import VPRDataModule
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.utils import config_manager


def _write_tiny_gsv(root: Path, *, places: int = 2, views: int = 4) -> None:
    dataframe_dir = root / "Dataframes"
    image_dir = root / "Images" / "TestCity"
    dataframe_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    rows = []
    for place_offset in range(places):
        place_id = 10 + place_offset
        for view in range(views):
            rows.append(
                {
                    "place_id": place_id,
                    "city_id": "TestCity",
                    "panoid": f"p{place_offset}-{view}",
                    "year": 2020 + view,
                    "month": view + 1,
                    "northdeg": view * 10,
                    "lat": float(place_offset + 1),
                    "lon": float(view + 1),
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(dataframe_dir / "TestCity.csv", index=False)

    # GSVCitiesDataset names files from the place-indexed dataframe row.
    # Widely separated solid colours make every view easy to identify after
    # JPEG encoding and resizing.
    for row in rows:
        indexed_row = pd.Series(row, name=row["place_id"])
        filename = GSVCitiesDataset.get_img_name(indexed_row)
        place_offset = int(row["place_id"]) - 10
        view = int(str(row["panoid"]).rsplit("-", maxsplit=1)[1])
        colour = (
            20 + 90 * place_offset + 13 * view,
            40 + 11 * view,
            210 - 17 * view,
        )
        Image.new("RGB", (37, 29), colour).save(
            image_dir / filename,
            quality=100,
            subsampling=0,
        )


def _image_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    resized = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class _InvertingStudentTransform:
    """A visible stand-in for ordinary (possibly random) student transforms."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: Image.Image) -> torch.Tensor:
        self.calls += 1
        return 1.0 - _image_tensor(image, (64, 64))


class _CleanTeacherTransform:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: Image.Image) -> torch.Tensor:
        self.calls += 1
        return _image_tensor(image, (280, 280))


def test_crop_teacher_returns_one_clean_sampled_view_per_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_tiny_gsv(tmp_path)
    student_transform = _InvertingStudentTransform()
    teacher_transform = _CleanTeacherTransform()
    dataset = GSVCitiesDataset(
        dataset_path=tmp_path,
        cities=["TestCity"],
        img_per_place=4,
        random_sample_from_each_place=True,
        transform=student_transform,
        return_metadata=False,
        return_crop_semantic_view=True,
        teacher_transform=teacher_transform,
    )

    # Patch only after construction: the city-level dataframe shuffle has
    # already happened.  __getitem__ now uses a known non-trivial sampled
    # order, so index zero is an explicit contract rather than an accident of
    # CSV ordering.
    def prescribed_sample(frame, n=None, *args, **kwargs):
        assert n == 4
        return frame.iloc[[2, 0, 3, 1]]

    monkeypatch.setattr(pd.DataFrame, "sample", prescribed_sample)

    images, labels, metadata = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
    )

    assert images.shape == (2, 4, 3, 64, 64)
    assert labels.shape == (2, 4)
    # Crucially, this is [P,C,H,W], not [P,K,C,H,W].
    assert metadata["crop_semantic_teacher_image"].shape == (2, 3, 280, 280)
    assert metadata["crop_semantic_view_index"].tolist() == [0, 0]
    assert "teacher_images" not in metadata
    assert student_transform.calls == 2 * 4
    assert teacher_transform.calls == 2

    clean = metadata["crop_semantic_teacher_image"]
    for place_index in range(2):
        clean_colour = clean[place_index, :, 0, 0]
        # The clean teacher tensor and inverted student tensor identify the
        # same raw photograph only for the first sampled view.
        assert torch.allclose(
            clean_colour,
            1.0 - images[place_index, 0, :, 0, 0],
            atol=1e-6,
            rtol=0.0,
        )
        for other_view in range(1, 4):
            assert not torch.allclose(
                clean_colour,
                1.0 - images[place_index, other_view, :, 0, 0],
                atol=1e-3,
                rtol=0.0,
            )


def test_datamodule_wires_full_280_crop_teacher_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_tiny_gsv(tmp_path, places=1)

    monkeypatch.setattr(
        config_manager,
        "get_dataset_path",
        lambda dataset_name, dataset_type: tmp_path,
    )
    datamodule = VPRDataModule(
        train_set_name="gsv-cities",
        cities=["TestCity"],
        train_image_size=(96, 96),
        teacher_image_size=(280, 280),
        batch_size=1,
        img_per_place=4,
        random_sample_from_each_place=False,
        val_set_names=[],
        num_workers=0,
        return_crop_semantic_view=True,
        augmentation_mode="photometric",
    )
    dataset = datamodule._get_train_dataset()

    assert dataset.return_metadata is True
    assert dataset.return_crop_semantic_view is True
    assert dataset.return_teacher_view is False
    images, _, metadata = dataset[0]
    assert images.shape == (4, 3, 96, 96)
    assert metadata["crop_semantic_teacher_image"].shape == (3, 280, 280)
    assert metadata["crop_semantic_view_index"].item() == 0

