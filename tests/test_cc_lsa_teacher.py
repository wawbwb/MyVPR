from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from scripts.train_cc_lsa_teacher import REGISTERED_CONFIG, load_config
from src.models.cc_lsa import (
    CC_LSA_TEACHER_SCHEMA,
    CC_LSA_TEACHER_VERSION,
    LocalSemanticAlignmentEncoder,
    crop_image_grid,
    load_lsa_checkpoint,
    lsa_cosine_loss,
    module_state_sha256,
    pool_region_grid,
    tensor_mapping_sha256,
)


def test_crop_and_region_pooling_share_row_major_geometry() -> None:
    images = torch.arange(4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    images = images.repeat(1, 3, 1, 1)
    crops = crop_image_grid(images, rows=2, columns=2)
    assert crops.shape == (4, 3, 2, 2)
    assert crops[:, 0].mean(dim=(1, 2)).tolist() == [2.5, 4.5, 10.5, 12.5]

    # Four 2-D patch descriptors in the same TL, TR, BL, BR order.
    local = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    pooled = pool_region_grid(local, rows=2, columns=2)
    torch.testing.assert_close(pooled, local)


def test_lsa_loss_stops_target_gradient_and_is_float32() -> None:
    local = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]],
        requires_grad=True,
    )
    targets = local.detach().clone().requires_grad_(True)
    loss = lsa_cosine_loss(local, targets, rows=2, columns=2)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert local.grad is not None
    assert targets.grad is None


def test_pool_region_grid_rejects_non_divisible_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        pool_region_grid(torch.randn(1, 36, 4), rows=4, columns=4)


def test_lsa_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        lsa_cosine_loss(
            torch.randn(2, 4, 8), torch.randn(2, 3, 8), rows=2, columns=2
        )


def test_teacher_config_is_exactly_the_frozen_registration(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert load_config(project_root / "config" / "cc_lsa_teacher.yaml") == REGISTERED_CONFIG

    altered = copy.deepcopy(REGISTERED_CONFIG)
    altered["training"]["epochs"] = 6
    altered_path = tmp_path / "altered.yaml"
    altered_path.write_text(yaml.safe_dump(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen pre-registration"):
        load_config(altered_path)


class _FakeResidualBlock(nn.Module):
    def __init__(self, width: int, value: float) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        nn.init.constant_(self.linear.weight, value)
        nn.init.constant_(self.linear.bias, value / 10.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + torch.tanh(self.linear(tokens))


class _FakeTransformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [_FakeResidualBlock(width, 0.01), _FakeResidualBlock(width, 0.02)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            tokens = block(tokens)
        return tokens


class _FakeVisual(nn.Module):
    image_size = (8, 8)
    patch_size = (4, 4)

    def __init__(self) -> None:
        super().__init__()
        width = 4
        self.conv1 = nn.Conv2d(3, width, kernel_size=4, stride=4, bias=False)
        with torch.no_grad():
            for channel in range(width):
                self.conv1.weight[channel].fill_(0.01 * (channel + 1))
        self.class_embedding = nn.Parameter(torch.full((width,), 0.1))
        self.positional_embedding = nn.Parameter(
            torch.arange(20, dtype=torch.float32).reshape(5, width) / 100.0
        )
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _FakeTransformer(width)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.1, 0.1]]
            )
        )


class _FakeClipTeacher(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.visual = _FakeVisual()
        self.visual.requires_grad_(False)


def test_lsa_delta_has_gradient_and_round_trips_without_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import src.models.clip_teacher as clip_teacher

    monkeypatch.setattr(clip_teacher, "CLIPTeacherEncoder", _FakeClipTeacher)
    model = LocalSemanticAlignmentEncoder(
        model_name="mock",
        pretrained="mock",
        hf_mirror=None,
        trainable_last_n_blocks=1,
    )
    assert model.native_grid_size == (2, 2)
    assert model.semantic_dim == 3
    assert not model.visual.transformer.resblocks[0].linear.weight.requires_grad
    assert model.visual.transformer.resblocks[1].linear.weight.requires_grad
    base_hash = module_state_sha256(model.visual)

    images = torch.randn(2, 3, 8, 8)
    targets = torch.randn(2, 4, 3)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.05,
    )
    before = model(images).detach().clone()
    loss = lsa_cosine_loss(model(images), targets)
    loss.backward()
    assert model.visual.transformer.resblocks[0].linear.weight.grad is None
    last_gradient = model.visual.transformer.resblocks[1].linear.weight.grad
    assert last_gradient is not None and bool(last_gradient.abs().sum() > 0)
    assert model.visual.proj.grad is not None
    optimizer.step()
    after = model(images).detach()
    assert not torch.equal(before, after)

    delta = model.trainable_state_dict()
    checkpoint = {
        "schema": CC_LSA_TEACHER_SCHEMA,
        "version": CC_LSA_TEACHER_VERSION,
        "model": {
            "model_name": "mock",
            "pretrained": "mock",
            "hf_mirror": None,
            "trainable_last_n_blocks": 1,
        },
        "base_model_state_sha256": base_hash,
        "trainable_state_dict": delta,
        "trainable_state_sha256": tensor_mapping_sha256(delta),
        "teacher_contract": {"verdict": "PASS"},
    }
    checkpoint_path = tmp_path / "lsa.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded, _ = load_lsa_checkpoint(checkpoint_path)
    torch.testing.assert_close(loaded(images), after)

    corrupted = copy.deepcopy(checkpoint)
    first_name = next(iter(corrupted["trainable_state_dict"]))
    corrupted["trainable_state_dict"][first_name].view(-1)[0] += 1
    corrupted_path = tmp_path / "corrupted.pt"
    torch.save(corrupted, corrupted_path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_lsa_checkpoint(corrupted_path)
