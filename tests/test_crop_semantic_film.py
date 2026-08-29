from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.crop_semantic_film import (
    CropCLSSemanticTarget,
    CropSemanticFiLM,
    warm_start_crop_semantic_film_model,
)


def test_crop_semantic_film_is_exact_zero_start() -> None:
    module = CropSemanticFiLM(
        in_channels=8, hidden_dim=4, semantic_dim=6, alpha=0.1
    )
    tokens = torch.randn(3, 16, 8)

    modulated, semantic, scale = module(tokens)

    assert torch.equal(modulated, tokens)
    assert semantic.shape == (3, 16, 6)
    assert scale.shape == tokens.shape
    assert torch.count_nonzero(scale) == 0
    diagnostics = module.diagnostics()
    assert diagnostics["crop_film_modulation_rms"].item() == 0.0
    assert diagnostics["crop_film_modulation_abs_max"].item() == 0.0


def test_crop_semantic_film_bypass_is_nested_and_exact() -> None:
    module = CropSemanticFiLM(8, 4, 6)
    nn.init.constant_(module.channel_scale.weight, 0.2)
    tokens = torch.randn(2, 4, 8)
    changed, _, _ = module(tokens)
    assert not torch.equal(changed, tokens)

    with module.bypass():
        with module.bypass():
            bypassed, _, raw_scale = module(tokens)
            assert torch.equal(bypassed, tokens)
            assert torch.count_nonzero(raw_scale) > 0
            assert module.diagnostics()["crop_film_bypassed"].item() == 1.0
    assert not module.bypassed
    restored, _, _ = module(tokens)
    torch.testing.assert_close(restored, changed)


def test_crop_semantic_film_can_skip_teacher_only_projection() -> None:
    module = CropSemanticFiLM(8, 4, 6)
    tokens = torch.randn(2, 4, 8)

    modulated, semantic, raw_scale = module(
        tokens, return_semantic=False
    )

    assert semantic is None
    assert modulated.shape == tokens.shape
    assert raw_scale.shape == tokens.shape
    assert "crop_semantic_token_std" not in module.diagnostics()


def test_crop_semantic_film_projects_only_selected_teacher_views() -> None:
    module = CropSemanticFiLM(8, 4, 6)
    tokens = torch.randn(8, 4, 8)
    selected = torch.tensor([0, 4])

    modulated, semantic, raw_scale = module(
        tokens, semantic_batch_indices=selected
    )

    assert modulated.shape == tokens.shape
    assert raw_scale.shape == tokens.shape
    assert semantic.shape == (2, 4, 6)


def test_crop_semantic_film_has_vpr_and_semantic_gradients() -> None:
    module = CropSemanticFiLM(8, 4, 6)
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    modulated, semantic, _ = module(tokens)
    (modulated.sum() + semantic.square().mean()).backward()

    assert module.channel_scale.weight.grad is not None
    assert torch.count_nonzero(module.channel_scale.weight.grad) > 0
    assert module.semantic_projection.weight.grad is not None
    assert torch.count_nonzero(module.semantic_projection.weight.grad) > 0
    assert module.bottleneck.weight.grad is not None
    assert torch.count_nonzero(module.bottleneck.weight.grad) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0},
        {"hidden_dim": 0},
        {"semantic_dim": 0},
        {"alpha": 0.0},
        {"alpha": 0.21},
    ],
)
def test_crop_semantic_film_rejects_invalid_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        CropSemanticFiLM(**kwargs)


def _controlled_tokens() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A 2x2 patch grid: aligned TL descriptors differ across the two places;
    # the opposite BR regions swap them. This makes both controls orthogonal.
    student = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]],
            [[0.0, 1.0], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]],
        ],
        requires_grad=True,
    )
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    quadrants = torch.tensor([0, 0])
    return student, teacher, quadrants


@pytest.mark.parametrize(
    ("mode", "expected_loss", "expected_selected_cosine"),
    [
        ("aligned", 0.0, 1.0),
        ("wrong_region", 1.0, 0.0),
        ("wrong_place", 1.0, 0.0),
    ],
)
def test_crop_cls_target_modes_are_causally_separated(
    mode: str, expected_loss: float, expected_selected_cosine: float
) -> None:
    student, teacher, quadrants = _controlled_tokens()
    target = CropCLSSemanticTarget(mode=mode)

    loss, stats = target(
        student,
        teacher_embeddings=teacher,
        region_indices=quadrants,
    )

    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    torch.testing.assert_close(
        stats["crop_semantic_cosine"],
        torch.tensor(expected_selected_cosine),
    )
    torch.testing.assert_close(
        stats["crop_semantic_aligned_cosine"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["crop_semantic_aligned_minus_wrong_region"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["crop_semantic_aligned_minus_wrong_place"], torch.tensor(1.0)
    )
    loss.backward()
    assert student.grad is not None


def test_region_schedule_is_shared_and_resume_stable() -> None:
    first = CropCLSSemanticTarget.region_indices(
        5, global_step=7, device=torch.device("cpu")
    )
    resumed = CropCLSSemanticTarget.region_indices(
        5, global_step=7, device=torch.device("cpu")
    )
    assert torch.equal(first, torch.full((5,), 3))
    assert torch.equal(resumed, first)


class _MeanTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))
        self.calls = 0
        self.last_shape = None

    def forward(self, crops: torch.Tensor):
        self.calls += 1
        self.last_shape = tuple(crops.shape)
        means = crops.mean(dim=(-2, -1))
        return means[:, :2], torch.empty(0, device=crops.device)


def test_crop_cls_target_lazy_teacher_uses_one_crop_per_place() -> None:
    built = []

    def factory():
        teacher = _MeanTeacher()
        built.append(teacher)
        return teacher

    target = CropCLSSemanticTarget(
        mode="aligned",
        teacher_factory=factory,
        expected_teacher_image_size=(8, 8),
    )
    images = torch.zeros(2, 3, 8, 8)
    images[0, 0, :4, :4] = 1.0
    images[1, 1, :4, :4] = 1.0
    student = torch.zeros(2, 4, 2)
    student[0, 0] = torch.tensor([1.0, 0.0])
    student[1, 0] = torch.tensor([0.0, 1.0])

    loss, _ = target(
        student,
        teacher_images=images,
        region_indices=torch.tensor([0, 0]),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert len(built) == 1
    assert built[0].calls == 1
    assert built[0].last_shape == (2, 3, 4, 4)
    assert built[0].training is False
    assert not any(parameter.requires_grad for parameter in built[0].parameters())


def test_crop_cls_teacher_is_encoded_in_bounded_chunks() -> None:
    teacher = _MeanTeacher()
    target = CropCLSSemanticTarget(
        mode="aligned",
        teacher=teacher,
        teacher_chunk_size=2,
        expected_teacher_image_size=(8, 8),
    )
    images = torch.zeros(5, 3, 8, 8)
    images[:, 0, :4, :4] = 1.0
    student = torch.zeros(5, 4, 2)
    student[:, 0, 0] = 1.0

    loss, _ = target(
        student,
        teacher_images=images,
        region_indices=torch.zeros(5, dtype=torch.long),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert teacher.calls == 3
    assert teacher.last_shape == (1, 3, 4, 4)


def test_plain_crop_target_keeps_teacher_out_of_student_state_dict() -> None:
    target = CropCLSSemanticTarget(
        mode="aligned",
        teacher=_MeanTeacher(),
        expected_teacher_image_size=(8, 8),
    )

    class _Student(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.student_weight = nn.Parameter(torch.ones(()))
            self.crop_target = target

    student = _Student()
    assert set(student.state_dict()) == {"student_weight"}


def test_wrong_place_single_place_is_graph_connected_zero() -> None:
    student = torch.randn(1, 4, 3, requires_grad=True)
    teacher = torch.randn(1, 3)
    target = CropCLSSemanticTarget(mode="wrong_place")
    loss, stats = target(
        student,
        teacher_embeddings=teacher,
        region_indices=torch.tensor([0]),
    )
    assert loss.item() == 0.0
    assert stats["crop_semantic_wrong_place_valid"].item() == 0.0
    loss.backward()
    assert student.grad is not None


class _FilmBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.legacy = nn.Linear(2, 2)
        self.crop_semantic_film = CropSemanticFiLM(2, 2, 2)


class _WarmStartModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FilmBackbone()
        self.aggregator = nn.Linear(2, 2)
        self.semantic_region_gate = nn.Linear(2, 1)


def _ru_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.crop_semantic_film.")
    }


def _ru_hyper_parameters() -> dict:
    return {
        "seed": 42,
        "datamodule": {
            "train_set_name": "gsv-cities",
            "cities": "all",
            "train_image_size": [280, 280],
            "augmentation_mode": "photometric",
            "batch_size": 40,
            "img_per_place": 4,
        },
        "backbone": {"class": "DinoV2"},
        "aggregator": {"class": "BoQ"},
        "trainer": {"max_epochs": 40},
        "distillation": {
            "semantic_region": {
                "enabled": True,
                "mode": "repeatability_uniqueness_only",
                "lambda_target": 0.02,
                "alpha": 0.2,
            }
        },
    }


def _save_ru_checkpoint(
    path: Path,
    model: nn.Module,
    state: dict[str, torch.Tensor] | None = None,
) -> str:
    torch.save(
        {
            "state_dict": _ru_state(model) if state is None else state,
            "hyper_parameters": _ru_hyper_parameters(),
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_crop_semantic_warm_start_allows_exact_new_prefix(
    tmp_path: Path,
) -> None:
    model = _WarmStartModel()
    checkpoint = tmp_path / "ru.ckpt"
    digest = _save_ru_checkpoint(checkpoint, model)

    report = warm_start_crop_semantic_film_model(
        model, checkpoint, expected_sha256=digest
    )

    expected_new = {
        key
        for key in model.state_dict()
        if key.startswith("backbone.crop_semantic_film.")
    }
    assert set(report["new_keys"]) == expected_new
    assert report["loaded_keys"] == len(_ru_state(model))


def test_crop_semantic_warm_start_rejects_legacy_mismatch(
    tmp_path: Path,
) -> None:
    model = _WarmStartModel()
    state = _ru_state(model)
    state.pop("aggregator.bias")
    checkpoint = tmp_path / "missing.ckpt"
    _save_ru_checkpoint(checkpoint, model, state)

    with pytest.raises(RuntimeError, match="unsafe RU crop-semantic warm start"):
        warm_start_crop_semantic_film_model(model, checkpoint)


def test_crop_semantic_warm_start_rejects_existing_new_branch(
    tmp_path: Path,
) -> None:
    model = _WarmStartModel()
    state = _ru_state(model)
    state["backbone.crop_semantic_film.channel_scale.bias"] = (
        model.backbone.crop_semantic_film.channel_scale.bias.detach().clone()
    )
    checkpoint = tmp_path / "already-film.ckpt"
    _save_ru_checkpoint(checkpoint, model, state)

    with pytest.raises(RuntimeError, match="must not contain crop-semantic"):
        warm_start_crop_semantic_film_model(model, checkpoint)
