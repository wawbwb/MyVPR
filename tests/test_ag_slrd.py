from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.audit_semantic_layout_complementarity import (
    RetrievalResult,
    build_audit,
    search_descriptors,
)
from scripts.train_ag_slrd_semantic_teacher import (
    _amp_optimizer_step,
    _metric_loss_fp32,
)
from src.dataloaders.train.semantic_layout import (
    SemanticLayoutPlaceDataset,
    place_split_remainder,
)
from src.models.ag_slrd import (
    SemanticLayoutEncoder,
    build_teacher_checkpoint,
    load_semantic_layout_teacher,
)
from src.semantic_layout_cache import (
    ADE20K_CLASS_NAME_NORMALIZATION,
    ADE20K_PATCH_CACHE_SCHEMA,
    ADE20K_PATCH_CACHE_VERSION,
    ADE20K_CLASSES,
    ADE20K_TO_SEMANTIC_LAYOUT,
    DYNAMIC_SUPERCLASS_ID,
    SEMANTIC_LAYOUT_CACHE_SCHEMA,
    SEMANTIC_LAYOUT_CACHE_VERSION,
    SEMANTIC_LAYOUT_CLASSES,
    remap_ade20k_labels,
    semantic_layout_mapping_record,
    validate_ade20k_class_names,
    validate_ade20k_patch_cache,
    validate_semantic_layout_cache,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_mapping_is_exhaustive_and_preserves_dynamic() -> None:
    assert len(ADE20K_CLASSES) == 150
    assert len(ADE20K_TO_SEMANTIC_LAYOUT) == 150
    assert set(ADE20K_TO_SEMANTIC_LAYOUT) == set(range(12))
    dynamic_source = {
        "person", "car", "bus", "truck", "van", "animal", "bicycle"
    }
    for name in dynamic_source:
        source_id = ADE20K_CLASSES.index(name)
        assert ADE20K_TO_SEMANTIC_LAYOUT[source_id] == DYNAMIC_SUPERCLASS_ID
    source = np.arange(150, dtype=np.uint8).reshape(2, 3, 25)
    mapped = remap_ade20k_labels(source)
    assert mapped.dtype == np.uint8
    assert mapped.shape == source.shape
    assert int(mapped.max()) < len(SEMANTIC_LAYOUT_CLASSES)


def test_official_class_whitespace_is_accepted_but_reordering_is_not() -> None:
    official = list(ADE20K_CLASSES)
    official[7] = "bed "
    assert validate_ade20k_class_names(official) == ADE20K_CLASSES
    assert ADE20K_CLASS_NAME_NORMALIZATION == "strip_ascii_outer_whitespace_v1"

    reordered = official.copy()
    reordered[7], reordered[8] = reordered[8], reordered[7]
    with pytest.raises(ValueError, match="names/order differ"):
        validate_ade20k_class_names(reordered)


def test_patch_cache_accepts_pinned_hf_trailing_whitespace(tmp_path: Path) -> None:
    cache = tmp_path / "ade20k"
    cache.mkdir()
    labels = np.zeros((2, 70, 70), dtype=np.uint8)
    confidence = np.full((2, 70, 70), 255, dtype=np.uint8)
    shuffled = np.asarray([1, 0], dtype=np.int32)
    np.save(cache / "labels.npy", labels)
    np.save(cache / "confidence.npy", confidence)
    np.save(cache / "shuffled_indices.npy", shuffled)
    source_classes = list(ADE20K_CLASSES)
    source_classes[7] = "bed "
    manifest = {
        "schema": ADE20K_PATCH_CACHE_SCHEMA,
        "version": ADE20K_PATCH_CACHE_VERSION,
        "complete": True,
        "num_images": 2,
        "grid_size": [70, 70],
        "num_classes": 150,
        "classes": source_classes,
        "cities": [
            {
                "name": "TestCity",
                "offset": 0,
                "count": 2,
                "eligible_count": 2,
                "sha256": "a" * 64,
            }
        ],
        "array_sha256": {
            name: _sha256(cache / name)
            for name in (
                "labels.npy",
                "confidence.npy",
                "shuffled_indices.npy",
            )
        },
    }
    (cache / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    loaded_manifest, arrays, _ = validate_ade20k_patch_cache(cache)

    assert loaded_manifest["classes"][7] == "bed "
    assert arrays["labels"].shape == (2, 70, 70)


def test_layout_encoder_normalizes_and_learns_class_weights() -> None:
    model = SemanticLayoutEncoder(
        num_classes=12,
        embed_dim=8,
        channels=(8, 16, 32),
        descriptor_dim=24,
    )
    labels = torch.randint(0, 12, (3, 16, 16), dtype=torch.long)
    descriptor, auxiliary = model(labels, return_aux=True)

    assert descriptor.shape == (3, 24)
    assert torch.allclose(
        descriptor.float().norm(dim=1), torch.ones(3), atol=1e-5
    )
    weights = auxiliary["class_weights"]
    assert weights.shape == (3, 12)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)
    descriptor.square().mean().backward()
    assert model.class_weight_bias.grad is not None
    assert bool(torch.isfinite(model.class_weight_bias.grad).all())


def test_layout_encoder_rejects_float_or_invalid_labels() -> None:
    model = SemanticLayoutEncoder(
        embed_dim=4, channels=(8, 8, 16), descriptor_dim=16
    )
    with pytest.raises(TypeError, match="integer dtype"):
        model(torch.zeros(1, 8, 8, dtype=torch.float32))
    invalid = torch.zeros(1, 8, 8, dtype=torch.long)
    invalid[0, 0, 0] = 12
    with pytest.raises(ValueError, match="neither a valid class"):
        model(invalid)


class _DtypeRecordingLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_dtype: torch.dtype | None = None

    def forward(
        self, descriptors: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        del labels
        self.seen_dtype = descriptors.dtype
        return descriptors.square().mean(), 0.0


def test_metric_loss_is_forced_to_float32() -> None:
    loss_function = _DtypeRecordingLoss()
    descriptors = torch.randn(8, 16, dtype=torch.float16, requires_grad=True)
    labels = torch.arange(8)

    loss, _ = _metric_loss_fp32(loss_function, descriptors, labels)

    assert loss_function.seen_dtype == torch.float32
    assert loss.dtype == torch.float32
    loss.backward()
    assert descriptors.grad is not None
    assert bool(torch.isfinite(descriptors.grad).all())


def test_amp_step_backs_off_and_recovers_from_nonfinite_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameters = (parameter,)
    optimizer = torch.optim.AdamW(parameters, lr=0.1)
    scaler = torch.amp.GradScaler(
        "cpu", init_scale=128.0, growth_interval=100
    )

    # Initialise AdamW's moments, then prove a skipped update mutates neither
    # the parameter nor optimizer state.
    scaler.scale(parameter.square().sum()).backward()
    initial = _amp_optimizer_step(
        parameters=parameters,
        optimizer=optimizer,
        scaler=scaler,
        max_grad_norm=10.0,
    )
    assert initial["applied"] is True
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square().sum()).backward()
    assert parameter.grad is not None
    parameter.grad.fill_(torch.inf)
    before = parameter.detach().clone()
    state_before = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in optimizer.state[parameter].items()
    }
    overflow = _amp_optimizer_step(
        parameters=parameters,
        optimizer=optimizer,
        scaler=scaler,
        max_grad_norm=10.0,
    )

    assert overflow["applied"] is False
    assert overflow["scale_before"] == 128.0
    assert overflow["scale_after"] == 64.0
    assert torch.equal(parameter.detach(), before)
    for name, value in state_before.items():
        current = optimizer.state[parameter][name]
        if torch.is_tensor(value):
            assert torch.equal(current, value)
        else:
            assert current == value

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square().sum()).backward()
    recovered = _amp_optimizer_step(
        parameters=parameters,
        optimizer=optimizer,
        scaler=scaler,
        max_grad_norm=10.0,
    )

    assert recovered["applied"] is True
    assert not torch.equal(parameter.detach(), before)


def test_teacher_checkpoint_round_trip(tmp_path: Path) -> None:
    model = SemanticLayoutEncoder(
        embed_dim=4, channels=(8, 8, 16), descriptor_dim=16
    )
    checkpoint = build_teacher_checkpoint(
        model,
        mode="aligned",
        epoch=9,
        global_step=123,
        cache_provenance={"mapping_sha256": "a" * 64},
        data_config={"split": "test"},
        trainer_config={"epochs": 10},
    )
    path = tmp_path / "teacher.pt"
    torch.save(checkpoint, path)

    restored, payload = load_semantic_layout_teacher(path)

    assert restored.export_config() == model.export_config()
    assert payload["mode"] == "aligned"
    labels = torch.randint(0, 12, (2, 8, 8))
    assert torch.allclose(model(labels), restored(labels))


def _write_layout_cache(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "gsv"
    dataframe_dir = dataset_root / "Dataframes"
    dataframe_dir.mkdir(parents=True)
    frame = pd.DataFrame({"place_id": [10] * 4 + [20] * 4})
    csv_path = dataframe_dir / "TestCity.csv"
    frame.to_csv(csv_path, index=False)

    cache = tmp_path / "cache"
    cache.mkdir()
    labels = np.stack(
        [np.full((70, 70), index, dtype=np.uint8) for index in range(8)]
    )
    shuffled = np.asarray([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32)
    np.save(cache / "labels.npy", labels)
    np.save(cache / "shuffled_indices.npy", shuffled)
    manifest = {
        "schema": SEMANTIC_LAYOUT_CACHE_SCHEMA,
        "version": SEMANTIC_LAYOUT_CACHE_VERSION,
        "complete": True,
        "num_images": 8,
        "grid_size": [70, 70],
        "num_classes": 12,
        "classes": list(SEMANTIC_LAYOUT_CLASSES),
        "ignore_index": 255,
        "mapping": semantic_layout_mapping_record(),
        "source_manifest_sha256": "b" * 64,
        "cities": [
            {
                "name": "TestCity",
                "offset": 0,
                "count": 8,
                "sha256": _sha256(csv_path),
                "eligible_count": 8,
            }
        ],
        "index": {
            "type": "gsv_city_csv_row",
            "eligible_min_views": 4,
        },
        "array_sha256": {
            "labels.npy": _sha256(cache / "labels.npy"),
            "shuffled_indices.npy": _sha256(cache / "shuffled_indices.npy"),
        },
    }
    (cache / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return dataset_root, cache


def test_cache_and_dataset_keep_aligned_and_wrong_place_matched(
    tmp_path: Path,
) -> None:
    dataset_root, cache = _write_layout_cache(tmp_path)
    manifest, arrays, hashes = validate_semantic_layout_cache(cache)
    assert manifest["num_images"] == 8
    assert arrays["labels"].shape == (8, 70, 70)
    assert set(hashes) == {"labels.npy", "shuffled_indices.npy"}

    aligned = SemanticLayoutPlaceDataset(
        dataset_root,
        cache,
        views_per_place=4,
        mode="aligned",
        split="all",
        random_sample=False,
        verify_cache_hashes=False,
    )
    shuffled = SemanticLayoutPlaceDataset(
        dataset_root,
        cache,
        views_per_place=4,
        mode="shuffled",
        split="all",
        random_sample=False,
        verify_cache_hashes=False,
    )
    aligned_layouts, aligned_labels, aligned_meta = aligned[0]
    shuffled_layouts, shuffled_labels, shuffled_meta = shuffled[0]
    assert torch.equal(aligned_labels, shuffled_labels)
    assert torch.equal(aligned_meta["receiver_cache_indices"], shuffled_meta["receiver_cache_indices"])
    assert not torch.equal(aligned_layouts, shuffled_layouts)
    assert torch.equal(shuffled_meta["source_cache_indices"], torch.arange(4, 8))
    aligned.close()
    shuffled.close()


def test_place_hash_split_is_stable_and_has_no_overlap() -> None:
    buckets = {
        place_split_remainder(
            "City", place_id, seed=42, modulus=10
        )
        for place_id in range(200)
    }
    assert buckets == set(range(10))
    first = place_split_remainder("City", 123, seed=42, modulus=10)
    assert first == place_split_remainder("City", 123, seed=42, modulus=10)


def _retrieval(hits: list[bool], ranks: list[int]) -> RetrievalResult:
    count = len(hits)
    hit_array = np.asarray(hits, dtype=bool)
    return RetrievalResult(
        top1=np.arange(count, dtype=np.int64),
        hits_at_1=hit_array,
        hits_at_5=hit_array.copy(),
        hits_at_10=hit_array.copy(),
        first_positive_rank=np.asarray(ranks, dtype=np.int64),
        positive_negative_margin=np.where(hit_array, 0.2, -0.2).astype(np.float32),
    )


def test_complementarity_gate_requires_both_negative_controls() -> None:
    ru = _retrieval(
        [True, True, False, False, True, False, True, True, False, False],
        [1, 1, 4, 5, 1, 3, 1, 1, 7, 6],
    )
    aligned = _retrieval(
        [True, True, True, True, True, False, True, True, False, False],
        [1, 1, 1, 1, 1, 2, 1, 1, 5, 4],
    )
    wrong = _retrieval(
        [True, True, False, False, True, False, True, True, False, False],
        [1, 1, 5, 6, 1, 4, 1, 1, 8, 7],
    )
    shuffled_teacher = wrong
    audit = build_audit(
        ru=ru,
        aligned=aligned,
        wrong_layout=wrong,
        shuffled_teacher=shuffled_teacher,
        min_semantic_only=2,
        min_teacher_better_rate=0.05,
        expected_ru_correct=5,
        aligned_holdout_batch_r1=0.8,
        shuffled_holdout_batch_r1=0.1,
    )
    assert audit["verdict"] == "PASS"
    assert audit["paired"]["aligned_vs_ru"]["left_only_correct@1"] == 2

    control_too_good = build_audit(
        ru=ru,
        aligned=aligned,
        wrong_layout=aligned,
        shuffled_teacher=shuffled_teacher,
        min_semantic_only=2,
        min_teacher_better_rate=0.05,
        expected_ru_correct=5,
        aligned_holdout_batch_r1=0.8,
        shuffled_holdout_batch_r1=0.1,
    )
    assert control_too_good["verdict"] == "FAIL"


def test_exact_cosine_search_reports_positive_rank() -> None:
    descriptors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.9, 0.1],
            [-0.9, 0.1],
        ],
        dtype=np.float32,
    )
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    result = search_descriptors(
        descriptors,
        num_references=3,
        positives=(np.asarray([0]), np.asarray([2])),
        query_chunk_size=1,
    )
    assert result.hits_at_1.tolist() == [True, True]
    assert result.first_positive_rank.tolist() == [1, 1]
    assert bool(np.all(result.positive_negative_margin > 0))


def test_collapsed_descriptor_ties_do_not_fake_positive_rank_one() -> None:
    descriptors = np.ones((5, 2), dtype=np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    result = search_descriptors(
        descriptors,
        num_references=3,
        positives=(np.asarray([2]), np.asarray([1])),
    )
    assert result.top1.tolist() == [0, 0]
    assert result.hits_at_1.tolist() == [False, False]
    assert result.first_positive_rank.tolist() == [3, 2]
