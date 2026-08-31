from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_sweep_module():
    """Load the pure sweep logic without importing models, CUDA, or FAISS."""

    placeholder = type("Placeholder", (), {})
    audit_stub = _module(
        "scripts.audit_residual_clip_paired",
        DescriptorStore=placeholder,
        PairedImageDataset=placeholder,
        _canonical_paths=lambda values: np.asarray(
            [str(value).replace("\\", "/") for value in values],
            dtype=np.str_,
        ),
        _sha256_file=lambda _path: "stub-sha256",
        aggregate_feature_map=lambda _model, value: value,
        search_and_score=lambda *_args, **_kwargs: None,
        validate_aligned_checkpoint=lambda *_args, **_kwargs: {},
        verify_first_batch_equivalence=lambda *_args, **_kwargs: 0.0,
        verify_frozen_base=lambda *_args, **_kwargs: {},
        write_csv=lambda *_args, **_kwargs: None,
    )
    condition_stub = _module(
        "scripts.eval_condition_robustness",
        build_transform=lambda *_args, **_kwargs: None,
        choose_device=lambda *_args, **_kwargs: None,
        load_inference_model_from_ckpt=lambda *_args, **_kwargs: None,
    )
    dynamic_prior_stub = _module(
        "scripts.dynamic_category_prior",
        role_preserving_derangement=lambda *_args, **_kwargs: None,
    )
    dataset_stub = _module(
        "src.dataloaders.valid.mapillary_sls",
        MapillarySLSDataset=placeholder,
    )
    boq_stub = _module("src.models.aggregators.boq", BoQ=placeholder)
    torch_stub = _module(
        "torch",
        Tensor=object,
        nn=SimpleNamespace(Module=object),
        device=object,
    )
    torch_utils_stub = _module("torch.utils")
    torch_data_stub = _module("torch.utils.data", DataLoader=placeholder)
    tqdm_stub = _module("tqdm", tqdm=lambda iterable, **_kwargs: iterable)

    module_name = "_residual_clip_semantic_scale_sweep_under_test"
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sweep_residual_clip_semantic_scale.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    dependencies = {
        "scripts.audit_residual_clip_paired": audit_stub,
        "scripts.eval_condition_robustness": condition_stub,
        "scripts.dynamic_category_prior": dynamic_prior_stub,
        "src.dataloaders.valid.mapillary_sls": dataset_stub,
        "src.models.aggregators.boq": boq_stub,
        "torch": torch_stub,
        "torch.utils": torch_utils_stub,
        "torch.utils.data": torch_data_stub,
        "tqdm": tqdm_stub,
    }
    try:
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


SWEEP = _load_sweep_module()


def _validation_args(
    tmp_path: Path, semantic_gammas: list[float]
) -> argparse.Namespace:
    checkpoint = tmp_path / "aligned.ckpt"
    ru_checkpoint = tmp_path / "ru.ckpt"
    msls_path = tmp_path / "msls"
    checkpoint.write_bytes(b"aligned")
    ru_checkpoint.write_bytes(b"ru")
    msls_path.mkdir(exist_ok=True)
    return argparse.Namespace(
        checkpoint=checkpoint,
        ru_checkpoint=ru_checkpoint,
        msls_path=msls_path,
        output=tmp_path / "new-output",
        scratch_dir=None,
        keep_descriptors=False,
        device="cpu",
        batch_size=2,
        num_workers=0,
        image_size=(280, 280),
        seed=42,
        semantic_gammas=semantic_gammas,
        descriptor_dtype="float32",
        k_values=(1, 5),
        rank_k=10,
        minimum_net_queries=4,
        expected_bypass_r1=91.22,
        baseline_tolerance_pp=0.15,
        equivalence_tolerance=1e-5,
        no_amp=True,
    )


def test_cli_defaults_build_the_exploratory_eighteen_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_residual_clip_semantic_scale.py",
            "--checkpoint",
            "aligned.ckpt",
            "--ru-checkpoint",
            "ru.ckpt",
            "--output",
            "audit-output",
        ],
    )
    args = SWEEP.parse_args()
    assert tuple(args.semantic_gammas) == (0.0, 0.5, 1.0, 2.0, 4.0)
    assert args.minimum_net_queries == 8
    assert args.descriptor_dtype == "float32"

    specs = SWEEP.build_variant_specs(args.semantic_gammas)
    expected_names = ["bypass", "zero_clip_g0"]
    for label in ("0p5", "1", "2", "4"):
        expected_names.extend(
            f"{mode}_g{label}" for mode in SWEEP.SEMANTIC_MODES
        )
    assert len(specs) == 18
    assert [spec.name for spec in specs] == expected_names
    assert len({spec.name for spec in specs}) == 18


@pytest.mark.parametrize(
    "gammas",
    (
        [0.0, 0.5],
        [0.5, 1.0],
        [],
    ),
)
def test_variant_specs_require_exact_gamma_zero_and_one(
    gammas: list[float],
) -> None:
    with pytest.raises(ValueError, match="include exact endpoints 0 and 1"):
        SWEEP.build_variant_specs(gammas)


@pytest.mark.parametrize(
    "gammas, message",
    (
        ([-0.01, 0.0, 1.0], r"lie in \[0,8\]"),
        ([0.0, 1.0, 8.01], r"lie in \[0,8\]"),
        ([0.0, 1.0, float("nan")], "finite real numbers"),
        ([0.0, 1.0, float("inf")], "finite real numbers"),
        ([0.0, 0.5], "include exact endpoints 0 and 1"),
        ([0.5, 1.0], "include exact endpoints 0 and 1"),
    ),
)
def test_cli_validation_rejects_invalid_semantic_gamma_protocol(
    tmp_path: Path, gammas: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SWEEP.validate_args(_validation_args(tmp_path, gammas))


def test_cli_validation_canonicalises_duplicate_scales(tmp_path: Path) -> None:
    args = _validation_args(tmp_path, [4.0, 1.0, 0.5, 0.0, 2.0, 1.0])
    assert SWEEP.validate_args(args) == (0.0, 0.5, 1.0, 2.0, 4.0)


def test_cli_validation_requires_r5_for_deterministic_tie_break(
    tmp_path: Path,
) -> None:
    args = _validation_args(tmp_path, [0.0, 1.0])
    args.k_values = (1,)

    with pytest.raises(ValueError, match="must include 5"):
        SWEEP.validate_args(args)


def _retrieval_result(correct_at_one: int, num_queries: int) -> dict:
    hits_at_one = np.arange(num_queries) < correct_at_one
    hits_at_five = np.arange(num_queries) < min(num_queries, correct_at_one + 1)
    first_rank = np.where(hits_at_one, 1, 11).astype(np.int64)
    return {
        "recalls": {
            1: float(hits_at_one.mean()),
            5: float(hits_at_five.mean()),
        },
        "hits": {1: hits_at_one, 5: hits_at_five},
        "top1": np.zeros(num_queries, dtype=np.int64),
        "top1_distance": np.zeros(num_queries, dtype=np.float32),
        "first_positive_rank": first_rank,
        "positive_found_within_rank_k": hits_at_one.copy(),
        "positive_negative_margin": np.where(
            hits_at_one, 1.0, -1.0
        ).astype(np.float32),
    }


def test_evaluate_pairs_every_control_at_the_same_gamma_and_applies_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = SWEEP.build_variant_specs((0.0, 0.5, 1.0, 2.0, 4.0))
    paths = {spec.name: tmp_path / f"{spec.name}.npy" for spec in specs}
    names_by_path = {str(path): name for name, path in paths.items()}
    specs_by_name = {spec.name: spec for spec in specs}
    aligned_correct = {0.5: 5, 1.0: 6, 2.0: 7, 4.0: 6}
    num_queries = 10

    def fake_search(descriptor_path: Path, **_kwargs) -> dict:
        spec = specs_by_name[names_by_path[str(descriptor_path)]]
        correct = (
            aligned_correct[float(spec.semantic_gamma)]
            if spec.mode == "aligned"
            else 2
        )
        return _retrieval_result(correct, num_queries)

    monkeypatch.setattr(SWEEP, "search_and_score", fake_search)
    dataset = SimpleNamespace(
        num_references=3,
        num_queries=num_queries,
        ground_truth=tuple(np.asarray([0]) for _ in range(num_queries)),
        dbImages=np.asarray([f"cph/database/r{i}.jpg" for i in range(3)]),
        qImages=np.asarray([f"sf/query/q{i}.jpg" for i in range(num_queries)]),
    )
    store = SimpleNamespace(paths=paths)

    summary, paired, selection, query_rows, results = SWEEP.evaluate(
        store,
        dataset,
        specs,
        k_values=(1, 5),
        rank_k=10,
        minimum_net_queries=4,
    )

    assert len(summary) == 18
    assert len(results) == 18
    assert len(query_rows) == 18 * num_queries
    assert len(paired) == 4 * len(SWEEP.COMPARATOR_MODES)
    selection_by_gamma = {
        float(row["semantic_gamma"]): row for row in selection
    }
    assert {
        gamma: int(row["minimum_net_queries"])
        for gamma, row in selection_by_gamma.items()
    } == {0.5: 3, 1.0: 4, 2.0: 5, 4.0: 4}
    assert {
        gamma: int(row["eligible"])
        for gamma, row in selection_by_gamma.items()
    } == {0.5: 0, 1.0: 1, 2.0: 1, 4.0: 1}

    for gamma in (0.5, 1.0, 2.0, 4.0):
        rows = [row for row in paired if row["semantic_gamma"] == gamma]
        assert {row["right_mode"] for row in rows} == set(
            SWEEP.COMPARATOR_MODES
        )
        for row in rows:
            left_spec = specs_by_name[row["left"]]
            assert left_spec.mode == "aligned"
            assert left_spec.semantic_gamma == gamma
            if row["right_mode"] in SWEEP.SEMANTIC_MODES:
                assert specs_by_name[row["right"]].semantic_gamma == gamma


def test_select_scale_orders_by_worst_net_then_recall_and_smaller_gamma() -> None:
    rows = [
        {
            "semantic_gamma": 0.5,
            "minimum_net_queries": 4,
            "aligned_r@1": 0.95,
            "aligned_r@5": 0.99,
            "eligible": 1,
        },
        {
            "semantic_gamma": 1.0,
            "minimum_net_queries": 5,
            "aligned_r@1": 0.90,
            "aligned_r@5": 0.99,
            "eligible": 1,
        },
        {
            "semantic_gamma": 2.0,
            "minimum_net_queries": 5,
            "aligned_r@1": 0.92,
            "aligned_r@5": 0.97,
            "eligible": 1,
        },
        {
            "semantic_gamma": 4.0,
            "minimum_net_queries": 5,
            "aligned_r@1": 0.92,
            "aligned_r@5": 0.98,
            "eligible": 1,
        },
    ]
    assert SWEEP.select_scale(rows) == (4.0, 4.0)

    rows[-1]["aligned_r@5"] = rows[-2]["aligned_r@5"]
    assert SWEEP.select_scale(rows) == (2.0, 2.0)


def test_select_scale_returns_no_winner_when_every_scale_misses_threshold() -> None:
    rows = [
        {
            "semantic_gamma": 0.5,
            "minimum_net_queries": 2,
            "aligned_r@1": 0.93,
            "aligned_r@5": 0.97,
            "eligible": 0,
        },
        {
            "semantic_gamma": 1.0,
            "minimum_net_queries": 3,
            "aligned_r@1": 0.92,
            "aligned_r@5": 0.98,
            "eligible": 0,
        },
    ]

    assert SWEEP.select_scale(rows) == (None, 1.0)
    with pytest.raises(ValueError, match="selection rows are empty"):
        SWEEP.select_scale([])
