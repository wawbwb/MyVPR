#!/usr/bin/env python
"""Train the place-supervised AG-SLRD semantic-layout teacher.

This is an intentionally standalone Phase-0 program.  It does not construct
the RGB student, read MSLS, or expose a relational-distillation loss.  The
aligned and shuffled-label configurations therefore test semantic-layout
sufficiency before any student implementation is authorised.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloaders.train.semantic_layout import (  # noqa: E402
    AG_SLRD_SPLIT_ALGORITHM,
    SemanticLayoutPlaceDataset,
)
from src.models.ag_slrd import (  # noqa: E402
    SemanticLayoutEncoder,
    build_teacher_checkpoint,
    file_sha256,
)
from src.semantic_layout_cache import (  # noqa: E402
    SEMANTIC_LAYOUT_CACHE_SCHEMA,
    SEMANTIC_LAYOUT_CACHE_VERSION,
    SEMANTIC_LAYOUT_CLASSES,
    SEMANTIC_LAYOUT_GRID_SIZE,
)


CONFIG_TOP_LEVEL = {"seed", "mode", "cache", "data", "model", "trainer"}
CACHE_KEYS = {
    "dir",
    "expected_schema",
    "expected_version",
    "expected_grid",
    "expected_num_classes",
}
DATA_KEYS = {
    "dataset_root",
    "cities",
    "views_per_place",
    "places_per_batch",
    "num_workers",
    "split_algorithm",
    "holdout_modulus",
    "holdout_remainder",
}
MODEL_KEYS = {
    "num_classes",
    "embed_dim",
    "channels",
    "descriptor_dim",
    "ignore_index",
}
TRAINER_KEYS = {
    "epochs",
    "optimizer",
    "lr",
    "weight_decay",
    "precision",
    "device",
    "output_dir",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _expect_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ from the frozen contract: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load a canonical teacher YAML and fail closed on protocol drift."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AG-SLRD teacher config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = _expect_mapping(config, name="config")
    _require_exact_keys(config, CONFIG_TOP_LEVEL, name="config")

    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 42:
        raise ValueError("AG-SLRD Phase 0 requires seed=42")
    mode = str(config["mode"]).lower()
    if mode not in {"aligned", "shuffled"}:
        raise ValueError("mode must be aligned or shuffled")
    config["mode"] = mode

    cache = _expect_mapping(config["cache"], name="cache")
    data = _expect_mapping(config["data"], name="data")
    model = _expect_mapping(config["model"], name="model")
    trainer = _expect_mapping(config["trainer"], name="trainer")
    _require_exact_keys(cache, CACHE_KEYS, name="cache")
    _require_exact_keys(data, DATA_KEYS, name="data")
    _require_exact_keys(model, MODEL_KEYS, name="model")
    _require_exact_keys(trainer, TRAINER_KEYS, name="trainer")

    if cache["expected_schema"] != SEMANTIC_LAYOUT_CACHE_SCHEMA:
        raise ValueError("cache.expected_schema differs from source contract")
    if cache["expected_version"] != SEMANTIC_LAYOUT_CACHE_VERSION:
        raise ValueError("cache.expected_version differs from source contract")
    if tuple(cache["expected_grid"]) != SEMANTIC_LAYOUT_GRID_SIZE:
        raise ValueError("cache.expected_grid differs from source contract")
    if cache["expected_num_classes"] != len(SEMANTIC_LAYOUT_CLASSES):
        raise ValueError("cache.expected_num_classes differs from source contract")

    if data["views_per_place"] != 4 or data["places_per_batch"] != 40:
        raise ValueError("canonical teacher training requires P=40, K=4")
    if data["split_algorithm"] != AG_SLRD_SPLIT_ALGORITHM:
        raise ValueError("unsupported place split algorithm")
    if data["holdout_modulus"] != 10 or data["holdout_remainder"] != 0:
        raise ValueError("canonical teacher training requires the frozen 90/10 split")
    if (
        isinstance(data["num_workers"], bool)
        or not isinstance(data["num_workers"], int)
        or data["num_workers"] < 0
    ):
        raise ValueError("data.num_workers must be a non-negative integer")

    if model != {
        "num_classes": 12,
        "embed_dim": 32,
        "channels": [64, 128, 256],
        "descriptor_dim": 512,
        "ignore_index": 255,
    }:
        raise ValueError("model configuration differs from the frozen Phase-0 model")
    if _positive_int(trainer["epochs"], name="trainer.epochs") != 10:
        raise ValueError("canonical teacher training requires exactly 10 epochs")
    if str(trainer["optimizer"]).lower() != "adamw":
        raise ValueError("canonical teacher optimizer must be AdamW")
    for field in ("lr", "weight_decay"):
        value = float(trainer[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"trainer.{field} must be finite and non-negative")
    if not math.isclose(float(trainer["lr"]), 1e-3, abs_tol=1e-15):
        raise ValueError("canonical teacher lr must be 0.001")
    if not math.isclose(float(trainer["weight_decay"]), 1e-4, abs_tol=1e-15):
        raise ValueError("canonical teacher weight_decay must be 0.0001")
    if trainer["precision"] != "16-mixed":
        raise ValueError("canonical teacher precision must be 16-mixed")
    expected_suffix = f"ag_slrd_semantic_teacher/{mode}"
    output = str(trainer["output_dir"]).replace("\\", "/").rstrip("/")
    if not output.endswith(expected_suffix):
        raise ValueError(
            f"trainer.output_dir must end with {expected_suffix!r}"
        )

    config["cache"] = cache
    config["data"] = data
    config["model"] = model
    config["trainer"] = trainer
    config["_config_path"] = str(path)
    return config


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("canonical 16-mixed teacher training requires CUDA")
    if device.index is None:
        raise ValueError("specify an explicit CUDA index; use cuda:1")
    if device.index == 0:
        raise ValueError("GPU 0 is faulty on the training machine; use cuda:1")
    if device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device} is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _batch_recall_one(
    descriptors: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    descriptors = torch.nn.functional.normalize(descriptors.float(), dim=1)
    similarities = descriptors @ descriptors.T
    similarities.fill_diagonal_(-torch.inf)
    nearest = similarities.argmax(dim=1)
    return (labels[nearest] == labels).float().mean()


def _flatten_batch(
    batch: tuple[torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    layouts, place_labels, _ = batch
    if layouts.ndim != 4 or place_labels.ndim != 2:
        raise ValueError("teacher batch must have layouts (P,K,H,W) and labels (P,K)")
    if layouts.shape[:2] != place_labels.shape:
        raise ValueError("teacher layout and place-label batch shapes disagree")
    return (
        layouts.flatten(0, 1).to(device=device, non_blocking=True),
        place_labels.flatten().to(device=device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_holdout(
    model: SemanticLayoutEncoder,
    loader: DataLoader,
    loss_function: VPRLossFunction,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    recall_sum = 0.0
    batches = 0
    amp_enabled = device.type == "cuda"
    for batch in tqdm(loader, desc="Holdout", leave=False):
        layouts, labels = _flatten_batch(batch, device)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            descriptors = model(layouts)
            loss, _ = loss_function(descriptors, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite semantic teacher holdout loss")
        loss_sum += float(loss.detach())
        recall_sum += float(_batch_recall_one(descriptors, labels))
        batches += 1
    if batches == 0:
        raise RuntimeError("semantic teacher holdout loader yielded no batches")
    return {
        "holdout_loss": loss_sum / batches,
        "holdout_batch_r1": recall_sum / batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "run one train and one holdout batch without writing outputs; "
            "this is not a reportable experiment"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The metric-learning dependency is needed only for an actual training
    # run; config validation and ``--help`` remain lightweight.
    from src.losses.vpr_losses import VPRLossFunction

    config = load_and_validate_config(args.config)
    seed = int(config["seed"])
    mode = config["mode"]
    cache_cfg = config["cache"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    trainer_cfg = config["trainer"]
    device = choose_device(str(trainer_cfg["device"]))
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    cache_dir = Path(cache_cfg["dir"]).expanduser().resolve()
    dataset_root = Path(data_cfg["dataset_root"]).expanduser().resolve()
    train_dataset = SemanticLayoutPlaceDataset(
        dataset_root,
        cache_dir,
        cities=data_cfg["cities"],
        views_per_place=int(data_cfg["views_per_place"]),
        mode=mode,
        split="train",
        split_algorithm=data_cfg["split_algorithm"],
        split_seed=seed,
        holdout_modulus=int(data_cfg["holdout_modulus"]),
        holdout_remainder=int(data_cfg["holdout_remainder"]),
        random_sample=True,
        verify_cache_hashes=True,
    )
    holdout_dataset = SemanticLayoutPlaceDataset(
        dataset_root,
        cache_dir,
        cities=data_cfg["cities"],
        views_per_place=int(data_cfg["views_per_place"]),
        mode=mode,
        split="holdout",
        split_algorithm=data_cfg["split_algorithm"],
        split_seed=seed,
        holdout_modulus=int(data_cfg["holdout_modulus"]),
        holdout_remainder=int(data_cfg["holdout_remainder"]),
        random_sample=False,
        verify_cache_hashes=False,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(data_cfg["places_per_batch"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": int(data_cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )

    # Dataset construction and cache validation do not consume torch RNG, but
    # reset explicitly so aligned/shuffled controls initialise identically.
    seed_everything(seed)
    model = SemanticLayoutEncoder(**model_cfg).to(device)
    loss_function = VPRLossFunction(
        loss_fn_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer_cfg["lr"]),
        weight_decay=float(trainer_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    output_dir = Path(trainer_cfg["output_dir"]).expanduser().resolve()
    if not args.smoke_test:
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite teacher run: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Device: {device}; mode: {mode}")
    print(
        f"Places: train={len(train_dataset)}, holdout={len(holdout_dataset)}; "
        f"P={data_cfg['places_per_batch']}, K={data_cfg['views_per_place']}"
    )
    print(
        f"Cache: {cache_dir}; labels={train_dataset.array_hashes['labels.npy'][:12]}..."
    )
    trainable = sum(parameter.numel() for parameter in model.parameters())
    print(f"Semantic-layout teacher parameters: {trainable:,}")

    history: list[dict[str, float | int]] = []
    global_step = 0
    epochs = 1 if args.smoke_test else int(trainer_cfg["epochs"])
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        recall_sum = 0.0
        batch_count = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            total=(1 if args.smoke_test else len(train_loader)),
        )
        for batch in progress:
            layouts, labels = _flatten_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                descriptors = model(layouts)
                loss, batch_accuracy = loss_function(descriptors, labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite semantic teacher training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite semantic teacher gradient norm")
            scaler.step(optimizer)
            scaler.update()
            batch_r1 = _batch_recall_one(descriptors.detach(), labels)
            loss_sum += float(loss.detach())
            recall_sum += float(batch_r1)
            batch_count += 1
            global_step += 1
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                r1=f"{float(batch_r1):.3f}",
                acc=f"{float(batch_accuracy):.3f}",
            )
            if args.smoke_test:
                break
        if batch_count == 0:
            raise RuntimeError("semantic teacher training loader yielded no batches")

        if args.smoke_test:
            # One deterministic holdout batch is enough to verify the complete
            # forward/loss path; no result is written or considered evidence.
            one_holdout = [next(iter(holdout_loader))]
            holdout_metrics = evaluate_holdout(
                model, one_holdout, loss_function, device
            )
        else:
            holdout_metrics = evaluate_holdout(
                model, holdout_loader, loss_function, device
            )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / batch_count,
            "train_batch_r1": recall_sum / batch_count,
            **holdout_metrics,
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}: train_loss={row['train_loss']:.4f}, "
            f"train_batch_R1={row['train_batch_r1']:.4f}, "
            f"holdout_loss={row['holdout_loss']:.4f}, "
            f"holdout_batch_R1={row['holdout_batch_r1']:.4f}"
        )

    if args.smoke_test:
        print("SMOKE TEST PASS (no checkpoint written)")
        return

    manifest_path = cache_dir / "manifest.json"
    cache_provenance = {
        "dir": str(cache_dir),
        "manifest_sha256": file_sha256(manifest_path),
        "schema": train_dataset.manifest["schema"],
        "version": train_dataset.manifest["version"],
        "grid_size": list(train_dataset.manifest["grid_size"]),
        "num_classes": train_dataset.manifest["num_classes"],
        "mapping_sha256": train_dataset.manifest["mapping"]["sha256"],
        "source_manifest_sha256": train_dataset.manifest[
            "source_manifest_sha256"
        ],
        "array_sha256": dict(train_dataset.array_hashes),
    }
    data_record = dict(data_cfg)
    data_record.update(
        {
            "dataset_root": str(dataset_root),
            "train_places": len(train_dataset),
            "holdout_places": len(holdout_dataset),
        }
    )
    trainer_record = dict(trainer_cfg)
    trainer_record["output_dir"] = str(output_dir)
    checkpoint = build_teacher_checkpoint(
        model,
        mode=mode,
        epoch=epochs - 1,
        global_step=global_step,
        cache_provenance=cache_provenance,
        data_config=data_record,
        trainer_config=trainer_record,
        optimizer_state=optimizer.state_dict(),
    )
    checkpoint_path = output_dir / "final.pt"
    temporary_checkpoint = output_dir / "final.pt.tmp"
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint_path)

    history_path = output_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    run_record = {
        "schema": "openvpr_ag_slrd_semantic_teacher_run",
        "version": 1,
        "complete": True,
        "created_utc": utc_now(),
        "mode": mode,
        "config_path": config["_config_path"],
        "seed": seed,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "cache": cache_provenance,
        "data": data_record,
        "model": model.export_config(),
        "trainer": trainer_record,
        "selection": "final epoch; MSLS was not read",
        "final_metrics": history[-1],
    }
    atomic_json(output_dir / "run.json", run_record)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        persisted_config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        yaml.safe_dump(
            persisted_config, handle, allow_unicode=True, sort_keys=False
        )
    print(f"Teacher checkpoint: {checkpoint_path}")
    print(f"Run record: {output_dir / 'run.json'}")


if __name__ == "__main__":
    main()
