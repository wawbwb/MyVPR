#!/usr/bin/env python
"""Train and contract-test the SemVPR-style CC-LSA local CLIP teacher."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cc_lsa_gate_a import (  # noqa: E402
    canonical_json_sha256,
    deterministic_permutation,
    file_sha256,
    implementation_sha256,
)
from src.dataloaders.train.cc_lsa import CCLSATargetDataset, load_target_manifest  # noqa: E402
from src.models.cc_lsa import (  # noqa: E402
    CC_LSA_TEACHER_SCHEMA,
    CC_LSA_TEACHER_VERSION,
    LocalSemanticAlignmentEncoder,
    lsa_cosine_loss,
    module_state_sha256,
    pool_region_grid,
    tensor_mapping_sha256,
)
from src.models.clip_teacher import CLIPTeacherEncoder  # noqa: E402


CONFIG_SCHEMA = "openvpr_cc_lsa_teacher_config"
CONFIG_VERSION = 1
RUN_SCHEMA = "openvpr_cc_lsa_teacher_run"
RUN_VERSION = 1
PRETRAINING_CONTRACT_SCHEMA = "openvpr_cc_lsa_pretraining_contract"
PRETRAINING_CONTRACT_VERSION = 1

REGISTERED_CONFIG: dict[str, Any] = {
    "schema": CONFIG_SCHEMA,
    "version": CONFIG_VERSION,
    "experiment": {"name": "CC_LSA_teacher", "seed": 42},
    "data": {
        "dataset_root": "datasets/gsv_cities",
        "target_cache": ".cache/cc_lsa/gsv_crop_cls_v1",
        "image_size": [280, 280],
        "region_grid": [2, 2],
    },
    "model": {
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "hf_mirror": "https://hf-mirror.com",
        "trainable_last_n_blocks": 2,
    },
    "training": {
        "output_dir": "logs/cc_lsa/teacher_seed42",
        "device": "cuda:1",
        "batch_size": 32,
        "num_workers": 8,
        "epochs": 5,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.01,
        "amp_dtype": "bfloat16",
    },
    "contract": {
        "raw_clip_r1_gain_pp": 5.0,
        "wrong_region_r1_gain_pp": 10.0,
        "token_permutation_r1_gain_pp": 10.0,
        "correct_wrong_cosine_margin": 0.05,
        "median_effective_rank": 16.0,
        "median_position_std": 0.02,
        "diagnostic_images": 512,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def exact_keys(value: Any, expected: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        found = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - found)}, "
            f"unexpected={sorted(found - expected)}"
        )
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    exact_keys(
        config,
        {"schema", "version", "experiment", "data", "model", "training", "contract"},
        context="teacher config",
    )
    if config["schema"] != CONFIG_SCHEMA or config["version"] != CONFIG_VERSION:
        raise ValueError("unsupported CC-LSA teacher config schema/version")
    exact_keys(config["experiment"], {"name", "seed"}, context="experiment")
    exact_keys(
        config["data"],
        {"dataset_root", "target_cache", "image_size", "region_grid"},
        context="data",
    )
    exact_keys(
        config["model"],
        {"model_name", "pretrained", "hf_mirror", "trainable_last_n_blocks"},
        context="model",
    )
    exact_keys(
        config["training"],
        {
            "output_dir",
            "device",
            "batch_size",
            "num_workers",
            "epochs",
            "learning_rate",
            "weight_decay",
            "amp_dtype",
        },
        context="training",
    )
    exact_keys(
        config["contract"],
        {
            "raw_clip_r1_gain_pp",
            "wrong_region_r1_gain_pp",
            "token_permutation_r1_gain_pp",
            "correct_wrong_cosine_margin",
            "median_effective_rank",
            "median_position_std",
            "diagnostic_images",
        },
        context="contract",
    )
    if config["experiment"]["seed"] != 42:
        raise ValueError("registered CC-LSA teacher seed is 42")
    if config["data"]["image_size"] != [280, 280]:
        raise ValueError("registered CC-LSA image size is 280x280")
    if config["data"]["region_grid"] != [2, 2]:
        raise ValueError("registered CC-LSA region grid is 2x2")
    if config["training"]["amp_dtype"] != "bfloat16":
        raise ValueError("registered CC-LSA AMP dtype is bfloat16")
    if config != REGISTERED_CONFIG:
        raise ValueError(
            "CC-LSA teacher config differs from the frozen pre-registration; "
            f"expected={canonical_json_sha256(REGISTERED_CONFIG)}, "
            f"found={canonical_json_sha256(config)}"
        )
    return config


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            raise ValueError("specify an explicit CUDA device, e.g. cuda:1")
        if device.index == 0:
            raise ValueError("GPU 0 is faulty on the training machine; use cuda:1")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} is unavailable")
    return device


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Avoid version-dependent nondeterministic fused attention backward.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def make_loader(
    dataset: CCLSATargetDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def train_epoch(
    model: LocalSemanticAlignmentEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    description: str,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    rows = 0
    progress = tqdm(loader, desc=description)
    for batch_index, (images, targets, _) in enumerate(progress):
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            local = model(images)
            loss = lsa_cosine_loss(local, targets, rows=2, columns=2)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("non-finite CC-LSA gradient norm")
        optimizer.step()
        count = len(images)
        total_loss += float(loss.detach()) * count
        rows += count
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if rows == 0:
        raise RuntimeError("CC-LSA training loader produced no rows")
    return total_loss / rows


def _alignment_correct(predicted: torch.Tensor, targets: torch.Tensor) -> int:
    similarity = torch.einsum("brd,bsd->brs", targets, predicted)
    expected = torch.arange(predicted.shape[1], device=predicted.device)
    return int((similarity.argmax(dim=-1) == expected).sum())


@torch.inference_mode()
def evaluate_contract(
    model: LocalSemanticAlignmentEncoder,
    raw_teacher: CLIPTeacherEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    diagnostic_images: int,
) -> dict[str, float]:
    model.eval()
    raw_teacher.eval()
    counts = {"lsa": 0, "raw": 0, "wrong": 0, "permuted": 0}
    region_total = 0
    cosine_correct = 0.0
    cosine_wrong = 0.0
    loss_total = 0.0
    image_total = 0
    position_stds: list[float] = []
    effective_ranks: list[float] = []
    for images, targets, global_indices in tqdm(loader, desc="LSA contract"):
        images = images.to(device=device, non_blocking=True)
        targets = F.normalize(
            targets.to(device=device, non_blocking=True).float(), dim=-1
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            local = model(images)
            _, raw_tokens = raw_teacher(images)
            raw_local = raw_teacher.project_patch_tokens(raw_tokens).float()
        local = F.normalize(local.float(), dim=-1)
        raw_local = F.normalize(raw_local.float(), dim=-1)
        predicted = pool_region_grid(local, rows=2, columns=2)
        raw_predicted = pool_region_grid(raw_local, rows=2, columns=2)
        opposite = torch.tensor([3, 2, 1, 0], device=device)
        wrong_predicted = predicted[:, opposite]
        permuted_maps = []
        for row, global_index in zip(local, global_indices.tolist()):
            permutation = deterministic_permutation(
                row.shape[0], image_key=int(global_index), seed=42
            )
            permuted_maps.append(row[torch.as_tensor(permutation, device=device)])
        permuted_predicted = pool_region_grid(
            torch.stack(permuted_maps), rows=2, columns=2
        )
        counts["lsa"] += _alignment_correct(predicted, targets)
        counts["raw"] += _alignment_correct(raw_predicted, targets)
        counts["wrong"] += _alignment_correct(wrong_predicted, targets)
        counts["permuted"] += _alignment_correct(permuted_predicted, targets)
        region_total += int(targets.shape[0] * targets.shape[1])
        cosine_correct += float((predicted * targets).sum())
        cosine_wrong += float((wrong_predicted * targets).sum())
        batch_loss = 1.0 - (predicted * targets).sum(dim=-1)
        loss_total += float(batch_loss.sum())
        image_total += len(images)

        remaining = diagnostic_images - len(position_stds)
        if remaining > 0:
            diagnostic = local[:remaining]
            std = diagnostic.std(dim=1, unbiased=False).mean(dim=1)
            position_stds.extend(float(value) for value in std.cpu())
            for row in diagnostic:
                centered = row - row.mean(dim=0, keepdim=True)
                gram = centered @ centered.T
                eigenvalues = torch.linalg.eigvalsh(gram.float()).clamp_min(0)
                numerator = eigenvalues.sum().square()
                denominator = eigenvalues.square().sum().clamp_min(1e-12)
                effective_ranks.append(float((numerator / denominator).cpu()))
    if image_total == 0 or region_total == 0:
        raise RuntimeError("CC-LSA holdout loader produced no rows")
    return {
        "holdout_loss": loss_total / region_total,
        "lsa_crop_region_r1": counts["lsa"] / region_total,
        "raw_clip_crop_region_r1": counts["raw"] / region_total,
        "wrong_region_crop_region_r1": counts["wrong"] / region_total,
        "token_permutation_crop_region_r1": counts["permuted"] / region_total,
        "correct_crop_cosine": cosine_correct / region_total,
        "wrong_region_cosine": cosine_wrong / region_total,
        "correct_wrong_cosine_margin": (cosine_correct - cosine_wrong) / region_total,
        "median_effective_rank": float(np.median(effective_ranks)),
        "median_position_std": float(np.median(position_stds)),
        "diagnostic_images": len(position_stds),
    }


def contract_verdict(metrics: dict[str, float], thresholds: dict[str, Any]) -> dict[str, Any]:
    lsa_r1 = metrics["lsa_crop_region_r1"]
    checks = {
        "beats_raw_clip": {
            "value_pp": 100 * (lsa_r1 - metrics["raw_clip_crop_region_r1"]),
            "threshold_pp": float(thresholds["raw_clip_r1_gain_pp"]),
        },
        "beats_wrong_region": {
            "value_pp": 100 * (lsa_r1 - metrics["wrong_region_crop_region_r1"]),
            "threshold_pp": float(thresholds["wrong_region_r1_gain_pp"]),
        },
        "beats_token_permutation": {
            "value_pp": 100 * (lsa_r1 - metrics["token_permutation_crop_region_r1"]),
            "threshold_pp": float(thresholds["token_permutation_r1_gain_pp"]),
        },
        "cosine_margin": {
            "value": metrics["correct_wrong_cosine_margin"],
            "threshold": float(thresholds["correct_wrong_cosine_margin"]),
        },
        "effective_rank": {
            "value": metrics["median_effective_rank"],
            "threshold": float(thresholds["median_effective_rank"]),
        },
        "position_std": {
            "value": metrics["median_position_std"],
            "threshold": float(thresholds["median_position_std"]),
        },
    }
    for record in checks.values():
        if "value_pp" in record:
            record["pass"] = record["value_pp"] >= record["threshold_pp"]
        else:
            record["pass"] = record["value"] >= record["threshold"]
    passed = all(bool(record["pass"]) for record in checks.values())
    return {"verdict": "PASS" if passed else "FAIL", "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = choose_device(str(config["training"]["device"]))
    dataset_root = (PROJECT_ROOT / config["data"]["dataset_root"]).resolve()
    target_cache = (PROJECT_ROOT / config["data"]["target_cache"]).resolve()
    output_dir = (PROJECT_ROOT / config["training"]["output_dir"]).resolve()
    if output_dir.exists() and not args.smoke_test:
        raise FileExistsError(f"refusing to overwrite teacher output: {output_dir}")
    target_manifest = load_target_manifest(
        target_cache, dataset_root=dataset_root
    )
    train_dataset = CCLSATargetDataset(dataset_root, target_cache, split="train")
    holdout_dataset = CCLSATargetDataset(dataset_root, target_cache, split="holdout")
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        device=device,
    )
    holdout_loader = make_loader(
        holdout_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
        device=device,
    )
    model_cfg = config["model"]
    model = LocalSemanticAlignmentEncoder(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        hf_mirror=model_cfg["hf_mirror"],
        trainable_last_n_blocks=int(model_cfg["trainable_last_n_blocks"]),
    ).to(device)
    base_hash = module_state_sha256(model.visual)
    if target_manifest["teacher"]["base_model_state_sha256"] != base_hash:
        raise ValueError("crop target base CLIP differs from LSA initialisation")
    if model.native_grid_size != (14, 14) or model.semantic_dim != 512:
        raise ValueError("registered LSA teacher must expose a 14x14x512 local map")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = 1 if args.smoke_test else int(config["training"]["epochs"])
    pretraining_contract_path: Path | None = None
    if not args.smoke_test:
        output_dir.mkdir(parents=True, exist_ok=False)
        pretraining_contract_path = output_dir / "pretraining_contract.json"
        initial_delta = model.trainable_state_dict()
        pretraining_contract = {
            "schema": PRETRAINING_CONTRACT_SCHEMA,
            "version": PRETRAINING_CONTRACT_VERSION,
            "locked_before_first_optimizer_step": True,
            "created_utc": utc_now(),
            "config": config,
            "config_file_sha256": file_sha256(config_path),
            "config_canonical_sha256": canonical_json_sha256(config),
            "registered_config_canonical_sha256": canonical_json_sha256(
                REGISTERED_CONFIG
            ),
            "target_manifest_sha256": file_sha256(target_cache / "manifest.json"),
            "target_path_sequence_sha256": target_manifest["path_sequence_sha256"],
            "num_train_places": len(train_dataset),
            "num_holdout_places": len(holdout_dataset),
            "base_model_state_sha256": base_hash,
            "initial_trainable_state_sha256": tensor_mapping_sha256(initial_delta),
            "trainable_parameter_names": list(model.trainable_parameter_names()),
            "implementation_sha256": implementation_sha256(),
            "numeric_protocol": {
                "amp_dtype": "bfloat16",
                "gradient_scaler": False,
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "gradient_clip_norm": 1.0,
                "attention_backend": "math_sdpa",
            },
        }
        atomic_json(pretraining_contract_path, pretraining_contract)
        print(f"Locked pre-training contract: {pretraining_contract_path}")
    history = []
    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            description=f"CC-LSA epoch {epoch + 1}/{epochs}",
            max_batches=1 if args.smoke_test else None,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss})
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}")
    if args.smoke_test:
        print("SMOKE TEST PASS (one finite update; no checkpoint written)")
        return

    raw_teacher = CLIPTeacherEncoder(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        hf_mirror=model_cfg["hf_mirror"],
    ).to(device)
    metrics = evaluate_contract(
        model,
        raw_teacher,
        holdout_loader,
        device=device,
        diagnostic_images=int(config["contract"]["diagnostic_images"]),
    )
    teacher_contract = contract_verdict(metrics, config["contract"])
    teacher_contract.update(
        {
            "metrics": metrics,
            "thresholds": config["contract"],
            "target_manifest_sha256": file_sha256(target_cache / "manifest.json"),
            "config_sha256": file_sha256(config_path),
        }
    )
    checkpoint_path = output_dir / "final.pt"
    trainable_state = model.trainable_state_dict()
    if pretraining_contract_path is None:
        raise RuntimeError("full training has no locked pre-training contract")
    pretraining_contract_sha256 = file_sha256(pretraining_contract_path)
    checkpoint = {
        "schema": CC_LSA_TEACHER_SCHEMA,
        "version": CC_LSA_TEACHER_VERSION,
        "created_utc": utc_now(),
        "model": model_cfg,
        "base_model_state_sha256": base_hash,
        "trainable_parameter_names": list(model.trainable_parameter_names()),
        "trainable_state_dict": trainable_state,
        "trainable_state_sha256": tensor_mapping_sha256(trainable_state),
        "teacher_contract": teacher_contract,
        "pretraining_contract_sha256": pretraining_contract_sha256,
        "target_cache": {
            "path": str(target_cache),
            "manifest_sha256": file_sha256(target_cache / "manifest.json"),
        },
        "config": config,
        "config_canonical_sha256": canonical_json_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "history": history,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    run = {
        "schema": RUN_SCHEMA,
        "version": RUN_VERSION,
        "complete": True,
        "created_utc": utc_now(),
        "verdict": teacher_contract["verdict"],
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": file_sha256(checkpoint_path),
        },
        "teacher_contract": teacher_contract,
        "pretraining_contract": {
            "path": pretraining_contract_path.name,
            "sha256": pretraining_contract_sha256,
        },
        "history": history,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }
    atomic_json(output_dir / "lsa_teacher_contract.json", teacher_contract)
    atomic_json(output_dir / "run.json", run)
    print(json.dumps(teacher_contract, ensure_ascii=False, indent=2))
    print(f"LSA teacher contract: {teacher_contract['verdict']}")
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
