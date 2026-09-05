#!/usr/bin/env python
"""Calibrate CC-LSA semantic Q95 on fixed GSV same-city RU-hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cc_lsa_config import load_gate_a_config  # noqa: E402
from src.cc_lsa_features import (  # noqa: E402
    mutual_nearest_edges_batch,
    CC_LSA_DINO_FEATURE_STAGE,
    extract_ru_descriptor_and_local,
    normalise_local_tokens,
)
from src.cc_lsa_gate_a import (  # noqa: E402
    implementation_sha256,
    CC_LSA_CALIBRATION_SCHEMA,
    CC_LSA_VERSION,
    file_sha256,
)
from src.dataloaders.train.cc_lsa import CCLSATargetDataset, load_target_manifest  # noqa: E402
from src.models.cc_lsa import load_lsa_checkpoint, module_state_sha256  # noqa: E402
from src.models.clip_teacher import CLIPTeacherEncoder  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/gsv_cities"))
    parser.add_argument("--ru-checkpoint", type=Path, required=True)
    parser.add_argument("--lsa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def _select_calibration_indices(
    cache_dir: Path, *, maximum_per_city: int, seed: int
) -> np.ndarray:
    cities = np.load(cache_dir / "city.npy", allow_pickle=False)
    paths = np.load(cache_dir / "paths.npy", allow_pickle=False)
    split = np.load(cache_dir / "split.npy", allow_pickle=False)
    groups: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(split == 1).tolist():
        groups[str(cities[index])].append(int(index))
    selected: list[int] = []
    for city, indices in sorted(groups.items()):
        indices.sort(
            key=lambda index: hashlib.sha256(
                f"cc_lsa_calibration_v1\0{seed}\0{paths[index]}".encode()
            ).digest()
        )
        chosen = indices[:maximum_per_city]
        if len(chosen) < 2:
            raise ValueError(f"GSV calibration city {city!r} has fewer than two holdout places")
        selected.extend(chosen)
    return np.asarray(sorted(selected), dtype=np.int64)


def _normalise(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if not bool(np.isfinite(array).all()) or bool(np.any(norm <= 1e-12)):
        raise RuntimeError("calibration feature is non-finite or zero norm")
    return array / norm


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    config_path = args.config.expanduser().resolve()
    config = load_gate_a_config(config_path)
    dataset_root = args.dataset_root.expanduser().resolve()
    ru_checkpoint = args.ru_checkpoint.expanduser().resolve()
    lsa_checkpoint_path = args.lsa_checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    scratch = args.scratch_dir.expanduser().resolve()
    if output.exists() or scratch.exists():
        raise FileExistsError("calibration output and scratch directories must not exist")
    if not ru_checkpoint.is_file() or not lsa_checkpoint_path.is_file():
        raise FileNotFoundError("RU or LSA checkpoint is missing")
    device = choose_device(args.device)
    target_cache = (PROJECT_ROOT / config["calibration"]["gsv_target_cache"]).resolve()
    target_manifest = load_target_manifest(target_cache)
    selected = _select_calibration_indices(
        target_cache,
        maximum_per_city=int(config["calibration"]["max_holdout_images_per_city"]),
        seed=int(config["seed"]),
    )
    all_dataset = CCLSATargetDataset(dataset_root, target_cache, split="all")
    loader = DataLoader(
        Subset(all_dataset, selected.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt

    ru_model = load_inference_model_from_ckpt(ru_checkpoint, device)
    lsa_model, lsa_checkpoint = load_lsa_checkpoint(
        lsa_checkpoint_path, map_location=device, require_contract_pass=True
    )
    if lsa_checkpoint["target_cache"]["manifest_sha256"] != file_sha256(target_cache / "manifest.json"):
        raise ValueError("calibration target-cache split differs from the LSA training cache")
    model_cfg = lsa_checkpoint["model"]
    raw_teacher = CLIPTeacherEncoder(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        hf_mirror=model_cfg.get("hf_mirror"),
    ).to(device)
    raw_hash = module_state_sha256(raw_teacher.visual)
    if raw_hash != lsa_checkpoint["base_model_state_sha256"]:
        raise ValueError("raw CLIP base differs from the LSA checkpoint base")
    if raw_hash != target_manifest["teacher"]["base_model_state_sha256"]:
        raise ValueError("GSV crop targets were produced by a different CLIP base")

    scratch.mkdir(parents=True)
    np.save(scratch / "selected_indices.npy", selected, allow_pickle=False)
    arrays: dict[str, np.memmap] | None = None
    next_row = 0
    with torch.inference_mode():
        for images, _, global_indices in tqdm(loader, desc="Extract GSV calibration features"):
            expected_global = selected[next_row : next_row + len(images)]
            if not np.array_equal(np.asarray(global_indices), expected_global):
                raise RuntimeError("GSV calibration DataLoader changed selected order")
            images = images.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                descriptors, dino_local = extract_ru_descriptor_and_local(ru_model, images)
                lsa_local = lsa_model(images)
                _, raw_tokens = raw_teacher(images)
                raw_local = normalise_local_tokens(
                    raw_teacher.project_patch_tokens(raw_tokens)
                )
            values = {
                "ru_descriptors": descriptors.float().cpu().numpy(),
                "dino_local": dino_local.float().cpu().numpy(),
                "lsa_local": lsa_local.float().cpu().numpy(),
                "raw_clip_local": raw_local.float().cpu().numpy(),
            }
            if arrays is None:
                arrays = {
                    name: np.lib.format.open_memmap(
                        scratch / f"{name}.npy",
                        mode="w+",
                        dtype="float16",
                        shape=(len(selected), *value.shape[1:]),
                    )
                    for name, value in values.items()
                }
            rows = slice(next_row, next_row + len(images))
            for name, value in values.items():
                if not bool(np.isfinite(value).all()):
                    raise RuntimeError(f"non-finite calibration feature: {name}")
                arrays[name][rows] = value.astype(np.float16)
            next_row += len(images)
    if arrays is None or next_row != len(selected):
        raise RuntimeError(f"calibration extraction stopped at {next_row}/{len(selected)}")
    for array in arrays.values():
        array.flush()
    del arrays

    descriptors = _normalise(np.load(scratch / "ru_descriptors.npy", mmap_mode="r"))
    dino = np.load(scratch / "dino_local.npy", mmap_mode="r")
    lsa = np.load(scratch / "lsa_local.npy", mmap_mode="r")
    raw = np.load(scratch / "raw_clip_local.npy", mmap_mode="r")
    city_values = np.load(target_cache / "city.npy", allow_pickle=False)[selected]
    hard_pairs = np.empty((len(selected), 2), dtype=np.int64)
    for city in sorted(set(str(value) for value in city_values.tolist())):
        rows = np.flatnonzero(city_values.astype(str) == city)
        city_scores = descriptors[rows] @ descriptors[rows].T
        np.fill_diagonal(city_scores, -np.inf)
        global_keys = selected[rows]
        for local_row, row_index in enumerate(rows.tolist()):
            order = np.lexsort((global_keys, -city_scores[local_row].astype(np.float64)))
            donor_row = int(rows[int(order[0])])
            hard_pairs[row_index] = (selected[row_index], selected[donor_row])

    lsa_edges: list[np.ndarray] = []
    raw_edges: list[np.ndarray] = []
    selected_to_row = {int(value): index for index, value in enumerate(selected.tolist())}
    for receiver_global, donor_global in tqdm(hard_pairs, desc="Calibrate semantic Q95"):
        receiver = selected_to_row[int(receiver_global)]
        donor = selected_to_row[int(donor_global)]
        left, right, _ = mutual_nearest_edges_batch(
            dino[receiver], np.asarray(dino[donor])[None], device=device
        )[0]
        lsa_q = _normalise(lsa[receiver])
        lsa_c = _normalise(lsa[donor])
        raw_q = _normalise(raw[receiver])
        raw_c = _normalise(raw[donor])
        lsa_edges.append(np.einsum("nd,nd->n", lsa_q[left], lsa_c[right]))
        raw_edges.append(np.einsum("nd,nd->n", raw_q[left], raw_c[right]))
    lsa_cosine = np.concatenate(lsa_edges).astype(np.float32)
    raw_cosine = np.concatenate(raw_edges).astype(np.float32)
    quantile = float(config["calibration"]["semantic_quantile"])
    tau_lsa = float(np.quantile(lsa_cosine, quantile, method="higher"))
    tau_raw = float(np.quantile(raw_cosine, quantile, method="higher"))
    if not (-1.0 < tau_lsa < 1.0 and -1.0 < tau_raw < 1.0):
        raise RuntimeError("calibrated semantic threshold is outside (-1,1)")

    output.mkdir(parents=True)
    np.save(output / "selected_indices.npy", selected, allow_pickle=False)
    np.save(output / "hard_pairs.npy", hard_pairs, allow_pickle=False)
    np.save(output / "lsa_edge_cosine.npy", lsa_cosine, allow_pickle=False)
    np.save(output / "raw_clip_edge_cosine.npy", raw_cosine, allow_pickle=False)
    array_files = (
        "selected_indices.npy",
        "hard_pairs.npy",
        "lsa_edge_cosine.npy",
        "raw_clip_edge_cosine.npy",
    )
    record = {
        "schema": CC_LSA_CALIBRATION_SCHEMA,
        "version": CC_LSA_VERSION,
        "complete": True,
        "created_utc": utc_now(),
        "source": "GSV fixed holdout only; no MSLS labels or scores",
        "implementation_sha256": implementation_sha256(),
        "matching_backend": "torch_fp32_no_tf32_argmax_first",
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "gsv_target_cache": {
            "path": str(target_cache),
            "manifest_sha256": file_sha256(target_cache / "manifest.json"),
        },
        "ru_checkpoint": {"path": str(ru_checkpoint), "sha256": file_sha256(ru_checkpoint)},
        "lsa_checkpoint": {
            "path": str(lsa_checkpoint_path),
            "sha256": file_sha256(lsa_checkpoint_path),
            "contract_verdict": lsa_checkpoint["teacher_contract"]["verdict"],
        },
        "raw_clip_base_model_state_sha256": raw_hash,
        "dino_feature_stage": CC_LSA_DINO_FEATURE_STAGE,
        "selected_images": len(selected),
        "directed_hard_pairs": len(hard_pairs),
        "semantic_quantile": quantile,
        "lsa_edge_count": len(lsa_cosine),
        "raw_clip_edge_count": len(raw_cosine),
        "tau_lsa": tau_lsa,
        "tau_raw_clip": tau_raw,
        "array_sha256": {filename: file_sha256(output / filename) for filename in array_files},
    }
    atomic_json(output / "calibration.json", record)
    print(f"tau_lsa={tau_lsa:.6f}; tau_raw_clip={tau_raw:.6f}")
    print(f"Calibration written to: {output}")


if __name__ == "__main__":
    main()
