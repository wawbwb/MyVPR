#!/usr/bin/env python
"""Cache the fixed RU candidates and local CC-LSA Gate-A MSLS features."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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

from scripts.audit_semantic_layout_complementarity import validate_descriptor_sidecar  # noqa: E402
from scripts.cache_msls_ag_slrd_layouts import load_msls_index  # noqa: E402
from src.cc_lsa_config import load_gate_a_config  # noqa: E402
from src.cc_lsa_features import (  # noqa: E402
    CC_LSA_DINO_FEATURE_STAGE,
    CC_LSA_LOCAL_GRID,
    extract_ru_descriptor_and_local,
    normalise_local_tokens,
)
from src.cc_lsa_gate_a import (  # noqa: E402
    implementation_sha256,
    CC_LSA_FEATURE_SCHEMA,
    CC_LSA_VERSION,
    build_same_city_wrong_place_donors,
    file_sha256,
    ru_candidate_diagnostics,
    stable_ru_topk,
)
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
    parser.add_argument("--ru-checkpoint", type=Path, required=True)
    parser.add_argument("--ru-descriptors", type=Path, required=True)
    parser.add_argument("--lsa-checkpoint", type=Path, required=True)
    parser.add_argument("--msls-path", type=Path, default=Path("datasets/msls-val"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--allow-failed-teacher-contract", action="store_true",
                        help="Exploratory only: retain the failed teacher contract and continue.")
    return parser.parse_args()


def _normalised(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    if not bool(np.isfinite(array).all()) or bool(np.any(norm <= 1e-12)):
        raise ValueError("RU descriptor matrix is non-finite or zero norm")
    return array / norm


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.flush_every) < 1 or args.num_workers < 0:
        raise ValueError("batch/flush must be positive and workers non-negative")
    config_path = args.config.expanduser().resolve()
    config = load_gate_a_config(config_path)
    device = choose_device(args.device)
    ru_checkpoint = args.ru_checkpoint.expanduser().resolve()
    ru_descriptors_path = args.ru_descriptors.expanduser().resolve()
    lsa_checkpoint_path = args.lsa_checkpoint.expanduser().resolve()
    msls_path = args.msls_path.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (ru_checkpoint, ru_descriptors_path, lsa_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    paths, num_references, num_queries, positives, index_record = load_msls_index(msls_path)
    thresholds = config["thresholds"]
    if num_queries != thresholds["expected_queries"]:
        raise ValueError(f"registered Gate A expects {thresholds['expected_queries']} queries")
    ru_sidecar = validate_descriptor_sidecar(ru_descriptors_path, expected_kind="ru")
    if ru_sidecar.get("msls_index") != index_record:
        raise ValueError(
            "RU descriptor sidecar has missing/different MSLS index hashes; "
            "re-extract into a NEW output using extract_ag_slrd_msls_descriptors.py ru"
        )
    if (
        ru_sidecar.get("image_size") != config["features"]["image_size"]
        or ru_sidecar.get("order") != "msls_standard_database_then_queries"
    ):
        raise ValueError("RU descriptor preprocessing or canonical order differs")
    sidecar_checkpoint = ru_sidecar.get("checkpoint")
    if (
        not isinstance(sidecar_checkpoint, dict)
        or sidecar_checkpoint.get("sha256") != file_sha256(ru_checkpoint)
    ):
        raise ValueError("RU descriptors were extracted from a different checkpoint")
    if (
        int(ru_sidecar["num_references"]) != num_references
        or int(ru_sidecar["num_queries"]) != num_queries
    ):
        raise ValueError("RU descriptor sidecar role counts differ from MSLS index")
    descriptors = np.load(ru_descriptors_path, mmap_mode="r", allow_pickle=False)
    if descriptors.shape[0] != len(paths) or descriptors.ndim != 2:
        raise ValueError("RU descriptor matrix does not match canonical MSLS rows")
    candidates, candidate_scores = stable_ru_topk(
        descriptors,
        num_references=num_references,
        top_k=int(config["score"]["top_k"]),
    )
    diagnostics = ru_candidate_diagnostics(candidates, positives)
    ru_correct = int(diagnostics["hit_at_1"].sum())
    if ru_correct != int(thresholds["expected_ru_correct"]):
        raise ValueError(
            f"RU top-1 reproduction failed: {ru_correct}/{num_queries}, "
            f"expected {thresholds['expected_ru_correct']}/{num_queries}"
        )
    wrong_donors = build_same_city_wrong_place_donors(
        paths.tolist(),
        num_references=num_references,
        positives=positives,
        seed=int(config["seed"]),
        candidates=candidates,
    )
    query_globals = num_references + np.arange(num_queries, dtype=np.int64)
    needed = np.unique(np.concatenate((query_globals, candidates.reshape(-1))))
    feature_indices = np.unique(np.concatenate((needed, wrong_donors[needed])))
    global_to_feature = np.full(len(paths), -1, dtype=np.int32)
    global_to_feature[feature_indices] = np.arange(len(feature_indices), dtype=np.int32)
    if bool(np.any(global_to_feature[needed] < 0)) or bool(
        np.any(global_to_feature[wrong_donors[needed]] < 0)
    ):
        raise RuntimeError("feature subset omitted a required receiver or donor")

    from scripts.eval_condition_robustness import (
        build_transform,
        load_inference_model_from_ckpt,
    )
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset

    ru_model = load_inference_model_from_ckpt(ru_checkpoint, device)
    lsa_model, lsa_checkpoint = load_lsa_checkpoint(
        lsa_checkpoint_path, map_location=device,
        require_contract_pass=not args.allow_failed_teacher_contract,
    )
    if tuple(lsa_model.native_grid_size) != CC_LSA_LOCAL_GRID:
        raise ValueError("LSA native grid differs from the registered 14x14 grid")
    model_cfg = lsa_checkpoint["model"]
    if lsa_checkpoint["teacher_contract"].get("verdict") not in {"PASS", "FAIL"}:
        raise ValueError("teacher contract verdict is missing or invalid")
    raw_teacher = CLIPTeacherEncoder(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        hf_mirror=model_cfg.get("hf_mirror"),
    ).to(device)
    raw_hash = module_state_sha256(raw_teacher.visual)
    if raw_hash != lsa_checkpoint["base_model_state_sha256"]:
        raise ValueError("raw CLIP base differs from LSA checkpoint base")
    local_count = int(CC_LSA_LOCAL_GRID[0] * CC_LSA_LOCAL_GRID[1])
    dino_dim = int(ru_model.backbone.out_channels)
    semantic_dim = int(lsa_model.semantic_dim)
    expected_bytes = len(feature_indices) * local_count * (
        dino_dim + 2 * semantic_dim
    ) * np.dtype("float16").itemsize
    parent = output.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    free_bytes = shutil.disk_usage(parent).free
    print(
        f"Feature subset: {len(feature_indices)}/{len(paths)} images; "
        f"estimated local arrays: {expected_bytes / 2**30:.2f} GiB; "
        f"free: {free_bytes / 2**30:.2f} GiB"
    )
    existing_bytes = sum(
        (output / filename).stat().st_size for filename in
        ("dino_local.npy", "lsa_local.npy", "raw_clip_local.npy")
        if (output / filename).is_file()
    )
    if free_bytes < max(0, expected_bytes - existing_bytes) * 1.1:
        raise RuntimeError("insufficient free space for CC-LSA feature cache")

    signature = {
        "exploratory_override": bool(args.allow_failed_teacher_contract),
        "teacher_contract": lsa_checkpoint["teacher_contract"],
        "schema": CC_LSA_FEATURE_SCHEMA,
        "version": CC_LSA_VERSION,
        "config_sha256": file_sha256(config_path),
        "ru_checkpoint_sha256": file_sha256(ru_checkpoint),
        "ru_descriptor_sha256": file_sha256(ru_descriptors_path),
        "lsa_checkpoint_sha256": file_sha256(lsa_checkpoint_path),
        "msls_index": index_record,
        "num_references": num_references,
        "num_queries": num_queries,
        "top_k": int(config["score"]["top_k"]),
        "feature_rows": len(feature_indices),
        "local_grid": list(CC_LSA_LOCAL_GRID),
        "dino_dim": dino_dim,
        "semantic_dim": semantic_dim,
        "dino_feature_stage": CC_LSA_DINO_FEATURE_STAGE,
        "dtype": "float16",
        "implementation_sha256": implementation_sha256(),
    }
    manifest_path = output / "manifest.json"
    progress_path = output / "progress.json"
    array_specs = {
        "dino_local.npy": (len(feature_indices), local_count, dino_dim),
        "lsa_local.npy": (len(feature_indices), local_count, semantic_dim),
        "raw_clip_local.npy": (len(feature_indices), local_count, semantic_dim),
    }
    if output.exists():
        if not manifest_path.is_file() or not progress_path.is_file():
            raise FileExistsError(f"existing output is not resumable: {output}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if {name: manifest.get(name) for name in signature} != signature:
            raise ValueError("existing CC-LSA feature-cache signature differs")
        for filename, expected in {
            "candidates.npy": candidates, "candidate_scores.npy": candidate_scores,
            "feature_indices.npy": feature_indices, "global_to_feature.npy": global_to_feature,
            "wrong_place_indices.npy": wrong_donors,
            "ru_best_positive_rank.npy": diagnostics["best_positive_rank"],
        }.items():
            actual = np.load(output / filename, allow_pickle=False)
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError(f"cached immutable index differs: {filename}")
        if manifest.get("complete") is True:
            for filename, digest in manifest.get("array_sha256", {}).items():
                if file_sha256(output / filename) != digest:
                    raise ValueError(f"complete cache SHA mismatch: {filename}")
            print(f"Complete CC-LSA feature cache already exists: {output}")
            return
        with progress_path.open("r", encoding="utf-8") as handle:
            progress_record = json.load(handle)
            next_row = int(progress_record["next_row"])
        manifest["max_descriptor_cosine_error"] = float(progress_record.get("max_descriptor_cosine_error", 0.0))
        for filename, expected_shape in array_specs.items():
            array = np.load(output / filename, mmap_mode="r+", allow_pickle=False)
            if array.shape != expected_shape or array.dtype != np.float16:
                raise ValueError(f"partial feature array differs: {filename}")
    else:
        output.mkdir(parents=True)
        manifest = dict(signature)
        manifest.update({"complete": False, "created_utc": utc_now()})
        atomic_json(manifest_path, manifest)
        np.save(output / "candidates.npy", candidates, allow_pickle=False)
        np.save(output / "candidate_scores.npy", candidate_scores, allow_pickle=False)
        np.save(output / "feature_indices.npy", feature_indices, allow_pickle=False)
        np.save(output / "global_to_feature.npy", global_to_feature, allow_pickle=False)
        np.save(output / "wrong_place_indices.npy", wrong_donors, allow_pickle=False)
        np.save(output / "ru_best_positive_rank.npy", diagnostics["best_positive_rank"], allow_pickle=False)
        for filename, shape in array_specs.items():
            np.lib.format.open_memmap(
                output / filename, mode="w+", dtype="float16", shape=shape
            ).flush()
        next_row = 0
        atomic_json(progress_path, {"next_row": 0, "updated_utc": utc_now()})
    if not 0 <= next_row <= len(feature_indices):
        raise ValueError("feature-cache resume cursor is outside its range")
    feature_arrays = {
        filename: np.load(output / filename, mmap_mode="r+", allow_pickle=False)
        for filename in array_specs
    }
    dataset = MapillarySLSDataset(
        dataset_path=msls_path,
        input_transform=build_transform(tuple(config["features"]["image_size"])),
    )
    loader = DataLoader(
        Subset(dataset, feature_indices[next_row:].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    descriptor_reference = _normalised(descriptors)
    max_descriptor_error = float(manifest.get("max_descriptor_cosine_error", 0.0))
    batches_since_flush = 0
    with torch.inference_mode():
        for images, global_indices in tqdm(loader, desc="Cache CC-LSA MSLS features"):
            global_array = np.asarray(global_indices, dtype=np.int64)
            expected_global = feature_indices[next_row : next_row + len(global_array)]
            if not np.array_equal(global_array, expected_global):
                raise RuntimeError("feature DataLoader changed canonical subset order")
            images = images.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                generated_descriptors, dino_local = extract_ru_descriptor_and_local(
                    ru_model, images
                )
                lsa_local = lsa_model(images)
                _, raw_tokens = raw_teacher(images)
                raw_local = normalise_local_tokens(
                    raw_teacher.project_patch_tokens(raw_tokens)
                )
            generated = generated_descriptors.float().cpu().numpy()
            cosine = np.einsum("nd,nd->n", generated, descriptor_reference[global_array])
            max_descriptor_error = max(
                max_descriptor_error, float(np.max(np.abs(1.0 - cosine)))
            )
            if max_descriptor_error > 5e-4:
                raise RuntimeError(
                    "shared RU forward differs from the canonical descriptor cache: "
                    f"max cosine error {max_descriptor_error:.3e}"
                )
            rows = slice(next_row, next_row + len(global_array))
            values = {
                "dino_local.npy": dino_local,
                "lsa_local.npy": lsa_local,
                "raw_clip_local.npy": raw_local,
            }
            for filename, tensor in values.items():
                array = tensor.detach().float().cpu().numpy()
                if not bool(np.isfinite(array).all()):
                    raise RuntimeError(f"non-finite local feature: {filename}")
                feature_arrays[filename][rows] = array.astype(np.float16)
            next_row += len(global_array)
            batches_since_flush += 1
            if batches_since_flush >= args.flush_every:
                for array in feature_arrays.values():
                    array.flush()
                atomic_json(
                    progress_path,
                    {"next_row": next_row, "updated_utc": utc_now(),
                     "max_descriptor_cosine_error": max_descriptor_error},
                )
                batches_since_flush = 0
    for array in feature_arrays.values():
        array.flush()
    del feature_arrays
    if next_row != len(feature_indices):
        raise RuntimeError(f"feature cache stopped at {next_row}/{len(feature_indices)}")
    atomic_json(progress_path, {"next_row": next_row, "updated_utc": utc_now()})
    immutable_files = (
        "candidates.npy",
        "candidate_scores.npy",
        "feature_indices.npy",
        "global_to_feature.npy",
        "wrong_place_indices.npy",
        "ru_best_positive_rank.npy",
        *array_specs.keys(),
    )
    manifest["complete"] = True
    manifest["completed_utc"] = utc_now()
    manifest["ru_correct_at_1"] = ru_correct
    manifest["ru_reachable_at_10"] = int(diagnostics["hit_at_10"].sum())
    manifest["ru_reachable_at_100"] = int(diagnostics["reachable"].sum())
    manifest["max_descriptor_cosine_error"] = max_descriptor_error
    manifest["raw_clip_base_model_state_sha256"] = raw_hash
    manifest["array_sha256"] = {
        filename: file_sha256(output / filename) for filename in immutable_files
    }
    atomic_json(manifest_path, manifest)
    print(
        f"CC-LSA feature cache complete: RU={ru_correct}/{num_queries}, "
        f"rows={len(feature_indices)}, max descriptor error={max_descriptor_error:.3e}"
    )
    print(f"Cache written to: {output}")


if __name__ == "__main__":
    main()
