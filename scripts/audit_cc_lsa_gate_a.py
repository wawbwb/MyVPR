#!/usr/bin/env python
"""Run the preregistered CC-LSA Pair-VPR Gate-A sufficiency audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_msls_ag_slrd_layouts import load_msls_index  # noqa: E402
from src.cc_lsa_config import load_gate_a_config  # noqa: E402
from src.cc_lsa_features import mutual_nearest_edges_batch  # noqa: E402
from src.cc_lsa_gate_a import (  # noqa: E402
    CC_LSA_AUDIT_SCHEMA,
    CC_LSA_CALIBRATION_SCHEMA,
    CC_LSA_FEATURE_SCHEMA,
    CC_LSA_VERSION,
    build_same_city_wrong_place_donors,
    count_weight_matched_scores,
    deterministic_permutation,
    dino_full_score,
    file_sha256,
    implementation_sha256,
    gate_a_verdict,
    rerank_variant,
    ru_candidate_diagnostics,
    semantic_pair_score,
)


FIXED_VARIANTS = (
    "aligned_lsa",
    "dino_full",
    "raw_clip",
    "token_permutation",
    "wrong_place",
)


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
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--msls-path", type=Path, default=Path("datasets/msls-val"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--allow-failed-teacher-contract", action="store_true",
                        help="Report EXPLORATORY_PASS/FAIL; never rewrite teacher admission as PASS.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _normalise(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not bool(np.isfinite(value).all()) or bool(np.any(norm <= 1e-12)):
        raise RuntimeError("cached local feature is non-finite or zero norm")
    return value / norm


def _weights(
    query: np.ndarray,
    candidate: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    tau: float,
) -> np.ndarray:
    cosine = np.einsum("nd,nd->n", query[left], candidate[right])
    return np.clip((cosine - tau) / (1.0 - tau), 0.0, 1.0).astype(np.float32)


def _validate_cache(
    cache_dir: Path,
    *,
    config_sha: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = _load_json(cache_dir / "manifest.json")
    if manifest.get("schema") != CC_LSA_FEATURE_SCHEMA:
        raise ValueError("unsupported CC-LSA feature-cache schema")
    if manifest.get("version") != CC_LSA_VERSION or manifest.get("complete") is not True:
        raise ValueError("CC-LSA feature cache is incomplete/unsupported")
    if manifest.get("config_sha256") != config_sha:
        raise ValueError("feature cache was built with a different Gate-A config")
    if manifest.get("implementation_sha256") != implementation_sha256():
        raise ValueError("feature cache implementation differs from current source")
    names = (
        "candidates",
        "candidate_scores",
        "feature_indices",
        "global_to_feature",
        "wrong_place_indices",
        "ru_best_positive_rank",
        "dino_local",
        "lsa_local",
        "raw_clip_local",
    )
    arrays: dict[str, np.ndarray] = {}
    declared = manifest.get("array_sha256")
    if not isinstance(declared, dict):
        raise ValueError("feature manifest has no array hashes")
    for name in names:
        filename = f"{name}.npy"
        path = cache_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if declared.get(filename) != file_sha256(path):
            raise ValueError(f"feature array SHA mismatch: {filename}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    nr, nq = int(manifest["num_references"]), int(manifest["num_queries"])
    rows, count = int(manifest["feature_rows"]), int(np.prod(config["features"]["local_grid"]))
    top_k = int(config["score"]["top_k"])
    shapes = {
        "candidates": (nq, top_k), "candidate_scores": (nq, top_k),
        "feature_indices": (rows,), "global_to_feature": (nr + nq,),
        "wrong_place_indices": (nr + nq,), "ru_best_positive_rank": (nq,),
        "dino_local": (rows, count, int(manifest["dino_dim"])),
        "lsa_local": (rows, count, int(manifest["semantic_dim"])),
        "raw_clip_local": (rows, count, int(manifest["semantic_dim"])),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"feature array shape mismatch: {name}")
        if name.endswith("local"):
            if arrays[name].dtype != np.float16:
                raise ValueError(f"local feature dtype must be float16: {name}")
        elif name != "candidate_scores" and arrays[name].dtype.kind not in "iu":
            raise ValueError(f"index array must contain integers: {name}")
    indices = np.asarray(arrays["feature_indices"])
    if np.any(indices < 0) or np.any(indices >= nr + nq) or np.any(np.diff(indices) <= 0):
        raise ValueError("feature indices must be unique increasing canonical rows")
    expected_mapping = np.full(nr + nq, -1, dtype=np.int64)
    expected_mapping[indices] = np.arange(rows)
    if not np.array_equal(arrays["global_to_feature"], expected_mapping):
        raise ValueError("global_to_feature is not the inverse of feature_indices")
    candidates = np.asarray(arrays["candidates"])
    if np.any(candidates < 0) or np.any(candidates >= nr) or any(
        np.unique(row).size != top_k for row in candidates
    ):
        raise ValueError("candidate rows contain an invalid or duplicate reference")
    scores = np.asarray(arrays["candidate_scores"])
    if not np.isfinite(scores).all() or np.any(scores[:, 1:] > scores[:, :-1]):
        raise ValueError("candidate scores must be finite and descending")
    if not np.array_equal(np.sort(arrays["wrong_place_indices"]), np.arange(nr + nq)):
        raise ValueError("wrong-place donors must form a bijection")
    return manifest, arrays


def _mnn_batch(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return mutual_nearest_edges_batch(query, candidates, device=device)


def score_all_pairs(
    arrays: dict[str, np.ndarray],
    *,
    num_references: int,
    tau_lsa: float,
    tau_raw: float,
    top_edges: int,
    token_seed: int,
    matched_seeds: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    candidates = np.asarray(arrays["candidates"], dtype=np.int64)
    mapping = np.asarray(arrays["global_to_feature"], dtype=np.int64)
    wrong = np.asarray(arrays["wrong_place_indices"], dtype=np.int64)
    query_count, top_k = candidates.shape
    scores = {
        name: np.empty((query_count, top_k), dtype=np.float32)
        for name in FIXED_VARIANTS
    }
    matched = np.empty(
        (len(matched_seeds), query_count, top_k), dtype=np.float32
    )
    permutation_cache: dict[int, np.ndarray] = {}
    for query_index in tqdm(range(query_count), desc="Score CC-LSA Gate-A pairs"):
        query_global = num_references + query_index
        candidate_globals = candidates[query_index]
        query_row = int(mapping[query_global])
        candidate_rows = mapping[candidate_globals]
        if query_row < 0 or bool(np.any(candidate_rows < 0)):
            raise RuntimeError("feature cache omitted a query or candidate")
        dino_query = arrays["dino_local"][query_row]
        dino_candidates = arrays["dino_local"][candidate_rows]
        edges = _mnn_batch(dino_query, dino_candidates, device=device)
        lsa_query = _normalise(arrays["lsa_local"][query_row])
        lsa_candidates = _normalise(arrays["lsa_local"][candidate_rows])
        raw_query = _normalise(arrays["raw_clip_local"][query_row])
        raw_candidates = _normalise(arrays["raw_clip_local"][candidate_rows])
        wrong_query_global = int(wrong[query_global])
        wrong_candidate_globals = wrong[candidate_globals]
        wrong_query_row = int(mapping[wrong_query_global])
        wrong_candidate_rows = mapping[wrong_candidate_globals]
        if wrong_query_row < 0 or bool(np.any(wrong_candidate_rows < 0)):
            raise RuntimeError("feature cache omitted a wrong-place donor")
        wrong_query = _normalise(arrays["lsa_local"][wrong_query_row])
        wrong_candidates = _normalise(arrays["lsa_local"][wrong_candidate_rows])
        if query_global not in permutation_cache:
            permutation_cache[query_global] = deterministic_permutation(
                len(lsa_query), image_key=query_global, seed=token_seed
            )
        permuted_query = lsa_query[permutation_cache[query_global]]
        for candidate_offset, candidate_global in enumerate(candidate_globals.tolist()):
            if candidate_global not in permutation_cache:
                permutation_cache[candidate_global] = deterministic_permutation(
                    len(lsa_query), image_key=candidate_global, seed=token_seed
                )
            left, right, visual = edges[candidate_offset]
            aligned_weights = _weights(
                lsa_query, lsa_candidates[candidate_offset], left, right, tau_lsa
            )
            raw_weights = _weights(
                raw_query, raw_candidates[candidate_offset], left, right, tau_raw
            )
            token_weights = _weights(
                permuted_query,
                lsa_candidates[candidate_offset][permutation_cache[candidate_global]],
                left,
                right,
                tau_lsa,
            )
            wrong_weights = _weights(
                wrong_query,
                wrong_candidates[candidate_offset],
                left,
                right,
                tau_lsa,
            )
            scores["aligned_lsa"][query_index, candidate_offset] = semantic_pair_score(
                visual, aligned_weights, top_l=top_edges
            )
            scores["dino_full"][query_index, candidate_offset] = dino_full_score(
                visual, top_l=top_edges
            )
            scores["raw_clip"][query_index, candidate_offset] = semantic_pair_score(
                visual, raw_weights, top_l=top_edges
            )
            scores["token_permutation"][query_index, candidate_offset] = semantic_pair_score(
                visual, token_weights, top_l=top_edges
            )
            scores["wrong_place"][query_index, candidate_offset] = semantic_pair_score(
                visual, wrong_weights, top_l=top_edges
            )
            matched[:, query_index, candidate_offset] = count_weight_matched_scores(
                visual,
                aligned_weights,
                query_key=query_global,
                candidate_key=candidate_global,
                seeds=matched_seeds,
                top_l=top_edges,
            )
    return scores, matched


def write_per_query(
    path: Path,
    *,
    paths: np.ndarray,
    num_references: int,
    candidates: np.ndarray,
    ru: dict[str, np.ndarray],
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = [
        "query_index",
        "query_path",
        "ru_top1",
        "ru_top1_path",
        "ru_correct",
        "ru_best_positive_rank_capped101",
        "ru_reachable_top100",
    ]
    for name in FIXED_VARIANTS:
        fields.extend(
            (
                f"{name}_top1",
                f"{name}_best_positive_rank_capped101",
                f"{name}_strict_correction",
                f"{name}_rank_improved",
                f"{name}_positive_negative_margin",
            )
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_index in range(len(candidates)):
            ru_top1 = int(candidates[query_index, 0])
            row: dict[str, Any] = {
                "query_index": query_index,
                "query_path": str(paths[num_references + query_index]),
                "ru_top1": ru_top1,
                "ru_top1_path": str(paths[ru_top1]),
                "ru_correct": int(ru["hit_at_1"][query_index]),
                "ru_best_positive_rank_capped101": int(
                    ru["best_positive_rank"][query_index]
                ),
                "ru_reachable_top100": int(ru["reachable"][query_index]),
            }
            for name in FIXED_VARIANTS:
                result = results[name]
                row.update(
                    {
                        f"{name}_top1": int(result["reranked_top1"][query_index]),
                        f"{name}_best_positive_rank_capped101": int(
                            result["best_positive_rank"][query_index]
                        ),
                        f"{name}_strict_correction": int(
                            result["correction"][query_index]
                        ),
                        f"{name}_rank_improved": int(
                            result["rank_improved"][query_index]
                        ),
                        f"{name}_positive_negative_margin": float(
                            result["positive_negative_margin"][query_index]
                        ),
                    }
                )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_gate_a_config(config_path)
    config_sha = file_sha256(config_path)
    feature_cache = args.feature_cache.expanduser().resolve()
    calibration_dir = args.calibration.expanduser().resolve()
    msls_path = args.msls_path.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Gate-A audit: {output}")
    device = choose_device(args.device)
    manifest, arrays = _validate_cache(
        feature_cache,
        config_sha=config_sha,
        config=config,
    )
    calibration_path = calibration_dir / "calibration.json"
    calibration = _load_json(calibration_path)
    expected_override = bool(args.allow_failed_teacher_contract)
    for record in (manifest, calibration):
        if record.get("exploratory_override", False) != expected_override:
            raise ValueError("exploratory admission mode differs between command and cache")
        teacher_verdict = record.get("teacher_contract", {}).get("verdict")
        if teacher_verdict not in {"PASS", "FAIL"}:
            raise ValueError("cache has no valid teacher contract")
        if teacher_verdict != "PASS" and not expected_override:
            raise ValueError("teacher contract failed; explicit exploratory flag is required")
    if manifest["teacher_contract"] != calibration["teacher_contract"]:
        raise ValueError("feature and calibration teacher contracts disagree")
    if calibration.get("schema") != CC_LSA_CALIBRATION_SCHEMA:
        raise ValueError("unsupported CC-LSA calibration schema")
    if calibration.get("version") != CC_LSA_VERSION or calibration.get("complete") is not True:
        raise ValueError("CC-LSA calibration is incomplete/unsupported")
    if calibration.get("config", {}).get("sha256") != config_sha:
        raise ValueError("calibration used a different Gate-A config")
    if calibration.get("implementation_sha256") != implementation_sha256():
        raise ValueError("calibration implementation differs from current source")
    if calibration.get("raw_clip_base_model_state_sha256") != manifest.get("raw_clip_base_model_state_sha256"):
        raise ValueError("calibration and MSLS features use different raw CLIP weights")
    if calibration.get("ru_checkpoint", {}).get("sha256") != manifest["ru_checkpoint_sha256"]:
        raise ValueError("calibration and MSLS feature cache use different RU checkpoints")
    if calibration.get("lsa_checkpoint", {}).get("sha256") != manifest["lsa_checkpoint_sha256"]:
        raise ValueError("calibration and MSLS feature cache use different LSA checkpoints")
    declared = calibration.get("array_sha256", {})
    if set(declared) != {
        "selected_indices.npy", "hard_pairs.npy", "lsa_edge_cosine.npy", "raw_clip_edge_cosine.npy"
    }:
        raise ValueError("calibration array hash set is incomplete or unsupported")
    for filename, digest in declared.items():
        if file_sha256(calibration_dir / filename) != digest:
            raise ValueError(f"calibration array SHA mismatch: {filename}")
    for key, filename in (("tau_lsa", "lsa_edge_cosine.npy"), ("tau_raw_clip", "raw_clip_edge_cosine.npy")):
        values = np.load(calibration_dir / filename, allow_pickle=False)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError(f"invalid calibration edge cosines: {filename}")
        expected = float(np.quantile(values, config["calibration"]["semantic_quantile"], method="higher"))
        if not -1.0 < expected < 1.0 or float(calibration[key]) != expected:
            raise ValueError(f"{key} differs from the registered GSV quantile")

    paths, num_references, num_queries, positives, index_record = load_msls_index(msls_path)
    if (
        num_queries != int(config["thresholds"]["expected_queries"])
        or num_queries != int(manifest["num_queries"])
        or num_references != int(manifest["num_references"])
    ):
        raise ValueError("MSLS role counts differ from the registered feature cache")
    if manifest["msls_index"] != index_record:
        raise ValueError("feature cache MSLS index provenance differs")
    candidates = np.asarray(arrays["candidates"], dtype=np.int32)
    if candidates.shape != (num_queries, int(config["score"]["top_k"])):
        raise ValueError("candidate matrix shape differs from registered Gate A")
    ru = ru_candidate_diagnostics(candidates, positives)
    if not np.array_equal(arrays["ru_best_positive_rank"], ru["best_positive_rank"]):
        raise ValueError("cached RU positive ranks differ from canonical ground truth")
    expected_wrong = build_same_city_wrong_place_donors(
        paths.tolist(), num_references=num_references, positives=positives,
        seed=int(config["seed"]), candidates=candidates,
    )
    if not np.array_equal(arrays["wrong_place_indices"], expected_wrong):
        raise ValueError("wrong-place donors differ from the registered mapping")
    score_cfg = config["score"]
    matched_seeds = tuple(
        range(score_cfg["matched_seeds"]["start"], score_cfg["matched_seeds"]["stop"])
    )
    scores, matched_scores = score_all_pairs(
        arrays,
        num_references=num_references,
        tau_lsa=float(calibration["tau_lsa"]),
        tau_raw=float(calibration["tau_raw_clip"]),
        top_edges=int(score_cfg["top_edges"]),
        token_seed=int(score_cfg["token_permutation_seed"]),
        matched_seeds=matched_seeds,
        device=device,
    )
    results = {
        name: rerank_variant(candidates, values, positives)
        for name, values in scores.items()
    }
    matched_results = [
        rerank_variant(candidates, matched_scores[seed_index], positives)
        for seed_index in range(len(matched_seeds))
    ]
    matched_correction_counts = np.asarray(
        [int(result["correction"].sum()) for result in matched_results], dtype=np.int32
    )
    matched_rank_counts = np.asarray(
        [int(result["rank_improved"].sum()) for result in matched_results], dtype=np.int32
    )
    variant_hits = {
        name: np.asarray([
            int(top1) in set(positive.tolist())
            for top1, positive in zip(result["reranked_top1"], positives)
        ], dtype=bool)
        for name, result in results.items()
    }
    thresholds = config["thresholds"]
    verdict = gate_a_verdict(
        ru_hit_at_1=ru["hit_at_1"],
        ru_reachable=ru["reachable"],
        aligned=results["aligned_lsa"],
        fixed_controls={name: results[name] for name in FIXED_VARIANTS if name != "aligned_lsa"},
        matched_correction_counts=matched_correction_counts,
        matched_rank_improved_counts=matched_rank_counts,
        expected_ru_correct=int(thresholds["expected_ru_correct"]),
        minimum_reachable_errors=int(thresholds["minimum_reachable_errors"]),
        minimum_corrections=int(thresholds["minimum_corrections"]),
        control_gap=int(thresholds["control_gap_queries"]),
        minimum_rank_improved=int(thresholds["minimum_rank_improved"]),
    )
    output.mkdir(parents=True)
    if expected_override:
        verdict["score_criteria_verdict"] = verdict["verdict"]
        verdict["verdict"] = "EXPLORATORY_" + verdict["verdict"]
        verdict["teacher_admission_verdict"] = manifest["teacher_contract"]["verdict"]
    np.savez(
        output / "pair_scores.npz",
        **scores,
        dino_count_weight_matched=matched_scores,
    )
    write_per_query(
        output / "per_query.csv",
        paths=paths,
        num_references=num_references,
        candidates=candidates,
        ru=ru,
        results=results,
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "strict_corrections", "rank_improved", "reranked_r1_count", "reranked_r1", "ru_correct_regressions"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "ru",
                "strict_corrections": "",
                "rank_improved": "",
                "reranked_r1_count": int(ru["hit_at_1"].sum()),
                "reranked_r1": float(ru["hit_at_1"].mean()),
                "ru_correct_regressions": 0,
            }
        )
        for name in FIXED_VARIANTS:
            result = results[name]
            positives_top1 = [
                int(top1) in set(positive.tolist())
                for top1, positive in zip(result["reranked_top1"], positives)
            ]
            writer.writerow(
                {
                    "variant": name,
                    "strict_corrections": int(result["correction"].sum()),
                    "rank_improved": int(result["rank_improved"].sum()),
                    "reranked_r1_count": int(sum(positives_top1)),
                    "reranked_r1": float(np.mean(positives_top1)),
                    "ru_correct_regressions": int((ru["hit_at_1"] & ~np.asarray(positives_top1)).sum()),
                }
            )
    with (output / "random_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "strict_corrections", "rank_improved"))
        writer.writeheader()
        for seed, corrections, ranks in zip(
            matched_seeds, matched_correction_counts, matched_rank_counts
        ):
            writer.writerow(
                {"seed": seed, "strict_corrections": int(corrections), "rank_improved": int(ranks)}
            )
    summary = {
        "schema": CC_LSA_AUDIT_SCHEMA,
        "version": CC_LSA_VERSION,
        "complete": True,
        "created_utc": utc_now(),
        "development_only": True,
        "exploratory_override": expected_override,
        "teacher_contract": manifest["teacher_contract"],
        "implementation_sha256": implementation_sha256(),
        "development_note": (
            "MSLS-val was previously inspected; this is a go/no-go screen, not an independent confirmation."
        ),
        "verdict": verdict,
        "config": {"path": str(config_path), "sha256": config_sha},
        "feature_cache": {
            "path": str(feature_cache),
            "manifest_sha256": file_sha256(feature_cache / "manifest.json"),
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": file_sha256(calibration_path),
            "tau_lsa": float(calibration["tau_lsa"]),
            "tau_raw_clip": float(calibration["tau_raw_clip"]),
        },
        "ru": {
            "correct_at_1": int(ru["hit_at_1"].sum()),
            "reachable_at_10": int(ru["hit_at_10"].sum()),
            "reachable_at_100": int(ru["reachable"].sum()),
            "reachable_ru_errors": int((~ru["hit_at_1"] & ru["reachable"]).sum()),
            "reachable_ru_errors_at_10": int((~ru["hit_at_1"] & ru["hit_at_10"]).sum()),
        },
        "variants": {
            name: {
                "strict_corrections": int(result["correction"].sum()),
                "rank_improved": int(result["rank_improved"].sum()),
                "reranked_r1_count": int(variant_hits[name].sum()),
                "reranked_r1": float(variant_hits[name].mean()),
                "ru_correct_regressions": int((ru["hit_at_1"] & ~variant_hits[name]).sum()),
            }
            for name, result in results.items()
        },
        "matched_control": {
            "seeds": list(matched_seeds),
            "correction_counts": matched_correction_counts.tolist(),
            "rank_improved_counts": matched_rank_counts.tolist(),
            "percentile_method": "numpy.percentile(method='higher')",
        },
    }
    atomic_json(output / "summary.json", summary)
    verdict_lines = [
        "# CC-LSA Pair-VPR Gate A",
        "",
        f"Verdict: {verdict['verdict']}",
        f"Reason: {verdict['reason']}",
        f"RU: {verdict['ru_correct']}/{num_queries}",
        f"Reachable RU errors: {verdict['reachable_ru_errors']}",
        f"Aligned strict corrections: {verdict['aligned_corrections']}",
        f"Aligned rank improvements: {verdict['aligned_rank_improved']}",
        "",
    ]
    for name, record in verdict["checks"].items():
        verdict_lines.append(f"{'PASS' if record['pass'] else 'FAIL'}  {name}: {record}")
    (output / "verdict.txt").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    print("\n".join(verdict_lines))
    print(f"Audit written to: {output}")


if __name__ == "__main__":
    main()
