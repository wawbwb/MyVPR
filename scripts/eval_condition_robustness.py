"""Compare two MixVPR checkpoints on illumination-sensitive MSLS subsets.

This script evaluates two checkpoints on:
1) msls-val-night (night -> day)
2) msls-val-season (seasonal / strong appearance shift)

Usage:
    python scripts/eval_condition_robustness.py

    python scripts/eval_condition_robustness.py \
      --my-ckpt my_MIXVPR_R1[0.8662]_R5[0.9297].ckpt \
      --origin-ckpt origin_MIXVPR_R1[0.8784]_R5[0.9297].ckpt
"""

import argparse
import copy
import importlib
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloaders.valid.msls_condition import MSLSConditionDataset
from src.models.distillation import SpatialAttentionHead
from src.utils.metrics import compute_recall_performance


DEFAULT_MY_CKPT = "my_MIXVPR_R1[0.8662]_R5[0.9297].ckpt"
DEFAULT_ORIGIN_CKPT = "origin_MIXVPR_R1[0.8784]_R5[0.9297].ckpt"

VAL_IMAGE_SIZE = (320, 320)
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class InferenceModel(torch.nn.Module):
    """Student-only inference wrapper used by both baseline and phase C.

    The CLIP teacher is intentionally absent at inference.  A phase-C model
    keeps only its learned spatial gate and follows
    ``backbone -> spatial gate -> aggregator``.
    """

    def __init__(self, backbone, aggregator, spatial_attn_head=None):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.spatial_attn_head = spatial_attn_head

    def forward(self, x):
        featmap = self.backbone(x)
        if self.spatial_attn_head is not None:
            featmap, _ = self.spatial_attn_head(featmap)
        output = self.aggregator(featmap)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate two MixVPR checkpoints on night/season condition splits"
    )
    parser.add_argument("--my-ckpt", default=DEFAULT_MY_CKPT, help="path to your best checkpoint")
    parser.add_argument("--origin-ckpt", default=DEFAULT_ORIGIN_CKPT, help="path to original best checkpoint")
    parser.add_argument("--msls-path", default="datasets/msls-val", help="path to msls-val folder")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="recall@k list, e.g. --k-values 1 5 10",
    )
    return parser.parse_args()


def get_instance(module_name, class_name, params):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)


def build_transform():
    return T2.Compose(
        [
            T2.ToImage(),
            T2.Resize(size=VAL_IMAGE_SIZE, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ]
    )


def get_spatial_attn_config(config):
    """Read the phase-C inference config while keeping old checkpoints valid."""
    # Current experiment configs store the head below ``distillation``.  The
    # top-level fallback makes exported/inference-only configs easy to support.
    distill_cfg = config.get("distillation", {}) or {}
    return config.get("spatial_attn", distill_cfg.get("spatial_attn", {})) or {}


def extract_required_submodule_state(state_dict, prefix, module_name):
    """Extract a submodule and fail before evaluation if it was not saved."""
    submodule_state = OrderedDict(
        (key[len(prefix):], value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )
    if not submodule_state:
        raise RuntimeError(
            f"Checkpoint enables {module_name}, but contains no weights with "
            f"the required prefix {prefix!r}. Refusing to evaluate a different "
            "backbone -> aggregator model silently."
        )
    return submodule_state


def strip_compiled_model_prefix(state_dict):
    """Canonicalise checkpoints saved from ``torch.compile(model)``."""
    compiled_prefix = "_orig_mod."
    if not any(key.startswith(compiled_prefix) for key in state_dict):
        return state_dict
    return OrderedDict(
        (
            key[len(compiled_prefix):]
            if key.startswith(compiled_prefix)
            else key,
            value,
        )
        for key, value in state_dict.items()
    )


def load_inference_model_from_ckpt(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]

    backbone = get_instance(
        config["backbone"]["module"],
        config["backbone"]["class"],
        copy.deepcopy(config["backbone"]["params"]),
    )

    agg_params = copy.deepcopy(config["aggregator"]["params"])
    if "in_channels" in agg_params and agg_params["in_channels"] is None:
        agg_params["in_channels"] = backbone.out_channels
    aggregator = get_instance(
        config["aggregator"]["module"],
        config["aggregator"]["class"],
        agg_params,
    )

    spatial_cfg = get_spatial_attn_config(config)
    spatial_attn_head = None
    if spatial_cfg.get("enabled", False):
        spatial_attn_head = SpatialAttentionHead(
            in_channels=backbone.out_channels,
            num_heads=spatial_cfg.get("num_heads", 1),
            gate_strength=spatial_cfg.get("gate_strength", 1.0),
        )

    model = InferenceModel(
        backbone=backbone,
        aggregator=aggregator,
        spatial_attn_head=spatial_attn_head,
    )

    full_state_dict = strip_compiled_model_prefix(checkpoint["state_dict"])
    backbone_state = extract_required_submodule_state(
        full_state_dict, prefix="backbone.", module_name="backbone"
    )
    aggregator_state = extract_required_submodule_state(
        full_state_dict, prefix="aggregator.", module_name="aggregator"
    )

    try:
        model.backbone.load_state_dict(backbone_state, strict=True)
        model.aggregator.load_state_dict(aggregator_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint backbone/aggregator weights do not match its saved "
            f"configuration: {exc}"
        ) from exc

    if model.spatial_attn_head is not None:
        spatial_state = extract_required_submodule_state(
            full_state_dict,
            prefix="spatial_attn_head.",
            module_name="SpatialAttentionHead",
        )
        try:
            model.spatial_attn_head.load_state_dict(spatial_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "SpatialAttentionHead is enabled, but its checkpoint weights "
                f"do not match the configured head: {exc}"
            ) from exc

    model = model.to(device)
    model.eval()
    return model


def extract_descriptors(model, dataloader, device):
    all_desc = []
    model.eval()
    use_amp = device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="  Extracting", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                desc = model(images)
            all_desc.append(desc.cpu().numpy())
    return np.concatenate(all_desc, axis=0)


def evaluate_on_dataset(model, dataset, device, batch_size, num_workers, k_values):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    descriptors = extract_descriptors(model, loader, device)
    recalls = compute_recall_performance(
        descriptors,
        dataset.num_references,
        dataset.num_queries,
        dataset.ground_truth,
        k_values=k_values,
    )
    return recalls


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def print_result_table(results, dataset_names, model_names, k_values):
    print("\n" + "=" * 90)
    print("结果对比（Δ = My - Origin）")
    print("=" * 90)

    header = f"{'Dataset':<35} | {'Metric':<8} | {model_names[0]:>16} | {model_names[1]:>16} | {'Δ':>10}"
    print(header)
    print("-" * len(header))

    for ds_name in dataset_names:
        for k in k_values:
            v0 = results[model_names[0]][ds_name][k]
            v1 = results[model_names[1]][ds_name][k]
            d = v0 - v1
            sign = "+" if d >= 0 else ""
            print(f"{ds_name:<35} | R@{k:<6} | {v0:>16.4f} | {v1:>16.4f} | {sign}{d:>9.4f}")
        print("-" * len(header))


def main():
    args = parse_args()
    device = choose_device(args.device)

    my_ckpt = Path(args.my_ckpt)
    origin_ckpt = Path(args.origin_ckpt)
    msls_path = Path(args.msls_path)

    if not my_ckpt.exists():
        raise FileNotFoundError(f"my checkpoint not found: {my_ckpt}")
    if not origin_ckpt.exists():
        raise FileNotFoundError(f"origin checkpoint not found: {origin_ckpt}")
    if not msls_path.exists():
        raise FileNotFoundError(f"msls path not found: {msls_path}")

    transform = build_transform()
    datasets = OrderedDict(
        {
            "msls-val-night": MSLSConditionDataset(
                condition="night", dataset_path=msls_path, input_transform=transform
            ),
            "msls-val-season": MSLSConditionDataset(
                condition="season", dataset_path=msls_path, input_transform=transform
            ),
        }
    )

    checkpoints = OrderedDict(
        {
            "My Semantic-Gated": my_ckpt,
            "Origin MixVPR": origin_ckpt,
        }
    )

    print("=" * 90)
    print("MSLS 条件鲁棒性评测（夜间/光照显著变化）")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"My ckpt: {my_ckpt}")
    print(f"Origin ckpt: {origin_ckpt}")
    for ds_name, ds in datasets.items():
        print(f"{ds_name}: {ds.num_queries} queries, {ds.num_references} db images")

    results = {}
    for model_name, ckpt_path in checkpoints.items():
        print("\n" + "=" * 90)
        print(f"Loading model: {model_name}")
        model = load_inference_model_from_ckpt(ckpt_path, device)

        results[model_name] = {}
        for ds_name, ds in datasets.items():
            if ds.num_queries == 0:
                print(f"  {ds_name}: skipped (0 queries)")
                results[model_name][ds_name] = {k: 0.0 for k in args.k_values}
                continue

            print(f"  Evaluating: {ds_name}")
            recalls = evaluate_on_dataset(
                model,
                ds,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                k_values=args.k_values,
            )
            results[model_name][ds_name] = recalls
            metric_str = "  ".join([f"R@{k}={recalls[k]:.4f}" for k in args.k_values])
            print(f"    {metric_str}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_result_table(
        results=results,
        dataset_names=list(datasets.keys()),
        model_names=list(checkpoints.keys()),
        k_values=args.k_values,
    )


if __name__ == "__main__":
    main()
