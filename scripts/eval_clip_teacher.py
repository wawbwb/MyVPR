"""
Evaluate CLIP teacher (ViT-B-16) as a zero-shot VPR feature extractor
on the same condition-specific MSLS-val subsets (night / season).

Usage:
    conda run -n VPR python scripts/eval_clip_teacher.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T2
from tqdm import tqdm

from src.models.clip_teacher import CLIPTeacherEncoder
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.msls_condition import MSLSConditionDataset
from src.utils.metrics import compute_recall_performance

VAL_IMAGE_SIZE = (320, 320)
BATCH_SIZE = 100
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def build_transform():
    return T2.Compose([
        T2.ToImage(),
        T2.Resize(size=VAL_IMAGE_SIZE, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])


def extract_descriptors(teacher, dataloader, device):
    all_desc = []
    teacher.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="  Extracting", leave=False):
            images = images.to(device)
            t_global, _ = teacher(images)   # (B, D_global), already L2-normed
            all_desc.append(t_global.float().cpu().numpy())
    return np.concatenate(all_desc, axis=0)


def evaluate_on_dataset(teacher, dataset, device):
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=False, pin_memory=True, drop_last=False,
    )
    descriptors = extract_descriptors(teacher, loader, device)
    recalls = compute_recall_performance(
        descriptors,
        dataset.num_references,
        dataset.num_queries,
        dataset.ground_truth,
        k_values=[1, 5, 10],
    )
    return recalls


def main():
    device = torch.device(DEVICE)
    transform = build_transform()
    msls_path = str(Path("datasets/msls-val").resolve())

    datasets = {
        "msls-val (full)":                     MapillarySLSDataset(dataset_path=msls_path, input_transform=transform),
        "msls-val-night (n2d, 夜间→白天)":     MSLSConditionDataset(condition="night", dataset_path=msls_path, input_transform=transform),
        "msls-val-season (跨季节)":             MSLSConditionDataset(condition="season", dataset_path=msls_path, input_transform=transform),
    }

    print("=" * 65)
    print("   CLIP Teacher (ViT-B-16) 零样本 VPR 评测")
    print("=" * 65)
    for name, ds in datasets.items():
        print(f"  {name}: {ds.num_queries} queries, {ds.num_references} DB")

    print("\nLoading CLIP ViT-B-16 teacher ...")
    teacher = CLIPTeacherEncoder(model_name="ViT-B-16", pretrained="openai")
    teacher = teacher.to(device)
    teacher.eval()
    print(f"  Global dim: {teacher.global_dim}, Token dim: {teacher.token_dim}")

    results = {}
    for ds_name, dataset in datasets.items():
        if dataset.num_queries == 0:
            print(f"  {ds_name}: SKIPPED (0 queries)")
            results[ds_name] = {1: 0, 5: 0, 10: 0}
            continue
        print(f"\n  Evaluating: {ds_name}")
        recalls = evaluate_on_dataset(teacher, dataset, device)
        results[ds_name] = recalls
        print(f"    R@1={recalls[1]:.4f}  R@5={recalls[5]:.4f}  R@10={recalls[10]:.4f}")

    # ── 对比表格 ────────────────────────────────────────
    # 上次运行的学生模型结果（硬编码方便对比，也可以传参）
    student_results = {
        "Distilled MixVPR": {
            "msls-val (full)":                 {1: 0.8716, 5: 0.9230, 10: 0.9378},
            "msls-val-night (n2d, 夜间→白天)": {1: 0.0545, 5: 0.3091, 10: 0.4364},
            "msls-val-season (跨季节)":        {1: 0.8674, 5: 0.9170, 10: 0.9200},
        },
        "Original MixVPR": {
            "msls-val (full)":                 {1: 0.8784, 5: 0.9297, 10: 0.9432},
            "msls-val-night (n2d, 夜间→白天)": {1: 0.0909, 5: 0.3091, 10: 0.4364},
            "msls-val-season (跨季节)":        {1: 0.8674, 5: 0.9170, 10: 0.9190},
        },
    }

    print("\n\n" + "=" * 80)
    print("   Teacher vs Student 综合对比")
    print("=" * 80)
    print(f"{'Dataset':<36} {'K':>3} | {'CLIP Teacher':>13} | {'Distill MixVPR':>14} | {'Origin MixVPR':>13}")
    print("─" * 88)

    for ds_name in datasets:
        for k in [1, 5, 10]:
            t_val = results[ds_name][k]
            d_val = student_results["Distilled MixVPR"].get(ds_name, {}).get(k, float('nan'))
            o_val = student_results["Original MixVPR"].get(ds_name, {}).get(k, float('nan'))
            print(f"  {ds_name:<34} R@{k:<2}| {t_val:>12.4f}  | {d_val:>13.4f}  | {o_val:>12.4f}")
        print()


if __name__ == "__main__":
    main()
