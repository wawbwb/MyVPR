"""
Step 1: Teacher Sanity Check — 对比两种预处理管线下 CLIP Teacher 的 VPR 检索能力

管线 A: CLIPTeacherEncoder._preprocess
        ImageNet-norm 320×320 → 反归一化 → bicubic resize 224 → CLIP-norm
管线 B: open_clip 官方 preprocess
        PIL Image → center-crop 224×224 → CLIP-norm（无 ImageNet 归一化中间步骤）

如果管线 B 明显优于管线 A，说明当前蒸馏训练中 teacher 收到的预处理信号存在偏差。

Usage:
    conda run -n VPR python scripts/step1_teacher_sanity.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T2
from tqdm import tqdm
from PIL import Image
import open_clip

from src.models.clip_teacher import CLIPTeacherEncoder
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.msls_condition import MSLSConditionDataset
from src.utils.metrics import compute_recall_performance

BATCH_SIZE = 100
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
MSLS_PATH = str(Path("datasets/msls-val").resolve())


# ── 管线 B 专用 Dataset：直接用 open_clip 标准 preprocess ──────────────
class MSLSRawImageDataset(Dataset):
    """从 MapillarySLSDataset/MSLSConditionDataset 拿路径列表，
       但用 open_clip 标准 preprocess 而非 ImageNet 归一化。"""

    def __init__(self, ref_dataset, clip_preprocess):
        self.dataset_path = ref_dataset.dataset_path
        self.image_paths = ref_dataset.image_paths
        self.num_references = ref_dataset.num_references
        self.num_queries = ref_dataset.num_queries
        self.ground_truth = ref_dataset.ground_truth
        self.dataset_name = ref_dataset.dataset_name + "-clippp"
        self.preprocess = clip_preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.dataset_path / self.image_paths[idx]).convert("RGB")
        return self.preprocess(img), idx


# ── 提特征 ───────────────────────────────────────────────────────────
@torch.no_grad()
def extract_with_teacher_preprocess(teacher, dataloader, device):
    """管线 A：走 CLIPTeacherEncoder.forward（含 _preprocess）"""
    descs = []
    for images, _ in tqdm(dataloader, desc="  [A] Teacher _preprocess", leave=False):
        images = images.to(device)
        g, _ = teacher(images)
        descs.append(g.float().cpu().numpy())
    return np.concatenate(descs)


@torch.no_grad()
def extract_with_clip_preprocess(visual, dataloader, device):
    """管线 B：open_clip 标准 preprocess → visual.forward → proj → L2-norm"""
    import torch.nn.functional as F
    descs = []
    for images, _ in tqdm(dataloader, desc="  [B] CLIP standard preprocess", leave=False):
        images = images.to(device)
        g = visual(images)
        g = F.normalize(g, dim=-1)
        descs.append(g.float().cpu().numpy())
    return np.concatenate(descs)


# ── 评估 ─────────────────────────────────────────────────────────────
def eval_recalls(descs, dataset):
    return compute_recall_performance(
        descs, dataset.num_references, dataset.num_queries,
        dataset.ground_truth, k_values=[1, 5, 10],
    )


def main():
    device = torch.device(DEVICE)

    # ── 加载模型 ──
    print("Loading CLIP ViT-B-16 ...")
    teacher = CLIPTeacherEncoder(model_name="ViT-B-16", pretrained="openai")
    teacher = teacher.to(device).eval()

    model_ref, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    clip_visual = model_ref.visual.to(device).eval()

    # ── 构建 ImageNet-norm 数据集（管线 A 用） ──
    imagenet_transform = T2.Compose([
        T2.ToImage(),
        T2.Resize(size=(320, 320), interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])

    dataset_specs = {
        "msls-val (full)":   MapillarySLSDataset(dataset_path=MSLS_PATH, input_transform=imagenet_transform),
        "msls-val-night":    MSLSConditionDataset(condition="night", dataset_path=MSLS_PATH, input_transform=imagenet_transform),
        "msls-val-season":   MSLSConditionDataset(condition="season", dataset_path=MSLS_PATH, input_transform=imagenet_transform),
    }

    # ── 逐数据集对比 ──
    print("\n" + "=" * 80)
    print("  Step 1: CLIP Teacher 预处理对齐检查")
    print("  管线 A: CLIPTeacherEncoder._preprocess (ImageNet→反归一化→resize224→CLIP-norm)")
    print("  管线 B: open_clip 官方 preprocess (PIL→center-crop 224→CLIP-norm)")
    print("=" * 80)

    results = {}
    for ds_name, ds_a in dataset_specs.items():
        if ds_a.num_queries == 0:
            print(f"\n  {ds_name}: SKIPPED (0 queries)")
            continue

        print(f"\n  ── {ds_name} ({ds_a.num_queries} queries) ──")

        # 管线 A
        loader_a = DataLoader(ds_a, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              shuffle=False, pin_memory=True)
        descs_a = extract_with_teacher_preprocess(teacher, loader_a, device)
        r_a = eval_recalls(descs_a, ds_a)

        # 管线 B：用标准 CLIP preprocess 读取同样的图片
        ds_b = MSLSRawImageDataset(ds_a, clip_preprocess)
        loader_b = DataLoader(ds_b, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              shuffle=False, pin_memory=True)
        descs_b = extract_with_clip_preprocess(clip_visual, loader_b, device)
        r_b = eval_recalls(descs_b, ds_b)

        results[ds_name] = {"A": r_a, "B": r_b}

        for k in [1, 5, 10]:
            delta = r_b[k] - r_a[k]
            sign = "+" if delta >= 0 else ""
            print(f"    R@{k:<3}  管线A: {r_a[k]:.4f}   管线B: {r_b[k]:.4f}   Δ(B-A): {sign}{delta:.4f}")

    # ── 对齐偏差诊断 ──
    print("\n" + "=" * 80)
    print("  诊断结论")
    print("=" * 80)

    # 检查两条管线在 full 数据集上的差距
    if "msls-val (full)" in results:
        gap = results["msls-val (full)"]["B"][1] - results["msls-val (full)"]["A"][1]
        if abs(gap) < 0.02:
            print("  ✓ 管线 A/B 在 full 集上 R@1 差距 < 2%，预处理基本对齐。")
            print("    Teacher 本身零样本 VPR 能力就很有限，不是预处理的问题。")
        else:
            print(f"  ✗ 管线 A/B 在 full 集上 R@1 差距 = {gap:.4f}")
            if gap > 0:
                print("    管线 B（标准 CLIP preprocess）更好，_preprocess 存在信息损失。")
                print("    建议：蒸馏训练时改用标准 CLIP preprocess 或 center-crop。")
            else:
                print("    管线 A 反而更好（resize 保留更多信息），当前 _preprocess 可保留。")

    if "msls-val-night" in results:
        r_a_n = results["msls-val-night"]["A"][1]
        r_b_n = results["msls-val-night"]["B"][1]
        print(f"\n  Night 子集: 管线A R@1={r_a_n:.4f}, 管线B R@1={r_b_n:.4f}")
        if r_b_n < 0.15:
            print("  → CLIP Teacher 在夜间场景零样本检索能力非常受限 (<15% R@1)。")
            print("    蒸馏 CLIP 全局特征对夜间场景的改善预期有限。")
            print("    更有前景的方向：利用 CLIP 的语义能力做动态区域过滤 (Step 3)。")
        else:
            print("  → CLIP Teacher 在夜间有一定检索能力，蒸馏有改善空间。")


if __name__ == "__main__":
    main()
