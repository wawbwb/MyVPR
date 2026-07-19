# VPR-conditioned CLIP Semantic Reliability

> **状态（2026-07-19）：该路线已停止，仅保留用于复现。** 最新 40 epoch
> 实验的 MSLS R@1 最佳为 86.49，后 20 epoch 均值为 86.23；相对 C0
> 分别下降 1.08 和 0.96 个点，而 Pitts30k 基本持平。这是稳定的低平台，
> 不再继续扫描 KL、temperature 或 positive-only。当前路线见
> `doc/CLIP_SEMANTIC_ALIAS.md`。

本路线替代旧的 Phase C（直接复制 CLIP CLS-to-patch attention）以及原计划的
D1/D2。旧 D1/D2 配置不再作为后续实验路线。

## 核心目标

对于一张图像中的 CLIP patch `i`：

```text
positive(i) = 同地点其他视角中，patch i 的最佳语义匹配相似度的均值
negative(i) = CLIP 语义最相似的异地点中，patch i 的最佳匹配相似度
reliability(i) = positive(i) - negative(i)
target = softmax(reliability / temperature)
```

正样本匹配不比较相同空间坐标，而是在另一视图的全部 patch 中寻找最近邻，
因此允许视角变化。异地点首先用每个地点的 CLIP 全局 prototype 选择，再为每张
anchor 图像选择该负地点中全局语义最相似的视图。道路、天空、树木等同时出现在
正负地点中的通用内容会被抵消，而同地点稳定且不易在混淆地点出现的区域会得到
更高权重。

CLIP、正负匹配和 target 构建均无梯度。默认配置还会阻断 reliability KL 到
backbone 的直接梯度；KL 训练空间门控头，原 VPR loss 仍通过门控检索路径更新
空间头和 backbone。验证/推理只执行 backbone、学生空间头和 MixVPR，不执行
CLIP。

## 主要文件

- `src/models/semantic_reliability.py`：位置无关的正匹配、地点级语义负样本挖掘和可靠性 target。
- `src/models/clip_teacher.py`：将 CLIP patch token 投影到归一化语义空间。
- `src/models/distillation.py`：在旧 attention KL 与新 reliability target 间按配置切换。
- `src/core/vpr_framework.py`：传递 place labels、可选阻断 KL 到 backbone，并记录诊断量。
- `config/mixvpr_distill_vpr_semantic_reliability.yaml`：首个主实验配置。
- `config/mixvpr_distill_vpr_semantic_stability.yaml`：移除异地点项的正样本消融。

## 服务器运行顺序

```bash
git pull origin main
conda activate VPR

pytest -q tests/test_phase_c_attention.py tests/test_semantic_reliability.py

python run.py --train --dev \
  --config config/mixvpr_distill_vpr_semantic_reliability.yaml \
  --devices 0 --precision 32-true --batch_size 2 --img_per_place 2

python run.py --train --dev \
  --config config/mixvpr_distill_vpr_semantic_reliability.yaml \
  --devices 0 --precision 16-mixed

python run.py --train \
  --config config/mixvpr_distill_vpr_semantic_reliability.yaml \
  --devices 0
```

第一个 dev run 检查边界和 fp32 数值；第二个使用完整 batch 检查混合精度和显存；
两者都通过后再启动 40 epoch。若完整 batch OOM，先尝试 `--batch_size 80`，但正式
对比时 C0 也必须使用相同 batch size 重跑。

不要为本配置添加 `--compile`：动态 label 分组、top-k 地点挖掘和分块 patch
匹配会导致不稳定的图重编译，`run.py` 会对此主动报错。

## TensorBoard 必看指标

- `loss_attn_distill`
- `reliability_pos_sim`、`reliability_neg_sim`、`reliability_margin`
- `reliability_positive_margin_frac`
- `reliability_target_entropy_norm`、`reliability_target_peak`
- `reliability_hard_negative_place_sim`
- `reliability_valid_anchor_frac`（正常 P×K batch 应接近 1）
- `student_attn_entropy`、`student_attn_peak`
- `student_gate_std`、`student_gate_max`

如果 `reliability_valid_anchor_frac` 在正常 batch 中小于 1，应先检查 sampler/labels，
不要继续训练。如果 target 接近完全均匀（归一化熵长期接近 1、peak 接近 `1/196`），
说明正负相似度差没有形成有效监督。

## 主实验通过后的消融与多种子

只有主配置单种子达到或超过 C0，才继续正样本消融：

```bash
python run.py --train \
  --config config/mixvpr_distill_vpr_semantic_stability.yaml \
  --devices 0 --seed 42
```

若完整正负方法优于 positive-only，说明“抑制语义混淆地点中的公共内容”确实有
独立贡献。随后对主方法与 C0 至少运行 `seed=1,42,3407`；已有 seed 42 的结果
无需重复：

```bash
python run.py --train --config config/mixvpr_distill_vpr_semantic_reliability.yaml --devices 0 --seed 1
python run.py --train --config config/mixvpr_distill_vpr_semantic_reliability.yaml --devices 0 --seed 3407

python run.py --train --config config/mixvpr_distill_attn_arch_only.yaml --devices 0 --seed 1
python run.py --train --config config/mixvpr_distill_attn_arch_only.yaml --devices 0 --seed 3407
```

## 首轮实验的解释边界

当前 GSV-Cities batch 只提供离散 place label，没有把 GPS/panoid 元数据传入训练步。
因此首轮实现能够排除同 label，却还不能排除“不同 place ID、但地理上相邻或来自
同一 panorama”的假负样本。首轮实验应被视为方法 pilot：若结果优于 C0，再增加
GPS 安全半径/panoid 过滤并复现实验，之后才能将提升归因于可靠的语义混淆负样本。

## 本地提交（只提交本次方法代码）

以下命令刻意不使用 `git add .`，避免把本地的大型实验 txt 日志一起提交：

```bash
git status --short

git add run.py \
  src/core/vpr_framework.py \
  src/models/clip_teacher.py \
  src/models/distillation.py \
  src/models/semantic_reliability.py \
  tests/test_semantic_reliability.py \
  config/mixvpr_distill_vpr_semantic_reliability.yaml \
  config/mixvpr_distill_vpr_semantic_stability.yaml \
  doc/VPR_SEMANTIC_RELIABILITY.md \
  doc/EXPERIMENT_GUIDE.md \
  doc/CODE_CHANGES.md

git diff --cached --check
git diff --cached --stat
git commit -m "feat: add VPR-conditioned CLIP semantic reliability"
git push origin main
```
