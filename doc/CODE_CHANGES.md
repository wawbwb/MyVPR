# 代码变更 — 语义先验蒸馏框架 (阶段 C–F)

> **Status (2026-07-19): archived historical record.** The C/D/E/F routes and
> semantic-reliability spatial target have both been stopped. The active
> CLIP semantic-alias implementation and commands are documented in
> `doc/CLIP_SEMANTIC_ALIAS.md`.

本文档记录了为整合 CLIP 语义先验蒸馏计划剩余实验（阶段 C、D、E、F）所做的所有代码更改。所有更改都是**附加的且由配置驱动的**：现有的 A/B 阶段配置保持不变继续工作，每一个新功能都通过 YAML 进行切换。您无需编辑 Python 代码即可运行以下任何实验。

框架简介：一个冻结的 CLIP ViT-B-16 **图像**编码器作为教师模型。它提供 (1) 全局语义向量以及 (2) 局部语义贡献分布（即其 CLS 到 patch 的注意力）。一个小型学生头在没有文本输入的情况下学习将注意力集中在何处，因此推理阶段保持仅依赖视觉且为单阶段。

## 1. 修改的文件

### 1.1 `src/models/clip_teacher.py`
增加了对 CLIP 最后一层 **CLS-to-patch 注意力**（教师模型的局部语义贡献分布）的提取。

- 新增 `_run_last_block_with_attn(blk, x)`：复现了一个 open_clip `ResidualAttentionBlock` 的前向传播，但在调用其 `nn.MultiheadAttention` 时设置了 `need_weights=True, average_attn_weights=True`，因此在捕获每个头平均后的注意力图的同时，数值块的输出保持不变。
- 修改 `_encode(x, return_attn=False)`：当 `return_attn=True` 时，会手动迭代残差块（遵循 transformer 的 NLD↔LND 布局），并且最后一个块返回注意力。返回值为 `(t_global, patch_tokens, cls_attn)`，其中 `cls_attn` 的形状为 `(B, num_patches)`，代表 CLS 对所有 patch token 的注意力。
- 修改 `forward(x, return_attn=False)`：新增标志；在请求时返回 3 元组。默认行为（返回 2 元组）保持不变，因此不会影响现有的调用者。

### 1.2 `src/models/distillation.py`
利用阶段 C/D 的空间注意力（spatial-attention）流水线扩展了 `DistillationModule`。所有新增内容均由 `use_spatial_attn` 门控；当它为 `False` 时，模块的行为与之前完全一样（兼容 B 阶段）。

- 新的构造函数参数：`use_spatial_attn=False`、`num_attn_heads=1`、`diverge_margin=0.5`。
- 新的子模块 `attn_head`：`nn.Conv2d(student_feat_channels, num_attn_heads, 1)`（仅在 `use_spatial_attn=True` 时构建；可训练，并在推理时保留）。
- `spatial_attention(featmap)` → `(weighted_featmap, attn)`。每个头应用空间 softmax；权重图通过 `N = Hs*Ws` 进行缩放，使平均乘数为 1.0（因为原始的 softmax 会将激活值缩小约 1/N 从而破坏度量学习）。在多头情况下，会对每个头的加权图求平均，因此聚合器的输入通道数保持不变。
- `attention_kl_loss(student_attn, cls_attn, Hs, Ws)` (**C2**)：将教师模型的注意力双线性缩放到学生网格并重新归一化；在对注意力头求平均的学生分布上计算 KL(teacher‖student)。
- `consistency_loss(student_attn, P, K)` (**D1**)：每个地点的 K 个视图之间的 `1 − 平均非对角余弦值`。
- `divergence_loss(student_attn, labels)` (**D2**)：批次中所有不同地点配对上的 `平均 relu(cos − margin)`。
- `forward(...)` 现在接受 `student_attn, labels, P, K` 参数，并且当空间注意力处于激活状态时，会在返回的字典中添加 `loss_attn`、`loss_consist`、`loss_diverge`（与现有的 `loss_global`、`loss_region` 并列）。

**设计注意事项 (D2)：** 计划中引用了 `MultiSimilarityMiner` 的困难负样本（hard negatives）。这些索引存在于 `VPRLossFunction` 内部且未被暴露；为了避免与损失函数的内部细节产生耦合，散度损失对**每一个**不同地点配对进行惩罚，这是挖掘出的困难负样本的严格超集。这在 `divergence_loss` 的内联注释中进行了说明。

### 1.3 `src/core/vpr_framework.py` (`VPRFrameworkDistill`)
- 新的构造函数参数：`lambda_attn`、`lambda_consist`、`lambda_diverge`、`anneal_start_epoch`、`anneal_end_epoch`。
- `self.apply_spatial_attn` 是从蒸馏模块中读取的。
- **重写 `forward(x)`**：当空间注意力被激活时，主干网络特征在进入聚合器**之前**通过 `spatial_attention` 重新加权。这保持了训练和验证/推理的描述符路径完全一致（推理时从不使用 CLIP）。
- `_anneal_factor()` (**E2**)：在 `anneal_start_epoch` 之前为 1.0，线性衰减到 `anneal_end_epoch` 时的 0.0；当禁用退火时返回 1.0。
- 重构了 `training_step`：主干网络 →（可选的空间注意力）→ 聚合器；在重新加权的描述符上计算 VPR 损失；将 `student_attn, labels, P, K` 传递给蒸馏模块；使用 `scale = warmup_scale * anneal_factor` 结合每个项的 lambda 权重将所有损失项组合起来。记录新的损失项（`loss_attn_distill`、`loss_consist_distill`、`loss_diverge_distill`）。

### 1.4 `run.py`
- 使用从 `distillation.spatial_attn` / `distillation.divergence` 读取的 `use_spatial_attn`、`num_attn_heads`、`diverge_margin` 来构建 `DistillationModule`。
- 将 `lambda_attn`（= `spatial_attn.lambda_kl`）、`lambda_consist`（= `consistency.lambda`）、`lambda_diverge`（= `divergence.lambda`）、`anneal_start_epoch`/`anneal_end_epoch`（= `anneal.*`）传递给 `VPRFrameworkDistill`。
- 所有新的键都使用带有 `.get(...)` 默认值的方式读取，因此较旧的配置仍然有效。

### 1.5 `src/core/vpr_datamodule.py`
- **Bug 修复：** `_get_val_dataset` 以前会优先匹配 `"msls"`，这导致掩盖了 `msls-val-night` / `msls-val-season` 分支（使之成为死代码）。现在会在通用的 `msls` 分支**之前**优先检查特定条件的分支，因此可以正确解析 `msls-val-night` 和 `msls-val-season`。
- 注册了新的验证集：`tokyo247` → `Tokyo247Dataset`，`nordland` → `NordlandDataset`（并在文件顶部添加了相应的导入）。

### 1.6 `config/data/config.yaml`
- 增加了 `val:` 条目 `tokyo247` 和 `nordland`。**在运行阶段 F 之前，请将这些路径更新为您本地下载的位置**（如果路径丢失，加载器会抛出异常）。

---

## 2. 新增文件

### 2.1 验证数据加载器（遵循 `pittsburgh.py` 接口）
- `src/dataloaders/valid/tokyo247.py` — **Tokyo 24/7** 数据集（白天/日落/夜晚不变性）。预期加载 `tokyo247_dbImages.npy`、`tokyo247_qImages.npy` 和 `tokyo247_gt_25m.npy`。
- `src/dataloaders/valid/nordland.py` — **Nordland** 数据集（四季）。支持两种布局：(A) 预计算的 `nordland_dbImages.npy` / `nordland_qImages.npy` / `nordland_gt.npy`；(B) 两个并行且文件名帧对齐的季节文件夹（`db_season`/`query_season`，具有可选 `tolerance` 容差的同一性真实标签）。如果 `.npy` 文件存在，则自动使用布局 A。

### 2.2 实验配置 (`config/`)

| 文件                                               | 阶段 | 新增内容                                        |
| -------------------------------------------------- | ---- | ----------------------------------------------- |
| `mixvpr_distill_attn_kl.yaml`                      | C2   | 空间注意力头 + CLIP 注意力 KL (`lambda_kl=0.1`) |
| `mixvpr_distill_attn_kl_l{0.01,0.05,0.1,0.2}.yaml` | C3   | `lambda_kl` 参数扫描                            |
| `mixvpr_distill_consistency.yaml`                  | D1   | + 跨视图一致性 (`lambda=0.05`)                  |
| `mixvpr_distill_divergence.yaml`                   | D2   | + 跨地点散度 (`lambda=0.02`, `margin=0.5`)      |
| `mixvpr_distill_multihead.yaml`                    | D3   | 4 个注意力头                                    |
| `mixvpr_distill_full.yaml`                         | E1   | B+C+D 的最佳组合                                |
| `mixvpr_distill_full_anneal.yaml`                  | E2   | E1 + 蒸馏退火 (第 20→30 轮)                     |
| `mixvpr_distill_full_resnet101.yaml`               | E3   | 在 ResNet-101 上的 E1 (无代码更改 — 已支持)     |
| `mixvpr_distill_full_benchmark.yaml`               | F    | 在所有 6 个验证集上评估的 E1                    |

---

## 3. 新的配置键（在 `distillation:` 层级下）

```yaml
distillation:
  spatial_attn:          # 阶段 C/D — 可学习的学生注意力头
    enabled: false       # 构建 + 应用 SpatialAttentionHead
    num_heads: 1         # 4 → 多头 (D3)；每个头的特征图求平均
    lambda_kl: 0.0       # KL 散度(教师注意力 ‖ 学生注意力)的权重
  consistency:           # 阶段 D1
    lambda: 0.0          # 跨视图注意力一致性的权重
  divergence:            # 阶段 D2
    lambda: 0.0          # 跨地点注意力散度的权重
    margin: 0.5          # 余弦相似度上的 hinge margin
  anneal:                # 阶段 E2
    start_epoch: null    # 开始衰减所有蒸馏权重
    end_epoch: null      # 从此轮开始蒸馏权重 = 0
```

所有键都是可选的；省略某个代码块即禁用该项（权重设为 0）。 现有的键（`enabled`、`mode`、`teacher`、`proj_dim`、`tau`、`lambda_global`、`lambda_region`、`distill_warmup_steps`、`dynamic_categories`）保持不变。

## 4. 注意事项

- **检查点监视器：** 训练阶段监视的是 `msls-val/R1`，因此请在 `val_set_names` 中保留 `msls-val`（所有提供的配置都已遵守此规则）。
- **Lambda 占位符：** C/D/E 阶段的配置文件提供了合理的默认值，但包含 `# set to your best ...` 的注释提示 — 请在运行 E 阶段之前填入您在 B2/B3/C3 中跑出的最佳参数。
- **GPU 设备：** `run.py` 中硬编码了 `devices=[1]`。如果您在不同的 GPU 上进行训练，请修改该处代码。
- **内存溢出 (OOM)：** 教师模型会增加显存消耗；如果需要，请减小 `batch_size`（例如从 100 降至 80），或者在命令行中传入 `--batch_size 80`。
