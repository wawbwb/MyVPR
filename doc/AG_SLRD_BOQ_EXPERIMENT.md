# Advantage-Gated Semantic-Layout Relational Distillation（AG-SLRD-BoQ）

状态：**PHASE 0 IMPLEMENTED / NOT RUN**
设计冻结日期：2026-09-01

## 1. 研究问题

本项目已有 aligned semantic teacher 多次不胜 shuffled/random/wrong-place，且 RSCD aligned 也只比 uniform DropBlock 少退化，没有超过 RU 或 no-mask。共同问题不是“语义分支一定太弱”，而是 generic CLIP/类别信号对某些样本并不包含有利的地点实例知识，却被无条件施加给模型。

AG-SLRD 只检验一个更窄的问题：

> 当一个由地点损失训练的 semantic-layout teacher 在某个真实 query-positive pair 上确实比 RU 更可靠时，只蒸馏它的相对检索关系，能否提高纯 RGB RU+BoQ？

它不再使用 semantic mask、attention bias、FiLM、CLIP residual、generic descriptor matching 或 semantic hard-negative mining。

## 2. 论文依据与项目适配

- [StructVPR（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_StructVPR_Distill_Structural_Knowledge_With_Weighting_Samples_for_Visual_Place_CVPR_2023_paper.html)分别用 VPR loss 训练 RGB/SEG branch，并依据 teacher/student 对 query-positive pair 的排名选择、加权蒸馏样本；其核心观察是某些 semantic sample 会伤害蒸馏。
- [StructVPR++（TPAMI 2025）](https://arxiv.org/abs/2503.06601)进一步把 label-specific features 从 global descriptor 解耦，在图像对之间显式对齐语义，并继续采用 sample-wise weighting。论文官方 DOI 为 [10.1109/TPAMI.2025.3556859](https://doi.org/10.1109/TPAMI.2025.3556859)。
- [DistilVPR（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28905)强调 VPR 的跨模态蒸馏应利用样本间关系，而不是假定异构 embedding 可逐元素对齐。
- [SemVPR（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html)明确指出 vanilla CLIP 的 raw local map 不适合作为局部教师；其有效证据属于 dense LSA、局部蒸馏和 semantic-aware aggregation 的组合。因此本方案不再复用 raw CLIP patch 或稀疏 crop 近似。

AG-SLRD 是这些原则在当前 DINOv2-RU-BoQ 代码库上的最小可证伪适配，不是任一论文的完整复现。

## 3. Phase 0：semantic-layout teacher 与充分性审计

### 3.1 输入与 teacher

从冻结 SegFormer ADE20K 生成 70×70 `uint8` label cache，保留比现有 20×20 cache 更完整的轮廓。将 150 类按预先固定映射合并为少量 VPR superclasses，以降低过分割噪声；dynamic 必须保留为独立 superclass，不能因类别名称被固定删除或惩罚。

teacher 只看 semantic layout：

```text
coarse label map
  -> lightweight layout encoder
  -> global structural feature + per-class masked features
  -> learned per-image class weights
  -> normalized semantic-layout descriptor
```

teacher 使用与 RGB 模型相同的地点标签和 VPR loss 训练。类别权重由地点损失学习，不使用手写“建筑重要、车辆有害”分数。

### 3.2 不允许跳过的充分性审计

冻结 semantic teacher 与 RU，在未参与 teacher 拟合的样本及 MSLS-val 上计算：

| 统计 | 含义 |
| --- | --- |
| both-correct | 两模态都能检索正确 |
| RU-only | 语义 teacher 会伤害的样本 |
| semantic-only | teacher 可向 RU 提供的 oracle 互补样本 |
| both-wrong | 两者都没有可用信号 |
| oracle union | 任一模态正确时的 R@1 上界 |
| teacher-better query-rank rate | 在未见过的 MSLS query 上，teacher 的真实 positive rank 优于 RU 的比例 |

同时用跨地点循环置换的 label map 做负对照，确认 aligned semantic-layout relation 不是仅凭类别频率或城市先验得到同样结果。

以下任一条件满足就停止，不实现 student：

- MSLS `semantic-only < 8/740`；
- MSLS `teacher-better query-rank < 5%`；
- aligned teacher 对 RU 错误的富集不优于 shuffled label control；
- teacher 在 held-out place/city 上没有高于 shuffled 的真实 positive ranking 能力。

## 4. Phase 1：advantage-gated relational distillation

### 4.1 样本权重

对同一组 query、positive 与跨地点 negative，计算：

```text
m_T = min_positive_cos_T - max_negative_cos_T
m_S = min_positive_cos_S - max_negative_cos_S
```

只有 `m_T > 0` 且 `m_T > m_S` 时允许 `w > 0`。`w` 随 teacher reliability `m_T` 与 advantage `m_T - m_S` 单调增加并有固定上界；其函数、温度和裁剪范围必须在查看 retrieval 结果前冻结。teacher、margin 与权重均 stop-gradient。

也可离线采用 StructVPR++ 的 rank partition：teacher 正样本排名为 `x`、RU 排名为 `y`，teacher 排名超过阈值的 noisy group 权重严格为 0。实现时只能选择 margin 或 rank 中的一种主协议，不得在 validation 结果后切换。

### 4.2 关系目标

蒸馏 query-positive-negative 的相对关系，不拟合 teacher descriptor：

```text
delta_T = positive_cos_T - hardest_negative_cos_T
delta_S = positive_cos_S - hardest_negative_cos_S

L_rel = sum_i w_i * SmoothL1(delta_S_i, stopgrad(delta_T_i))
        / max(sum_i w_i, 1)

L = L_VPR + lambda * L_rel
```

teacher descriptor 已由 global structural feature 与 label-specific weighted features组成，因此 `delta_T` 包含语义布局关系；student 始终是 RGB `DINOv2 -> RU -> BoQ`。训练结束后删除 teacher/cache 路径，student checkpoint 不依赖 segmentation。

`lambda` 不能靠 MSLS recall 扫描。500-step preflight 只允许验证：非零权重覆盖率、teacher/student margin 分布、关系梯度非零且有限、RGB descriptor drift、wrong-pair 干预有效，以及 clean validation forward 与原 RU 接口一致。

## 5. 严格对照与运行顺序

### 5.1 第一阶段三组

| 组别 | 设置 | 回答的问题 |
| --- | --- | --- |
| `matched_continue` | 相同 RU warm start、trainable scope、步数，`lambda=0` | 继续训练本身的变化 |
| `random_gate` | 匹配 aligned 的非零 pair 数和权重直方图，但随机分配给 pair | 收益是否只来自稀疏加权/正则化 |
| `aligned_advantage` | teacher 对真实 pair 的优势决定权重与 relation | 选择性语义关系是否成为候选 |

三组统一使用完整 GSV-Cities、280×280、P=40、K=4、seed 42、3 epochs、同一 RU checkpoint、batch 顺序、optimizer 与 trainable scope。

只有 aligned 在 R@1 上同时比冻结 RU、matched-continue 和 random-gate 多至少 4/740 query，才进入第二阶段。

### 5.2 第二阶段因果对照

`shuffled_teacher` 保留 aligned 的非零权重数和权重多重集，但把 teacher relation 跨地点循环置换。aligned 必须再比它多至少 4 query，才能把收益归因于正确 semantic-layout correspondence。

最终候选门槛：

- overall：aligned 相对 RU、matched、random、shuffled 分别至少 `+4/740` R@1 query；或
- difficult condition：season-full-db 至少 `+1.0 pp`，同时 overall 最多下降 `1/740` query。

通过后才运行 3 seeds、较长训练和其他 aggregator。未通过则不扫描 `lambda`、温度、superclass 数量或 teacher 深度。

## 6. 终止边界

若 Phase 0 没有发现 semantic-only 互补样本，说明当前标签源没有提供可转移的额外地点信息；若 Phase 0 通过而 Phase 1 失败，说明该信息无法通过当前 relation distillation 改善强 RU+BoQ。两种情况下都应停止继续设计 BoQ semantic adapter。

仍可独立开展的工作是按论文原架构完整复现 SemVPR 的 dense LSA + CLS aggregation，或使用作者代码复现 StructVPR++；这些属于论文复现，不应与 AG-SLRD 的 BoQ 适配结果混为一谈。

## 7. Phase 0 实现与运行协议

### 7.1 已实现组件

| 文件 | 作用 |
| --- | --- |
| `src/semantic_layout_cache.py` | 固定 ADE20K-150 → 12 个 VPR superclass 映射；`dynamic` 独立保留；校验 schema、70×70、shape、dtype 与 SHA256 |
| `scripts/build_gsv_semantic_layout_cache.py` | 验证并转换 GSV 70×70 SegFormer cache，不允许把旧 20×20 cache 上采样冒充结构标签 |
| `scripts/cache_msls_ag_slrd_layouts.py` | 按 `18871 database + 740 query` 标准顺序生成 MSLS coarse layout cache |
| `src/models/ag_slrd.py` | 纯 semantic-layout encoder：坐标感知卷积、4×4 空间池化、label-specific pooling 与逐图可学习类别权重 |
| `src/dataloaders/train/semantic_layout.py` | 不读取 RGB 的 GSV place-balanced memmap loader；固定 SHA256 90/10 place split |
| `scripts/train_ag_slrd_semantic_teacher.py` | aligned/shuffled 两组 P=40、K=4、10 epoch 教师训练；实时进度条；只保存预注册 final epoch |
| `scripts/extract_ag_slrd_msls_descriptors.py` | 严格同序提取冻结 RU 与 semantic teacher 描述子并写入 provenance sidecar |
| `scripts/audit_semantic_layout_complementarity.py` | 精确 cosine 检索、四格/oracle/rank/margin 统计和 Phase-0 自动 PASS/FAIL |

两份正式配置为 `config/ag_slrd_semantic_teacher_aligned.yaml` 和
`config/ag_slrd_semantic_teacher_shuffled.yaml`。除输入选择与输出目录外，二者的 seed、split、模型、batch、epoch、优化器完全相同。

### 7.2 四个审计输入

MSLS 审计不把含义不同的 shuffled 对照混为一组，而是固定为：

1. `ru`：冻结 RU+BoQ；
2. `aligned_semantic`：aligned-trained teacher + aligned MSLS layout；
3. `wrong_layout`：同一个 aligned-trained teacher + shuffled MSLS layout；
4. `shuffled_teacher`：shuffled-trained teacher + aligned MSLS layout。

每个 `.npy` 必须带提取器生成的 `.npy.json`，审计会校验 descriptor SHA256、teacher training mode 和 layout selection，防止命令复制或文件重命名后静默比较错组。

默认 PASS 条件为全部同时满足：

- 冻结 RU 精确复现 `675/740`；
- aligned semantic-only 至少 `8/740`；
- aligned positive rank 优于 RU 的 query 比例至少 `5%`；
- aligned 对 RU 错误的补回数量分别高于 wrong-layout 与 shuffled-teacher；
- aligned 的 positive rank 胜负数分别优于两个对照。
- aligned 在固定 SHA256 GSV holdout 的 matched batch-R@1 高于 shuffled-trained teacher。

任一项失败即保持 `PHASE 0 FAIL`，不实现 Phase 1 student，也不扫描 superclass、网络宽度、epoch 或损失权重。

### 7.3 训练机命令

先生成不能由旧 20×20 cache 替代的 GSV 70×70 raw cache：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/cache_gsv_patch_semantics.py \
  --dataset-root datasets/gsv_cities \
  --output .cache/ade20k_patch_labels/segformer_b0_ade20k_grid70 \
  --device cuda:1 \
  --batch-size 16 \
  --num-workers 8 \
  --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
  --revision 489d5cd81a0b59fab9b7ea758d3548ebe99677da \
  --grid-size 70 70 \
  --target-image-size 280 280 \
  --eligible-min-views 4 \
  --flush-every 10 \
  --amp

python scripts/build_gsv_semantic_layout_cache.py \
  --dataset-root datasets/gsv_cities \
  --source-cache .cache/ade20k_patch_labels/segformer_b0_ade20k_grid70 \
  --output .cache/ade20k_semantic_layout/gsv_grid70
```

先跑测试与单 batch smoke，再依次正式训练两组：

```bash
python -m pytest -q \
  tests/test_ag_slrd.py \
  tests/test_ag_slrd_configs.py \
  tests/test_ag_slrd_msls_cache.py

python -u scripts/train_ag_slrd_semantic_teacher.py \
  --config config/ag_slrd_semantic_teacher_aligned.yaml \
  --smoke-test

mkdir -p doc/ag_slrd_runs
set -o pipefail

python -u scripts/train_ag_slrd_semantic_teacher.py \
  --config config/ag_slrd_semantic_teacher_aligned.yaml \
  2>&1 | tee doc/ag_slrd_runs/teacher_aligned_10ep.txt

python -u scripts/train_ag_slrd_semantic_teacher.py \
  --config config/ag_slrd_semantic_teacher_shuffled.yaml \
  2>&1 | tee doc/ag_slrd_runs/teacher_shuffled_10ep.txt
```

最后生成 MSLS layout、提取四份描述子并审计：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/cache_msls_ag_slrd_layouts.py \
  --msls-path datasets/msls-val \
  --output .cache/ade20k_semantic_layout/msls_val_grid70 \
  --device cuda:1 \
  --batch-size 16 \
  --num-workers 8 \
  --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
  --revision 489d5cd81a0b59fab9b7ea758d3548ebe99677da \
  --target-image-size 280 280 \
  --seed 42 \
  --flush-every 10 \
  --amp

export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
export ALIGNED_TEACHER='logs/ag_slrd_semantic_teacher/aligned/final.pt'
export SHUFFLED_TEACHER='logs/ag_slrd_semantic_teacher/shuffled/final.pt'
export LAYOUT_CACHE='.cache/ade20k_semantic_layout/msls_val_grid70'
mkdir -p .cache/ag_slrd_descriptors

python -u scripts/extract_ag_slrd_msls_descriptors.py ru \
  --checkpoint "$RU_CKPT" \
  --msls-path datasets/msls-val \
  --image-size 280 280 \
  --output .cache/ag_slrd_descriptors/ru.npy \
  --device cuda:1 --batch-size 32 --num-workers 8

python -u scripts/extract_ag_slrd_msls_descriptors.py semantic \
  --checkpoint "$ALIGNED_TEACHER" --layout-cache "$LAYOUT_CACHE" \
  --selection aligned \
  --output .cache/ag_slrd_descriptors/aligned.npy \
  --device cuda:1 --batch-size 256 --num-workers 8

python -u scripts/extract_ag_slrd_msls_descriptors.py semantic \
  --checkpoint "$ALIGNED_TEACHER" --layout-cache "$LAYOUT_CACHE" \
  --selection shuffled \
  --output .cache/ag_slrd_descriptors/wrong_layout.npy \
  --device cuda:1 --batch-size 256 --num-workers 8

python -u scripts/extract_ag_slrd_msls_descriptors.py semantic \
  --checkpoint "$SHUFFLED_TEACHER" --layout-cache "$LAYOUT_CACHE" \
  --selection aligned \
  --output .cache/ag_slrd_descriptors/shuffled_teacher.npy \
  --device cuda:1 --batch-size 256 --num-workers 8

python scripts/audit_semantic_layout_complementarity.py \
  --ru-descriptors .cache/ag_slrd_descriptors/ru.npy \
  --aligned-descriptors .cache/ag_slrd_descriptors/aligned.npy \
  --wrong-layout-descriptors .cache/ag_slrd_descriptors/wrong_layout.npy \
  --shuffled-teacher-descriptors .cache/ag_slrd_descriptors/shuffled_teacher.npy \
  --aligned-run logs/ag_slrd_semantic_teacher/aligned/run.json \
  --shuffled-run logs/ag_slrd_semantic_teacher/shuffled/run.json \
  --positives datasets/msls-val/msls_val_gt_25m.npy \
  --num-references 18871 \
  --output doc/ag_slrd_phase0_audit
```

只需下载并归档 `doc/ag_slrd_runs/`、两个 teacher 的 `run.json/history.csv`，以及 `doc/ag_slrd_phase0_audit/`。raw cache 和 descriptor `.npy` 是可再生副产物，不提交 Git。
