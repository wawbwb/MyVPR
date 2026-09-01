# Reliability-Calibrated Semantic Counterfactual Dropout（RSCD-BoQ）

## 1. 目的与边界

已有实验反复出现“aligned 不优于 random/shuffled/wrong control”。这说明错误对应有时只是切断了有害的场景共性，或提供了普通随机正则化；它不能证明随机语义具有地点信息。

RSCD 因此不再把语义作为地点标签、推理特征、蒸馏 target 或 attention bias。SegFormer ADE20K 只在训练前提供局部类别，用于选择训练期反事实遮挡；验证和推理始终执行原始的：

```text
DINOv2 -> pretrained RU gate -> BoQ
```

模型 checkpoint 不新增 RSCD tensor，部署时不需要 SegFormer、缓存或 mask。

## 2. 固定方法

对每个 ADE20K 类别，从完整 GSV-Cities 的现有 20×20 cache 统计：

```text
repeatability = sum_place m(m-1) / sum_place m(n-1)
frequency     = 出现该类的 eligible place / 全部 eligible place
nuisance      = 1 - repeatability * (1 - frequency)
```

其中 `n` 是一个地点的视图数，`m` 是出现该类的视图数。support 小于 100 的类别不参与语义采样，不能因为证据不足而自动成为“高 nuisance 类”。统计 JSON 与 manifest、三个 `.npy` 和所有 city CSV 的 SHA256 绑定。

训练 mask 只使用置信度不低于 0.5、四个 token 同类的非重叠 2×2 block。每张图的共同配额为：

```text
min(15, aligned 合格 block 数, cross-place donor 合格 block 数)
```

所以 active 三组在每张图上都具有完全相同的 token 数、block 数和 block 尺寸；最大覆盖率是 `15 * 4 / 400 = 15%`。

```text
Z_mask = where(mask, stopgrad(spatial_mean(Z)), Z)
g_clean = BoQ(RU(Z))          # no_grad
g_mask  = BoQ(RU(Z_mask))

L = L_MS(g_mask)
  + 0.05 * SmoothL1(pairwise_cos(g_mask),
                    stopgrad(pairwise_cos(g_clean)))
```

两条分支共享一次 DINO forward。RU gate 和 BoQ 分别运行 clean/masked 两次；VPR loss 只作用在 masked 分支。

## 3. 四个严格匹配的实验

| 配置 | 作用 |
| --- | --- |
| `boq_dinov2_rscd_no_mask.yaml` | 相同双分支和继续训练范围，但 `M=0` |
| `boq_dinov2_rscd_uniform_block.yaml` | 同配额 2×2 uniform DropBlock placebo |
| `boq_dinov2_rscd_shuffled_semantic.yaml` | 同配额 cross-place donor semantic mask |
| `boq_dinov2_rscd_aligned_rscd.yaml` | 当前图像 reliability-calibrated semantic mask |

四组除 `distillation.rscd.mode` 外逐字段相同：完整 GSV-Cities、280×280、P=40、K=4、photometric augmentation、seed 42、3 epochs，从同一个 RU checkpoint 继续训练最后两个 DINO block、RU gate 和 BoQ。

学习率固定为 `2e-6`、无 warmup、无 milestone。这是 RU checkpoint 在原 10/20 epoch 两次衰减后的实际学习率，而不是重新用 `2e-4` 热启动。

## 4. 运行顺序

### 4.1 生成类别可靠性统计

```bash
python -u scripts/compute_rscd_class_reliability.py \
  --dataset-root datasets/gsv_cities \
  --cache-dir .cache/ade20k_patch_labels/segformer_b0_ade20k_grid20 \
  --min-confidence 0.5 \
  --min-support 100
```

默认输出：

```text
.cache/ade20k_patch_labels/segformer_b0_ade20k_grid20/rscd_class_stats.json
```

如果该文件已存在，脚本会拒绝覆盖；只有明确重算时才加 `--overwrite`。

### 4.2 512 图离线 matched-mask 审计

```bash
python -u scripts/audit_rscd_masks.py \
  --dataset-root datasets/gsv_cities \
  --cache-dir .cache/ade20k_patch_labels/segformer_b0_ade20k_grid20 \
  --stats-path .cache/ade20k_patch_labels/segformer_b0_ade20k_grid20/rscd_class_stats.json \
  --output doc/rscd_mask_audit \
  --num-images 512 \
  --seed 42 \
  --global-step 0
```

必须得到 `Verdict: PASS`。报告会验证 hash、eligible row、cross-place donor、逐图等配额、完整非重叠 2×2 block、15% 上限、bit-exact determinism，以及 aligned 与两个 placebo 确实不同。

### 4.3 500-step 实现预检

```bash
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
mkdir -p doc/rscd_runs
set -o pipefail

python -u run.py \
  --config config/boq_dinov2_rscd_preflight.yaml \
  --train \
  --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/rscd_runs/preflight_500steps.txt
```

随后审计 TensorBoard：

```bash
python -u scripts/audit_rscd_preflight.py \
  --logdir logs/dinov2_vitb14/BoQ_rscd_aligned_rscd_preflight \
  --output doc/rscd_runs/preflight_audit.json
```

只有 `Verdict: PASS` 才进入正式训练。该审计检查 step 490、mask accounting、三组梯度、descriptor drift、relation loss，以及验证路径的 clean descriptor 误差不超过 `1e-6`。

### 4.4 第一阶段正式筛选

分别运行，不使用易被终端粘贴破坏的 shell 循环：

```bash
python -u run.py \
  --config config/boq_dinov2_rscd_no_mask.yaml \
  --train \
  --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/rscd_runs/no_mask_3ep.txt
```

```bash
python -u run.py \
  --config config/boq_dinov2_rscd_uniform_block.yaml \
  --train \
  --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/rscd_runs/uniform_block_3ep.txt
```

```bash
python -u run.py \
  --config config/boq_dinov2_rscd_aligned_rscd.yaml \
  --train \
  --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/rscd_runs/aligned_rscd_3ep.txt
```

只有 aligned 同时比 RU、no-mask 和 uniform 多命中至少 4/740 个 R@1 query，才运行：

```bash
python -u run.py \
  --config config/boq_dinov2_rscd_shuffled_semantic.yaml \
  --train \
  --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/rscd_runs/shuffled_semantic_3ep.txt
```

## 5. 预注册解释规则

- `aligned > RU/no-mask/uniform/shuffled` 且分别至少多 4 个 R@1 query：语义选择策略成为候选因果收益。
- `uniform > RU/no-mask`，但 aligned 不胜 uniform：可保留为非语义 DropBlock 工程改进，不能声称语义有效。
- `aligned ≈ shuffled > uniform`：区域分布可能有用，但正确图像—语义对应没有被证明。
- 四组均无收益：停止当前数据和 BoQ 下的语义路线，不继续扫权重。
- 另一条候选通道是 season-full-db 至少 `+1.0 pp`，同时 overall 相对 RU 不下降超过 1/740 query。

所有结论先基于相同 seed 的 causal screen；达到候选门槛后才进行 3 seeds 和其他聚合器复现。

## 6. 正式结果与停止决定（2026-09-01）

### 6.1 运行完整性

离线 matched-mask 审计与 500-step 梯度/推理合同审计均为 `PASS`。这说明正式负结果不能用“语义 mask 没有真正执行”“梯度断开”或“验证路径错误地保留了 mask”解释。

以下结果来自训练机的 checkpoint 清单；原始三组训练日志尚未同步回本机仓库，清单已原样归档到 `doc/rscd_runs/formal_checkpoint_inventory.txt`。表中的命中数按 MSLS-val 的 740 个 query 换算。

| 组别 | Epoch | R@1 | R@1 命中 | R@5 | R@5 命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冻结 RU source | 26 | **91.22** | **675/740** | 95.14 | 704/740 |
| no-mask | 0 | 90.81 | 672/740 | 95.14 | 704/740 |
| no-mask | 1 | **90.95** | **673/740** | **95.27** | **705/740** |
| no-mask | 2 | 90.68 | 671/740 | 95.00 | 703/740 |
| uniform-block | 0 | **90.41** | **669/740** | **94.86** | **702/740** |
| uniform-block | 1 | 90.41 | 669/740 | 94.73 | 701/740 |
| uniform-block | 2 | 90.41 | 669/740 | 94.73 | 701/740 |
| aligned-RSCD | 0 | **90.95** | **673/740** | **94.86** | **702/740** |
| aligned-RSCD | 1 | 90.68 | 671/740 | 94.86 | 702/740 |
| aligned-RSCD | 2 | 90.27 | 668/740 | 95.00 | 703/740 |

每组以 R@1 为主指标选择 checkpoint；R@1 持平时才用 R@5 破同分。因此 no-mask 选 epoch 1、uniform-block 选 epoch 0、aligned-RSCD 选 epoch 0。

### 6.2 预注册判定

| 比较 | R@1 差值 | 命中数差值 | 解释 |
| --- | ---: | ---: | --- |
| aligned vs 冻结 RU | -0.27 pp | -2 | 没有超过起点模型 |
| aligned vs no-mask | +0.00 pp | 0 | 没有超过匹配的继续训练对照 |
| aligned vs uniform-block | +0.54 pp | +4 | 语义遮挡比均匀 DropBlock 少破坏 4 个 query |

结论为：

```text
RSCD-BoQ
Implementation / contract: PASS
Semantic causal screen: FAIL
Overall status: COMPLETED / FAIL
```

aligned 只胜过 uniform-block，却与 no-mask 持平并低于冻结 RU；这最多说明数据驱动语义遮挡比同配额均匀遮挡破坏更小，不能证明语义带来检索收益。aligned 的 R@1 还从 epoch 0 的 90.95% 连续降到 90.68% 和 90.27%，没有延长训练的证据。

按第 4.4 节与第 5 节的预注册规则：

- 不运行 `shuffled_semantic`；aligned 没有同时比 RU、no-mask、uniform 多至少 4 个 query；
- 不运行 condition evaluation；overall 相对 RU 已下降 2 query，超过候选通道允许的 1 query；
- 不增加 seed、不延长 epoch、不复制到 SALAD，也不扫描 mask coverage、relation-loss 权重或学习率。

该结论否定的是当前 GSV-Cities → MSLS、RU+BoQ 设置下的 RSCD 实现，不是“训练期语义扰动在理论上永远无效”。
