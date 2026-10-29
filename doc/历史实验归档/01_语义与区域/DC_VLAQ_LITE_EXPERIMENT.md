# DINO-anchor Residual CLIP Fusion → VLAQ（DC-VLAQ-lite）

更新日期：2026-08-31

状态：**Phase A 已完成，机制判定 FAIL；Phase B/VLAQ 未实现并按预注册规则终止。本文档作为归档保留。**

## 1. 研究问题

前序实验反复把 CLIP 当作训练期 teacher、target、mask、样本选择器或调制条件，推理描述子仍主要由 DINO/RU/BoQ 产生。它们均未通过 aligned 相对无语义与错误语义对照的因果判据。尤其 Crop-CLS FiLM 的 aligned 为 90.95%，低于 RU 91.22%，也低于 wrong-place 91.08%。

本方案改问一个更直接的问题：**冻结 CLIP 的 dense token 是否包含能在 VPR loss 下直接补充 DINO token 的信息？** CLIP 不再提供额外监督，而是成为描述子前向路径的一部分。

主要依据是 [DC-VLAQ: Query–Residual Aggregation for Robust Visual Place Recognition](https://arxiv.org/html/2601.12729)。该文在 DINOv2 空间中残差注入 CLIP，并用 VLAQ 聚合；其消融报告 residual fusion 优于 addition、cross-attention 和 FiLM，且 DINO+CLIP 优于 DINO-only。它是 2026-01-19 发布的 arXiv v1 预印本，尚未同行评审，也未发现可直接复用并审计的官方代码，所以本项目只借鉴机制，并以严格 matched controls 自行验证，不直接接受论文结论。

## 2. 与已有路线的区别

| 项目 | 已失败的 Crop-CLS FiLM | 本方案 |
| --- | --- | --- |
| CLIP 角色 | 训练期 crop teacher | 训练和推理中的 dense 第二特征流 |
| 优化目标 | VPR loss + crop cosine | 仅 VPR loss |
| 语义注入 | bottleneck 生成乘性 FiLM | DINO-anchor additive residual |
| 语义空间 | 稀疏 2×2 region | 同图 dense patch grid |
| 推理成本 | 不运行 CLIP | 必须运行冻结 CLIP |
| 第一阶段聚合 | BoQ | 仍用 BoQ，隔离融合变量 |
| 第二阶段聚合 | — | 只有融合通过后才实现 VLAQ |

该方法不是后处理：CLIP patch token 在 global descriptor 形成前参与每张 query/database 图像的前向计算，语义分支被移除后描述子会改变。

当前实现使用 OpenCLIP `ViT-B-16/openai` 的 14×14 patch token，经 CLIP 自身 joint-space projection 得到 512D 特征，再用双线性插值对齐 DINO 的 20×20 网格。冻结 CLIP 由非注册 provider 按配置重建，不进入 student checkpoint；可训练的 `P_C/W_zero` 则进入 checkpoint，并由条件评测 loader 严格恢复。

## 3. Phase A：Residual-CLIP BoQ

### 3.1 固定计算图

```text
280×280 photometric image
       +--> DINOv2 --> raw patch tokens D_raw [20×20×768] ----+
       |                                                       |
       +--> frozen CLIP --> patch tokens C --> P_C --> 20×20 --+--> residual adapter
                                                               |
Z = D_raw + W_zero(P_C(norm(C)) - norm(D_raw)) <----------------+
       |
pretrained RU gate --> pretrained BoQ --> global descriptor --> Multi-Similarity loss
```

仓库适配公式固定为：

```text
R = P_C(L2Norm(C)) - L2Norm(D_raw)
Z = D_raw + W_zero(R)
```

其中 `P_C` 把实际 CLIP patch 维度映射为 768D；DINO 的归一化支路固定为 identity，不再增加一个可训练 `P_D`；`W_zero` 的权重和 bias 为零。使用 `D_raw` 作主干锚点是对论文公式的工程适配：它保证初始 `Z == D_raw`，从而能逐元素复现 RU；不应把归一化 DINO token 直接替换原始 token，否则 zero-start 并不等于历史基线。需要注意，`W_zero` 训练后仍可能只利用固定的 `-L2Norm(D_raw)` 项，因此“aligned 胜 RU”本身不能证明 CLIP 有效，必须再胜过 global-only、wrong-region 与 wrong-place 三组同容量控制。

固定约束：

- DINO 与 CLIP 看同一个 photometric augmentation 后的图像，避免 clean cache 与训练图的空间错位；
- CLIP patch grid 用一个确定性插值对齐到 20×20，不做逐图单位方差化；
- CLIP 全程冻结，但保留在训练和推理；
- 不使用 ADE20K、crop-CLS loss、confidence、mask、semantic gate、attention bias、hard negative 或 text prompt；
- 第一阶段冻结 DINO、RU gate 与 BoQ，只训练 `P_C/W_zero`，避免把 backbone/aggregator 再训练带来的波动误认为语义收益；
- 使用当前 RU checkpoint、full GSV-Cities、280×280、P=40、K=4、seed 42 和相同 photometric augmentation。

### 3.2 Preflight

先运行一个固定 500 optimizer-step、无 checkpoint 的 preflight。必须同时满足：

1. 严格验证 RU checkpoint SHA256 和逐 key 载入；
2. step 0 使用 fp32 比较 residual enabled/bypass 描述子，最大绝对误差 `<=1e-6`；
3. `P_C`、`W_zero` 的梯度出现非零且全部有限；
4. residual RMS、descriptor drift 在训练后非零且有限；
5. CLIP 参数梯度始终为空，eval mode 不被 Lightning 切回 train；
6. aligned、wrong-region、wrong-place 变体在进入 adapter 前具有相同张量形状、范数处理和插值路径。

这里不再要求 aligned token 与 wrong-place token 存在任意人为 cosine margin；检索效果由正式 matched run 判断。

### 3.3 正式短筛与因果对照

第一轮只运行 `aligned_residual` 3 epochs。若它未达到下面任一候选门槛，立即停止：

- MSLS overall R@1 相对冻结 RU 至少 `+0.5 pp`，即至少多 4/740 query；或
- season-full-db R@1 至少 `+1.0 pp`，同时 overall 不比 RU 低超过 `0.2 pp`。

通过后才各运行 3 epochs 的三个 matched controls：

| 对照 | 操作 | 排除的替代解释 |
| --- | --- | --- |
| `global_only` | 每张图把同一个 CLIP global token 复制到所有 patch | 收益只来自全局场景语义，而非局部结构 |
| `wrong_region` | 对 CLIP grid 使用固定、非恒等、可逆的空间置换 | 局部位置对应并不重要 |
| `wrong_place` | 在 batch 的不同 place 间稳定轮换整张 CLIP grid | 正确图像/地点语义并不重要 |

四组必须拥有相同的 CLIP forward、投影与 adapter 参数量、数据顺序、优化器、步数和初始化。不得用“关闭 CLIP forward”作为 architecture control，因为那会改变计算路径和归一化统计。

三个破坏操作只作用于**训练期**。验证、条件评测和部署时，四种 checkpoint 都使用当前图像自己的 aligned CLIP grid；因此描述子始终是单张图像的确定函数，不会因验证 batch 的组成或顺序改变。这里检验的是“学习到的收益是否依赖训练期正确的局部/地点对应”，不是在部署时向模型输入别的地点。

最终进入 Phase B 的门槛：

1. aligned 相对 RU 满足上述 overall 或 season 候选门槛；
2. aligned R@1 相对 `global_only`、`wrong_region`、`wrong_place` **分别**至少高 `0.5 pp`；
3. 结果不是通过事后选择不同 epoch 获得：统一使用首次达到各组最高 MSLS R@1 的 checkpoint，并同时报告完整逐 epoch 曲线和命中 query ID 差集。

任一条件失败即归档为 FAIL，不扫 residual scale、CLIP layer、插值方式、学习率或训练 epoch。

### 3.4 已实现文件与执行顺序

- 核心融合、冻结 provider、严格 RU warm-start：`src/models/residual_clip_fusion.py`；
- DINO 最终 patch map 接入：`src/models/backbones/dinov2.py`；
- zero-start、梯度、残差、descriptor drift 与 CLIP 冻结审计：`src/core/vpr_framework.py`；
- 运行协议和 fail-closed 配置校验：`run.py`；
- preflight 审计：`scripts/audit_residual_clip_preflight.py`；
- 配置：`config/boq_dinov2_residual_clip_{preflight,aligned,global_only,wrong_region,wrong_place}.yaml`；
- 条件评测：`scripts/eval_condition_robustness.py` 已支持严格恢复 residual branch，并在评测时重建冻结 CLIP；
- 契约测试：`tests/test_residual_clip*.py` 与 `tests/test_condition_eval_loader.py`。

固定执行顺序为：测试 → 500-step preflight → aligned 3 epochs → 门槛判断 → 仅在通过时运行三个 controls。不得跳过 aligned 停止点直接批量训练 controls。

### 3.5 训练机结果与最终判定（2026-08-31）

preflight 的 zero-start、梯度、residual RMS、descriptor drift 和 CLIP 冻结检查均通过。正式训练中，aligned、global-only、wrong-region、wrong-place 四组从同一个 RU checkpoint 启动，只训练 983,808 个 residual 参数并各跑满 3 epochs；每组每个 epoch 的 MSLS-val 都为 R@1/R@5/R@10/R@15 = 91.22/95.14/96.08/96.49，与冻结 RU 持平。

aligned checkpoint 的 full-database paired intervention 进一步得到：

| 变体 | R@1 | R@5 | R@10 | R@15 | aligned 净胜 R@1 query |
| --- | ---: | ---: | ---: | ---: | ---: |
| bypass | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| aligned | 91.22 | 95.14 | 96.08 | 96.49 | — |
| zero-CLIP | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| global-only | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| wrong-region | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| wrong-image cross-city | 91.22 | 95.14 | 96.08 | 96.49 | 0 |

aligned 相对上述五个对照分别改变了 4、4、2、2、5 个 query 的 top-1 reference，但没有改变任何 R@1 正误，说明分支已连接、排序会变化，却没有可用的检索增益。night-full-db 与 RU 完全相同；season-full-db 仅多 2/988 个 R@1 query（88.87% 对 88.66%），低于 +1.0 pp 门槛。

为排除“方向正确但残差太小”，又将 checkpoint 中已经学习的 CLIP-only residual 以 `gamma={0.5,1,2,4}` 推理。所有 gamma 和 aligned/global-only/wrong-region/wrong-image control 的 R@1 均为 675/740，R@5 均为 704/740；`gamma=4` 时 aligned 与 wrong-region 的 R@10 反而各少 1 query。筛选器输出 `NO_CANDIDATE`，`best_observed_gamma=0.5` 只是并列后的排序值。

最终判定：**Phase A FAIL。** 当前 learned residual 不是单纯权重不足；不再扫 residual scale、CLIP layer、学习率或 epoch，也不实现 Phase B/VLAQ。该结论仅适用于本项目的 `ViT-B-16/openai + DINO-anchor residual + frozen RU/BoQ` 适配，不外推为所有 CLIP/VPR 融合都无效。

## 4. Phase B：Matched VLAQ（已终止）

原计划只有 Phase A 通过后才实现 VLAQ，避免一次同时改变融合和聚合而无法归因。Phase A 已失败，因此本节不再执行；下面只保留原始设计。按照论文思想，learned query 先产生 token assignment，再聚合 token 相对 query center 的残差：

```text
s_jk     = q_k^T z_j / sqrt(d)
alpha_jk = softmax_j(s_jk)
v_k      = sum_j alpha_jk * (z_j - q_k)
g        = L2Norm(concat_k(v_k))
```

最小实验矩阵：

| 模型 | token 输入 | 目的 |
| --- | --- | --- |
| `dino_vlaq` | DINO-only | 隔离 VLAQ 相对 BoQ 的收益 |
| `residual_clip_vlaq` | aligned DINO+CLIP residual | 测量 CLIP 在同一 VLAQ 下的额外收益 |
| `wrong_place_vlaq` | wrong-place residual | 验证收益需要正确语义对应 |

三组使用同一 descriptor 维度、VLAQ block/query 数、trainable scope、初始化和 5 epochs。只有 `residual_clip_vlaq` 同时胜过 `dino_vlaq` 与 `wrong_place_vlaq` 至少 0.5 pp，且胜过历史 RU，才进入 3 seeds；3 seeds 的均值与置信区间通过后才考虑 40 epochs。

论文使用 2 个 VLAQ block、每个 64 queries、P=110/K=4、训练 40 epochs、280 训练和 322 评测。本项目第一轮不得直接照搬其 P=110 或 322 评测，因为这会破坏与 RU 的 matched comparison；这些设置只能在因果筛选通过后作为独立扩展。

## 5. 成本与实际失败解释

- 该设计没有 teacher target 冲突，zero-start 也严格复现了 RU，因此本次 FAIL 不是由初始破坏基线造成的。
- 每张 database/query 图都要额外执行冻结 CLIP，推理、显存和建库成本明显增加；在没有精度收益时不存在部署价值。
- aligned 不胜 RU，说明当前 CLIP token 在该数据、分辨率和 RU+BoQ 几何下没有表现出足够互补信息。
- paired intervention 证明 adapter 会改变少量排序，但 aligned 不胜任何错误对照；gamma sweep 又排除了 `0.5–4` 范围内“仅仅权重不足”的解释。

## 6. 归档时的备选方向

如果部署明确不允许保留 CLIP，备选为 **reliability-gated semantic-layout relational distillation**：参考 [StructVPR（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_StructVPR_Distill_Structural_Knowledge_With_Weighting_Samples_for_Visual_Place_CVPR_2023_paper.html) 的结构知识/样本加权，以及 [DistilVPR（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28905) 的跨模态关系蒸馏，只在可靠样本上匹配 batch 内相对关系，不做逐 patch feature 拟合。推理仍只保留 DINO+BoQ。

在 Residual-CLIP 运行前，该备选的优先级低于直接融合。Phase A 失败后也不直接启动新的蒸馏，而是先测试总结文档第 10 节的 RSCD-BoQ：把可靠性用于训练期结构化反事实扰动，并用严格等覆盖的随机 placebo 隔离普通正则化。关系蒸馏只作为其中所有组共享的小权重一致性项，不再单独充当语义主监督。

## 7. 当前执行决定

1. Phase A residual core、preflight、四组正式训练、paired intervention、condition audit 和 gamma sweep 均已完成；
2. aligned 没有净胜 RU 或任何错误语义对照，Phase A 归档为 FAIL；
3. 不再运行 40 epochs、多 seed 或更多 residual 超参数，不实现 VLAQ；
4. 下一候选改为训练期 Reliability-Calibrated Semantic Counterfactual Dropout，方案和 matched controls 见 `doc/SEMANTIC_VPR_EXPERIMENT_SUMMARY.md` 第 10 节。

## 8. 历史训练机 runbook（已完成，不要重复运行）

以下命令从仓库根目录运行；`RU_CKPT` 必须是已审计的 RU checkpoint：

```bash
conda activate VPR
cd ~/workspace/OpenVPRLab

export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
export EXPECTED_RU_SHA='38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e'
export HF_ENDPOINT='https://hf-mirror.com'
test -f "$RU_CKPT"
echo "$EXPECTED_RU_SHA  $RU_CKPT" | sha256sum -c -
```

先运行契约测试：

```bash
python -m pytest -q \
  tests/test_residual_clip.py \
  tests/test_residual_clip_dinov2.py \
  tests/test_residual_clip_configs.py \
  tests/test_residual_clip_audit.py \
  tests/test_condition_eval_loader.py
```

再运行并审计 500-step preflight：

```bash
mkdir -p doc/residual_clip_runs
set -o pipefail
python run.py \
  --config config/boq_dinov2_residual_clip_preflight.yaml \
  --train \
  --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/residual_clip_runs/preflight_500steps.txt

python scripts/audit_residual_clip_preflight.py \
  --logdir logs/dinov2_vitb14/BoQ_residual_clip_aligned_preflight \
  --output doc/residual_clip_runs/preflight_audit.json
```

审计必须输出 `Verdict: PASS`。随后只运行 aligned：

```bash
python run.py \
  --config config/boq_dinov2_residual_clip_aligned.yaml \
  --train \
  --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/residual_clip_runs/aligned_3ep.txt

find logs/dinov2_vitb14/BoQ_residual_clip_aligned \
  -type f -name '*.ckpt' -print
```

overall 首个候选门槛是最佳 R@1 至少 `91.76%`（相对 RU 多命中至少 4/740 query）。若未达到但 overall 不低于 `91.02%`，可把最佳 checkpoint 设为 `ALIGNED_CKPT` 后运行一次已修正 full-database 条件评测，检查 season 是否达到相对 RU `+1.0 pp`：

```bash
export ALIGNED_CKPT='替换为上一步列出的最佳 aligned checkpoint'
python scripts/eval_condition_robustness.py \
  --my-ckpt "$ALIGNED_CKPT" \
  --origin-ckpt "$RU_CKPT" \
  --my-name 'Residual-CLIP aligned' \
  --origin-name 'Frozen RU' \
  --msls-path datasets/msls-val \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda:1 \
  --image-size 280 280 \
  2>&1 | tee doc/residual_clip_runs/aligned_condition_eval.txt
```

只有 aligned 通过 overall 或 season 候选门槛，才运行 controls：

```bash
for MODE in global_only wrong_region wrong_place; do
  python run.py \
    --config "config/boq_dinov2_residual_clip_${MODE}.yaml" \
    --train \
    --init-checkpoint "$RU_CKPT" \
    2>&1 | tee "doc/residual_clip_runs/${MODE}_3ep.txt"
done
```

最后要求 aligned 相对每个 control 的最佳 MSLS R@1 都至少高 `0.5 pp`；否则 Phase A 归档为 FAIL，不实现 VLAQ。
