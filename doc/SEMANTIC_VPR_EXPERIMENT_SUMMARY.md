# 语义增强 VPR 实验总结

更新日期：2026-08-31

## 1. 总结结论

当前结果允许继续研究“语义如何进入 VPR”，但不支持继续微调已经失败的注入方式。到目前为止，没有一种语义方案同时满足以下两个条件：

1. aligned semantic 主实验优于匹配的无语义基线；
2. aligned semantic 主实验优于 shuffled/random 等破坏语义对应关系的对照。

因此，现有实验不能证明“语义在 VPR 中理论上无效”，但已经较有力地否定了本项目中这些具体实现：全局 CLIP 描述子拟合、CLIP 单图空间 attention 拟合、标量 patch reliability、CLIP 语义负样本/正样本选择、CLIP affinity 平滑区域目标，以及固定动态类别负偏置。继续只扫 `alpha`、`lambda`、`beta` 或温度，信息收益很低。

Query-conditioned Semantic BoQ 已完成 10-epoch 首筛。aligned 在 MSLS-val 的最佳 R@1 为 91.22%，与 architecture-only 和冻结 RU 均持平；在 night-full-db 与 season-full-db 上，R@1/R@5/R@10 相对 matched architecture-only 全部相差 0 query。该实现没有通过预注册门槛，按规则停止，不运行 shuffled/random、多 seed、40 epochs、SALAD 复制或 `lambda`/bias-scale 扫描。

参考 SemVPR 设计的 **Crop-CLS 连续局部语义蒸馏 + patch×channel FiLM** 也已完成四组 seed-42、5-epoch 正式训练。aligned 最佳 R@1 为 90.95%，比 architecture-only 高 0.54 pp（4/740 query），但仍比冻结 RU 低 0.27 pp（2 query），并比 wrong-place 低 0.13 pp（1 query）。因此训练机制正常，且局部配对比 wrong-region 有一定作用，但正确地点语义并非取得结果所必需，语义因果归因判定为 **FAIL**。按修订后的预注册停止规则，本应在 aligned 低于 RU 0.2 pp 时停止；已经完成的 wrong-region/wrong-place 只能作为超出停止点的探索性诊断，不进入正式通过判据。完整结果见第 8 节和 `doc/CROP_CLS_SEMANTIC_FILM_BOQ.md`。

**DINO-anchor Residual CLIP Fusion → VLAQ（DC-VLAQ-lite）** 的 Phase A 现已完成并判定为 **FAIL**。aligned、bypass、zero-CLIP、global-only、wrong-region 和 wrong-image 在 MSLS-val 上均为 R@1 91.22%；把已学习 CLIP 残差的推理倍率从 0.5 扫到 4，R@1/R@5 仍完全不变，且 `gamma=4` 时 aligned 的 R@10 反而少 1 query。因此问题不是“语义权重还不够大”，不再实现 Phase B/VLAQ。完整结果见第 9 节。

从所有随机/打乱对照中得到的新启示不是“随机语义优于真实语义”，而是：**现有 aligned teacher 往往施加了与地点实例判别不一致的偏置；破坏对应关系会解除这一偏置，并可能退化成普通随机正则化。** 大多数 corrupted control 只胜过 aligned，并未稳定胜过匹配视觉基线。下一候选因此改为 **Reliability-Calibrated Semantic Counterfactual Dropout（RSCD-BoQ）**：语义只决定训练时优先扰动哪些不稳定/高频区域，不向描述子注入语义特征；用等覆盖、等形状的 uniform/shuffled dropout 作主对照。只有 aligned semantic policy 同时胜过视觉基线和这些随机对照，才能归因为语义。见第 10 节。

除特别说明外，下面的结果主要来自 seed 42 的单次运行。单 seed 的负结果足以按预注册规则停止明显失败的路线，但不足以支撑小幅正收益的论文结论。

## 2. 基线与结果口径

| 系列 | 模型/设置 | MSLS-val 最佳 R@1 | R@5 | R@10 | 作用 |
| --- | --- | ---: | ---: | ---: | --- |
| MixVPR | ResNet50 + MixVPR（A2） | 87.84 | 92.97 | 94.32 | MixVPR 匹配基线 |
| DINOv2-BoQ | 原始 RandAugment | 91.22 | 95.68 | 96.22 | BoQ 历史结果，不是 semantic-region 的直接消融基线 |
| DINOv2-SALAD | 原始训练 | 91.08 | 95.68 | 96.49 | 聚合器参考 |
| DINOv2-BoQ | photometric-only | 90.27 | — | — | semantic-region 的直接增广匹配基线 |
| DINOv2-BoQ | repeatability+uniqueness（RU） | 91.22 | — | — | 后续局部语义实验的最强匹配无语义基线 |

semantic-region 使用缓存的空间网格，几何 RandAugment 会破坏网格对齐，因此应与 photometric-only 和 RU 对照比较，不能把原始 RandAugment BoQ 当作唯一直接基线。MSLS-val 有 740 个 query，约 0.14 个百分点只对应一个 query；小于这一量级的变化不应解释为稳定收益。

## 3. MixVPR 阶段尝试

### 3.1 全局 CLIP 描述子蒸馏

方法是在训练期让 VPR 全局描述子拟合冻结 CLIP CLS 表示，推理期移除 CLIP。

| 实验 | 语义损失权重 | MSLS-val 最佳 R@1 | 相对 A2 |
| --- | ---: | ---: | ---: |
| B1 / B2 | 0.05 | 87.03 | -0.81 pp |
| B2 | 0.01 | 87.30 | -0.54 pp |
| B2 | 0.20 | 86.49 | -1.35 pp |

`B2_0.1.md` 没有完整训练指标，不能把它当作有效结果。已完成的权重均未超过 A2，说明 CLIP 全局图像语义与地点实例判别目标不够一致；继续在这一损失上细扫权重没有依据。

### 3.2 单图局部 attention 蒸馏及其正则项

该系列让一个学生空间 gate 拟合冻结 CLIP 的单图 patch attention。C0 只增加 gate 容量，不使用 CLIP，是必要的 architecture-only 对照。

| 实验 | 方法 | MSLS-val 最佳 R@1 | 相对 A2 |
| --- | --- | ---: | ---: |
| C0 | spatial gate，无 CLIP | 87.57 | -0.27 pp |
| C2 | CLIP attention KL，`lambda=0.1` | 86.35 | -1.49 pp |
| C3 | CLIP attention KL，`lambda=0.01` | 86.89 | -0.95 pp |
| C3 | CLIP attention KL，`lambda=0.05` | 86.89 | -0.95 pp |
| C3 | CLIP attention KL，`lambda=0.10` | 86.35 | -1.49 pp |
| C3 | CLIP attention KL，`lambda=0.20` | 86.35 | -1.49 pp |
| D1 | 再加跨视图 consistency | 86.35 | -1.49 pp |
| D2 | 再加 divergence 正则 | 86.35 | -1.49 pp |

`C3_0.01.txt` 与 `C3_0.05.txt` 是采用的完整运行记录；同名 `.md` 是较早记录。最小权重仍落后于无 CLIP 的 C0，故问题不只是语义损失过强。

### 3.3 VPR-conditioned semantic reliability

方法先用同地点跨视图正匹配与不同地点难负样本构造 patch reliability，再用 CLIP patch 表示辅助匹配，监督学生空间分布。最佳 MSLS R@1 为 86.49；后 20 epoch 平均为 86.23，而 C0 为 87.20，平均低 0.96 pp。Pitts30k 后 20 epoch 平均基本不变（93.35 对 93.35）。这说明退化在 MSLS 上持续存在，并非仅是某个 checkpoint 的偶然波动。

### 3.4 CLIP semantic alias 负样本挖掘

方法不让学生拟合 CLIP，而是让 CLIP 在 batch 内选择“语义相似但地点不同”的难负地点；推理结构与 A2 相同。random 使用同一有效候选池随机选负样本，shuffled 保留分数分布但破坏地点对应。

| 选择方式 | MSLS-val 最佳 R@1 | R@5 | 相对 A2 |
| --- | ---: | ---: | ---: |
| aligned CLIP | 87.43 | 92.84 | -0.41 pp |
| random | 88.11 | 93.51 | +0.27 pp |
| shuffled CLIP | 87.30 | 93.11 | -0.54 pp |

random 优于 aligned CLIP，因而即使附加关系损失偶有收益，也不能归因于 CLIP 语义。该路线未通过语义归因门槛。

### 3.5 CLIP semantic positive 正样本选择

方法在同一地点的多视图中，用 CLIP 选择语义差异最大的正样本对并施加辅助约束；另设 random、shuffled 和 student 选择。20-epoch、seed-42 筛选结果为：random 87.57、shuffled 87.30、student 87.43、aligned CLIP 87.03。aligned CLIP 最差，故没有进入更长训练或超参数扫描。

## 4. DINOv2-BoQ local semantic-region

该方法先以同地点跨视图 DINO token repeatability 和 batch 内不同地点 uniqueness 构造视觉可靠性，再用冻结 CLIP 的局部语义 affinity 在高置信区域内传播可靠性。学生 gate 为 `1 + 0.2 * tanh(score)`，零初始化，推理时不需要 CLIP。

六组实验均使用完整 GSV-Cities、280×280、P=40、K=4、40 epochs、seed 42。

| 配置 | 最佳 R@1 | epoch 20–39 平均 | 最终 R@1 |
| --- | ---: | ---: | ---: |
| photometric | 90.27 | 89.925 | 89.86 |
| repeatability only | 90.95 | 90.604 | 90.81 |
| repeatability+uniqueness（RU） | 91.22 | 90.562 | 90.54 |
| semantic only | 90.54 | 89.893 | 90.14 |
| shuffled semantics | 91.22 | 90.983 | 90.95 |
| full aligned semantics | 89.86 | 89.709 | 89.73 |

full aligned 比 RU 和 shuffled 的最佳值均低 1.36 pp，且低于 photometric baseline，明确未通过预注册判据。训练流程本身成功，但语义假设的这一实现失败。

### 4.1 传播目标诊断

固定同一 batch 和同一份 raw DINO 特征后，aligned/shuffled 诊断发现：

- full/shuffled 会走 `base10 -> base14 -> out10`，RU 则直接使用 10×10；即使语义 mask 全空，round trip 相对 RU 的 target MAE 仍为 0.10062，空间相关系数为 0.9722，存在插值混杂；
- 当前逐图单位方差化让 dense aligned target 约 36% 饱和；改为 RU-preserving additive + center-only 后约为 4%；
- `confidence >= 0.6` 时 patch coverage 为 12.64%，保留 aligned delta 的 19.94%，相对 RU 的 target change 为 0.01583，aligned-shuffled target difference 为 0.02734，饱和率为 5.75%；
- `confidence >= 0.7` 已接近 semantic-off，160 张图中有 3 张没有 active patch；
- retained mask 仍散落在道路、植被、普通立面和车辆等区域，未显示稳定的地点判别指向。

硬稀疏和移除单位方差化能修复目标构造的机械问题，但不能把离线 target 变化当成检索提升证据。鉴于原 aligned 训练已经显著失败，不建议仅为 `0.5/0.6/0.7` 再完整重训。

## 5. 动态类别负先验

### 5.1 mask 教师与数据协议

使用 torchvision DeepLabV3-MobileNetV3 的 VOC 标签，把 person、bicycle、car、motorbike、bus、train 的 hard-argmax 像素面积汇聚到 20×20 patch。修复 MSLS 条件 query 协议后，cache 覆盖 18,871 个 reference 和 1,694 个 query，共 20,565 张图；其中 standard query 740、night 55、winter2summer 586、summer2winter 402。

动态区域平均覆盖 6.303%，8.46% 图像完全没有动态 mask；car 单类约占 5.260%。VOC 标签缺少 rider 和 truck，因此这只是粗粒度动态先验。条件结果使用“条件 query + 完整标准 database”的自定义审计协议，不应表述为官方 MSLS condition benchmark。

### 5.2 冻结 checkpoint 的 attention-logit 干预

在 RU checkpoint 上以 `beta=0.5` 把 aligned mask 作为 BoQ cross-attention 的负 bias，同时计算 zero-bias、shuffled 和 exact-value random 对照。

| 数据集 | baseline R@1 | aligned | shuffled | random | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| MSLS-val，740 queries | 91.22 | 91.08（-1 query） | 90.68（-4） | 91.08（-1） | aligned 不优于 zero/random |
| night-full-db，55 | 89.09 | 89.09 | 89.09 | 89.09 | 无变化 |
| winter2summer-full-db，586 | 90.27 | 90.27 | 90.61 | 90.27 | shuffled 更高 2 queries |
| summer2winter-full-db，402 | 86.32 | 86.32 | 86.57 | 86.57 | controls 更高 1 query |

screen verdict 为 FAIL。固定、类别共享的动态负先验没有改善检索；不应继续扫 `beta`。

### 5.3 baseline BoQ 原始 attention mass

审计的 baseline 实际为 `DINO -> 已训练 RU semantic_region_gate -> BoQ(attention_bias=None)`，不是 photometric-only 模型。全 20,565 张图的 consensus attention：动态面积 0.06303、aligned attention mass 0.06381、aligned enrichment 1.01245；shuffled enrichment 1.19018，random enrichment 0.99889。standard 740 queries 上 aligned enrichment 为 1.02845，仍低于 shuffled 的 1.20069。

12 个 attention head 中 aligned 胜过 shuffled 的数量为 0；128 个 learned query slot 中仅 6 个胜过 shuffled。attention 本身较分散：归一化熵约 0.938、effective patch fraction 约 0.694、top-10% patch mass 约 0.306、top-20% 约 0.496。

因此，没有证据表明 RU baseline 系统性地过度关注图中的动态物体。aligned 略高于 random 可由车辆和 attention 都偏向图像下部解释，而 role/condition-preserving shuffled 更高，进一步否定了“图像特定动态物体吸走 attention”这一共同负先验。

## 6. 已排除的结论与仍开放的问题

现有结果支持：

- 本项目中已测试的 CLIP 全局/局部直接拟合和 affinity propagation 不能稳定提高 MSLS；
- aligned 不胜 shuffled/random 时，不能把变化归因于语义；
- semantic-region 的插值 round trip 与逐图单位方差化是实现层面的真实混杂，但修复它们只说明 target 更干净，不保证 retrieval 会转正；
- BoQ query/head 的偏好高度异质，固定类别、所有 query 共用的负 bias 过于粗糙；但本次 query-specific ADE20K class bias 同样没有改善检索，因此“改成 query-specific”本身也不是充分条件；
- 当前证据否定了“冻结 RU/BoQ，仅靠 hard-class attention-logit adapter”、本项目的“稀疏 crop-CLS 蒸馏 + FiLM + RU/BoQ 联合微调”，以及“把冻结 CLIP 保留为推理特征流、零起点残差接入现有 RU+BoQ”三种实现；尚未检验的是 SemVPR 的完整 dense LSA teacher、原论文 CLS aggregation，以及语义只作为训练期结构化扰动策略的路线。

现有结果不支持：

- “语义理论上不能用于 VPR”；
- “CLIP 完全没有空间信息”；
- “所有动态物体都应该被抑制”；
- “只需减小权重就能救回现有方案”。

## 7. Query-conditioned Semantic BoQ：已完成首筛，FAIL

该路线让 SegFormer ADE20K hard label 监督轻量语义学生头，再由每个 BoQ learned query 学习独立的类别偏好，并把 query-specific signed bias 加到 cross-attention logits。它不是后处理，但 DINOv2、RU gate 与原 BoQ 全部冻结，仅训练 semantic head 和 query-to-class adapter。

### 7.1 数据与运行完整性

- 教师为 150 类 SegFormer ADE20K，固定 revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da`；cache manifest 为 complete；
- cache confidence coverage：`>=0.5` 为 89.0344%，`>=0.6` 为 81.0140%，`>=0.7` 为 73.0192%；
- cache SHA256 前缀：`labels=4841aa2cca57...`、`confidence=f676ae84ba71...`、`shuffled_indices=9065ad66a7c1...`；
- RU checkpoint SHA256 为 `38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e`；两组都严格载入 233 个历史 tensor，并初始化 7 个新 tensor；
- architecture-only 与 aligned 均使用完整 GSV-Cities（524,701 张图）、280×280、P=40、K=4、10 epochs、seed 42；各有 350,658 个可训练参数，RU 基座冻结。

### 7.2 常规验证结果

下表采用各组**首次达到最佳 MSLS R@1** 的 checkpoint，避免事后按 Pitts 指标挑选。

| 模型 | checkpoint | MSLS R@1 | R@5 | R@10 | Pitts R@1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Architecture-only | epoch 0 | 91.22 | 95.14 | 96.08 | 94.10 |
| Aligned semantics | epoch 2 | 91.22 | 95.27 | 95.95 | 94.11 |

architecture-only 的 10/10 epochs 均为 MSLS R@1 91.22%；aligned 只有 epoch 2、3 达到 91.22%，其余 8/10 epochs 为 91.08%，即少 1/740 query。aligned 相对 architecture-only 和冻结 RU 的最佳 MSLS R@1 都是 `+0.00 pp`。Pitts 的 `+0.01 pp` 约为一个 query 的舍入尺度，不能解释为有效收益。

### 7.3 困难条件 full-db 审计

这里的比较对象是 matched architecture-only checkpoint，不是冻结源 RU。评测采用“条件 query + 18,871 张完整标准 database”的自定义审计协议，不是官方 condition benchmark。

| 条件 | Queries | Architecture-only R@1/R@5/R@10 | Aligned R@1/R@5/R@10 | 命中数差异 |
| --- | ---: | ---: | ---: | ---: |
| Night | 55 | 89.09 / 98.18 / 100.00 | 89.09 / 98.18 / 100.00 | 0 / 0 / 0 query |
| Season | 988 | 88.66 / 93.83 / 94.03 | 88.66 / 93.83 / 94.03 | 0 / 0 / 0 query |

对应命中数为 night 49/55、54/55、55/55，season 876/988、927/988、929/988。六项 top-k 命中数完全相同，只能说明排名没有跨过这些 recall 阈值，不能据此断言两个描述子逐元素相同。

### 7.4 终止判定

```text
Query-conditioned Semantic BoQ: FAIL

- aligned 未优于 architecture-only；
- overall 未达到 +0.3 pp；
- season 未达到 +1.0 pp，六项条件指标实际均为 0 query 差异；
- shuffled/random 因预注册提前停止规则而未运行。
```

因此不延长到 40 epochs，不增加 seed，不做 SALAD 复制，也不扫描 semantic-loss 权重、confidence threshold 或 bias scale。现有文本日志没有 TensorBoard 语义分支统计，所以不能断言语义分支“完全没学到”；可以确定的是，它没有带来可测的检索收益。

## 8. Crop-CLS Local Semantic FiLM-BoQ：正式筛选完成，FAIL

### 8.1 SemVPR 真正有效的组成

[Efficient Visual Place Recognition Through Multimodal Semantic Knowledge Integration（SemVPR，ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html)不是把分割类别直接变成 mask 或 attention bias。它在 DINOv2 中间层增加连续局部语义描述子分支，用经过 Local Semantics Alignment（LSA）的 CLIP 局部 feature map 做逐 patch cosine 蒸馏，再通过 Semantic-Aware Aggregation（SAA）把语义描述子映射为逐 patch、逐 channel 的乘性调制，最后才聚合全局描述子。教师只在训练期使用。

LSA 很关键：论文明确指出原始 CLIP 只受全局图文目标训练，raw local feature map 不适合直接充当局部教师；它先把图像划成区域，让教师的区域平均 patch feature 对齐相应 crop 的 CLIP CLS。Pitts30k 的 1024 维消融中，SemVPR w/o SAA、w/o LSA、完整模型的 R@1 分别为 91.7、93.5、94.2，说明连续局部表示与 feature-level aggregation 都是方法的一部分。同一维度下 w/o NDL 为 94.4、完整模型为 94.2；NDL 主要服务于低维描述子压缩，本项目当前只验证语义是否增益，不应在首筛中加入这一额外混杂。方法公式与完整消融见[论文 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.pdf)。

与本轮失败方案的关键差别是：

| 维度 | Query-conditioned 失败方案 | 新的 SemVPR-lite 首筛 |
| --- | --- | --- |
| 教师信号 | ADE20K hard class + confidence | CLIP crop CLS 连续 512D embedding |
| 空间语义 | 分割类别网格 | crop 对应区域的连续概念表示 |
| 注入位置 | BoQ cross-attention logit | BoQ 前的 patch×channel visual feature |
| 注入形式 | 每 query 一个有界标量 bias | signed channel-wise FiLM |
| 局部 CLIP 可靠性 | 依赖类别教师，不做 CLIP LSA | 直接用 crop CLS，避开 raw CLIP patch map |
| 首筛训练范围 | 冻结 RU/DINO/BoQ，仅 class head/bias | 恢复 RU 原训练范围，联合训练 continuous head/FiLM |

### 8.2 最短可行实现

推荐名称为 **Crop-CLS Local Semantic FiLM-BoQ（SemVPR-lite）**。它是对 SemVPR 最小核心的低成本因果筛选，不是官方实现或完整复现；目前也没有在论文官方页面发现可直接复用的作者代码。

```text
first 10 frozen DINO blocks --> 1x1 bottleneck --> semantic descriptor
                |                    |--> crop-CLS cosine loss
                +--> channel-wise FiLM --> last 2 DINO blocks --> RU gate --> BoQ
```

具体固定为：

- 语义教师复用训练机已有的 OpenCLIP `ViT-B-16/openai`，不用 raw patch token；
- 每个 P=40、K=4 batch 中，每个 place 只选一个 view，并按稳定轮换选择一个 2×2 象限；clean 140×140 crop resize 到 224 后取冻结 CLIP CLS，因此每 step 只增加 40 个 crop forward，无需生成新的全量大 cache；
- DINO wrapper 暴露进入最后两个 block 前的 20×20 token；学生支路为 `768 -> 128 -> 512` 连续语义投影，并由同一 128D bottleneck 经零初始化 `128 -> 768` 产生逐 patch、逐 channel 调制；
- 融合采用 `X_fused = X_k * (1 + 0.1 * tanh(delta))`，再送入最后两个 DINO block。零初始化保证第一个 forward 复现 RU，而后 VPR loss 可以直接决定哪些 token channel 应增强或抑制；
- 损失固定为 `L = L_MS + 0.05 * L_crop_cosine`，500-step linear warmup；teacher 看 clean crop，student 看 photometric augmentation，二者空间几何一致；
- 从 RU checkpoint warm start，冻结前 10 个 DINO block，但恢复 RU 原训练范围：最后 2 个 DINO block、RU gate、BoQ 与新语义支路共同训练。新支路约 26.4 万参数；推理完全移除 CLIP，并跳过仅用于蒸馏的 `128 -> 512` projection，实际执行的额外路径约 19.7 万参数的 bottleneck/FiLM（checkpoint 仍保留小型 projection 权重以便严格恢复）。

该设计刻意不使用 ADE20K hard class、confidence threshold、CLIP raw patch attention/affinity、空间插值、逐图单位方差化、固定动态负先验或 attention-logit bias，因而不会机械重复前面已经失败的路径。

### 8.3 预飞、对照与停止规则

先做固定 500 step 预飞，不计为正式结果：

1. 连续记录的因果指标至少到达 optimizer step 490，证明 500-step run 没有中途退出；
2. step 0 的 fp32 descriptor 相对 RU 最大绝对误差必须 `<=1e-6`；
3. 训练后 aligned crop cosine 必须比同图 wrong-region 和同 batch wrong-place 至少高 0.05；
4. channel-scale/semantic-projection gradient、modulation RMS 与 descriptor drift 必须都非零。

任一条件失败，先修实现或停止，不启动正式训练。通过后按顺序运行：

1. architecture-only：相同推理架构、数据 I/O 与 VPR trainable scope，`lambda_crop=0`；跳过并冻结不参与描述子生成的 teacher-only `128 -> 512` projection，5 epochs；
2. aligned：正确 crop CLS，5 epochs；
3. 若 overall 尚未达到 `+0.3 pp`、但相对 RU/architecture-only 下降均不超过 `0.2 pp`，先做低成本 condition evaluation；season 达不到 `+1.0 pp` 时停止；
4. 达到 overall 或 season 候选门槛后，运行 wrong-region：把已计算的 crop CLS 固定配给同图对角象限的 student region，5 epochs；
5. 继续通过后运行 wrong-place：把同 batch 异地点的已计算 crop CLS 固定轮换给当前 region，5 epochs。

最少 2 个、最多 4 个短 run，不扫描 `lambda` 或 FiLM scale。最终只有 aligned 同时胜过 RU、architecture-only、wrong-region、wrong-place，且 MSLS overall 至少 `+0.3 pp`；或 season 至少 `+1.0 pp` 且 overall 不低于 RU `0.2 pp`，才进入三个 seed。

实际 500-step run 到达 step 499。zero-start 误差为 0；aligned − wrong-region 尾五点均值为 0.05352；aligned − wrong-place 尾五点均值为 0.02755，最后值与全程最大值均为 0.03837；channel/projection gradient、modulation 和 descriptor drift 均非零。原审计据此输出 `FAIL`，该原始结果保留。

复核方法定义后发现，致命 wrong-place `>=0.05` 是本项目额外加入的地点特异性约束；SemVPR Eq. (10) 本身只使用对应 region/crop 的正余弦对齐，没有跨地点负项。跨地点道路、天空、植被和普通建筑相似时，wrong-place 并非可靠的语义负样本。因此这次结果解释为“机制预检通过、地点特异性探索项未过线”，而不是代码失败。为避免按结果调参，仍不改 `lambda`、FiLM scale 或 warmup，也不重跑预飞；在任何 retrieval 结果产生前固定只运行原有 architecture-only/aligned 两组。若 overall 未达到 `+0.3 pp` 但下降不超过 `0.2 pp`，先检查 season 是否达到 `+1.0 pp`；只有达到其中一个候选门槛才运行两组错误语义正式对照，最终检索归因门槛不变。

主要风险是完整 SemVPR 使用经过 LSA 训练的 dense teacher，并在最后一个 ViT block 内以 CLS 聚合；这个首筛则用稀疏 crop CLS 直接近似 LSA 锚点，并保留 RU+BoQ 聚合。若它失败，只能否定这套 BoQ 适配版 SemVPR-lite，不能否定完整 SemVPR；只有它接近或达到门槛，才值得训练真正的 dense LSA teacher或复现 SemVPR 的 CLS aggregator，并做 matched control。

### 8.4 正式训练结果（2026-08-30）

四组运行均从同一个 RU checkpoint 启动，使用 full GSV-Cities、280×280、P=40、K=4、seed 42 和 5 epochs。architecture-only 与 aligned 的本地原始日志确认：严格载入 233 个历史 tensor、只初始化 6 个 FiLM tensor、step-0 descriptor 最大误差为 0，并正常到达 `max_epochs=5`。下表统一采用**每组最高 R@1 checkpoint，并报告该 checkpoint 配对的 R@5**；wrong-region/wrong-place 数值来自训练机 checkpoint 清单，原始日志尚未下载归档。

| 模型 | 最佳 checkpoint | MSLS R@1 | R@5 | 相对 RU R@1 | 相对 architecture-only R@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冻结 RU | epoch 26 | 91.22（675/740） | 95.14（704/740） | — | +0.81 pp（+6 query） |
| Architecture-only | epoch 3 | 90.41（669/740） | 95.27（705/740） | -0.81 pp（-6） | — |
| Aligned crop-CLS | epoch 4 | 90.95（673/740） | 95.54（707/740） | -0.27 pp（-2） | +0.54 pp（+4） |
| Wrong-region | epoch 2 | 90.54（670/740） | 95.00（703/740） | -0.68 pp（-5） | +0.13 pp（+1） |
| Wrong-place | epoch 4 | **91.08（674/740）** | **95.68（708/740）** | -0.14 pp（-1） | +0.67 pp（+5） |

结果包含三个不同层面的结论：

1. **工程与优化成功。** 零起点、梯度、modulation、descriptor drift 和完整训练均正常；不能把负结果归因于“分支没接上”或训练卡死。
2. **aligned 相对 architecture-only 的 +0.54 pp 不能单独证明语义有效。** 它只说明加入 crop cosine 约束比没有该约束少退化 4 个 query。
3. **语义因果归因失败。** aligned 胜 wrong-region 0.41 pp（3 query），说明局部空间配对可能有作用；但 aligned 比 wrong-place 低 0.13 pp（1 query），正确地点配对不是获得该变化的必要条件。wrong-place 甚至是四个训练变体中最强者，更符合辅助正则化/优化轨迹差异，而不是地点相关语义增益。

按第 8.3 节在正式结果产生前固定的规则，aligned 相对 RU 下降 0.27 pp，已经超过允许的 0.2 pp，因此**正式停止点其实在 aligned 之后**。后续 wrong-region 与 wrong-place 是超出停止点完成的探索性诊断；它们加强了 FAIL 结论，但不能事后改变预注册流程。由此不再做 condition evaluation、更多 seed、40 epochs、SALAD 复制或 `lambda`/FiLM-scale 扫描。

### 8.5 结论边界

本实验否定的是这套具体组合：稀疏 2×2 crop-CLS 教师、每 step 每 place 一个 crop、cosine 蒸馏、最后两层 DINO 前的 FiLM，以及 RU/BoQ 联合微调。它没有复现 SemVPR 的 dense LSA teacher 和论文 CLS aggregation，因此不能外推为“SemVPR 无效”；但已有结果足以说明，不应再通过增加 crop 数、延长训练或微调 FiLM 强度来挽救当前实现。

## 9. DINO-anchor Residual CLIP Fusion → VLAQ：Phase A 完成，FAIL

### 9.1 为什么当时选择这条路线

[DC-VLAQ](https://arxiv.org/html/2601.12729) 把 DINOv2 与 CLIP 都保留在训练和推理中，以 DINO token 为锚，只把 CLIP 相对 DINO 的互补部分作为残差修正，再用 query-residual aggregation 生成全局描述子。它目前只是 2026-01-19 的 arXiv v1 预印本，尚不能当作经过同行评审的确定证据，也没有发现可直接审计复用的官方实现；但它给出的消融与本项目现状高度相关：在其协议下，MSLS R@1 的 naive addition、cross-attention、FiLM、residual 分别为 93.0、93.2、93.0、94.2；DINO-only 与 DINO+CLIP 分别为 93.1 和 94.2；BoQ 与 VLAQ 分别为 93.4 和 94.2。论文配置与本项目的 batch、评测分辨率和聚合器不同，这些绝对数值不能直接横向比较，只能作为选择下一机制的依据。

它与当前失败路线的本质差异是：

| 维度 | Crop-CLS FiLM（已失败） | Residual CLIP + VLAQ（本次实验） |
| --- | --- | --- |
| CLIP 身份 | 训练期 teacher，推理移除 | 真实第二特征流，训练和推理均保留 |
| 监督 | VPR loss + crop cosine | 只用地点检索损失决定是否采用 CLIP 信息 |
| 融合 | 语义 bottleneck 乘性调制 DINO | 以原始 DINO 为基座的零起点加性残差 |
| 空间信号 | 稀疏 2×2 crop | 同图 dense patch token |
| 聚合 | 现有 BoQ 绝对 query response | 第二阶段才测试 query-relative residual |
| 风险 | 教师目标与地点判别冲突 | 推理增加一套冻结 CLIP 的时间和显存 |

这仍然明确属于“把语义加入 VPR”，而不是后处理：CLIP patch token 在生成每张图的全局检索描述子之前参与前向计算，最终表示无法脱离语义分支独立得到。

### 9.2 Phase A 预注册：先筛 residual fusion，不立即重写聚合器

为把融合收益与新聚合器收益分开，第一阶段保留 RU gate 与 BoQ：

```text
same photometric image
    +--> DINOv2 20x20 raw token D ---------------------+
    +--> frozen CLIP patch token --> P_C --> align 20x20+--> D + W_zero(C_norm - D_norm)
                                                         --> RU gate --> BoQ --> VPR loss
```

仓库适配式固定为 `Z = D_raw + W_zero(P_C(C_norm) - D_norm)`，其中 DINO 的归一化支路是固定 identity，不引入可单独学习的 DINO adapter。这不是逐字复刻论文公式，而是为了同时满足两点：保留论文的 DINO-anchor residual 思想，并让 `W_zero=0` 时严格逐元素复现现有 RU 描述子。CLIP 与 DINO 看同一张 photometric-only 图，CLIP token 投影到 768D，并用确定性插值对齐到 20×20；不再使用 clean-image cache、类别标签、confidence threshold、teacher loss、mask、attention bias 或 FiLM。

最低成本执行顺序：

1. **500-step preflight**：冻结 RU/DINO/BoQ，只训练 CLIP 投影与零初始化 residual adapter；要求 step-0 descriptor 最大误差 `<=1e-6`，adapter/CLIP projection 梯度、residual RMS 与 descriptor drift 非零且有限。
2. **aligned residual，3 epochs**：仍只训练 residual branch，先判断 CLIP 是否含有能被现成 RU+BoQ 利用的互补信息。
3. 只有 aligned 达到候选门槛，才跑三组完全匹配对照：`global-only`（每个位置复制同一 CLIP 全局向量）、`wrong-region`（固定非恒等 patch permutation）、`wrong-place`（batch 内异地点稳定轮换）。所有组保持同一 CLIP forward、参数量、优化步数和 adapter。

Phase A 的候选门槛固定为：aligned 相对冻结 RU 至少 `+0.5 pp`（MSLS 740 queries 上至少 4 个 query），并相对三个对照分别至少 `+0.5 pp`；或者 season-full-db 至少 `+1.0 pp`，同时 overall 不低于 RU 超过 0.2 pp。任何一项没有候选收益即停止，不扫描残差 scale、学习率或层数。

### 9.3 Phase A 正式结果、成对干预与增权诊断（2026-08-31）

500-step preflight 通过了 zero-start、梯度、残差和 descriptor drift 审计。随后 aligned、global-only、wrong-region、wrong-place 都从同一个 RU checkpoint 训练 3 epochs，只训练 983,808 个 residual 参数；四组每个 epoch 的 MSLS 指标都停在 R@1/R@5/R@10/R@15 = 91.22/95.14/96.08/96.49。训练损失与 batch accuracy 正常变化，因此不是训练提前结束或分支没有梯度。

对 aligned 最终 checkpoint 做同一模型、同一 raw image、同一检索库的成对推理干预，得到：

| 推理变体 | R@1 | R@5 | R@10 | R@15 | aligned 相对该变体的净 R@1 query |
| --- | ---: | ---: | ---: | ---: | ---: |
| bypass | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| aligned CLIP | 91.22 | 95.14 | 96.08 | 96.49 | — |
| zero-CLIP | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| global-only | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| wrong-region | 91.22 | 95.14 | 96.08 | 96.49 | 0 |
| wrong-image cross-city | 91.22 | 95.14 | 96.08 | 96.49 | 0 |

这六组并非逐元素完全相同：aligned 相对 bypass、zero-CLIP、global-only、wrong-region、wrong-image 分别改变了 4、4、2、2、5 个 query 的 top-1 reference；但没有任何 R@1 正误翻转。也就是说 residual branch 已接通并会轻微改变排序，却没有形成可用的地点判别增益。

困难条件审计同样没有达到门槛：night-full-db 的 R@1/R@5/R@10 完全相同；season-full-db 的 aligned R@1 为 88.87%，RU 为 88.66%，仅多 2/988 query（+0.20 pp），R@5/R@10 相同。该探索性变化远低于预注册的 +1.0 pp，不能当作候选收益。

最后将 aligned checkpoint 中**已学习的 CLIP-only 残差**按 `gamma in {0.5, 1, 2, 4}` 放大，并在每个 gamma 下同时测 aligned、global-only、wrong-region、wrong-image：所有 18 个变体的 R@1 都是 675/740（91.22%），R@5 都是 704/740（95.14%）。`gamma<=2` 时 R@10 也全部为 711/740；`gamma=4` 时 aligned 与 wrong-region 降为 710/740，其余仍为 711/740。`best_observed_gamma=0.5` 只是全体并列后的排序产物，不是候选。

因此 Phase A 的机制判定为 **FAIL / NO_CANDIDATE**：现有 learned residual 在 gamma 0.5–4 范围内不是“影响太弱但方向正确”，放大它既没有产生 R@1 收益，也先在 R@10 暴露退化。该结论否定的是当前 `ViT-B-16/openai + DINO-anchor residual + 已冻结 RU/BoQ` 组合，不等价于 CLIP 或多模态 VPR 普遍无效。

### 9.4 Phase B：因 Phase A 失败而终止

VLAQ 先用 learned query 对 token 分配权重，再聚合 token 相对 query 的残差：

```text
alpha_jk = softmax_j(q_k^T z_j / sqrt(d))
v_k      = sum_j alpha_jk * (z_j - q_k)
```

原预注册要求做 `DINO-only VLAQ` 与 `DINO+CLIP residual VLAQ` 的 matched 5-epoch 对照，以分开“VLAQ 本身优于 BoQ”和“CLIP 语义带来额外收益”。由于 Phase A 没有任何候选增益，按规则 **不实现 VLAQ、不做 40 epochs 或多 seed**；以下公式只保留为历史设计记录。

### 9.5 已发表论文提供的边界

- [SemVPR（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html)证明了连续局部 CLIP 对齐与 feature-level aggregation 是可行方向，但本项目的稀疏 BoQ 适配已经失败，下一步不应继续近似同一 FiLM 路线。
- [StructVPR（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_StructVPR_Distill_Structural_Knowledge_With_Weighting_Samples_for_Visual_Place_CVPR_2023_paper.html)强调语义结构知识和按样本可靠性加权；[DistilVPR（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28905)则蒸馏样本间关系而非强迫不同模态逐点重合。若部署约束不允许推理时保留 CLIP，可把“可靠样本上的 semantic-layout relational distillation”作为无额外推理成本的备选，但它的优先级低于直接 residual screen，因为本项目已有多种 teacher-only 蒸馏/筛选负结果。

当前状态：**Phase A 已完成且 FAIL；Phase B/VLAQ 未实现并终止。** CLIP 使用 `ViT-B-16/openai` 的 14×14 joint-space patch token，插值到 DINO 20×20 网格；冻结 CLIP provider 不进入 checkpoint，`P_C/W_zero` 严格进入并恢复。独立预注册、实现索引和归档判定见 `doc/DC_VLAQ_LITE_EXPERIMENT.md`。

## 10. 随机/打乱对照的共同启示与下一方案

### 10.1 先纠正“随机组每次都有提升”

不同系列的训练长度和协议不同，下表只能在每一行内部比较，不能横向比较绝对 R@1：

| 路线 | aligned R@1 | 最强 random/shuffled/wrong 对照 | 匹配视觉基线 | 实际结论 |
| --- | ---: | ---: | ---: | --- |
| Semantic-alias negative | 87.43 | random 88.11 | A2 87.84 | random 胜 aligned 0.68 pp，也胜 A2 0.27 pp；这是最明确的一次随机正则候选，但仍是单 seed 小幅变化 |
| Semantic-positive | 87.03 | random 87.57 | A2 87.84 | random 只比 aligned 少退化，仍低于视觉基线 |
| Semantic-region | 89.86 | shuffled 91.22 | RU 91.22 | shuffled 完全解除 aligned 的 1.36 pp 退化，但没有超过 RU |
| Dynamic-category prior（overall） | 91.08 | random 91.08 | zero-bias 91.22 | aligned/random 同样少 1 query；条件子集上的 control 优势只有 1–2 query |
| Crop-CLS FiLM | 90.95 | wrong-place 91.08 | RU 91.22 | wrong-place 比 aligned 多 1 query，但仍比 RU 少 1 query |
| Residual-CLIP | 91.22 | 所有错误对照 91.22 | RU 91.22 | 全部持平；不存在随机收益 |

所以严格表述应是：**被破坏的语义对应经常胜过 aligned，却没有稳定胜过无语义基线。** 这不证明“随机语义含有地点信息”，而支持三个更保守的解释：

1. CLIP/分割类别描述的是场景或物体共性，VPR 需要区分具体地点；aligned teacher 可能把同类道路、车辆、植被、普通立面拉得更近，形成错误归纳偏置。
2. random/shuffled 会切断这条有害对应，或者通过随机采样、扰动与噪声降低共适应；收益若存在，更像 regularization，而不是 semantic knowledge。
3. 当前差值多为 1–4 个 query、单 seed 或验证集 checkpoint 最大值。它足以否定语义因果归因，却不足以证明一种稳定的随机正则化规律。

因此今后的 corrupted control 不是附属消融，而是“语义是否必要”的 placebo：aligned 不胜它，就不能声称收益来自语义。

### 10.2 论文给出的可用边界

- [StructVPR（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_StructVPR_Distill_Structural_Knowledge_With_Weighting_Samples_for_Visual_Place_CVPR_2023_paper.html)明确指出，并非所有样本都含有高质量、有帮助的语义结构知识，有些样本会伤害蒸馏，因此需要按可靠性筛选和加权。这与本项目“aligned 经常比破坏对照更差”的现象一致。
- [DistilVPR（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28905)蒸馏的是 self/cross agents 的特征关系，并在欧氏、球面和双曲空间建模，而不是要求跨模态局部特征逐点相等；这支持只约束检索关系、不再拟合 CLIP token。
- [DropBlock（NeurIPS 2018）](https://proceedings.neurips.cc/paper/2018/hash/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Abstract.html)说明空间相关特征适合整块丢弃，而非独立像素 dropout；[Random Erasing（AAAI 2020）](https://ojs.aaai.org/index.php/AAAI/article/view/7000)则把随机区域擦除作为遮挡增强。这两者为“随机组可能在提供结构化正则”提供通用机制，但不是 VPR 正收益的直接证据。
- [SALAD（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.html)用 dustbin cluster 学习丢弃非信息局部特征；[SemVPR（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html)表明训练期局部视觉—语义学习与 semantic-aware aggregation 可以有效。这说明可行方向是“选择性利用/丢弃”，不是给所有 aligned semantic patch 统一加权。

### 10.3 新候选：Reliability-Calibrated Semantic Counterfactual Dropout（RSCD-BoQ）

核心变化是：**语义不再作为地点标签、特征、蒸馏 target 或 attention bias，而只作为训练期反事实遮挡策略。** 模型被迫在语义上不稳定且跨地点高频的区域缺失时，仍利用其余局部结构完成地点判别。推理时关闭 dropout，网络仍是 DINOv2 + RU + BoQ，不运行分割器，因此不是检索后处理，也没有额外部署成本。

离线直接复用现有 GSV SegFormer-ADE20K 20×20 cache，不先验指定“车辆一定有害”。对每个类别 `c` 从训练集统计：

```text
repeatability(c) = 同一 place 的其他视图也出现 c 的条件概率
frequency(c)     = 包含 c 的 place 占全部 place 的比例
nuisance(c)      = 1 - repeatability(c) * (1 - frequency(c))
```

这里按 place/class presence 统计，不要求不同视角 patch 几何对齐。低 repeatability 或高 frequency 都会提高 nuisance；只有同时稳定且相对稀有的类别得到低分，因此建筑构件和标志不会仅凭类别名称被一刀切删除。

训练时在同一份 raw DINO 20×20 token 上构造 mask `M`，按 nuisance 对同标签连通区域采样；遮挡上限先固定为每图 15%，若合格区域不足则使用实际较小覆盖率，所有 control 必须逐图复制这个实际 token 数与连通块尺寸分布：

```text
Z_mask = (1 - M) * Z + M * stopgrad(mean_token(Z))
g_clean = BoQ(RU(Z))
g_mask  = BoQ(RU(Z_mask))

L = L_MS(g_mask)
  + 0.05 * SmoothL1(pairwise_cos(g_mask),
                    stopgrad(pairwise_cos(g_clean)))
```

关系一致性项只保持 batch 内地点关系，不要求 masked descriptor 或局部 token 逐点复制 clean 分支；其思想接近 DistilVPR 的 relation distillation。两条分支共享一次 DINO raw-token 前向，只额外执行轻量 RU+BoQ，从而控制首筛成本。

### 10.4 必须完全匹配的四组实验

| 组别 | mask 策略 | 回答的问题 |
| --- | --- | --- |
| `no_mask` | 同训练范围和双分支，`M=0` | 再训练/一致性本身是否带来变化 |
| `uniform_block` | 与 aligned **逐图相同覆盖率和连通块尺寸分布**，位置均匀随机 | 收益是否只是 DropBlock/Random Erasing |
| `shuffled_semantic` | 从异地点 donor 取同覆盖、同形状 mask | 语义形状有用但图像对应无关，还是对应本身必要 |
| `aligned_rscd` | 当前图像的 reliability-calibrated semantic mask | 正确语义是否优于所有非语义解释 |

四组必须使用同一 RU checkpoint、seed、batch 顺序、trainable scope、总步数、token replacement 和 relation loss。尤其不能让 aligned 的平均遮挡面积大于 random；否则只是更强扰动而不是语义差异。

最低成本顺序为：512 图离线 mask 审计 → 500-step contract preflight → `no_mask/uniform_block/aligned_rscd` 各 3 epochs；只有 aligned 同时至少比这两组多 4/740 个 R@1 query，才补跑 `shuffled_semantic`。最终语义候选要求 aligned 相对 RU、no-mask、uniform 和 shuffled **分别**至少多 4 query，或 season-full-db 至少 +1.0 pp 且 overall 不下降超过 1 query；通过后才做 3 seeds。

结果的解释在运行前固定：

| 结果模式 | 允许的结论 |
| --- | --- |
| aligned > uniform、shuffled、no-mask、RU | 语义选择策略有候选因果收益 |
| uniform > no-mask/RU，但 aligned 不胜 uniform | 找到的是非语义结构化正则，可保留工程改进，但不能写成语义贡献 |
| aligned ≈ shuffled > uniform | 区域形状/分布有用，正确语义对应未被证明 |
| 四组均无收益 | 停止当前数据与 BoQ 下的语义路线，不再换权重继续扫 |

这条路线仍有风险：动态先验实验已经说明“固定抑制某类”不成立。因此 RSCD 的关键不是换一组负类别，而是**数据驱动可靠性、随机训练扰动、推理无 mask，以及等覆盖随机 placebo**。若这四点不能严格实现，就不值得启动正式训练。

## 11. 证据索引与归档说明

保留的主要证据：

- MixVPR 训练原始输出：`doc/A2结果.md`、`doc/B1.md`、`doc/B2_*.md`、`doc/C0.txt`、`doc/C2.md`、`doc/C3_*`、`doc/D1.md`、`doc/D2.md`；
- semantic reliability/alias/positive：`doc/VPR_SEMANTIC_RELIABILITY.md`、`doc/semantic_alias_tb_raw.csv`、`doc/semantic_alias_tb_summary.csv`、对应训练日志；
- DINOv2-BoQ/SALAD 与六组 region 实验：`doc/boq_dinov2*.txt`、`doc/salad_dinov2.txt`、`doc/SEMANTIC_REGION_EXPERIMENT.md`；
- 传播诊断的 compact evidence：`doc/semantic_region_delta_batch0_clean/{run.json,summary.csv}` 与 `doc/semantic_region_counterfactual_batch0/{counterfactual_run.json,counterfactual_summary.csv,aligned_mask_montage.png}`；
- 动态先验：`doc/dynamic_category_mask_audit_full_db_condition_union/run.json` 与 `doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries/{run.json,summary.csv,paired_comparisons.csv}`；
- BoQ attention compact evidence：`run.json`、`summary.csv`、`head_summary.csv`、`query_slot_summary.csv`、`fc_energy_slot_weights.csv`、`mean_position_attention.jpg`、`attention_balanced_random.jpg`。
- Query-conditioned Semantic BoQ：`doc/QUERY_CONDITIONED_SEMANTIC_BOQ.md`、`doc/boq_dinov2_query_semantic_architecture_only.txt`、`doc/boq_dinov2_query_semantic_aligned.txt` 与 `doc/boq_dinov2_query_semantic_condition_eval.txt`。
- Crop-CLS Local Semantic FiLM-BoQ：`doc/CROP_CLS_SEMANTIC_FILM_BOQ.md`（实现、预注册、正式结果与 FAIL 结论）。
- Crop-CLS 原始证据：`doc/crop_semantic_film_runs/preflight_500steps.txt`、`preflight_audit.json`、`architecture_only_5ep.txt` 与 `aligned_5ep.txt`；wrong-region/wrong-place 当前仅有用户提供的训练机 checkpoint 清单，待原始日志下载后补归档。
- Residual-CLIP：`doc/DC_VLAQ_LITE_EXPERIMENT.md`（Phase A 预注册、实现索引、正式 FAIL 与停止决定）；训练机输出目录为 `doc/residual_clip_runs/paired_full_20260831_105942` 和 `doc/residual_clip_runs/semantic_gamma_sweep_20260831_123138`，其精确结果已固化在第 9 节，原始目录仍待同步回本机仓库。
- 下一候选 RSCD-BoQ：代码、四组严格匹配配置、类别可靠性统计、512 图离线 mask 审计和 500-step TensorBoard 合同审计均已实现；实验尚未运行，当前状态为 **IMPLEMENTED / PENDING PREFLIGHT**。完整操作协议见 `doc/RSCD_BOQ_EXPERIMENT.md`。

为减少仓库副产物，一次性诊断实现与大体积逐图数据在结论固化后清理。需要复现旧诊断时，可从以下 Git 提交恢复：semantic delta visualization `123d745`、counterfactual sweep `85e2816`、BoQ attention audit `ca158bd`、Phase-C smoke `8a08e81`、早期 CLIP sanity `4d19bfe`。训练日志、配置、核心模型代码和 checkpoint 加载路径不在清理范围内。
