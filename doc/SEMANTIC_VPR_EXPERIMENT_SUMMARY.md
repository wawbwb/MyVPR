# 语义增强 VPR 实验总结

更新日期：2026-08-29

## 1. 总结结论

当前结果允许继续研究“语义如何进入 VPR”，但不支持继续微调已经失败的注入方式。到目前为止，没有一种语义方案同时满足以下两个条件：

1. aligned semantic 主实验优于匹配的无语义基线；
2. aligned semantic 主实验优于 shuffled/random 等破坏语义对应关系的对照。

因此，现有实验不能证明“语义在 VPR 中理论上无效”，但已经较有力地否定了本项目中这些具体实现：全局 CLIP 描述子拟合、CLIP 单图空间 attention 拟合、标量 patch reliability、CLIP 语义负样本/正样本选择、CLIP affinity 平滑区域目标，以及固定动态类别负偏置。继续只扫 `alpha`、`lambda`、`beta` 或温度，信息收益很低。

Query-conditioned Semantic BoQ 已完成 10-epoch 首筛。aligned 在 MSLS-val 的最佳 R@1 为 91.22%，与 architecture-only 和冻结 RU 均持平；在 night-full-db 与 season-full-db 上，R@1/R@5/R@10 相对 matched architecture-only 全部相差 0 query。该实现没有通过预注册门槛，按规则停止，不运行 shuffled/random、多 seed、40 epochs、SALAD 复制或 `lambda`/bias-scale 扫描。

下一条仍有信息价值的路线是参考 SemVPR，但不照搬本轮已经失败的 hard class attention bias：使用 **crop-CLS 连续局部语义蒸馏 + patch×channel 特征调制**。该路线已完成 500-step 预飞：实现、空间对齐、梯度和描述子干预检查通过，但额外设置的 wrong-place margin 未达到 0.05，原始审计因此为 FAIL。由于该跨地点负 margin 并不是 SemVPR LSA 的训练条件，现保留原失败记录，并在任何 retrieval 结果产生前修订为只运行 architecture-only 与 aligned 两个 matched 5-epoch run；它们仍未运行，不能记作检索结果。完整解释见第 8 节和 `doc/CROP_CLS_SEMANTIC_FILM_BOQ.md`。

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
- 当前证据更具体地否定了“冻结 RU/BoQ，仅靠 hard-class attention-logit adapter 即可获得语义收益”，尚未检验 SemVPR 的连续局部特征蒸馏和 feature-level modulation。

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

## 8. 参考 SemVPR 的下一方案：Crop-CLS Local Semantic FiLM-BoQ

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

### 8.4 实现状态（2026-08-29）

已实现 `backbone.crop_semantic_film.*`、每 place 一张 clean teacher view、P 个选中 view 的 crop-CLS cosine target、真正的对角 wrong-region、不同地点轮换 wrong-place、严格 RU/SHA warm-start、step-0 fp32 descriptor 等价检查、显存受限的 teacher chunk、训练/推理分离和 TensorBoard 诊断。500-step preflight 已完成并按上文归档；当前状态是“等待 architecture-only/aligned matched retrieval”，不是“实验成功”。详见 `doc/CROP_CLS_SEMANTIC_FILM_BOQ.md`。

## 9. 证据索引与归档说明

保留的主要证据：

- MixVPR 训练原始输出：`doc/A2结果.md`、`doc/B1.md`、`doc/B2_*.md`、`doc/C0.txt`、`doc/C2.md`、`doc/C3_*`、`doc/D1.md`、`doc/D2.md`；
- semantic reliability/alias/positive：`doc/VPR_SEMANTIC_RELIABILITY.md`、`doc/semantic_alias_tb_raw.csv`、`doc/semantic_alias_tb_summary.csv`、对应训练日志；
- DINOv2-BoQ/SALAD 与六组 region 实验：`doc/boq_dinov2*.txt`、`doc/salad_dinov2.txt`、`doc/SEMANTIC_REGION_EXPERIMENT.md`；
- 传播诊断的 compact evidence：`doc/semantic_region_delta_batch0_clean/{run.json,summary.csv}` 与 `doc/semantic_region_counterfactual_batch0/{counterfactual_run.json,counterfactual_summary.csv,aligned_mask_montage.png}`；
- 动态先验：`doc/dynamic_category_mask_audit_full_db_condition_union/run.json` 与 `doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries/{run.json,summary.csv,paired_comparisons.csv}`；
- BoQ attention compact evidence：`run.json`、`summary.csv`、`head_summary.csv`、`query_slot_summary.csv`、`fc_energy_slot_weights.csv`、`mean_position_attention.jpg`、`attention_balanced_random.jpg`。
- Query-conditioned Semantic BoQ：`doc/QUERY_CONDITIONED_SEMANTIC_BOQ.md`、`doc/boq_dinov2_query_semantic_architecture_only.txt`、`doc/boq_dinov2_query_semantic_aligned.txt` 与 `doc/boq_dinov2_query_semantic_condition_eval.txt`。
- Crop-CLS Local Semantic FiLM-BoQ：`doc/CROP_CLS_SEMANTIC_FILM_BOQ.md`（实现与预注册协议；尚无训练结果）。
- Crop-CLS 预飞原始证据：`doc/crop_semantic_film_runs/preflight_500steps.txt` 与 `preflight_audit.json`。

为减少仓库副产物，一次性诊断实现与大体积逐图数据在结论固化后清理。需要复现旧诊断时，可从以下 Git 提交恢复：semantic delta visualization `123d745`、counterfactual sweep `85e2816`、BoQ attention audit `ca158bd`、Phase-C smoke `8a08e81`、早期 CLIP sanity `4d19bfe`。训练日志、配置、核心模型代码和 checkpoint 加载路径不在清理范围内。
