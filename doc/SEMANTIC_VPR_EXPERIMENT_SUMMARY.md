# 语义增强 VPR 实验总结

更新日期：2026-08-27

## 1. 总结结论

当前结果允许继续研究“语义如何进入 VPR”，但不支持继续微调已经失败的注入方式。到目前为止，没有一种语义方案同时满足以下两个条件：

1. aligned semantic 主实验优于匹配的无语义基线；
2. aligned semantic 主实验优于 shuffled/random 等破坏语义对应关系的对照。

因此，现有实验不能证明“语义在 VPR 中理论上无效”，但已经较有力地否定了本项目中这些具体实现：全局 CLIP 描述子拟合、CLIP 单图空间 attention 拟合、标量 patch reliability、CLIP 语义负样本/正样本选择、CLIP affinity 平滑区域目标，以及固定动态类别负偏置。继续只扫 `alpha`、`lambda`、`beta` 或温度，信息收益很低。

下一条仍值得做的路线是 **Query-conditioned Semantic BoQ**：把冻结分割教师产生的局部类别向量压缩为训练期监督，让不同 BoQ query 学习不同的语义条件化读取方式；推理时只保留学生分支。它应以 DINOv2-BoQ 的 repeatability+uniqueness（RU）模型为强匹配基线，并保留 shuffled-label、图内空间随机 label-confidence、architecture-only 三类对照。该路线现已实现，具体结构、配置、命令和判据见 `doc/QUERY_CONDITIONED_SEMANTIC_BOQ.md`。

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
- BoQ query/head 的偏好高度异质，固定类别、所有 query 共用的负 bias 过于粗糙。

现有结果不支持：

- “语义理论上不能用于 VPR”；
- “CLIP 完全没有空间信息”；
- “所有动态物体都应该被抑制”；
- “只需减小权重就能救回现有方案”。

## 7. 已实现、待运行：Query-conditioned Semantic BoQ

建议让冻结分割教师为每个 DINO patch 提供低维类别/属性向量，由一个轻量学生头预测这些向量；每个 BoQ learned query 再根据自身状态产生类别条件，形成 query-specific 的 key/value modulation 或 attention bias。这样语义不是检索后的重排，而是在描述子生成阶段参与局部聚合；同时避免所有 query 使用同一张“动态=坏”的 mask。

最短、信息量最高的执行顺序：

1. 冻结 RU checkpoint，只训练语义学生头和 query-conditioned adapter，先做 5–10 epoch screen；
2. matched controls：architecture-only、aligned label、shuffled label、图内空间随机 label-confidence，所有组参数量与训练预算一致；
3. 同时报告 MSLS overall、night/winter2summer/summer2winter full-db 和 Pitts30k；
4. aligned 必须优于 RU、architecture-only、shuffled、random；overall 至少 +0.3 pp，或某个足够大的困难子集至少 +1.0 pp 且 overall 不下降超过 0.2 pp，才进入 40 epochs；
5. 通过后再做三个 seed，最后才考虑 SALAD 复制。

若 aligned 与 random/shuffled 相同，应立即停止；若只有某几个 query slot 有稳定收益，再把 adapter 限制到这些 slot，而不是增加全局语义权重。

## 8. 证据索引与归档说明

保留的主要证据：

- MixVPR 训练原始输出：`doc/A2结果.md`、`doc/B1.md`、`doc/B2_*.md`、`doc/C0.txt`、`doc/C2.md`、`doc/C3_*`、`doc/D1.md`、`doc/D2.md`；
- semantic reliability/alias/positive：`doc/VPR_SEMANTIC_RELIABILITY.md`、`doc/semantic_alias_tb_raw.csv`、`doc/semantic_alias_tb_summary.csv`、对应训练日志；
- DINOv2-BoQ/SALAD 与六组 region 实验：`doc/boq_dinov2*.txt`、`doc/salad_dinov2.txt`、`doc/SEMANTIC_REGION_EXPERIMENT.md`；
- 传播诊断的 compact evidence：`doc/semantic_region_delta_batch0_clean/{run.json,summary.csv}` 与 `doc/semantic_region_counterfactual_batch0/{counterfactual_run.json,counterfactual_summary.csv,aligned_mask_montage.png}`；
- 动态先验：`doc/dynamic_category_mask_audit_full_db_condition_union/run.json` 与 `doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries/{run.json,summary.csv,paired_comparisons.csv}`；
- BoQ attention compact evidence：`run.json`、`summary.csv`、`head_summary.csv`、`query_slot_summary.csv`、`fc_energy_slot_weights.csv`、`mean_position_attention.jpg`、`attention_balanced_random.jpg`。

为减少仓库副产物，一次性诊断实现与大体积逐图数据在结论固化后清理。需要复现旧诊断时，可从以下 Git 提交恢复：semantic delta visualization `123d745`、counterfactual sweep `85e2816`、BoQ attention audit `ca158bd`、Phase-C smoke `8a08e81`、早期 CLIP sanity `4d19bfe`。训练日志、配置、核心模型代码和 checkpoint 加载路径不在清理范围内。
