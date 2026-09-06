# 新候选：视觉优先的语义矛盾成对验证

日期：2026-09-06。状态：**PROPOSAL ONLY / NOT IMPLEMENTED / NOT VALIDATED**。

## 1. 推荐与边界

先建立保留 RU 全局证据的 learned visual pair verifier，再问：**在外观相似的候选之间，可靠的跨图类别矛盾能否减少错误匹配？**

这不是继续训练失败的 CC-LSA Gate B，也不是再给 BoQ 加一个单图 adapter。不复用失败 LSA teacher，不改动已有 RU 主干。第一步是独立的纯视觉基准，不算语义创新；第二步通过匹配对照后才允许声称加入语义有用。

更保守、论文可比性更高的选择是先复现官方 Pair-VPR 或 R²Former；若保持当前 RU 和控制成本，则只能称下述实现为受论文启发的轻量适配，不能称完整复现。没有证据保证它一定优于当前 RU。

## 2. 论文依据与不是论文结论的部分

| 论文 | 可借鉴的具体原则 | 不可直接外推 |
| --- | --- | --- |
| [Pair-VPR](https://arxiv.org/abs/2410.06614) | 同时学习全局描述子与同地点 pair classifier，另有 place-aware Siamese masked-image pretraining | 冻结 RU 后训练小头并不等价于其完整方法或性能 |
| [R²Former](https://arxiv.org/abs/2304.03410) | 用局部相关性、attention、xy 坐标学习地点对判别 | 任意 MNN top-20 均值不是论文的 learned reranker |
| [StructVPR++](https://arxiv.org/abs/2503.06601) | 显式语义对齐，并对不可靠训练样本降权 | 不能据此假定本项目的类别矛盾一定可靠 |
| [SemVPR](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html) | 联合局部视觉/语义学习及语义感知聚合，教师只用于训练 | 单独学好 crop 对齐不等价于完整 SemVPR，也不证明可以用其相似度替代 RU |

下面的“矛盾证据 + 保留全局分数 + 拒绝不确定改排”是本项目根据失败结果提出的研究假设，不是以上论文已验证的组合。

## 3. 为什么与之前不同

| 既有失败经验 | 新设计约束 |
| --- | --- |
| 类别相同、CLIP 相似不等于同地点 | 不因“同类建筑/道路”直接给匹配加分 |
| FiLM、残差、mask 改动强基线仍无增益 | RU 完全冻结，只研究候选对 |
| teacher proxy 很好，但 RU-only 错误补不回来 | 训练/选模使用检索 hard-negative pair，不再以 crop R@1 或 rank 阈值代替地点指标 |
| CC-LSA 连 DINO-only 重排都明显退化 | 先建立 learned visual baseline，保留 RU cosine，不直接用局部分数替换排序 |
| corrupted control 偶然多纠正，却破坏更多 | 主指标为纠正数减破坏数，并报告逐查询配对结果 |

## 4. 阶段 V：只验证纯视觉成对判别

冻结 RU，缓存候选的原生 DINO 20×20 token，不为迁就 CLIP 网格先平均池化到 14×14。轻量 pair head 输入局部相关性、双方归一化坐标、匹配质量和 RU global cosine；若读取 attention，必须明确提取定义并对所有组一致，不能用未验证量冒充 attention。

训练数据来自 GSV 地点正对与 RU 检索得到的 hard negatives，而不是随机两图二分类。训练、模型选择和阈值校准使用相互隔离的 place split；尽可能额外保留未见城市。注意既有 RU 已见过 GSV 全集，因此这种 holdout 仅对新 head 隔离，不能称整个系统从未见过；最终必须使用独立 benchmark。

对候选输出 learned visual score `v(q,d)`，保留全局证据：

```text
s_visual(q,d) = s_RU(q,d) + alpha * bounded(v(q,d))
```

初始化残差为零，验证退回 RU 精确一致。alpha 与是否改排的置信阈值仅在 GSV calibration split 选定并冻结。该设计允许退回 RU，但**不能数学保证测试集不下降**。

首筛预算建议：缓存固定的至多 10,000 个训练 places，固定 head 和最多 3 epochs；这是低成本研究预算，不是论文默认参数。校准集必须按实际检索候选列表计分，不能以 balanced pair accuracy 代替 R@1。若这一步对冻结 RU 没有可重复净收益，则停止，不启动语义分支。

## 5. 阶段 S：语义只提供“不一致证据”

仅在阶段 V 有效后实施。用冻结分割器得到类别概率；优先保留足够细类别，不再预设道路、天空等类别必须抑制。已有 GSV ADE20K labels/confidence 可做先导诊断；它们不是完整类别概率，不能据此计算完整分布距离。正式用分布距离时需另行缓存相应概率，或明确改用置信度加权硬标签不一致。

匹配集合始终由同一视觉 verifier 建立，语义不得改变匹配数量或视觉候选，以便隔离贡献。在双方语义置信度较高且视觉匹配可信的对应上，计算类别不一致比例/分布差异、有效对应数和覆盖率。视野不重叠、低置信分割时应输出“未知”，而不是“不同地点”。

训练一个受限的非负 contradiction penalty：

```text
s_semantic(q,d) = s_visual(q,d) - beta * contradiction(q,d)
```

penalty 以真实地点对标签训练、在独立校准 split 选 beta，不把类别不一致当绝对负标签。同类不自动奖励；不同类也不是一票否决。即便 penalty 只减分，也可能误伤正样本，必须实测净退化。

首版分割器在推理时保留，以便直接检验语义证据，不提前加蒸馏变量。若最终有效，再单独研究训练期语义辅助监督、推理删除分割器；不能提前宣传零推理成本。

## 6. 最小公平实验

固定 RU、候选、视觉匹配、训练量、head 容量、更新步数和校准预算：

1. `visual-only`：阶段 V 模型。
2. `aligned-semantic`：相同视觉基础加真实矛盾证据。
3. `shuffled-semantic`：同城异地点语义 donor；保持置信度/覆盖率统计尽可能匹配，训练与推理各自记录打乱规则。
4. `matched-nonsemantic`：相同额外参数与输入尺度，仅使用视觉匹配质量或匹配分布随机量；避免把容量增益当语义增益。

先冻结已训练 visual head，仅训练附加小头，所有扩展组从完全相同权重开始。推理时还需在同一个 aligned checkpoint 内替换语义，区分“训练正则化效应”和“推理确实使用正确语义”。

所有查询统一推理，不能用 GT 只对 RU 的 65 个错误做改排。逐查询报告 corrections、regressions、net、R@1/5/10、候选可达上限、耗时；保存候选排名以检查提升是否来自相同 query。小幅收益需多 seed 和配对不确定性分析，不能把 +1 query 直接定性成功。

## 7. 停止条件与最终验证

- 视觉模型无法在隔离开发集稳定取得净收益：停止；不以加语义补救尚未成立的视觉底座。
- aligned 不胜 visual-only 或匹配破坏对照：语义增量失败，不通过权重扫描解释成成功。
- 不沿用 arbitrary teacher effective-rank 门槛；评价目标就是地点对判别与全查询净检索收益。
- MSLS-val 已反复查看，仅是探索性开发证据。新方案的正结论必须在冻结配置后用未参与设计的 benchmark 或官方 test 评测确认。
- 当前没有新训练结果，也不保证新路线有效。如果用户要求单阶段、无额外推理且不愿重做论文级训练，应暂停新增方案，而不是声称小型适配必然解决已有问题。
