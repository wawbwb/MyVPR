# SLGD：先验证层选择与空间粒度

## 问题与依据

当前教师是训练期视觉匹配器，不是语义模型：冻结 RU checkpoint 中 DINOv2 的最后一层，20×20 平均池化到 10×10，再训练 768→256→128 projector，用互为最近邻平均余弦的正负间隔损失监督。训练 512 步后冻结；未启动学生正式 A/B，也未加入 CLIP。

RU 为 DINOv2 + repeatability/uniqueness 残差空间门控 + BoQ。这个 checkpoint 不使用 CLIP 或分割标签。它既是已训练的检索参照，也是本轮图像编码权重来源。“教师直接重排整体胜过 RU”不是蒸馏成立的必要条件；需要检验可靠互补，而不是将小 projector 与完整长期训练 RU 的差距解释成语义路线失败。

[SegVPR 原文](https://arxiv.org/pdf/2201.09701) §3.1、补充 §7.2/Fig.7：固定选择 conv4/conv5，f4 产生空间注意力，两层加权、分别池化再拼接。论文做了层组合消融，不是在推理时利用正确数据库图片逐对挑选重合热图。借鉴点是先检验不同抽象层的检索/对应价值，不照搬 ResNet 空间尺度到 ViT。

[DINOv2 官方代码](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) 支持中间层。本项目 legacy forward 返回未额外施加最终 LayerNorm 的末层 patch；本实验直接 hook 真实 forward，各层统一为 raw post-block，L12 必须逐项等于 legacy 输出，避免官方中间层接口默认归一化造成混淆。ViT-B 的各 block 网格相同，不等于 ResNet 多空间分辨率。

## 上一轮实际结果

GSV 独立地点、25m GPS 排除负样本诊断，6144 查询/10240 图库：RU 6030、raw 局部重排 5373、训练教师 5632 正确。教师相对 raw +259，但与 RU 比纠错 13、退化 411。困难集校准69/评估88查询；教师相对 raw 净 -1/+2。正式标记 INSUFFICIENT_HARD_CASES。不是标准 MSLS/Pitts R@1；不是所有语义方法失效的证据。

## 锁定的诊断

1. 复用上一轮所有图片、视图、RU top20、正样本、GPS 排除、地点划分。层号为一基 4/6/8/10/12。RU 和 DINO 全部冻结；不训练 projector 或聚合头。
2. 在全部 6144 查询上对比五层 **10×10**，相同 MNN 平均余弦规则和至少3个匹配条件。同分保持 RU 原候选顺序，全部无效时退回 RU 第一名。
3. 新缓存只保存四个中间层的100个单位 token，FP16；L12 复用原始 raw 缓存。预计新增约9.38 GiB。一次 backbone forward 收集所有层；每次只把一层加载 RAM 打分。不保存完整20×20特征。
4. 校准/评估各锁定192查询：包含所有旧 RU 困难案例，其余按固定 SHA256 顺序填充。所有层/网格使用同一子集。超过上限则停止，不偷偷丢弃困难案例。子集刻意富集困难案例，不当作总体 R@1。
5. 在该384查询子集在线重提取400-token **20×20**，同时重算100-token10×10确认复现；只保存20候选分数，不落盘400-token大缓存。比较同层20×20与10×10，分离“层选择”和“池化丢失”问题。
6. 只在校准10×10选择层：相对 L12 困难正确数净增加、总体正确数不减少。符合者按困难净增、总体净增、较小层号依次排序；没有符合者保留L12。选择先保存，原生网格和评估结果不可反向调整选择。
7. 输出总体、RU困难、可达RU错误、纠错/退化、独立纠错地点数、正样本相对最强有效负样本的间隔。GT 只用于事后指标，不参与匹配或部署层选择。没有中间层直接套用已针对末层训练的 BoQ。

## 边界

本轮已看过旧 GSV 校准/评估统计，是探索性诊断，不是新盲测。多个视图共享地点，不能当独立样本宣称显著性；本轮不声称稳定收益、创新或语义有效。`CALIBRATION_SELECTED_EVALUATION_COMPLEMENT` 仅描述固定选层在现有评估的方向一致，下一步仍需新地点与标准 MSLS/Pitts 验证。没有收益记 `NO_REPLICATED_LAYER_COMPLEMENT`，不是整条路线失败。smoke 只判机械运行通过。

若层选择或原生网格有互补，再设计等维度、等训练预算的单层/双层共享聚合头实验，随后验证语义监督；不自动启动教师/学生全量训练。

## 运行与恢复（训练机）

入口 `bash scripts/run_slgd_layers.sh all`：29旧测试 + 新回归测试 → 源smoke上的新smoke → 完整诊断。现有 VPR 环境，物理GPU1，共享实验锁，不抢其他任务。

结果 `logs/slgd_layers_v1_audit/{summary,selection,pooled_outcomes,native_outcomes,progress,completed}.json`。新缓存 `.cache/slgd_layers_v1_audit`。旧权重、结果、缓存不会被修改。

中断后相同入口自动带 `--resume`，固定 contract 不变时复用哈希有效分片和匹配分数。损坏的新特征分片隔离保留再重算；旧源缓存损坏则停止，不擅自修改。完成输出自带6类结果哈希（实际数量以completed为准）。不在开发 PC 跑 Python/model/test。
