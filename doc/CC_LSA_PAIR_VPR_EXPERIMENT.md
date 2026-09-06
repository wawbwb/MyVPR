# Candidate-Conditioned LSA Semantic Pair-VPR（CC-LSA Pair-VPR）

状态：**TEACHER CONTRACT FAIL / EXPLORATORY GATE A FAIL / TERMINATED**
设计冻结日期：2026-09-02；实现交付日期：2026-09-06

代码、同步与完整运行命令见 [Gate A 运行手册](CC_LSA_GATE_A_RUNBOOK.md)。当前实现覆盖
LSA teacher、GSV-only calibration、冻结 RU 候选与六类对照审计。Gate B/C 的 pair
classifier 未实施，现按失败结果终止。2026-09-06 的教师与检索结果见
[结果归档](CC_LSA_RESULTS_ARCHIVE.md)：aligned 为 377/740，纠正 2 个 RU 错误、
破坏 300 个 RU 正确查询。下文保留原协议，不表示建议继续执行。

## 1. 边界与研究问题

AG-SLRD Phase 0 已证明：12 类 semantic-layout teacher 在 GSV holdout 上
能够学习真实地点关系，但在 MSLS 上只补回 RU 的 2 个错误，低于 shuffled
teacher 的 4 个。结合此前 global/local CLIP distillation、semantic-region、
class bias、FiLM、residual fusion、semantic dropout 等负结果，本项目停止继续
给当前 `DINOv2 -> RU -> BoQ` 叠加**单图 semantic adapter**。

新路线改问一个不同的问题：

> 单图语义不足以生成互补的全局描述子时，连续局部语义能否在 RU 已召回的
> query-candidate 图像对中，帮助判断二者是否来自同一地点？

RU 保持第一阶段全库召回器。语义只用于训练或审计候选图像对的局部对应，
不再为单张图生成类别权重、gate、FiLM、全局 semantic descriptor 或 mask。
该路线属于标准两阶段 VPR；它不是 AG-SLRD Phase 1，也不改变 AG-SLRD 的
停止结论。

## 2. 论文依据

- [SemVPR（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html)
  先用 crop CLS 监督 full-image local map 的 Local Semantic Alignment（LSA），
  再做 local semantic distillation 与 Semantic-Aware Aggregation。它说明不能把
  vanilla CLIP raw patch 直接视为高质量局部教师。
- [StructVPR++（TPAMI 2025）](https://arxiv.org/abs/2503.06601)把
  label-specific features 从 global descriptor 解耦，在图像对间显式进行
  semantic alignment，并用 sample-wise weighting 抑制有害 pair。
- [Pair-VPR（RA-L 2025）](https://arxiv.org/abs/2410.06614)联合学习 global
  descriptor 与 same-place pair classifier，用第二阶段对第一阶段候选重排；
  [官方实现](https://github.com/csiro-robotics/Pair-VPR)明确把它定义为两阶段
  VPR，而不是外部检索后处理脚本。
- [R²Former（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_R2Former_Unified_Retrieval_and_Reranking_Transformer_for_Place_Recognition_CVPR_2023_paper.html)
  的 pair scorer 同时利用 local correlation、attention 与二维坐标，并报告
  ViT token 可用于局部匹配。
- [EffoVPR（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6a1b224b153e55c40a6359f9c9fb9d8c-Abstract-Conference.html)
  表明 DINOv2 中间层局部特征可作为强 zero-shot reranker，且在夜间、季节与
  遮挡条件下具有鲁棒性。
- [EfficientVPR（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_EfficientVPR_Toward_Efficient_Visual_Place_Recognition_via_Scene-Aware_Prompt_Tuning_CVPR_2026_paper.html)
  进一步采用 instance-dependent key local feature enhancement；这与本项目
  “不能给所有同类区域统一语义权重”的经验一致。

CC-LSA Pair-VPR 是上述原则针对当前失败证据的组合设计，不声称是任何论文
的原样复现。

## 3. 与所有已失败方案的实质差异

| 已失败机制 | 本路线的改变 |
| --- | --- |
| 单图 global CLIP/layout descriptor | 只比较 query-candidate pair |
| raw CLIP patch 或稀疏 crop CLS | 先训练并冻结 dense LSA teacher |
| ADE20K hard class/12 类 layout | 使用连续局部语义，不输入类别名或手写类别权重 |
| 每图无条件 gate/FiLM/bias/residual | 只有跨视角可靠对应边产生 semantic loss |
| aligned 信号作用于 RU 已经正确的大多数图 | 先在 RU top-1 失败及 hard candidates 上检查互补性 |
| random/shuffled 只是附属消融 | DINO-only、raw-CLIP、spatial permutation、wrong-place 均为致命 placebo |
| 先训练再看 semantic 是否有用 | Gate A 不训练 RGB student，先证明 pair-level 信息充分 |

## 4. Dense LSA teacher

保留一份冻结的原始 vision-language 图像编码器 `F_clip`，训练一份 local
teacher `T_lsa`。对于 full image `I` 中预先固定的 crop/region `r`：

```text
u_r = normalize(F_clip(crop(I, r)).CLS)
t_r = normalize(pool(T_lsa(I).local_map, r))

L_LSA = mean_r [1 - cosine(t_r, stopgrad(u_r))]
```

训练协议必须在查看 MSLS 结果前冻结：crop 尺度、数量、重叠率、teacher
backbone、训练步数、学习率和输出网格不得根据 retrieval 结果调整。LSA 完成
后冻结 `F_clip/T_lsa`；二者不参与 Gate A 以后的参数更新。

最低 teacher 合同在第一次训练前写入 `pretraining_contract.json` 并记录 SHA256；
训练完成的实测判定写入 `lsa_teacher_contract.json`。固定 GSV train/holdout 的 place 清单
与 hash。默认量化门槛为：

1. 固定 holdout 的 crop-to-region R@1 比 raw-CLIP local map 至少高 5 pp，且比
   wrong-region、token-permutation 各至少高 10 pp；
2. correct-crop cosine 均值比固定 wrong-region 均值至少高 0.05；
3. 每图归一化 local descriptor 的位置有效秩中位数至少为 16，位置维平均标准差
   中位数至少为 0.02，不能退化成整图常量；
4. seed 42 只训练一次；除明确的代码/数值错误外，任一项失败即停止，不更换
   CLIP layer、crop 分布、teacher 宽度或训练步数。

上述阈值只使用 GSV place label 和预先冻结的 holdout，不读取 MSLS label。

## 5. Gate A：零 RGB-student 训练的 pair-sufficiency audit

### 5.1 固定候选与局部特征

1. 用冻结 RU checkpoint 生成 MSLS 标准 18,871 database + 740 query
   descriptor，并精确复现 `675/740` R@1；
2. 为每个 query 固定 RU top-100 candidates；
3. 对 query 和候选提取同一 DINOv2 层的归一化 local token、坐标以及冻结
   `T_lsa` 的连续 local semantic descriptor；
4. 所有缓存保存 image order、checkpoint/model SHA 和完整 provenance。

RU top-100 中不存在正样本的 query 不可能被任何 reranker 修复，必须单独报告
为 `unreachable`，不能从分母中静默删除。

### 5.2 参数冻结且可复现的 pair score

对于 query `q` 和 candidate `c`：

```text
C_vis(i,j) = cosine(z_q[i], z_c[j])
C_sem(i,j) = cosine(t_q[i], t_c[j])

E_vis = mutual_nearest_neighbors(C_vis)
tau_sem = Q95_{fixed GSV same-city different-place RU-hard pairs}(C_sem on E_vis)
w_sem(i,j) = clip((C_sem(i,j) - tau_sem) / (1 - tau_sem), 0, 1)
E_sem = {(i,j) in E_vis | w_sem(i,j) > 0}

edge(i,j) = ((C_vis(i,j) + 1) / 2) * w_sem(i,j)
S_pair(q,c) = sum(top-20(edge on E_sem)) / 20
```

不足 20 条边时用零补齐；`E_sem` 为空时得分为 0。`tau_sem` 是一个全局标量，
其 different-place RU-hard pair 清单、DINO/LSA 层、网格、归一化、数值精度与
代码/config SHA 均在看 Gate-A 输出前写入 manifest。raw-CLIP 使用同一 GSV
pair 清单和相同 Q95 规则得到自己的唯一标量。不得用 MSLS pair label、分数分布
或 retrieval 结果校准阈值、top-L、融合权重或分数归一化。

每个变体都只在固定 RU top-100 内按自己的 `S_pair` 降序重排；同分时依次使用
更小的原 RU rank 和 database index，保证结果确定。对一个 query，主判据为：

```text
max_{positive in top100} S_pair > max_{non-positive in top100} S_pair
```

这才记作一次真实 pair-only R@1 correction。与 RU 原错误 top-1 的 margin 只作
附加诊断，不计入“至少纠正 8 个”的门槛。

### 5.3 严格匹配的输入与稀疏度安慰剂

| 变体 | 改变内容 | 回答的问题 |
| --- | --- | --- |
| `aligned_lsa` | 正确 query/candidate LSA map | 正确连续局部语义是否有 pair 证据 |
| `dino_full` | 全部 `E_vis` 上取 top-20 normalized `C_vis`，零补齐 | 收益是否来自视觉局部匹配 |
| `dino_count_weight_matched` | 从同一 `E_vis` 抽取与 aligned 每 pair 完全相同的边数，并随机置换 aligned 权重多重集 | 收益是否只来自边数、稀疏化或权重分布 |
| `raw_clip` | 用未做 LSA 的 raw CLIP local map | LSA 是否必要 |
| `token_permutation` | 每图保留相同 semantic vector 多重集，但打乱 vector-to-DINO-token 注册 | 正确局部注册是否必要 |
| `wrong_place` | 从同城异地点 donor 提供 semantic map | 正确图像对应是否必要 |

`dino_count_weight_matched` 固定 100 个全局 seed（0--99）；每条边的抽样与权重
置换由 `SHA256(query_id, candidate_id, seed)` 决定，无放回抽样，报告 correction
和 rank-improvement 的 95th percentile。aligned 必须超过这个 95% 上界，而非
只胜某个幸运 seed。`token_permutation` 使用每图 SHA256 固定置换。`wrong_place`
使用固定同城、同 database/query 角色的 donor 置换。标准 MSLS index 没有完整的
database place ID，因此“异地点”严格指可从 GT 验证的关系不重叠：禁止 reference
donor 与 receiver 属于同一个已知 positive 集合；禁止 query donor 与 receiver 的
positive 集合相交。先构造 query donor，再为每个实际 top-100 候选附加约束，确保
其 database donor 也不是 donor query 的 GT-positive。该映射与候选一起冻结并审计。

所有控制复用相同 candidate list、DINO token、top-20、零补齐、tie-breaking 和
数值精度。由于 Gate A 不使用坐标项，`token_permutation` 只检验语义向量与
DINO token 的局部注册，不与几何函数混杂。

### 5.4 Gate-A 预注册通过条件

以下条件必须全部满足：

1. RU 精确复现 `675/740`；
2. 报告 65 个 RU top-1 错误中 positive 进入 top-10/top-100 的数量；top-100
   reachable 少于 8 时直接 `INSUFFICIENT_CANDIDATE_RECALL`；
3. `aligned_lsa` 依据 5.2 的全 top-100 最难负样本判据，真实 pair-only R@1
   correction 至少 8 个；
4. aligned correction 数分别比 `dino_full/raw_clip/token_permutation/
   wrong_place` 至少多 4 个，并且比 `dino_count_weight_matched` 100 seeds 的
   95th percentile 至少多 4 个；
5. 对每个 query 使用 best-positive rank；RU 与 pair rank 均截断为 101，
   unreachable 为 101，同分按 5.2 处理，740 个 query 全部保留在分母。aligned
   pair rank 优于原 RU rank 的 query 至少 `37/740 = 5%`；
6. 第 5 条的改善数也必须比 count/weight-matched 100 seeds 的 95th percentile
   至少多 4 个。

任一项失败即：

```text
CC-LSA Pair-VPR Gate A: FAIL
Pair classifier: NOT IMPLEMENTED
```

不训练 classifier，不修改 top-20/threshold，也不改用验证结果更好的 CLIP
layer。

### 5.5 数据使用边界

本项目此前已多次查看 MSLS-val 740-query 的总指标和逐 query 结果，因此它不再
是未触碰测试集。Gate A 与后续 Gate C 只能作为**开发集 go/no-go screen**，
不得单独支持论文式确认性结论。若 Gate A 通过，必须先冻结 Gate B/C 的完整
代码、模型、loss、miner、top-K、融合与 config SHA，再训练 student；最终正收益
必须在冻结后一次性提交到未用于本路线选型的 MSLS-test server，或另一个事先
登记的独立 benchmark。没有该独立确认时，最多称为 MSLS-val exploratory result。

## 6. Gate B：仅在 Gate A 通过后训练 RGB-only pair classifier

### 6.1 模型

RU 继续负责全库召回。轻量 pair classifier 只读取 query/candidate 的 RGB
DINO token、RU/BoQ attention summary 与二维坐标；推理不运行 CLIP 或 LSA：

```text
query RGB ---- DINO local tokens --\
                                  cross-image pair transformer -> same-place logit
candidate RGB - DINO local tokens --/
                  + visual correlation + xy + attention
```

训练期 LSA 对应边监督 cross-image attention/correspondence：

```text
L_pair = BCE(pair_logit, same_place_label)
L_corr = weighted local correspondence InfoNCE/BCE on reliable LSA edges
L_rank = softplus(score(hard_negative) - score(positive))

L = L_pair + lambda_corr * L_corr + lambda_rank * L_rank
```

LSA edge、confidence 和权重全部 stop-gradient。训练 pair 来自 GSV same-place
cross-view positives 与冻结 RU 挖出的 hard negatives；随机负样本不能代替 RU
hard negatives。

### 6.2 训练对照

| 组别 | 语义监督 |
| --- | --- |
| `rgb_pair` | 相同 pair classifier，仅 `L_pair/L_rank` |
| `random_edges` | 与 aligned 逐 batch 匹配边数和权重直方图，随机分配对应 |
| `aligned_lsa_pair` | 正确 LSA 对应边 |
| `wrong_place_edges` | 同步运行；用于判断正确语义对应是否必要 |

模型、candidate mining、batch、训练步数和可训练范围必须完全一致。

### 6.3 500-step 合同

- step 0 的 RGB pair baseline logits 必须可复现；
- `L_corr`、pair/correspondence gradients 有限且非零；
- aligned relation loss 在 GSV holdout 上至少比 random 与 wrong-place 低 10%；
- cross-image attention 不得塌缩到单 patch 或全均匀；
- LSA 分支在 inference checkpoint 中不存在；
- auxiliary loss 的尺度只可用固定 batch 的梯度范数校准一次，不看 MSLS。

## 7. Gate C：3-epoch seed-42 retrieval screen

仅当 Gate A/B 通过，依次训练 `rgb_pair`、`random_edges`、`aligned_lsa_pair`、
`wrong_place_edges`。
固定 top-100 reranking，报告 overall R@1/R@5/R@10、RU-correct regression、
RU-error correction、reachable oracle 和 condition full-db。

定义两个互斥的通过分支：

```text
overall_pass =
    aligned overall R@1 >= max(RU, rgb_pair, random_edges, wrong_place) + 4 queries
    AND corrected_RU_errors >= 8
    AND aligned_overall_correct - RU_overall_correct >= 4

season_pass =
    aligned season R@1 >= max(RU, rgb_pair, random_edges, wrong_place) + 1.0 pp
    AND aligned overall R@1 >= RU overall R@1 - 1 query
    AND aligned overall R@1 >= max(rgb_pair, random_edges, wrong_place) + 4 queries

Gate_C_pass = overall_pass OR season_pass
```

MSLS-val 上通过后才运行 3 seeds，但仍只算 exploratory screen；冻结后的独立
benchmark 才是确认性结果。否则结束该路线，不扫描 top-K、loss weight、teacher
layer 或训练 epoch。

## 8. 这是否仍属于“把语义加入 VPR”

是，但需要准确称为**训练期语义监督的两阶段 VPR**：

- 语义决定 pair classifier 学习哪些跨图局部对应，而不是在检索完成后用固定
  类别规则改分；
- pair classifier 是 VPR 模型的一部分，same-place classification 与 hard
  negative ranking 都由地点监督训练；
- 推理可以是 RGB-only：语义 teacher 只在训练中存在；
- 第一阶段 global retrieval + 第二阶段 learned pair verification 是 Pair-VPR、
  R²Former 等论文采用的正式 VPR 范式。

它确实不是单阶段 global descriptor。若项目约束绝对禁止第二阶段，则不应把
该方案改名包装成单阶段；更诚实的替代任务是按作者架构完整复现 SemVPR。
只有当本方案先证明 pair-level semantic complementarity 后，才有依据另立
“把 pair ranking 蒸馏回单阶段描述子”的实验。

## 9. 允许的结果解释

| 结果 | 结论 |
| --- | --- |
| aligned 在 Gate A 胜所有控制 | dense LSA 含 RU hard-pair 的互补局部证据 |
| DINO-only 与 aligned 相同 | 找到的是视觉局部匹配收益，不是语义收益 |
| raw-CLIP 与 aligned 相同 | 完整 LSA 并非必要，需停止语义归因 |
| spatial/wrong-place 与 aligned 相同或更好 | 正确语义对应不是必要条件，停止 |
| Gate A 过、Gate C 失败 | 信息存在，但当前 RGB pair distillation/模型未利用成功 |
| Gate C aligned 胜全部对照 | 进入多 seed，才可称为 semantic pair-VPR candidate |

本路线不能用来推翻此前任何 FAIL，也不能把 pair reranking 的正收益写成 BoQ
global descriptor 的语义提升。

## 10. 首次运行前冻结的实现细节

- 每个符合 K≥4 的 GSV place 由 SHA256 固定选择一张 clean image；按 place 划分
  train/holdout，避免同地点泄漏。calibration 由这个未参与梯度更新的 holdout
  每城固定选取至多 128 张，teacher 合同也在固定 holdout 检查；两者均不使用 MSLS。
- 冻结 crop teacher 为 `ViT-B-16/openai`，固定 `open_clip_torch==2.26.1`。
  280×280 clean image 切为四个不重叠 140×140 crop。LSA 从同一 CLIP 初始化，
  仅训练最后两个 transformer block、ln_post 与 projection；固定 seed 42、5 epochs、
  batch 32、AdamW lr=1e-5、weight_decay=0.01、BF16 与 math-SDPA。
- LSA/raw-CLIP native map 为 14×14×512。DINO 使用 RU checkpoint 的最后层
  20×20 patch map，在 semantic-region gate 和 BoQ 之前，经过一次
  adaptive average pooling 到 14×14 并逐 token L2 normalize。所有变体共用这份
  DINO map。该选层是本路线的固定设计，不声称复现 EffoVPR 的中间层选择。
- GSV/MSLS 输入均采用同一 torchvision-v2 bicubic+antialias clean transform；
  local cache 为 FP16，calibration 与 audit 匹配统一使用 Torch FP32、关闭 TF32，
  argmax 同分取较小 token index。Q95 使用 `method='higher'`。
- source/config/checkpoint/index/array SHA 都记录到相应 manifest；审计强制校验，
  并从冻结 GSV cosine 数组重算 Q95。续跑必须逐一核对候选、donor 与行映射。
- 报告同时输出纠正数、RU 原正确查询退化数及最终 R@1。Gate A 是信息充分性首筛，
  不是最终部署性能判据；Gate B/C 需继续检验是否能保住 RU 的已有命中。
- 分阶段脚本为 `scripts/run_cc_lsa_gate_a.sh`，带实时 tqdm/日志与失败即停止机制。
  按用户要求，最终修改版在训练机执行测试、smoke 与正式实验，本机未执行本次最终版测试。

## 11. 教师首轮结果与探索性继续（2026-09-06）

用户提供的训练机输出显示 62,514 个 crop target 已完成，teacher 完整训练 5 epochs。
GSV holdout crop-region R@1 为 97.0055%，raw CLIP 为 5.4905%，token-permutation
为 25.0198%；correct/wrong cosine margin 为 0.138674。空间标准差为 0.021424，
但前 512 张 holdout 图的空间 effective rank 中位数为 8.206716，低于预设 16。
原始教师合同因此保持 **FAIL**；这些不是 MSLS 检索结果。

后续按用户选择开放一次探索性 Gate A：复用同一 final.pt，不重训、不修改 teacher
配置、checkpoint 或合同。只有显式 `--allow-failed-teacher-contract` 才允许校准与
缓存加载该教师。校准、特征和审计必须采用一致的探索模式并记录完整原合同。
计分、对照和原 Gate-A 数值阈值保持不变；输出只能标为 `EXPLORATORY_PASS` 或
`EXPLORATORY_FAIL`，不能用于声称原先预注册的教师准入条件通过。

运行入口为 `bash scripts/run_cc_lsa_gate_a.sh exploratory`，从 GSV calibration
开始，随后提取 RU/MSLS features 并审计。输出为 `doc/cc_lsa_gate_a_exploratory`；
探索缓存目录也与原正式目录分开。最终修改未在本机执行测试。
