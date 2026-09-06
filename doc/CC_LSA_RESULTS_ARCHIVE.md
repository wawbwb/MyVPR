# CC-LSA：教师与探索性 Gate A 结果归档

归档日期：2026-09-06。状态：**TEACHER CONTRACT FAIL / EXPLORATORY GATE A FAIL / TERMINATED**。

## 1. 来源与边界

本记录转录用户提供的训练机输出、探索性审计输出和 `summary.csv`，不是重新运行结果，也不声称已独立核验完整逐查询审计。

- 教师目录：`logs/cc_lsa/teacher_seed42`。
- 探索性结果目录：`doc/cc_lsa_gate_a_exploratory`。
- 运行日志目录：`doc/cc_lsa_runs`。
- 探索性运行输出附件：`0ea97386-d05b-4f71-b70f-e6ab097aedf1/pasted-text.txt`；汇总表来自后续用户直接粘贴。
- 本次仅归档文档，不删除代码、缓存、日志或 checkpoint，不改变原审计阈值。

## 2. 教师训练完成，但原合同未通过

Crop CLS 共 62,514 行，对应 56,194 train / 6,320 holdout places。5 个 epoch 完整完成，train loss 依次为 0.081213、0.062668、0.058619、0.056193、0.054419；holdout loss 为 0.0556512。

| 教师诊断 | 实测 | 原合同 | 结果 |
| --- | ---: | ---: | --- |
| crop-region R@1：LSA | 97.0055% | — | 区域预测可学习 |
| crop-region R@1：raw CLIP | 5.4905% | aligned 增益 ≥5 pp | PASS |
| crop-region R@1：wrong-region | 0.0752% | aligned 增益 ≥10 pp | PASS |
| crop-region R@1：token-permutation | 25.0198% | aligned 增益 ≥10 pp | PASS |
| correct/wrong cosine margin | 0.138674 | ≥0.05 | PASS |
| median effective rank | 8.206716 | ≥16 | FAIL |
| median position std | 0.021424 | ≥0.02 | PASS |

Rank/position 诊断涉及 512 张图。阈值 16 是项目预设筛选规则，不是 SemVPR 的普适理论下限；rank 偏低也不能单独证明地点信息完全塌缩。97% 是 crop-region 任务，不是地点检索 R@1。

原教师 FAIL 保留；用户授权探索性运行后，显式允许 failed teacher 继续，不把探索性结果改写成正式准入通过。

## 3. 全量候选重排结果

MSLS-val：740 queries、18,871 DB；RU top-100 中有正确候选的 RU 错误查询为 52 个，另 13 个错误无法靠本候选集内重排修复。

| 变体 | R@1 命中 | R@1 | 纠正 RU 错误 | 破坏 RU 正确 | 净变化 | 正样本排名改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RU | 675 | 91.2162% | — | 0 | 0 | — |
| aligned LSA | 377 | 50.9459% | 2 | 300 | -298 | 13 |
| DINO-full | 504 | 68.1081% | 7 | 178 | -171 | 15 |
| raw CLIP | 455 | 61.4865% | 4 | 224 | -220 | 14 |
| token-permutation | 222 | 30.0000% | 6 | 459 | -453 | 15 |
| wrong-place | 161 | 21.7568% | 5 | 519 | -514 | 11 |

计数满足 `重排命中 = 675 + 纠正 - 破坏`。aligned 的净退化为 -40.2703 pp，不能以“补回两个查询”描述为正收益。

探索性合同也未通过：aligned corrections 2 < 8，rank-improved 13 < 37；100-seed count/weight-matched 对照的 correction P95 为 3、rank-improved P95 为 13。aligned 没有达到要求的纠错领先幅度，且其纠错数低于全部固定对照。最终为 `EXPLORATORY_FAIL / REGISTERED_CHECK_FAILED`。

## 4. 能得出什么，不能得出什么

1. 当前教师加手工 pair score 不能替代 RU 排序：不仅 aligned，DINO-only 也大量破坏正确结果。
2. 语义区域拟合成功不等于地点互补性成立。AG-SLRD 的布局预测和本次 crop 对齐都出现这一断层。
3. corrupted control 纠错更多不代表总体更好。本次 wrong-place 比 aligned 多纠正 3 个，却额外破坏 219 个正确结果。随机扰动偶尔翻对边界样本是可能解释，尚不是经过多 seed 证明的正则化收益。
4. 结果没有识别出唯一根因。教师表示、特征分辨率、局部匹配和分数校准均可能参与；不能仅凭汇总把全部损失归于 effective rank，也不能断言代码必然无误。
5. 本次未训练 learned pair classifier，不能宣称已否定 Pair-VPR/R²Former。另一方面，原合同失败也不能以“分类器还没训练”为理由放行原路线。

## 5. 决定

停止当前 CC-LSA Gate B/C、教师长训与局部语义权重扫描。保留结果和复现材料。

下一步是一个独立、尚未验证的 [视觉优先成对验证方案](VISUAL_FIRST_SEMANTIC_PAIR_VERIFICATION.md)：先解决纯视觉重排的净收益，再验证语义的条件增量，绝不复用当前 LSA 的失败准入结论作为正依据。
