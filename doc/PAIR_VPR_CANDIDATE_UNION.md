# 下一步：候选互补性筛查，不训练新模型

## 正式归档：筛查完成，暂停扩展（2026-09-07）

`doc/pairvpr_candidate_union_v1` 的 740 条逐查询记录及三个输出哈希已核对。

| 候选方案 | oracle 可达数 | 相对 Pair100 |
| --- | ---: | ---: |
| Pair100 | 724/740 | 0 |
| Pair120 | 724/740 | 0 |
| Pair100 ∪ RU20 | 725/740 | +1，仅 q98 |
| 并集补齐120 | 725/740 | +1，仅 q98 |

原始并集增加 4646 对候选，平均每查询 6.278，最多 19。新增可达的唯一查询 q98 原 RU top1 也错误。仅“补回原先不可达 GT”机制新增 1 个纠错机会，即 0.1351 pp；尚未真实评分，不能写成已实现收益，也不是所有候选扩展效应的上限。

结论：**COMPLETED / WEAK CANDIDATE COMPLEMENTARITY / NOT ADVANCED**。不是原“零新增即停止”规则的 FAIL；本次有一个新增，但信息收益与额外工作相比有限，决定不继续全量重排。停止的是这次配置，不是否定一切多检索器融合。下方方法和命令保留用于复现。

## 先归档几何诊断

2026-09-07：`pairvpr_match_geometry_v1` 完成 10 查询、58 对匹配的固定仿射一致性审计，输入/输出/脚本哈希匹配。

7 个远距离错误中，原评分最佳 GT 的内点数仅在 q34/q164/q439 超过错误 top1。q49 成功对照的错误候选有 87 内点，而已检查全部 GT 最多 80；q325 错误候选有 65，GT 最多 61。58 对真实对应均超过各自 5 次随机排列的内点数，但这无法区分真假地点。位置接近或布局相似也能产生一致对应。

结论：当前 MNN 数量/覆盖率/粗 token 仿射内点不足以支撑直接重排加权，不延长该参数扫描；不是判定所有精确几何验证或语义方法无效。保留 Pair-VPR 已实现的 693/740 结果。

## 唯一下一步问题

Pair-VPR global top100 的 oracle 为 724/740，最终仍有 16 个查询在候选内完全没有 GT。RU top20 能否补回其中部分？RU/refined 的 top1 oracle union=701 不回答这个问题，不能当作候选并集收益。

这一步明确是视觉候选补充，不是语义创新，也不预先实现新的训练方案。若研究必须限定为语义贡献，则不应把它包装成语义方法。

`scripts/audit_pairvpr_candidate_union.py` 读取已保存预测与 GT，校验原审计和索引哈希，再构造：

1. Pair top100：原候选基线。
2. Pair top120：同模型加候选的控制。
3. Pair top100 与 RU top20 去重并集：100–120 个候选。
4. 将并集按 Pair 原排名补齐到 120：与第 2 组严格相同候选数量。

候选构造不读取 GT；只在统计可达性时使用 GT。无评分融合、无 GPU、无新模型。报告新增可达查询、仍不可达查询、等预算双方独占的可达查询及增加的候选数量。不能把 oracle 当作实际 R@1，也不能把新增 GT 候选当作必然纠错。

若原始并集没有新增可达查询，停止“补充遗漏 GT”方向；这不排除已可达查询中新增更易匹配的正样本，但本筛查不测量这种效应。若有增益，再决定是否值得用固定官方 scorer 对全部查询检查纠错/误伤，并与 Pair120 比较。不能只评分获益查询。新增候选也可能带来更难的负样本。

本机不执行代码测试，按用户要求由训练机运行；只依赖现有 NumPy。

## 操作命令

本机 PowerShell：

```powershell
scp D:\Desktop\MyVPR\scripts\audit_pairvpr_candidate_union.py wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/scripts/
```

训练机：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python scripts/audit_pairvpr_candidate_union.py --audit doc/pairvpr_official_paired_audit_v1 --msls-path datasets/msls-val --output doc/pairvpr_candidate_union_v1
```

本机下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/pairvpr_candidate_union_v1 D:\Desktop\MyVPR\doc\
```

输出目录必须是新目录。下载后读取 summary.json 即可先决定是否继续；这是反复使用 MSLS-val 的开发探索，不是独立泛化验证。
