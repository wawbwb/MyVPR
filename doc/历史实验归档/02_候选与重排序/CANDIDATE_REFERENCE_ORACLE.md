# 候选图理想过滤诊断（CPU）

前一轮结构审计：train 的 35 个可达错误只有 12 个具有正候选连接，dev 的 15 个只有 6 个；
错误地点内部连接分别覆盖 31/35、15/15。截断 10 条边没有丢失正连接。
这不是新方法的召回结果，也不能证明所有候选图方法无效。

本轮复用完整 train/dev 缓存和已验证的互为 top2、最多 10 边图，不提取图片、不训练、不调用 GPU。
固定三个分支：frozen、visual_graph、label_filtered_graph。
最后一个使用数据库 place-ID 删除异地点连接，但保留错误地点内部连接，不使用 query GT 挑选正边。
**它仍然使用了测试侧数据库标签，只能用于诊断，不能部署、不能称为真实方法结果或通用性能上限。**
place-ID 相同不保证视觉重叠，不同也不绝对排除重叠。

固定公式（不根据本轮 train/dev 结果选参数）：

`p = softmax(frozen_pair_logits)`，温度 1；`score_i = p_i + 0.5 * max(p_j for neighbor j)`。

无邻居支持为零；最多一个最强邻居，避免度数直接投票；只传播一次。
每个候选自身的 query 分数始终保留。相同分数按原候选顺序打破平局，与 frozen 一致。
这只是一个预先固定的可检验规则，不是已经校准的最佳融合方式。

输出全量和子集纠错/退步 ID、净变化、top1 改变数，并对可达错误按“有/无正支持”和“有/无错误支持”分组。
预测数组保存原 logits、候选 ID、GT 及两个诊断分数；两者尺度不同，不要直接相减解释为 logit 残差。
保留原始审计和缓存，校验其完成清单、代码、模型、split 和逐查询图复现。
不会自动开始后续 GPU 实验。dev 已反复查看，不是独立确认集。

## 训练机操作

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -m unittest discover -s tests -p test_candidate_reference_oracle.py -v
python -u scripts/diagnose_candidate_reference_oracle.py --output doc/candidate_reference_oracle_v1
```

测试成功后再执行诊断。CPU 校验有磁盘 I/O；不需 RU_CKPT、显卡或新依赖。
本机按用户要求未执行测试。输出目录已存在时拒绝覆盖；中断后使用新的输出名称。

Windows 下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_reference_oracle_v1 D:/Desktop/MyVPR/doc/
```

若理想过滤图仍无净改善，不投入同一规则的全量 Pair-VPR 边计算；不是否定全部图方法。
若改善，也仅支持下一阶段真实边验证；必须与原始图对照、分析退步，不能凭标签辅助结果宣称新方法成功。
