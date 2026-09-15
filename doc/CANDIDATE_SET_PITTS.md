# GSV 训练 / Pitts 开发选模：候选集合重排首筛

本轮复用 `.cache/pitts_pair_admission_v1` 的完整冻结缓存，以及原 GSV 计划
`doc/candidate_set_plan_v1`。不修改原缓存生成代码，不重新提取 Pitts 特征。
GSV 训练缓存尚未生成时，需要提取一次 1024 条训练查询及其参考库。

Pitts 固定 1024 查询、完整 10000 参考图的冻结 Pair-VPR 基线是 985/1024，
其中 30 条错误在 Top20 内存在正样本，另 9 条不可达。训练输入只有 GSV，
Pitts 只用于每组模型的 epoch 选择和开发诊断。它不是独立测试集；
已经观察过该数据的表现，也不能把剩余 Pitts 查询直接宣称为独立测试。

## 固定实验

| 模型 | 候选间交互 | 视觉密度校正 | 重复一致性损失 |
|---|---|---|---|
| independent | 无 | 无 | 无 |
| set | 有 | 无 | 无 |
| density | 有 | 有 | 无 |
| consistency | 有 | 无 | 有 |
| competition | 有 | 有 | 有 |

各组接收相同的双向 pair CLS 和参考全局描述子，参数形状相同。
independent 的注意力只允许对角项，因此有效跨候选梯度不同。
固定随机种子 42、5 epoch、batch 8、AdamW lr=1e-4、weight_decay=0.001、
梯度范数裁剪 1；每组都附加相同随机 5 个候选进行增强。
多正样本 list loss 只对既有正样本又有负样本的 GSV 查询训练，至少需要 128 条。
Pitts 统计始终覆盖全部 1024 查询，包括不可达查询。

每个候选的分数残差固定为 ±4，视觉密度温度固定为 0.02，
一致性项权重固定为 1。每组仅从 5 个训练 epoch 中选择开发 R@1 最高者，
平局选择最早 epoch；冻结基线单独报告，不用 epoch 0 隐藏训练退化。
这只是小型重排头首筛，不更新 Pair-VPR 主干。

## 执行

训练机在 VPR 环境和仓库根目录运行：

```bash
git pull --ff-only origin main
bash scripts/run_candidate_set_pitts.sh
```

脚本固定物理 GPU 1（内部 cuda:0），依次核验 Pitts 缓存、执行 CPU/CUDA
预检、补齐 GSV 训练缓存、训练五组头、导出开发报告。
不运行旧 GSV dev 准入，也不提取旧 GSV test 缓存。
预检不允许测试被跳过，包含五组模型的 CUDA 前向、反向及更新检查。
缓存核验包含源代码/文件哈希、冻结模型身份、多正样本 GT、候选顺序和原查询索引。

中断后重跑同一命令。完成的训练组经哈希核验后跳过；
尚未完成的组从固定种子重新开始该组的 5 epoch，不冒充逐步断点续训。
若训练准入失败，脚本退出并将原因保存在训练目录的 `gate.json`。

## 下载与解释

Windows 执行：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_set_pitts_report_v1 C:/Users/zy/Desktop/MyVPR/doc/
```

报告包含 `summary.json`（相对冻结 Pair 的修复/退化）、
`ablation_comparisons.json`（各关键消融的逐查询配对比较）、
`stress.json`（候选顺序等变性、重复全局第一候选的敏感性）、
`scores.npz`、`query_mapping.json`、训练日志、选模记录、训练契约和准入统计。
修复/退化数组均是样本行号，可通过 query_mapping 转为原 Pitts 查询索引。

首先比较所有组与 985 条基线，再比较 set 与 independent，最后检查密度和
一致性项的额外贡献。只超过原始全局检索的 935 条不算此方法有效。
任何开发提升都需要后续锁定方法、多种子和独立评测；单次选模结果不能作为论文结论。
