# VPR-guided Semantic Learning：匹配结构的三组验证

## 假设与边界

参考 [Learning Semantics for Visual Place Recognition through Multi-Scale Attention](https://arxiv.org/abs/2201.09701)，测试“地点识别训练的注意力引导共享编码器的语义学习”。不预设车辆/行人应降低权重，不使用 CLIP，不在推理时运行分割头。

这是 DINO/RU/BoQ 上的机制适配，不是原始 SegVPR 复现。仍用真实 GSV 与 SegFormer ADE20K 伪标签；没有合成真值、ResNet 多层特征、原分割解码器或合成到真实的域适配。它只检验结构作用，不能回答合成数据是否更有效。先隔离结构，再决定是否值得增加精确语义数据。

## 网络与梯度

最终 DINO patch map F 经 1×1 降维，再经 3/5/7 卷积分支生成空间图 M=2 sigmoid(logit)。最终投影零初始化，M 初始严格等于 1，确保未经更新的模型复现 RU。多尺度指同一特征层的感受野，不是多层 DINO 特征，也不是 BoQ query attention。

检索：BoQ(RU_gate(F×M))。该分支训练 DINO 最后两个 block、BoQ 和 M；RU gate 参数冻结。

分割：plain 使用 head(F)，guided 使用 head(F×stop_gradient(M))。CE 更新分割头和共享 DINO，不能更新 M，也不会直接更新 BoQ。对图 M 的 detach 同时切断它对输入 F 的额外梯度，但保留乘积中 F 到 CE 的直接梯度。

| 组 | 检索注意力 | 分割监督 | 分割输入 |
| --- | --- | --- | --- |
| vpr_only | 同一新增 M | 无 | 不运行 |
| plain | 同一新增 M | 同一 ADE20K CE | F |
| guided | 同一新增 M | 同一 ADE20K CE | F×detach(M) |

所以 guided vs plain 隔离语义引导；guided vs vpr_only 隔离辅助任务的作用；三组 vs 冻结 RU 测实际净收益。plain 不是历史 SegAux 原封不动复跑，它也有新增检索注意力，以控制结构容量差异。

## 预先固定的实验设置

seed42，GSV 全城市，P40/K4，280×280，photometric-only，已完成 grid20 语义缓存。两组语义头各预热500步且不更改检索参数，联合训练5轮，语义权重1000步渐增至0.02。三组联合阶段重设同一数据/增强随机种子；vpr_only 无辅助预热。FP32，骨干/BoQ LR2e-6，注意力 LR1e-5，分割头 LR1e-4。

每组开始完整复现 MSLS 675/740、Pitts30k-val 7160/7608；每轮评估两个验证集，保存全部轮次和 last.ckpt。主比较固定最终第5轮，不事后按各组最佳轮次挑选。验证结果仅用于实验诊断；不能宣称这两个验证集是新的独立测试集。

成功判据：guided 在两集最终 R@1 均不低于 RU，且严格优于 plain 与 vpr_only 至少一个查询，并在至少一个集合净纠错为正。单种子筛查满足也只算继续验证信号，须补齐成对纠错/回归与多种子，不宣称稳定收益。若只提高分割准确率，或仅超过退化的对照而低于 RU，不算 VPR 成功。

## 训练机操作

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
bash scripts/run_vpr_guided_semantic.sh smoke
bash scripts/run_vpr_guided_semantic.sh train
```

脚本使用物理卡1，因此 Python 内部 --device 0。脚本先在训练机运行旧/新单元测试；smoke 每组2个联合更新，不是正式训练。正式训练依赖三组 smoke 完成。每个日志路径唯一；已有完成运行跳过，未完成正式运行若存在 last.ckpt 则恢复。不会删除旧结果、权重、缓存。

断点按已有 SegAux 的 epoch checkpoint 恢复优化器和学习率；不能精确恢复多进程数据 worker 的随机状态。中断次数应进入结果报告。若输出目录已存在但没有有效 checkpoint，先检查日志，不要删除或假装能够恢复。

推理必须用更新后的 scripts/eval_condition_robustness.py 加载，它严格重建注意力模块；旧的通用 RU 加载逻辑会漏掉该模块。训练期自动检查语义梯度不进入注意力，以及 VPR 注意力初始梯度非零。单元测试覆盖初始 RU 一致、优化器参数完整、梯度方向和严格 checkpoint 恢复；本机未运行这些测试。
