# GSV 困难监督修正：independent / set

2026-09-15。不是新语义路线，也不是全量 Pair-VPR 微调。

## 动机与范围

candidate_training_audit_v1：1024 查询中 Pair 已正确 1003，候选可达错误仅 5，
不可达 16；5 条错误占初始损失 83.06%，最高损失 11 条占 96.57%。
旧 Pitts 开发结果 independent=986，set=985，frozen=985，尚无集合交互收益。
本实验检验增加困难监督后是否改变这一结论，不预设有效。

默认从 GSV 原训练城市取 10240 地点，排除旧 Barcelona/WashingtonDC 城市留出。
先按固定地点哈希划分 8192 训练 / 2048 留出；每地点 1 查询、2 参考。
两边参考库也隔离；路径、地点、panorama 不得交叉。引用原 choose_views 的
最早查询/最近两张参考规则，不声称解决了所有跨季节或跨域差异。
留出集保持全部查询，不能按困难程度筛选。官方预训练可能见过这些地点，
这里只是本次 adaptation 的地点隔离，不是预训练未见数据保证。

## 预算与缓存

新库改变了候选集合，旧 pairs 不能直接复用。保留所有 v1 缓存与报告。
复用原严格权重加载、原生提取、双向 CLS、分片 hash 和续传实现；仅在新入口
进程扩展代码指纹，不修改旧源文件。新缓存不可交给旧入口混用。

10240 查询 × top20 × 双向 = **409600 次 pair forward**，另有 dense/global 提取。
FP32 evidence + db_vectors 约 **1.56 GiB**，加 global、元数据与小文件开销；
建议预留 4 GiB 磁盘，提取器收尾合并时建议有 8 GiB 以上可用 RAM。
不保存全库 dense token。GPU 时间取决于机器，可能数小时；不能当成几分钟 CPU 审计。
先 prepare 查看预算，再决定 cache。缓存中断可原命令继续，损坏分片会报错，
不会静默忽略或删除。不要同时运行两个写同一目录的进程。

## 挖掘与门槛

仅在训练池中选择：全部可达错误、等量最低 margin 正确、等量其余正确（固定哈希抽样）。
不复制错误凑数，不把不可达查询当成全负监督。记录所有可达错误，包括残差范围阻塞者。
训练至少 40 个错误地点、4 个错误城市；留出至少 10 个错误地点；三组必须足量。
这是预先登记的试验可行性门槛，不是统计显著性。失败退出 3 并保存报告，
不要降低门槛或查看 Pitts 错误来选择 GSV 训练样本。

## 两组训练

仅 independent / set；相同初始权重、样本顺序、重复增强、5 epochs、lr=1e-4、
batch=8、AdamW wd=.001、grad clip=1、FP32、±4 tanh。
多正样本 loss 用 softplus(logZneg-logZpos) 避免近零 float32 相减消失，两组一致。
checkpoint 按完整 GSV 留出 R1 选择，平局取最早，包含 epoch0 冻结基线。
逐 epoch 同时报告 selected-train 与完整 holdout 的纠错/回退。
每 epoch 原子保存模型、optimizer、RNG、已选最佳模型；--resume 从最近完整 epoch
恢复，中断 epoch 重跑，不声称从中断 batch 精确恢复。checkpoint 只加载自己产生的文件。
本阶段不访问 Pitts、不自动开启跨域评测。先区分训练拟合与 GSV 留出泛化。

## 训练机命令（每条单独执行）

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_candidate_hard_screen.py -v
python -m unittest discover -s tests -p test_candidate_set_torch.py -v
python -u scripts/candidate_hard_screen.py prepare --output doc/candidate_hard_plan_v1
```

确认预算后，顺序执行：

```bash
python -u scripts/candidate_hard_screen.py cache --split train --output .cache/candidate_hard_v1/train
python -u scripts/candidate_hard_screen.py cache --split dev --output .cache/candidate_hard_v1/dev
python -u scripts/candidate_hard_screen.py mine --output doc/candidate_hard_mining_v1
```

mine 通过且检查 summary 后才执行（不与前面命令自动串联）：

```bash
python -u scripts/candidate_hard_screen.py train --output logs/candidate_hard_v1
```

训练中断：

```bash
python -u scripts/candidate_hard_screen.py train --output logs/candidate_hard_v1 --resume
```

Windows 下载轻量挖掘报告：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_hard_mining_v1 D:/Desktop/MyVPR/doc/
```

训练完成后下载 JSON/小型预测文件（不需下载所有缓存）：

```powershell
New-Item -ItemType Directory -Force D:/Desktop/MyVPR/doc/candidate_hard_report_v1
scp 'wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/logs/candidate_hard_v1/*.json' D:/Desktop/MyVPR/doc/candidate_hard_report_v1/
scp 'wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/logs/candidate_hard_v1/*_holdout.npz' D:/Desktop/MyVPR/doc/candidate_hard_report_v1/
```

这只是部分下载，completed 内的 checkpoint 不在本地，不能声称完整 run 哈希核验通过。
本机按用户要求未执行测试/训练；新增 CPU 测试及旧 CUDA 模型测试由训练机执行。
