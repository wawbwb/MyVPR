# Frozen Pair-VPR：整图 top20 → top44 真实重排

等预算审计中，dev整图44/fullM/多视图并集均覆盖2021/2048；多视图并未证明额外价值。
本轮只评测固定整图top44，非新语义方法；dev已反复查看，结果只能作探索性证据。

对全部2048个查询：复用原top20双向配对分数，对排名21–44额外计算24对。
保持完整query/reference图、原预处理、模型、FP32、双向分数相加、稳定排序，不调权重、不训练。
新增49152对（98304次定向pair forward），另每query重算旧赢家一对（4096次定向forward）作为分数复现检查。
复现检查不是重算全top20；超过固定atol/rtol 1e-4立即停止，不静默放宽。

源数据与模型哈希校验；对本轮涉及的query/reference图逐一校验原hash。
配对需要重新编码dense特征，使用最多64条CPU内存LRU，不建立dense磁盘缓存。
真实运行时间含dense计算、传输和配对，不能只按配对forward数量估算。
逐query保存44个分数、ID、GT、复现分数及SHA，原缓存只读。
报告包含R@1/5/10/20、纠错/退步ID、净变化、新增可达实际转化，及每query赢家是否来自新增候选。
49,152对只是新增预算，top20不变仍可能被新负候选破坏，不能只评已知错误子集。

## 训练机操作

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_candidate_top44.py -v
```

通过后运行（本机未测试）：

```bash
python -u scripts/eval_candidate_top44.py --output doc/candidate_top44_eval_v1 --resume
```

支持中断续跑，重复同一命令；完整产物验证后直接退出。
匹配契约才允许恢复，坏分片报错，完整分片不重复算。最后不完整query重算最多25对。
不需要RU_CKPT；使用原官方Pair权重与源码身份。物理卡1、逻辑cuda:0。
输出原子npz+sha，中断在写入之间的分片会重算；不覆盖完整报告或忽略哈希不一致。
计时只记录最后这次进程的循环时间与新增query数，不能当作所有恢复进程的总耗时。

Windows下载完整报告：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_top44_eval_v1 D:/Desktop/MyVPR/doc/
```

预期旧基线1994/2048、top44候选覆盖2021/2048是历史参照，不是期望达到的实际正确数。
脚本从原cache重现基线，不通过硬编码准确率强行判定成功。
