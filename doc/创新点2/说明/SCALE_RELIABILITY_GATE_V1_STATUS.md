# 中心化尺度可靠性门控运行记录

**已完成并独立核验，2026-10-26日09:39:30汇总封存。** 三个种子的五组1024步及三套16版本验证全部结束，27测试通过；144份版本报告按原GT重算、最终状态/封存核对通过，83份轻量快照已下载并核验摘要。候选与冻结标量在全部种子和数据集上的R@1正误集合完全相同，R@5/10/20计数也未增加；正确组对应没有超过同权重推理打乱。本轮未补强独立门控优势，不替换旧候选。详见[SCALE_RELIABILITY_GATE_V1_RESULT.md](SCALE_RELIABILITY_GATE_V1_RESULT.md)。

以下为保留的启动与中途记录，不代表当前进度。

2026-10-26，北京时间09:29:05正式启动有限流水线，launcher PID855895。代码本地提交a4c5dd6，六个新增文件经SCP同步、包SHA256核对一致；没有通过GitHub拉取或覆盖旧源码。方案见[SCALE_RELIABILITY_GATE_V1.md](../../SCALE_RELIABILITY_GATE_V1.md)。

服务器27项测试通过（15项新门控/教师/优化器/边界测试，12项旧融合回归），原结果和缓存封存预检通过。43、44、45每种子8步真实缓存冒烟均通过；五组全部有梯度并更新，零输出精确等于该种子的冻结标量混合，中心化组均值误差小于1e-7。

09:31:54检查时，正式训练seed43已到544/1024步；候选组间门控标准差约0.00193，静态约0.00050、打乱约0.00015、MS-only约0.00067。此为训练过程诊断，不是检索收益。完整预算是3种子×5组×1024步，随后统一三套验证及汇总，不选最好种子或中间checkpoint。

另准备scripts/audit_scale_reliability_gate.py独立核验GT、预测、成对计数、最终头/训练状态/封存，以及主要查询区间；所有训练结束后再执行，不更改训练源码或结果。只在服务器测试和运行，本机仅编辑/管理文件。

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/scale_reliability_gate_v1_launcher/progress.json
tail -n 8 .cache/scale_reliability_gate_v1_launcher/train.log
# 完成后
cat logs/scale_reliability_gate_v1/aggregate/summary.json
CUDA_VISIBLE_DEVICES='' /home/wt/.conda/envs/VPR/bin/python scripts/audit_scale_reliability_gate.py
```

截至上述检查，尚无验证结论；新实验的旧开发集使用限制、固定平均比例/全局对照映射差异均见方案。后续完成状态及分析在本文件顶部和结果文档更新。
