# 键值协同校准V1：服务器启动记录

用户在方案后授权实现和实验。2026-10-23北京时间22:06:34启动，launcher PID3086489；实现/协议本地提交509e866，六个新文件通过SCP同步服务器，未声称已推送GitHub。创新点1及旧实验不改。

服务器先通过15项新模块测试，最终launcher通过50项新增/回归测试；计划核验通过；8步真实smoke全部7项检查通过，完成文件散列独立验证。四组零起点描述子误差均0；每组smoke有7/8步检索损失非零，全部新增参数收到非零梯度。冻结anchor SHA：85efdb822f8a805e5ba69dca0a27521c09954ce9cdee18e1c8cbc30d8212eee6。

22:06:58进入正式训练，独立检查时16/2560步；约1.31秒/步，CUDA分配峰值879.7MiB。按当前速度训练约56分钟，完整验证另需约20分钟，随负载变化，预计总计约70–90分钟。当前只是运行与机械验证，不是收益结论。

实测新增参数：plain_kv73728、value_only50752、local_kv75328、permuted_kv75328。Local与Permuted初始状态SHA相同，CPU非零权重测试确认局部打乱确实改变Value更新，并不改变Key更新；不是对GAP无效的伪打乱。其余两组容量差异按协议报告，不能称四组完全同容量。

第16步四组都挖到403个安全负对、涉及7个负锚查询，主损失约0.08754。Local组K/V相对更新约2.20e-4/6.37e-6，K/V输出投影梯度范数约2.09e-4/7.78e-6。这提示当前Key路径更新更强，但只是早期批次，不能宣布Value无效；保持预定配方，后续按日志检查全程变化，不按验证或早期单批修改缩放。

所有组从指定零残差重新开始正式训练，不继承smoke；冻结DINO/C-neck/原BoQ权重，仅新增K/V校准参数训练；完整64图挖掘、微批梯度回放。有效loss步、难负数、K/V梯度和描述子漂移已经记录。最终固定2560步后自动三套完整验证，基线要求完整Top20复现。

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/key_value_coadapt_v1_launcher/progress.json
cat logs/key_value_coadapt_v1/train/progress.json
```

最终报告logs/key_value_coadapt_v1/eval/summary.json。当前已启动，不需重复执行；后续先验证smoke/train/eval封存，再按KEY_VALUE_COADAPT_V1_PROPOSAL.md预定判据分析，不能用运行完成代替性能成功。
