# Moment Channel V1 启动状态（2026-10-23）

方案见[MOMENT_CHANNEL_V1.md](MOMENT_CHANNEL_V1.md)。实现提交fbc0585，六个新文件通过SCP归档同步至服务器；GitHub推送先前受403权限限制，本轮未将“已同步服务器”表述为“已推送远端”。创新点1与旧实验源码未改。

北京时间12:47:04启动有限流水线，PID1984971；CPU测试57 passed；计划核验通过；8步smoke通过7项机械检查并独立核验封存文件散列。四组描述子零起点误差均为0，所有参数收到非零梯度，冻结anchor与受保护源码不变。

12:47:48进入正式训练，首次独立读取时4/2560步。每步约2.33秒，CUDA分配峰值570.1MiB；预计训练约100分钟，加完整验证约2–2.5小时，资源负载可能改变耗时。四组同初始化SHA：8441aad7a296443a07cc408aac2759e715b8a3abeaa15ee4d05789abc566dc86，与旧structural-channel普通组初始化一致。冻结anchor SHA：85efdb822f8a805e5ba69dca0a27521c09954ce9cdee18e1c8cbc30d8212eee6。

第4步候选mean_norm的实际相对残差约0.0003597，通道均值相对误差5.61e-8，token范数相对误差2.45e-7；普通组对应误差2.41e-5、1.17e-4。这仅说明矩约束生效且更新非零，不能据此判断检索收益。

流水线状态：`.cache/moment_channel_v1_launcher/progress.json`；训练状态：`logs/moment_channel_v1/train/progress.json`；最终报告：`logs/moment_channel_v1/eval/summary.json`。最终报告尚未生成，本记录不是成功结论。后续读取时需先确认phase/封存状态，再分析完整验证和逐查询纠错退化。

服务器查看：

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/moment_channel_v1_launcher/progress.json
cat logs/moment_channel_v1/train/progress.json
```

当前任务已启动，无需重复运行启动命令。若异常退出，只能同版本、同资源策略恢复；不要覆盖权重、修改封存协议或按验证选检查点。
