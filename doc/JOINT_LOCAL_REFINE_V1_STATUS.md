# 联合局部—全局细化V1启动状态

**完成更新：2026-10-24 01:25:37完成全部阶段；06:57独立核验通过。联合候选GSV3050、MSLS693、Pitts7275，未达到预设条件。详见[最终分析](JOINT_LOCAL_REFINE_V1_RESULT.md)。以下保留启动记录。**

2026-10-24北京时间00:11:24启动，launcher PID3299391。实现与预注册提交b0967ce，六个新增文件已SCP同步服务器；未声称GitHub推送完成。原创新点1源码与检查点只读，联合训练仅更新独立副本。

服务器先通过12项新测试，最终launcher通过47项新增/回归测试。8步真实smoke通过全部9项机械检查并独立验证完成散列；四组零起点描述子误差均0，每组7/8步主损失非零。训练参数均得到非零梯度；原anchor、aux以及frozen_refine的neck副本未改。

实测可训练参数：neck_continue1229698、frozen_refine100928、joint_refine1330626、joint_permuted1330626。联合与打乱组严格同容量同初值；冻结组/继续训练组容量不同，不声称全组同参数。

首次核实正式训练5/1024步，每步约2.83秒，CUDA分配峰值986.7MiB；训练约48分钟，完整验证再约20分钟，资源负载可能改变耗时，预计总计约70–90分钟。原冻结anchor状态检查仍使用既有封存。

smoke第8步新细化残差均值约8.22e-8（joint），neck相对原创新点1漂移约2.24e-4；初始新模块变化很小但上游参数已有梯度，不据早期批次改变上限或训练配方。后续需检查其全程实际作用，不能把联合组收益自动归因新增模块。

当前已从全新指定初值正式启动，无需重复运行。有限流程将自动训练并完成三套完整验证，不按验证挑epoch。状态和结果：

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/joint_local_refine_v1_launcher/progress.json
cat logs/joint_local_refine_v1/train/progress.json
```

最终报告logs/joint_local_refine_v1/eval/summary.json。运行正常和机制检查不等于性能收益；最终按JOINT_LOCAL_REFINE_V1.md预定对照与判据解释。
