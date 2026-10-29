# 全训练图库监督V1启动状态

**完成更新：2026-10-24 08:21:10完成全部阶段，09:47独立核验通过。候选GSV3049、MSLS693、Pitts7267，未获得叠加收益。详见[最终结果](FULL_GALLERY_SUPERVISION_V1_RESULT.md)。以下保留启动记录。**

2026-10-24北京时间07:05:58启动，launcher PID4011459。实现与预注册提交b1c0703，六个新增文件已SCP同步到服务器；未声称GitHub推送完成。原创新点1、训练描述子缓存及旧实验全部只读。

服务器先通过18项新监督/旧有界模块回归测试，launcher通过41项新增/回归测试；计划核验通过，8步真实smoke全部8项机械检查通过，封存散列独立验证。四组零起点描述子误差均0；全部训练参数收到非零梯度，原anchor和aux未变。MS同结构两组容量与初始化配对一致。

07:06:45进入正式训练，首次独立读取2/1024步，每步约2.88秒，CUDA分配峰值1386.4MiB；预计训练约49分钟，完整验证另约20分钟，总计约70–90分钟，随资源负载变化。

第2步主损失约0.11834；原型两组加权CE约0.12606，对描述子加权梯度范数约0.1540；MS对照组原型梯度为0。当前批64条查询的原型Top1均正确，但CE仍非零，说明分类正确不等于softmax目标饱和；不能把这个训练准确率作为新收益证据。当前每查询8191个安全负类，正原型排除了自身。

模型方面不增加新的深度结构：continue两组训练neck副本，refine两组再加上轮100928参数的1%有界模块；改变的是有无全训练图库原型监督。主候选须胜过同监督无细化和同细化无原型对照，才能考虑组合贡献。

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/full_gallery_supervision_v1_launcher/progress.json
cat logs/full_gallery_supervision_v1/train/progress.json
```

最终报告logs/full_gallery_supervision_v1/eval/summary.json。当前已启动，无需重复执行。到时先核验三阶段封存、完整基线Top20，再按FULL_GALLERY_SUPERVISION_V1.md的预定条件分析；未获得性能结论，不自动调参或选检查点。
