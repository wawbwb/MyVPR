# 采样位移冻结机制验证V1启动记录

**已完成**：2026-10-25北京时间17:56:16，cpu_tests/calibrate/eval全部结束。20:11起独立核验；轻量报告已下载，13份封存文件SHA256核对通过。完整结果见[SPATIAL_SAMPLING_AUDIT_V1_RESULT.md](SPATIAL_SAMPLING_AUDIT_V1_RESULT.md)。

训练整图与训练集标定固定平移的11,420条查询Top1完全一致；训练局部、固定局部、每图均值的正确集合三套均相同，Pitts仍只纠正118。去均值局部场及其打乱版均回到基线R@1，未建立局部对应独立收益。自动verdict仅为READ_ONLY_MECHANISM_AUDIT_COMPLETE；当前版本不作为有效第二创新点。本次没有新训练。

以下为保留的启动与标定记录。

2026-10-25北京时间17:33:33启动，launcher PID3376459，实现及预注册提交be6c959。五个新增文件已SCP同步服务器；未声称已推送GitHub。原模型、原实验源码/权重均只读；没有优化器或训练。

服务器5项新增测试通过，launcher共20项新增/原重采样回归通过。17:34:03完成1024张训练图标定并进入三套完整验证，标定封存已独立核验。训练图上干预包装与原adapter前向误差≤1e-6。

标定结果（图像均值的统计，不是所有patch的符号统计）：

|来源|固定平均xy（patch）|图间标准差xy|平均空间标准差xy|图像均值正号比例xy|
|---|---|---|---|---|
|整图模型|(+0.989894,-0.987609)|(0.01020,0.01153)|(0,0)|(1,0)|
|局部模型|(-0.897630,-0.849261)|(0.06512,0.07658)|(0.12091,0.16314)|(0,0)|

整图位移在这1024训练图上接近固定方向和幅度，局部模型也有显著全局分量；仅凭标定不能断言全部验证图相同、不能证明常量已解释检索收益。两组横向方向相反，不能笼统说所有模型学到同一个固定平移。

后续八版本统一评测：innovation1、trained_global、constant_global、trained_local、constant_local、local_mean、local_centered、local_centered_shuffled。原三份基线/训练模型报告须完整Top20复现。新的干预全部报告，不按最高验证分数选方法。

启动时预计完整验证约25–35分钟，实际17:56:16完成。验证只生成机制诊断，不自动调参或开启新训练。

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/spatial_sampling_audit_v1_launcher/progress.json
cat logs/spatial_sampling_audit_v1/eval/progress.json
```

最终报告logs/spatial_sampling_audit_v1/eval/summary.json。当前已完成，无需重复运行。
