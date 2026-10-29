# 尺度融合三个新种子重复验证：运行记录

**最终训练及评测已完成**，修复后的汇总于2026-10-26日08:56:08封存并核验。候选43/44/45号在GSV均3051、MSLS均695，Pitts分别7288/7290/7290；相对创新点1为+2/0/+12–14，整体计数增益复现。相对等权平均仍有MSLS少1条的取舍，Pitts平均额外+2.33条的查询区间包含0。详见[SCALE_AGREEMENT_REPEAT_V1_RESULT.md](SCALE_AGREEMENT_REPEAT_V1_RESULT.md)。

原launcher在训练/验证全部完成后因NumPy bool_写JSON失败；已通过独立工具仅修复导出，原失败记录保留，最终入口为logs/scale_agreement_repeat_v1/aggregate_fixed/summary.json。当前完成状态为.cache/scale_agreement_repeat_v1_launcher/report_status.json。不要因原progress.json仍显示failed而重启训练；本轮没有为导出修复重训或改统计策略。

以下为保留的启动记录，不代表当前进度。

2026-10-26北京时间08:42:35启动，launcher PID750699。代码本地提交d1bc1e6，六份新增文件已SCP同步，LF规范化SHA256与服务器launcher全部一致。方案见[SCALE_AGREEMENT_REPEAT_V1.md](../../SCALE_AGREEMENT_REPEAT_V1.md)。

服务器17项统计/原融合回归测试通过；只读来源预检通过。43、44、45三个新种子各8步缓存冒烟全部通过，四组零起点误差为0，所有参数有梯度并更新，受保护源码不变。本机未执行测试或模型前向，没有重新编码图像。

08:44:06检查时，正式阶段已启动，seed43训练53/1024步；三个种子会各完成四组1024步、训练平均门控标定和三套八版本验证。原42号仅作历史参照，不计入新种子统计。随后自动计算成对查询及共享正样本成分bootstrap区间，全部结果保留，不按最好种子或对照替换候选。

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/scale_agreement_repeat_v1_launcher/report_status.json
cat logs/scale_agreement_repeat_v1/aggregate_fixed/summary.json
```

本轮已完成。原创新点1、原V1实验和大缓存未改动。执行期间只使用物理GPU1、20%allocator与原锁，没有终止其他作业或下载依赖，没有自动扩展新种子或参数。
