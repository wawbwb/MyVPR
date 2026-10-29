# 可靠性准入V1启动记录

**完成更新：21:20:58完成全部阶段，21:41独立复核通过。拟合集通道独有正确样本为0，未达准入，分类器未拟合。详见[最终分析](CHANNEL_RELIABILITY_V1_RESULT.md)。以下保留启动记录。**

2026-10-23北京时间21:11:26启动，PID2989311。实现和结果读取前协议本地提交d4e8800，四个新增文件已SCP同步服务器；未声称GitHub推送成功。CPU新增/固定融合回归11项测试通过。

首次查看时处于cache阶段，12/512分片，每片64图，共32768张原封存训练图。模型冻结、图像哈希及分片行序核验正常，未读取MSLS/Pitts或GSV适配留出标签做拟合。后续自动训练集检索和CPU可靠性分类器拟合/留出准入，到此结束。

该轮的拟合/留出划分只对可靠性分类器成立；冻结两分支以前在8192地点上训练过，结果只能作为偏乐观的训练信号检查。不得把通过准入称为未见场景泛化。

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/channel_reliability_v1_launcher/progress.json
cat .cache/channel_reliability_v1/channel/progress.json
```

最终结果doc/channel_reliability_v1/summary.json；完整协议CHANNEL_RELIABILITY_V1.md。当前已启动，无需重复执行。下一次应核验各阶段completed.json及来源散列后分析，不根据训练样本纠错数直接下成功结论。
