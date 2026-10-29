# 尺度一致性条件融合V1：启动与核验记录

**后续授权**：用户要求保留有增益候选并继续验证，已启动43/44/45三个新种子的重复实验，见[SCALE_AGREEMENT_REPEAT_V1_STATUS.md](SCALE_AGREEMENT_REPEAT_V1_STATUS.md)。原42号的结果与原预注册判定保持不变；重复实验分别报告整体增益和机制增益，不将旧判定概括为所有尺度融合均无效。

**最终已完成**：2026-10-26北京时间02:06:03。08:17起读取并独立核验，结论NO_MATCHED_SCALE_AGREEMENT_SIGNAL。候选三套R@1正确数3051/695/7289，相对创新点1为+2/0/+13，相对等权平均为0/-1/+2；Pitts与打乱组正确集合相同，低于整图及训练平均门控7290。完整结果及边界见[SCALE_AGREEMENT_FUSION_V1_RESULT.md](SCALE_AGREEMENT_FUSION_V1_RESULT.md)。轻量报告已下载核验，本次没有新训练。

以下保留启动时记录，其中缓存进度和预计耗时不代表当前状态。

2026-10-26北京时间01:02:44启动，launcher PID4154908。实现本地提交f24e09f；六个新文件已通过SCP同步，规范化LF后的摘要与服务器launcher记录全部一致。方案见[SCALE_AGREEMENT_FUSION_V1.md](../../SCALE_AGREEMENT_FUSION_V1.md)。

服务器42项测试通过，包含新增融合、既有跨分辨率接口及协方差回归；没有在本机运行模型或测试。只读plan通过，8步真实图像smoke于01:03:53前完成并丢弃，随后进入训练518缓存生成。

## 冒烟与独立检查

- static/global/group/shuffled四组各128个参数，初始张量一致，零起点对等权平均的最大绝对误差均0。
- 复用的322训练缓存与512张实际smoke输入的本轮前向最大绝对误差0。
- 四组所有参数均收到非零梯度并更新；各6/8步MS损失非零。原创新点1和受保护源码不变。
- 原anchor完整状态为85efdb822f8a805e5ba69dca0a27521c09954ce9cdee18e1c8cbc30d8212eee6；新增训练不更新它。
- smoke峰值PyTorch分配516.65MiB，不等于nvidia-smi完整进程显存，也不是后续全库检索峰值。
- 01:04:41独立核验5份smoke封存文件、13份实验依赖来源、7份保护源码，并核对原322缓存封存及创新点1checkpoint文件摘要，全部通过。

## 当前进度

01:04:41的快照为训练518缓存19/512分片，即1216/32768图。正式1024步融合头训练尚未开始；缓存完成后流水线自动进入，之后在全部训练图上标定候选平均门控，再缓存验证双尺度描述子并评测八版本。

旧322缓存只读复用，新缓存不保存patch，预计新增约5.3GiB。服务器空闲磁盘约3.2TiB、可用RAM约56GiB；当前阶段只使用物理GPU1，allocator20%加2GB余量，原排他锁，没有停止其他作业。

目前只确认实现和执行正常，没有召回收益结论。按已观察缓存速度，余下约1–1.5小时，随负载和I/O变化；主要开销是一次性冻结编码，小头训练不反复提取图像。完成后仅生成结果，不自动调参、扩预算或开始下一轮。

探索信号须在Pitts超过其余七版本，MSLS不低于其余版本，GSV不低于等权平均；仍是单种子开发结果，不是独立确认。训练集平均门控对照用于识别逐图条件化是否必要，不从验证集选系数。

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/scale_agreement_fusion_v1_launcher/progress.json
cat .cache/scale_agreement_fusion_v1/train_high/progress.json
cat logs/scale_agreement_fusion_v1/train/progress.json
cat logs/scale_agreement_fusion_v1/eval/summary.json
```

后两项在对应阶段开始/完成后存在。流水线已启动，无需重复运行启动命令。独立smoke记录保存在.cache/scale_agreement_fusion_v1_launcher/smoke_independent.json。
