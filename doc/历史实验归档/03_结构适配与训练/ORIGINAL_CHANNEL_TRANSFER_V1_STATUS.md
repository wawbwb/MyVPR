# 原版通道迁移V1启动记录

**完成更新：2026-10-23 18:02:43完成全部阶段；20:28独立复核通过。三种原版迁移方案均未超过创新点1，详见[最终分析](ORIGINAL_CHANNEL_TRANSFER_V1_RESULT.md)。以下保留原启动记录。**

2026-10-23北京时间16:09:29启动，launcher PID2438392。本地实现提交cafc338；六个新文件已SCP同步到服务器。未声称GitHub推送成功。原创新点1、原通道和moment实验源码/权重未改。

服务器43项CPU测试通过；计划核验确认seed42训练顺序SHA与原版相同；8步真实smoke通过全部7项检查，封存文件散列独立核验通过。direct与original_init描述子零起点误差0；transferred起点相对创新点1最大描述子差0.0013083，是载入已有非零残差的预期行为。三组所有参数都收到非零梯度，冻结anchor与保护源码保持不变。

16:09:58开始正式三组训练；独立查看时22/2560步，每步约1.94秒，CUDA分配峰值565.6MiB。训练估计约83分钟，随后完整验证预计约20分钟，随负载变化；最终以launcher完成状态为准，当前不能判断收益。

### 哈希核对及协议文字勘误

原初始和迁移权重实际校验的是机器读取的旧initial.json与summary.json，并非手写哈希常量，两项均精确通过。

- original_init：`9e1f2244086e2acfb68ba6f59b46afc07704a5ad68a40e7043bfd9ed96a160dd`
- transferred：`1e1a83a24a7e46f283f1dc66a8d80fc1cc8f0879154469d3f4aa7571e562659c`
- 冻结创新点1：`85efdb822f8a805e5ba69dca0a27521c09954ce9cdee18e1c8cbc30d8212eee6`

预注册文档手写的两个适配器哈希各在末尾重复了一字符（成为65字符），是终端换行抄录错误；以上为正确64字符值。代码始终读取历史记录并校验，没有使用错误文字值。为保持已封存协议不变，在此勘误而不修改运行中的源文件/合同，不影响初始化或权重来源。

direct组使用原模块spatial层名承载此前middle层的相同张量，因此模块state_hash因键名改变而不同；CPU测试已核对映射后前向严格相同。不能仅比较不同键名模块的状态哈希判断张量不同。

### 后续读取

```bash
cd /home/wt/workspace/OpenVPRLab
cat .cache/original_channel_transfer_v1_launcher/progress.json
cat logs/original_channel_transfer_v1/train/progress.json
```

完成报告位于logs/original_channel_transfer_v1/eval。需核验smoke/train/eval封存后分析五组结果。三训练组为direct、original_init、transferred；不训练对照为innovation1与transferred_frozen。结果解释和判据见ORIGINAL_CHANNEL_TRANSFER_V1.md，当前无需重复启动。
