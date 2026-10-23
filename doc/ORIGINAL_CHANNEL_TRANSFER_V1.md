# 原通道版本迁移到创新点1：预注册V1

用户要求纠正之前“同结构等于原版本”的混淆。旧spatial_adapter_repeat的channel由3×3深度卷积中心核初始化；之前structural/moment使用直接1×1初始化，且未加载旧训练权重。本轮调用原始paired_models/训练构建函数，并校验历史初始参数SHA，避免仅复刻结构。

原版本在官方BoQ上有Pitts收益，不能预先认定其在创新点1上成功。本轮固定seed42来源，不按三种子验证结果挑权重。迁移来源：logs/spatial_adapter_repeat_v1/train_seed42/last.pt，channel分支；原始封存完整性必须通过。

## 三训练组与不训练对照

所有组插在冻结创新点1的C-neck之后、冻结BoQ之前；使用原PatchAdapter（768→64→768，深度1×1，GELU，残差系数0.1），无门控/矩保持；99,264参数、单编码器、12288维。

|组|初始权重|本轮训练|
|---|---|---|
|direct|前一轮直接1×1初始化，映射到原模块同名对应层|2560步|
|original_init|调用原构建函数恢复历史seed42初始张量|2560步|
|transferred|加载历史seed42最终channel张量|2560步，重新建立优化器|
|transferred_frozen|同一个历史最终channel张量|不训练，仅完整验证|
|innovation1|原冻结创新点1|不训练，完整基线|

原初始channel SHA必须为历史initial.json记录的9e1f2244086e2acfb68ba6f59b46afc07704a5ad68a40e7043bfd9ed96a160ddd；原最终channel SHA必须匹配summary.json的1e1a83a24a7e46f283f1dc66a8d80fc1cc8f0879154469d3f4aa7571e562659cc。

注意原构建函数先构造官方网络再初始化适配器，该过程消耗随机数；只在适配器前设置seed42不足以复现。因此完整调用旧构建流程后检查张量哈希，再释放临时官方模型，加载创新点1。两个编码器不会共同参与训练。

原训练批次顺序SHA也必须与封存source contract完全一致。8192地点、4视图、16地点/步，5轮2560步，输入322，FP32，lr3e-4、AdamW wd0、clip1，GPS过滤MS损失+0.1相对创新点1的描述子余弦保持。三组共享当前批次冻结特征，各自独立参数、优化器及梯度。warm组此前已训练2560步，累计训练预算不同，不能把其相对fresh的收益归因为新结构。

8步smoke参数丢弃；正式三组重新从指定初始状态开始。仅direct/original_init应严格零起点，transferred不应被错误要求复现创新点1描述子。全部组检查参数更新与冻结保护。

## 判据与解释

只评固定最终步，GSV留出3072、MSLS740、Pitts7608和完整图库；保存Top20与逐查询结果，基线必须复现旧完整Top20。对original_init、transferred、transferred_frozen分别报告预指定问题，不按结果挑一个包装成唯一成功候选。

每种方案的探索性“叠加收益”要求：Pitts严格高于创新点1，MSLS和GSV均不低于创新点1。额外报告original_init与direct、transferred与direct以及transferred与transferred_frozen的配对结果。只有在满足基线保护且Pitts胜过direct时，才分别标记初始化/暖启动优势。该探索试验不提供盲测或稳定论文收益证明；增加种子或更换数据需后续另定，不自动扩展。

三组若失败，不能继续用同结构等同于原权重的说法。直接迁移若好而继续训练变差，提示训练过程有害；原初始化胜direct才支持初始化因素；两者均无增益则不支持本协议下的原版叠加。上述是待验证判别，不是预设结论。

## 执行

仅服务器物理GPU1，显存分配上限20%、额外2GB余量，沿用实验锁，不终止其他作业。launcher有限依次CPU测试、计划核验、8步smoke、三组2560步、五变体验证。

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_original_channel_transfer.py --allow-training --allow-shared-gpu
cat .cache/original_channel_transfer_v1_launcher/progress.json
```

代码、记录与输出独立；原创新点1、原channel和moment实验均只读。状态另记ORIGINAL_CHANNEL_TRANSFER_V1_STATUS.md，避免改动封存方案。
