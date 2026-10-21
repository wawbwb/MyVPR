# 创新点2候选：局部结构条件化通道适配器（SCCA）

## 1. 哪些证据值得继承

2026-10-22，复核训练机`logs/spatial_adapter_repeat_v1/aggregate/summary.json`：纯通道适配器在Pitts30k-val的三个种子正确数为7277、7276、7274，对官方7264分别净增13、12、10个；MSLS为693、693、694，对官方694分别-1、-1、0。说明轻量通道适配值得作为较强的起点，但不能称跨域稳定提升。其训练为完整5轮2560步，而不是1024步。

上一轮`logs/complementary_covariance_v1/eval/summary.json`中，对齐二阶分支对冻结创新点1：Pitts7277对7276净增1个（纠正10/退化9），MSLS691对695净减4个，GSV3049持平。对齐在Pitts优于两个二阶控制，但该弱信号不能证明异构关系已经有效。更早DINO+ResNet交错路线无收益，不能把它引用为已成功的双编码器方法。

本方案不再次直接融合两种特征，也不扩展描述子维度：用空间结构信息作为通道更新的条件，而不是作为需要冻结BoQ重新解释的值。

## 2. 参考与新颖性边界

- Hu等，Squeeze-and-Excitation Networks，CVPR2018：https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html 。全局上下文生成通道权重已有先例。
- Yang等，Gated Channel Transformation for Visual Recognition，CVPR2020：https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.html 。通道能量统计、归一化与门控已有先例。
- 本项目已读的第二篇私人ReID稿件提供“异构信息条件化交互”的启发，旧创新点1也已使用动态低秩及局部/全局门控。本轮候选区别仅为：不再加跨层/交错/二阶读出，在冻结创新点1之后，利用局部结构对比统计控制纯通道瓶颈更新。不能将“门控”“局部全局”或“低秩”本身当作首创。

这是待验证的网络方法候选，不承诺学术/专利新颖性；当前正信号也不能保证新组合一定上涨。无需再次上传或读取私人论文原件，服务器仅接收本轮源代码/配置/文档。

## 3. 结构与公式

```text
RGB -> 单个冻结DINO -> 冻结创新点1 -> F1(768x23x23)
                                            |
                                     1x1降维+GELU -> H(64x23x23)
                                            |             |
                             均值mu(H)       |     H-AvgPool3(H)的RMS
                                    \        |          /
                                     两类统计各做非仿射LN
                                            |
                                      128 -> 16 -> 64
                                            |
                                 g = 1 + 0.5*tanh(MLP)
                                            |
 H * g -> depthwise1x1 -> GELU -> 零初始化1x1升维 -> 0.1倍加回F1
                                            |
                                  冻结BoQ -> 12288维单位描述子
```

局部统计是`q_c=sqrt(mean_p[(H_c(p)-AvgPool3(H_c)(p))^2]+1e-6)`。AvgPool采用reflect边界，常数图不会因补零制造虚假高频。全局均值与局部对比RMS分别非仿射LayerNorm后输入低秩门控，g范围(0.5,1.5)。它只控制更新分支，不乘/屏蔽原主路，不宣称动态目标抑制。

空间上下文来自DINO；局部算子与全局统计是两类算子，不是两套独立预训练编码器。没有ResNet、Mamba、CLIP、SAM、额外描述子或重排。单图独立提取，训练/推理不依赖同批次其他图片。

末层up为0，初始化精确复现创新点1。继承纯通道适配器gain=.1，无新的硬残差RMS上限；这与此前rho=.025的交错网络不同，必须记录实际残差。仅训练新adapter，冻结全部创新点1和原BoQ。测试前后校验保护源码与tensor哈希。

## 4. 四组固定实验

|组别|控制条件|目的|
|---|---|---|
|channel|无门控，原纯通道瓶颈结构|检查原成功结构接到创新点1上是否仍有增量|
|global_gate|mu(H)+RMS(H)|普通全局条件化能否解释收益|
|structure_gate|mu(H)+局部对比RMS|新候选|
|scrambled_gate|mu(H)+逐图打乱通道身份的局部对比RMS|检查局部统计与被调制通道的对应关系|

三个门控组严格同参数同初始化；普通channel组参数更少，不能把新组对它的上涨单独归因结构。四组down/middle/up初始参数一致，无补空参数。门控MLP有效梯度在第二步后才会越过零初始化up传播，烟测检查每个参数都曾收到非零梯度。普通组与历史channel具有相同运算形式，但不声称随机初始张量与当年的3x3中心裁剪初始化完全一致；本轮四组统一新初始化。

打乱以图片SHA256为种子，每张图片独立，打乱的是局部统计的通道身份，不改变原H、均值、局部统计数值分布，不使用GT/GPS/查询序号。不再做无法影响通道池化的“空间打乱”伪对照。

## 5. 预注册预算与判据

四组共享只读编码、各自独立参数与AdamW，seed42。复用原封存GSV8192训练地点、1024适配留出地点、四视图、P16xK4，完整5轮2560步，输入322，FP32、microbatch2、lr3e-4、wd0、clip1。损失GPS过滤25米假负样本后的MS检索损失 + .1倍对创新点1描述子的余弦保持，不再添加辅助分支损失。全64图计算挖掘和梯度，再微批回放，不能微批内独立挖掘。

先8步真实smoke并丢弃，再全新初始化正式训练。固定最后一步，不选最好epoch/λ/seed；无效后不自动延长。1024步与2560步跨实验不可视为等预算，归因以本轮同2560步四组为准。

评测GSV适配留出、完整MSLS-val740/18871及Pitts30k-val7608/10000，基线必须复现创新点1的695/7276。数据均反复观察过，是开发而不是盲测。沿用通道路线的Pitts主要端点：structure_gate必须在Pitts同时严格胜过创新点1和其余三组，在MSLS不得低于创新点1及三组对照（仅计数保护，不是统计非劣效证明），才标记探索信号。报告GSV全结果、R@1/5/10/20、逐查询纠错/退化，不隐瞒不同数据集方向冲突。单种子成功也不等于稳定增益；是否做多种子复核另行决定，不自动追加。

## 6. 实现与运行边界

新模型`src/models/structural_channel_adapter.py`；训练/评测`scripts/structural_channel.py`；配置`config/structural_channel.json`；测试`tests/test_structural_channel.py`；有限后台启动器`scripts/launch_structural_channel.py`。

只在服务器测试/训练，本机只读写代码。GPU1分配上限20%，闲置不足上限+2GiB则退出，不终止其他任务、不建新环境、不更改旧代码。输出`logs/structural_channel_v1/{smoke,train,eval}`；状态`.cache/structural_channel_v1_launcher/progress.json`。每16步原子保存checkpoint/优化器/RNG/游标，重启后同合同恢复。

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_structural_channel.py --allow-training --allow-shared-gpu
```
