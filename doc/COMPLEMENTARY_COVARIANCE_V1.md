# 创新点2候选：局部—全局二阶互补描述子

## 1. 问题与研究假设

2026-10-21，先前单DINO交互分支V1、修正监督的V2，以及V2完整数据库去均值干预均未取得跨数据集一致收益。V2的原始分支在MSLS/Pitts验证集分别为693/7274，冻结创新点1为695/7276；去均值为695/7274，仅均值为693/7276。不能由此宣称“动态交互有效”或“去均值已经解决问题”。

新假设：不要再将小幅交互残差送回冻结BoQ。固定原有全局表示，另学一个受直接地点监督的互补描述子，使局部邻域响应与全局上下文响应的共现关系拥有独立的读出与梯度路径。

## 2. 参考依据与迁移边界

第一篇私人稿件的多尺度融合、分支监督支持局部结构建模；创新点1已经借鉴其跨层融合，不在本轮重复声称创新。第二篇私人稿件使用MambaVision与OSNet、阶段交互、动态低秩交互，并分别监督主干与融合描述子、测试时融合描述子。本轮仅借鉴“异构算子互补、低秩条件化、直接监督、描述子级融合”，不是忠实复现其双骨干/Mamba结构，也没有使用其数据或身份分类头。原件仅本地阅读，不上传服务器。

已知先例：

- Lin等，Bilinear CNN Models for Fine-Grained Visual Recognition，ICCV2015：https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html 。局部双流外积及池化不是本方案首创。
- Zhang等，Learning second-order statistics for place recognition based on robust covariance estimation of CNN features，Neurocomputing2020，DOI：https://doi.org/10.1016/j.neucom.2020.02.001 。VPR二阶统计也不是本方案首创。

候选区别仅在于：在已经冻结的跨层增强表示之上，学习局部/上下文异构算子的低秩交叉统计，作为直接受监督的独立补充描述子，而非改写原BoQ聚合。此组合是否具有学术/专利新颖性仍需专门检索；实验有效性也尚未证实。

## 3. 单阶段网络

```text
RGB -> 单个冻结DINO(L8,L10,L12) -> 冻结创新点1 -> F1
                                                    |-> 冻结BoQ -> d0(12288维)
                                                    |-> 局部3x3/5x5卷积分支 L
                                                    |-> 上下文点映射分支 G
                                                         |
                                    全局条件通道门控 + 各投影32维
                                                         |
                                    同位置中心化交叉协方差(32x32)
                                                         |
                                    平滑符号幂归一化 -> FC -> h(512维)
d = concat(d0, 0.25 h) / sqrt(1 + 0.25^2) -> 全库一次余弦检索
```

上下文来自DINO的全图自注意力，不额外实例化ResNet、CLIP、SAM或Mamba。两个分支不是两个预训练编码器。不做候选扩展或重排序，不使用查询与数据库间的交互来提取单图描述子。

设局部/上下文投影为L,G∈R^(N×32)，对每张图独立计算 C=(L-mean(L))^T(G-mean(G))/(N-1)。中心化针对新分支内部统计，不是此前直接给冻结BoQ输入残差去均值的操作。DINO已经隐含上下文，故不能把中心化解释成完全消除了动态物体。

仅训练新头；原描述子d0完全冻结，λ=0时拼接描述子前12288维精确等于d0，后512维为0。λ=.25时，新分支占余弦相似度权重约5.88%，不是25%。额外维度/存储量512/12288≈4.17%，不得声称同维度比较或无额外成本。

## 4. 最小匹配实验

三个分支相同参数量、相同初始化、相同图像/地点批次/优化器/训练步数：

|组别|聚合统计|回答的问题|
|---|---|---|
|mean_outer|L和G各自均值的外积|新增参数、维度与一阶统计是否已足够|
|aligned_cov|同位置中心化交叉协方差|对齐的二阶关系是否有额外收益|
|permuted_cov|逐图打乱G的位置后交叉协方差|收益是否依赖位置对应|

打乱由原始图片SHA256确定，独立于标签、GPS、批次和提取顺序；不是全数据共用一个可逆固定排列。它是可重复的反事实负对照，不是部署所需组件。不额外使用打乱分支挑选评测超参数。

损失为MS(d)+MS(h)，地点批次16×4，忽略不同地点ID但GPS距离小于25m的负样本。直接监督h，不再用相对于原主损失极小的梯度上限压制其学习。λ在前128步线性升至.25，之后固定。监督、描述子融合和统计结构同时改变；旧V2对新方案的比较不能分离这三项，只有本轮三组比较针对统计类型。

固定seed42、原封存GSV的8192训练地点/1024适配留出地点、322输入、同四视图、第一1024步；不是GSV全量训练。先8步真实烟测并丢弃，再从原seed重新开始正式三组。AdamW lr1e-4、weight_decay0、全局梯度范数裁剪1。完整64图批次计算挖掘损失，通过描述子梯度缓存/微批2图回放实现，禁止在微批内独立挖掘。

固定最终步评测GSV适配留出、完整MSLS-val和Pitts30k-val。后两者已被反复观察，是开发集，不是盲测。只有aligned_cov在两个验证集同时优于创新点1、mean_outer和permuted_cov，才记作“探索性二阶互补信号”，否则不宣称创新点2成立。单次一两个查询的收益仍须未来独立确认，不能当稳定性证明。不得见到失败再扩大本轮预算或搜索λ。

## 5. 运行、恢复与保护

新文件全部独立，旧创新点1源码和历史权重只读并封存校验。服务器GPU1，20%显存分配上限；共享资源时显式授权，要求空闲空间至少分配上限加2GiB。不创建新环境、不安装依赖、不终止其他任务。

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_complementary_covariance.py --allow-training --allow-shared-gpu
cat .cache/complementary_covariance_v1_launcher/progress.json
cat logs/complementary_covariance_v1/train/progress.json
```

有限流水线：服务器CPU测试 -> 计划校验 -> 8步烟测 -> 全新1024步三组训练 -> 三集评测。断线不影响运行，异常以failed状态停止。每16步原子保存头、AdamW、RNG、游标和合同校验；意外重启后重用同一启动命令，从最近完整检查点恢复。已完成的阶段仅校验，不重复训练。

代码：`src/models/complementary_covariance.py`；训练/评测：`scripts/complementary_covariance.py`；配置：`config/complementary_covariance.json`；单测：`tests/test_complementary_covariance.py`。本机不执行测试或训练，测试仅在训练机运行。
