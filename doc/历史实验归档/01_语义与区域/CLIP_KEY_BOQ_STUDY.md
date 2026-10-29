# CLIP-Key Guided BoQ：连续局部语义引导的隔离试验

## 目标与边界

在官方 DINOv2-BoQ 权重上，仅增加可训练的 query/key 投影和门控，检验 CLIP 的连续局部表征是否能改善 BoQ 注意力分配。不是已证实的创新/有效性结论，不是创新点1的替换版本。本方案不改动创新点1、旧区域池化实验及其权重、归档、契约。

旧 SSRP 试验中，aligned 在 MSLS 与另外两个训练对照相同，且三者均低于官方基线；Pitts-val 的微小组间差异不足以证明语义收益。因此本次避免区域均值池化、18类概率瓶颈、语义拟合辅助损失和原始 BoQ 微调。

## 模型

原始冻结 BoQ 每层 cross-attention 的视觉 logits 为 L，visual Value 为 V。冻结 OpenAI CLIP ViT-B/16 的末层使用 SCLIP CSA 方程，取文本类别 softmax **之前**的 512维 patch 特征 S。训练/推理都需要 CLIP；不再将“蒸馏质量”和“语义信息是否有用”混在一起。

S 为224输入的14×14网格；双线性插值到 DINO 的23×23，**不宣称插值增加真实空间分辨率**。先移除图像内全局均值，再用独立可训练投影将冻结 BoQ query 与局部 S 映射到64维空间，归一化后计算相似度。

```
C = 5 * cosine(Wq q, Wk LayerNorm(S - spatial_mean(S)))
B = 2 * tanh(beta_query) * tanh(C - spatial_mean(C))
A = softmax(L + B)
O = A V
```

每个 BoQ block 独立路由器；同一 query 的 bias 对所有 attention heads 共享。beta 零初始化，投影非零随机初始化。零起点返回原始无mask内核输出，同时保留有偏注意力对门控的导数，避免 CUDA MHA 的零mask内核差异破坏官方复现。门控是有符号、幅值有界的，并非显式车辆/行人抑制。

冻结整个原始网络，保持 eval 模式；不改动 visual Value、BoQ encoder/self-attention/FC 权重或运行模式。不做跨图像匹配，不做候选重排序，每张图独立输出12288维全局描述子。推理多一次 CLIP 编码，其时延/开销必须另测，不能称为免费语义。

## 公平三组与训练协议

- visual：固定随机投影把 DINO patch 映射到512维，再用相同的新增路由器；不使用 CLIP 信息。
- aligned：正确位置的连续 CLIP patch。
- shuffled：按图像路径确定、保持512维向量完整的空间无固定点置换；在插值前打乱。

三组新增参数量、随机初始化、投影空间维度、数据、优化器、最终步数完全相同。比较 aligned 对 visual 和 shuffled，而不只对官方基线。

只读复用已封存 GSV 计划：8192训练地点，1024适配留出地点，每地点4固定视图；训练只访问32768张训练图片，P16×K4，5epoch=每组2560步。官方预训练可能见过这些地点，所以留出不是“基础模型未见数据”的声明。

只使用带25米异ID负样本排除的地点 Multi-Similarity 检索损失。没有 KL、分类标签、attention拟合、语义辅助损失。记录零梯度步比例；不能因为分支有非零梯度就声称检索收益。训练不读取 MSLS/Pitts/Nordland/AmsterTime 的标签，Pitts-test 保持未用。

FP32，AdamW lr=3e-4、weight_decay=0，梯度范数截断1，microbatch=2，但损失使用完整64张检索批次。官方网络冻结，只保存新增参数+优化器+RNG+精确步数；每16步原子保存，可恢复到最近保存步。进程中断最多重算15步，不复用smoke的更新。

训练专用连续特征缓存约6.125 GiB，float16仅作磁盘存储、读入转FP32；不缓存全库DINO/CLIP密集特征。每128行flush并记录内容SHA256；可从已提交游标继续，坏块显式拒绝，不静默覆盖。训练/eval source、配置、权重、原始图片、计划、缓存均记录哈希。

## 文件与执行

新增：`src/models/clip_key_boq.py`、`src/models/clip_dense_keys.py`、`scripts/clip_key_study.py`、`scripts/launch_clip_key_study.py`、`config/clip_key_boq.yaml`、`tests/test_clip_key_boq.py`。不修改原始共享模型文件。

训练机 VPR 环境执行以下命令会串行跑服务器CPU测试、只读检查、缓存、三组smoke和三组正式训练。任一步失败立即停止，不自动评测，也不结束其他进程。

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_clip_key_study.py --allow-training
```

launcher 使用物理GPU1，子进程启动前设置 CUDA_VISIBLE_DEVICES=1，内部逻辑cuda:0；CUDA allocator上限为显卡总容量20%。仍会共享GPU算力，不保证不影响其他任务速度。显存不足时报错停止，不切换GPU、不杀进程。

查看 `.cache/clip_key_launcher_v1/progress.json` 与对应日志；各训练组 `logs/clip_key_boq_v1/<arm>/progress.json`、`last.pt`、`summary.json`、`completed.json`。缓存和smoke均不是正式训练完成。只有三组各自封存才表示本轮训练完成；`TRAINED_NOT_EVALUATED` 不是收益判定。

完成后先做官方/visual/aligned/shuffled在相同开发协议下的成对评测。仅当aligned同时优于容量对照与空间错位对照，且相对官方没有不可接受的回退，再多种子复核；不通过则归档，不在测试集上调参。

## 思想来源与差异

- [BoQ，CVPR 2024 官方代码](https://github.com/amaralibey/Bag-of-Queries)：保留查询聚合和原始视觉Value。
- [SCLIP，ECCV 2024](https://arxiv.org/abs/2312.01597)：借用末层QQ/KK相关注意力，取得连续dense表征。本实现无滑窗/PAMR，不宣称完整分割复现。
- [DenseCLIP，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rao_DenseCLIP_Language-Guided_Dense_Prediction_With_Context-Aware_Prompting_CVPR_2022_paper.html)：借鉴语言图像预训练特征用于密集视觉适配的方向，不照搬其文本类别监督。

先前项目已有查询语义bias实验失败，因此“attention加语义bias”本身不是本次可直接主张的新颖点；本次是针对其概率瓶颈/蒸馏/原始权重漂移的可证伪受控检验，收益和新颖性均待证。
