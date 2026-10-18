# CLIP-Key BoQ 固定最终模型评测

本次评测只新增独立文件，不修改已封存的训练源码、配置、模型、数据计划、创新点1或旧实验结果。

## 协议

- 组别：官方 DINOv2-BoQ、visual、aligned、shuffled。
- 训练模型一律使用已完成的第5轮/2560步 `last.pt`，禁止依据本轮评测挑选轮数、门控强度或融合权重。
- 开发集：GSV适配留出（1024参考、3072查询）、MSLS-val（18871参考、740查询）、Pitts30k-val（10000参考、7608查询）。不使用Pitts-test。
- 所有组别相同的原始图片顺序，读取同一RGB，分别全图缩放322/224，沿用训练时DINO/CLIP归一化，无center crop。GSV核对原始图片SHA；其他集对图片内容累计哈希。
- 同图共用冻结DINO及冻结CLIP编码结果，只是为减少重复计算；各组分别产生独立的12288维单位全局描述子，不做候选扩张、重排或成对推理。
- CLIP连续特征在推理中执行FP32→FP16→FP32，复现训练磁盘缓存的量化，避免无意改变输入表示；保持14×14→23×23插值和打乱先于插值。
- 整个官方网络冻结，直接严格加载官方完整检查点。新增路由参数逐项严格加载，并核对训练封存哈希。评测无优化器和反向传播。
- 统一以全数据库余弦相似度检索Top20，报告R@1/5/10/20、逐查询Top20，以及aligned相对其他三组的纠错、退化、净收益和Top1变化数。
- 若报告精确McNemar p值，限描述性：开发集重复使用、查询可能相关、单种子，不能据此宣称确认性显著或稳定泛化。

完整密集特征只在内存暂存，逐数据集释放，不在磁盘建立CLIP/DINO全库密集缓存。只保存小型排名和结果；MSLS四组全局描述子约3.6GiB CPU内存。GPU仍仅使用物理GPU1、allocator上限20%。

## 启动与结果

训练机已有VPR环境中：

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_clip_key_eval.py --evaluate
```

后台先运行服务器CPU测试，再顺序评测三个开发集。`.cache/clip_key_eval_launcher_v1/progress.json`为启动器状态，`logs/clip_key_boq_eval_v1/progress.json`为细粒度状态。每个完整数据集原子保存并封存校验，重启可跳过完成的数据集；未完成的数据集从头提取，不使用半成品排名。

结果：`logs/clip_key_boq_eval_v1/summary.json`、三个数据集的JSON、对应ready哈希与最终`completed.json`。评测代码/训练源码/运行时/权重/数据划分/GT/图片顺序均写入独立契约。训练结果不放进评测目录，也不改变训练封存。

评价必须区分：aligned是否胜过官方（实用收益）、是否胜过visual（CLIP附加信息）、是否胜过shuffled（空间对应价值）。三者都不能用训练损失下降替代。
