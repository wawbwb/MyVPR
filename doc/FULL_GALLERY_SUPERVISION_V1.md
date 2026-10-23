# 新候选：全训练图库地点原型监督的有界细化

2026-10-24用户要求换方案并继续实验。本轮改变学习目标，而不继续堆叠空间/通道结构。复用上轮LGGR式有界模块，让同一模块在不同监督下比较，判断全库地点判别是否带来额外有效信号。

## 依据与局限

两篇本地ReID稿使用ID分类+Triplet，多输出/融合模块受判别监督；当前VPR主干适配主要采用批内MS。上轮MS约90%步非零，不能说没有训练信号；本轮假设是批内16地点未充分覆盖全图库的相似地点，不是再次宣称零损失饱和。

使用训练地点代理的度量学习已有先例，例如Proxy Anchor（CVPR2020）：https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html 。本轮不是该方法复现，也不能把“加分类损失”本身称为论文创新。可检验的是：全库、去除自身的同地点原型监督能否让当前有界模块获得普通MS没有的净收益。

## 训练专用全库目标

从已封存创新点1 FP32训练缓存读取8192地点×4视图，无新图像提取。每地点四图描述子均值L2归一化作固定原型，不参与优化。对某查询，自身地点的正原型改用**另外三图**的均值归一化；避免把查询自己包含在正原型中。负地点仍使用四图原型。

所有查询同时与8192地点比较，cosine×20作softmax logits，交叉熵监督真实训练地点；无可学习分类器，不在推理保留原型。GPS排除使用球面中心及四图最大半径：只有查询到中心距离减半径≥25m的其他地点才作为安全负原型；这比仅中心距离更保守，同地点正类不屏蔽。

目标为MS + 0.1×全库原型CE；MS对照目标仍为MS。系数、scale固定不扫参，CE在MS为0时也可提供梯度。不用验证标签、错误查询ID或少量固定困难配对生成监督。记录CE、原型训练正确数、安全负类数、CE对描述子的梯度范数，防止训练头吸收损失或辅助目标没有到达最终描述子。固定原型可能限制表示变化或不够表达多视角，这是待验证风险。

## 四组2×2对照

|组|网络|监督|
|---|---|---|
|continue_ms|仅创新点1 neck副本|MS|
|refine_ms|neck副本+上轮有界细化|MS|
|continue_proto|仅neck副本|MS+0.1全库CE|
|refine_proto（预指定候选）|neck副本+同一有界细化|MS+0.1全库CE|

同结构两组同参数、同初值；原DINO/BoQ和归档anchor保持冻结，neck副本lr1e-5，细化lr1e-4；新细化约100928参数，实际残差≤当前特征1%，初始零残差。MS两组用于复核上轮继续训练/联合细化，不把旧实验总分当本轮唯一对照。

固定seed42、8192地点、16×4、322、FP32、AdamW wd0、clip1，原计划前1024步，8步smoke丢弃，固定末步，不选最佳epoch。完整64图计算目标，再微批2梯度回放。训练原型约384MiB GPU，无额外推理网络。GPU1分配20%+2GB余量、旧排他锁，不终止其他作业。缓存只读、来源SHA入合同。

## 判据与归因

末步完整GSV留出/MSLS/Pitts和完整图库，基线必须复现3049/695/7276及全部Top20。refine_proto在Pitts严格胜过基线及另外三组，MSLS不低于基线及三组，GSV不低于基线，才算探索信号。输出各条配对差值以观察结构×监督关系。

若只有continue_proto有收益，只支持新的训练目标，不支持细化结构必要；若refine_proto不胜continue_proto，不能把分类收益称作“模块叠加成功”。仅训练CE下降或多纠错但更多退化都不算成功。反复开发集单seed过关仍不证明稳定收益/新颖性。失败停止，不自动扫scale/CE权重或延长训练。

## 执行

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_full_gallery_supervision.py --allow-training --allow-shared-gpu
cat .cache/full_gallery_supervision_v1_launcher/progress.json
```

输出logs/full_gallery_supervision_v1，状态另记FULL_GALLERY_SUPERVISION_V1_STATUS.md。所有新文件独立，旧论文PDF仅本地参考，不上传。
