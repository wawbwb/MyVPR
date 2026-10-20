# CV-Slot-BoQ：跨视图对应教师预检

## 目的与边界

前几轮CLIP回归、聚合门控及变化对抗约束没有提供超过同结构视觉对照的明确收益。
本次不继续放大语义权重，而先验证更窄的假设：**在DINO给定的同一组候选对应边上，
正确CLIP连续patch特征能否比DINO单独使用、打乱CLIP权重更可靠地分配对应概率？**

这是冻结教师预检，不是VPR训练。PASS也不等于检索提升；FAIL只表示本次教师构造
未获得支持，不等于所有CLIP或语义方法无效。创新点1及旧实验只读，不叠加新模块。

## 固定协议

复用封存GSV的8192训练地点、1024地点不重合的开发留出地点。按城市轮转、地点哈希
确定性抽取各128个地点。不重新划分数据。该留出集合已参与历史实验，不称为盲测集。
每地点只提取4个视图：原第0/1张图，以及由第0张图产生的2个几何裁剪。
新增fp16数组约0.5GiB，不生成MSLS全数据库局部特征，不更新任何模型参数。

### 已知几何对应

原图缩放256×256，裁剪左上224×224和右下224×224，偏移32像素；后者亮度0.85、
对比度0.9。CLIP为原生14×14，真实对应为平移2个patch；每方向有144个重叠格子。
真值来自图像坐标，不来自CLIP或DINO相似度。DINO使用官方BoQ编码器的322输入，
其23×23末层原始patch图池化14×14并单位化。这是近似的局部特征对齐，不是假定
ViT的感受野等于单个patch。统计每图双向平均，再以**地点**进行2000次配对bootstrap。

### 真实视图对应的间接检查

第0张为查询、第1张为同地点正样本；同城市其他抽样地点第1张作为负样本池。
负样本与查询及正样本GPS均至少相隔25m。用冻结官方BoQ的全局相似度选择最多4个
困难负样本，三组复用同一集合。少于2个合格负样本则明确记为排除，不跨城补齐。
以对应概率加权的DINO相似度做双向图对评分，报告正样本胜过负样本的比例。
该评分只是对应可用性的启发式，不等价于真实局部几何真值或BoQ检索性能。

## 三组完全匹配的教师

先以DINO余弦相似度/0.1做行、列softmax乘积，建立双向top8交集的固定候选边。
空行弃权，不使用隐式网格回退。不能只留下单个MNN边：单边归一化后CLIP权重会抵消。

- DINO：只归一化DINO置信度。
- CLIP：乘以连续SCLIP-equation patch余弦的指数权重（温度0.2），再归一化。
- shuffled：每行在有效边内部随机打乱上述CLIP权重，再与同一DINO置信度相乘。

打乱组保留候选支持、候选数和每行CLIP权重的多重集合；**不保证最终联合分布熵相同**。
教师全部停止梯度，CLIP使用本地固定OpenAI ViT-B/16权重，末层CSA最小适配，
不声称完整复现SCLIP分割，也不使用文本类别、动态mask或人工语义标签。

## 运行前固定的进入条件

只以128地点留出结果做进入判断，训练子集仅报告，不调阈值。

1. 几何样本128地点、真实图对至少120地点。
2. 正确CLIP的真值对应概率相对DINO和shuffled各提高至少0.001，
   且各自地点配对bootstrap 95%区间下界大于0。
3. 几何top1准确率不低于DINO超过0.002。
4. 真实图对的成对胜率不低于DINO，并高于shuffled。

这些是本次小规模可行性筛选条件，不是普适统计检验或可自动适用于论文的显著性声明。
不在看到结果后修改同一运行的阈值。失败就停止此教师构造，不启动四组正式训练。

## 通过后的学生方案（尚未实现/启动）

RGB → 冻结DINO → 同结构纯通道适配器 → 视觉低秩BoQ路由残差 → 单个12288维描述子。
CLIP仅训练时参与跨视图关系教师；不回归CLIP特征、不在推理时增加CLIP。
辅助约束拟为双向槽位对应分布一致性，配合检索损失、描述子保持和槽位相关结构保持。
四组需同初始化、同训练预算：无对应监督、DINO教师、正确CLIP教师、打乱CLIP权重教师。
必须超过后两种匹配对照才能归因于正确CLIP关系。不能仅超过旧官方基线就宣称CLIP有效。

## 运行与产物

只在训练机VPR环境测试。启动器限定物理GPU1，CUDA分配上限20%，原生fp32，
没有自动安装、下载或重启其他作业。逐地点原子缓存、SHA256清单、原始图像哈希、
官方与CLIP权重/源码哈希、创新点1保护源码哈希均入合同；中断可`--resume`。
`logs/cv_slot_teacher_audit_v1`保存合同、每地点原始指标、summary和完成校验清单。
`logs/cv_slot_teacher_audit_launcher_v1`保存进程号及独立日志。

## 参考与区别

- [RKD, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)：关系蒸馏，而非逐点特征回归。
- [LoFTR, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.pdf)：借鉴双softmax置信度；本方案不具有LoFTR的几何训练真值。
- [SCLIP](https://arxiv.org/abs/2312.01597)：密集CLIP特征需适配，不能假定原始CLIP token完美定位。
- [StructVPR++](https://arxiv.org/html/2503.06601v1)、[SemVPR](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.pdf)：语义蒸馏已有先例，不能把大方向本身称为首创。

与早期VPR-conditioned semantic reliability/CC-LSA的区别是拟将跨图关系直接用于
BoQ槽位分配一致性，不再蒸馏单图空间gate或做逐候选局部重排；创新性与效果尚未验证。
