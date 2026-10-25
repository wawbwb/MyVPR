# 创新点1后接内容条件空间重采样：V1预注册

2026-10-25用户要求参考已有失败继续实验。本轮选择不同机制：新增模块只预测“从哪里取特征”，不生成新的通道值或修改BoQ内部，不继续分类器/原型监督。原创新点1与原BoQ全部冻结，只训练小型位移预测器。

## 依据与可证伪假设

局部卷积、通道门控、矩保持、原版channel迁移、KV联合校准、联合LGGR与全库监督都没有取得预定净收益。这些结果不证明几何重采样有效；它只是尚未被上述局部固定网格变换检验的方向。假设单图内容条件的局部采样可改善对视角变化的适应，必须超过整图位移与错位移场对照才能保留。

Deformable ConvNets v2已有内容条件采样先例：https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.html 。本文不是DCNv2复现，没有调制卷积9点核；可变形采样本身不是新颖贡献。单图位移也不等于已实现跨图像几何配准。

## 结构

RGB→冻结DINO及创新点1→F→新增采样模块→冻结BoQ→单12288维描述子。

逐位置非仿射LN，1×1降维768→32，GELU，深度3×3，GELU，32→2的1×1输出二维位移。全无bias，末层零初始化；24928参数。tanh令每坐标轴位移最多1个patch，不是说二维位移向量范数≤1。用PyTorch grid_sample双线性采样、border padding、align_corners=True，不安装额外算子。

为严格零起点，用同一采样内核的差值：

```
D = warp(F, offset) - warp(F, zero_offset)
R = 0.025 * stopgrad(RMS_channels(F)) * tanh(D / clamp_min(RMS_channels(F), 1e-6))
F_out = F + R
```

保留恒等主路，新增残差实际逐位置相对L2≤2.5%，学习权重不能抵消这个上限；该上限不保证检索排序不变。位移学习到零时仍保留对位移输出层的梯度，不用条件return切断梯度。无新值投影、无语义、无局部图像匹配或额外推理分支。

## 三组严格同参数对照

|组|应用的位移场|问题|
|---|---|---|
|global_shift|预测场的空间均值广播到整图|仅整图内容条件平移是否足够|
|local_shift（主候选）|原位置逐patch二维位移|逐位置适配是否有价值|
|shuffled_shift|同一预测场按图像SHA固定打乱位置，直接作用于原特征|位移与内容的空间对应是否重要|

同结构、同参数、同初始张量、相同数据顺序和loss。打乱不逆变换位移场，因为要破坏位移与被采样特征的对应；保持二维向量完整和数值分布。CPU测试在非零位移下必须确认其实际改变输出。global组的预测器全部参与训练，不注册无用参数凑容量。

## 固定实验

seed42、原8192训练地点/四视图、16×4完整批、322、FP32、微批2，封存计划前1024步；8步真实smoke丢弃。AdamW lr3e-4、wd0、clip1，仅GPS过滤MS检索损失，不继承上一轮原型监督，不搜索位移上限/混合系数，不选验证最佳checkpoint。

源模型/权重只读，CPU测试仅服务器；物理GPU1、20%分配上限+2GB余量，旧排他锁，不停止其他作业。记录位移均值/最大值/空间标准差、实际残差、描述子漂移、有效MS步与难负对数。grid_sample的CUDA反向可能有浮点非确定性；CPU断点与梯度回放验证不能等同于GPU逐位确定性承诺。

最终完整GSV适配留出、MSLS740、Pitts7608，原完整图库/GT。基线必须复现3049/695/7276及全部Top20。候选Pitts必须严格超过创新点1及两个对照，MSLS不低于基线及对照，GSV不低于基线，才算探索信号。若global解释收益，不支持局部自适应必要；若shuffled相同，不支持空间对应作用。单seed不证明稳定/论文新颖性，失败后不自动扩展。

## 执行

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_spatial_resample.py --allow-training --allow-shared-gpu
cat .cache/spatial_resample_v1_launcher/progress.json
```

输出logs/spatial_resample_v1，状态SPATIAL_RESAMPLE_V1_STATUS.md。论文创新候选是该位置/约束下的局部采样是否产生超出两个对照的检索价值，不能把已有可变形采样重命名为已成立创新。
