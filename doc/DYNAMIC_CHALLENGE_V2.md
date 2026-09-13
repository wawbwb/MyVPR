# 挑战集 V2：实例素材与低重叠错位

V1 的32个对象大多是车身局部，且有错位掩码与准确位置重叠87.6%的案例。V2修正这两个诊断条件，不提高抑制强度，不训练，不根据检索结果筛选素材或参数。V1代码与结果保留不动。

## 实例素材

使用 torchvision MaskRCNN_ResNet50_FPN_V2 的 COCO_V1 权重。官方权重约177.2 MB；首次训练机运行 bank 时如无缓存，会下载到 torch hub checkpoints，不安装新 Python 包。也可单独运行 bank 并以 --seg-weights 指定已下载的原始官方 state_dict。

来源：[torchvision 官方模型接口](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)。这属于可见实例分割，不能保证自动补全被遮挡的车辆。

按固定路径哈希扫描最多2048张GSV原图；原图只缩小至最长边1024、不先压成280。检测置信度≥.9，实例掩码阈值.5，不做高置信度像素侵蚀；保留至少9000个前景像素、bbox内占比≥40%、距原图边缘至少5像素的对象。15%贴入档仍需符合V1尺寸限制，放大比例不超过1.25。每张来源图最多一个对象，目标32个，少于8个停止，记录所有筛选拒绝计数。

候选类别为 car、truck、bus、person、bicycle、motorcycle。但固定大面积和长宽限制可能偏向车辆；报告实际类别分布，不声称一定包含行人。边界检查和实例分割只是提高质量，仍可能保留遮挡对象或不完整实例，需要结合自动预览解释。

## 对照规则

- known_shift：不越界、不绕回，按14像素步长搜索平移，前景重叠≤10%。形状和面积不变。
- clip_shift：保留V1的20×20周期平移，全部数值不变，soft overlap≤10%。仍可能绕边界，与known控制机制不同。
- 顺序由固定随机种子产生，只找满足掩码几何条件的第一个平移，不使用图像评分或GT。对象/纯色匹配对使用相同控制随机键；CLIP图不同时最终可接受的位移可能不同。
- 无合格位移：eligible=false，不输出伪造的正确/错误结果。数组对应位置为NaN占位，不应当作有效模型描述子。
- 主方案保留全部案例；与错位方案比较时，仅用双方共同eligible集合。summary给出各自分母、排除数，以及共同集合上的独有正确查询。不同分母的总正确数不能直接相减。
- 空掩码允许原样返回，但注明empty；例如干净图的known对照仍为空，不能当作抑制起效的实验。

新增 CLIP 在贴入区域的分数增量统计：合成图CLIP分数减同一干净查询CLIP分数，再在已知支持区取均值。只作事后解释，实际自动抑制仍使用整张合成图CLIP图，不使用GT支持范围裁剪。

## 保持不变

同一128个查询、3个种子、5%/15%面积、对象/纯色对照、原图与完整18871数据库、0.5归一化输入衰减、原RU权重及CLIP规则。继续检查PNG归一化往返、旧数据库哨兵和干净查询描述子/top1。新素材会改变具体合成图，因此V1与V2结果不能当成仅改变一个因素的严格消融。

## 运行

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
unset CUDA_VISIBLE_DEVICES
bash scripts/run_dynamic_challenge_v2.sh
```

物理GPU1，无训练。需生成新的对象、合成图和CLIP缓存；最终检索复用上轮冻结RU数据库。建议另预留2 GB；首次实例分割权重另需约177 MB。已有V2完成阶段会复用并校验哈希，未完成目录拒绝覆盖，没有阶段内断点恢复。

下载：

```text
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_challenge_eval_v2/report C:/Users/zy/Desktop/MyVPR/doc/dynamic_challenge_report_v2
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/.cache/dynamic_challenge_donors_v2 C:/Users/zy/Desktop/MyVPR/doc/dynamic_challenge_donors_v2
```

本机运行6个CPU测试（含V1三项和V2三项）及Python语法检查，没有运行实例模型、CLIP或RU。此次修正只提高诊断可解释性，不保证结果变正；不能把合成MSLS上的结果外推为自然重访准确率提升。
