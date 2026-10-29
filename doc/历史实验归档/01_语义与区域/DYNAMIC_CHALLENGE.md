# 自动动态干扰挑战集

目标是检验固定动态抑制操作在受控干扰下是否能挽救检索，不寻找保证正结果的配置。无训练、无新增人工标注、无强度扫描。

## 固定设计

- 从标准 MSLS-val 按查询路径 SHA256 选 128 个查询，选择不依赖 GT 或已有正确性。
- 保持全部 18871 张数据库描述子不变，复用上一轮 `doc/dynamic_invariance_msls_v1/frozen_descriptors.npy`。
- 每个查询用 3 个种子（11、29、47）、两档贴入像素面积（5%、15%）合成挑战。两档使用同一 donor、相同底部和水平中心，不截断对象；改变的是一个对象的面积，不是物体数量。
- 每个挑战同时生成对象版本和同位置、同形状、同面积的纯色版本，纯色取对象内部平均 RGB。后者去掉内部纹理，但仍保留轮廓，不能称完全无语义。
- 128 原图＋1536 合成图＝1664 个案例。每例 5 个抑制条件，共 8320 次查询检索。数据库与所有 GT 不变。

## 自动对象来源

复用已缓存的 torchvision DeepLabV3-MobileNetV3 权重，按固定路径哈希扫描最多 1024 张 GSV 图像，取得最多 32 个可用对象，少于 8 个则停止。选择 car、bus、person、bicycle、motorbike，置信度至少 .9，连通区域、边界、面积和放大后尺寸按代码固定检查。VOC teacher 没有单独 truck 类；不宣称 donor 覆盖所有动态类别或真实运动。

donor 图像与 MSLS 查询、数据库来自不同数据集。自动分割可能不完整，可能合并相邻对象；它用来构造合成输入，不是自然场景标注真值。合成时记录的贴入像素位置则精确已知。脚本生成透明对象预览及原图/对象/纯色对照页面，不要求人工修正。

权重缺失会报出路径，允许通过 bank 的 --seg-weights 指定已有文件；脚本不静默下载新权重。CLIP 使用原冻结 ClearCLIP 规则重新分析合成图，不允许根据已知贴入位置裁剪 CLIP 输出。

## 五个推理条件

1. none：原 RU，不抑制。
2. clip：依据自动 CLIP 20×20 动态像素覆盖率抑制。
3. clip_shift：将同一 CLIP 图周期平移，保留全部分数分布；可能绕边界、与原区域重叠，记录 soft overlap。
4. known：依据精确贴入像素支持区域抑制。
5. known_shift：对精确支持区域作不越界平移，形状、面积不变，记录重叠。

所有条件固定采用 `normalized_image * (1 - 0.5 * mask)`。RGB 意义是区域向 ImageNet 均值靠近，不是恢复隐藏建筑，也不是将黑色当作背景。known 在原图上为空；原图的保护性检查主要比较 clip、clip_shift 与 none。

known 与 clip 的分辨率、总抑制量、是否包含原有动态对象不同，因此它们的差距不能全部归因于 CLIP 定位误差。优先看各自相对自身空间对照的收益。已知位置抑制是这个操作的诊断对照，不是任意动态方法的性能上界。

## 结果如何判读

先按每个种子、每档面积分别报告原正确查询因合成干扰而变错的数量。基线未受损时，不将该组当作抑制有效性证据。

报告每种抑制救回多少“原图正确、干扰后错误”的查询、新增多少错误，以及相对干扰图未抑制版本的正负分数间隔。将原本就错的查询纠正另行统计，不能混入恢复率。对照对象和纯色版本的结果，以判断是否只是一般遮挡/边缘效应。多种子复用同一批查询，不把它们当独立样本扩大样本量。

只有对象干扰确实伤害基线，且定位正确的抑制比错位更能恢复，才支持该操作在此挑战中有用。若自动 CLIP 同样出现选择性恢复，说明自动方案在该条件下有信号。无论正负，都不能直接推广到自然车辆移动、真实重访或所有抑制方式。

## 运行

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
unset CUDA_VISIBLE_DEVICES
bash scripts/run_dynamic_challenge.sh
```

使用物理 GPU1：对象提取 → CPU 合成 → CLIP 缓存 → RU 评估。未修改任何旧训练/评估脚本的代码契约。完成的前置阶段复用并校验 SHA；未完成目录不自动覆盖、不支持阶段内部断点恢复，重新运行应先诊断错误。最终评估输出已存在则拒绝覆盖。

所有原图缩放沿用原 RU 的 torchvision v2 双三次 uint8 路径；生成 PNG 后检查归一化张量一致。评估再次检查三张数据库哨兵、128 个干净查询描述子及其旧 top1。失败即停止。

预留约 2 GB 输出空间；完整查询描述子和检索分数约 1.04 GB，其余为合成 PNG 和缓存。无需下载大数组或全部合成图。

本机已通过 3 个 CPU 测试（背景不变与面积、空间对照、严格配对恢复计数）及语法检查。无本机 torch，不宣称已经运行真实分割、CLIP 或 RU。

## 下载

Windows：

```text
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_challenge_eval_v1/report C:/Users/zy/Desktop/MyVPR/doc/dynamic_challenge_report_v1
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/.cache/dynamic_challenge_donors_v1 C:/Users/zy/Desktop/MyVPR/doc/dynamic_challenge_donors_v1
```

报告包括配对汇总、逐案例结果、CLIP 在贴入区域内外的分数、空间对照重叠和文件哈希。贴入区域内的 CLIP mass 只描述定位响应，不是自然场景 segmentation 精确率。合成图预览可在训练机输出目录 index.html 查看，或单独下载 preview_*.jpg；无需重新标注。
