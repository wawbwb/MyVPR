# 固定权重的最后一层 CLS 语义干预

本脚本不训练、不修改官方源码、不改变全局检索。以原 top100 排名确定每个查询的最佳正候选、最佳负候选，覆盖 31 个可达错误和 693 个成功查询。16 个不可达错误被明确排除。两候选诊断只说明固定分数间隔，不能报告 R@1。

## 干预

仅修改最后 decoder block 交叉注意力的 CLS 行，对 memory 侧 token 加 `log(1 - 0.5*d)`。正反方向各用对应 memory 图像的语义图，原分数仍相加。原模型完成一次 forward 后，复用前缀及最后层残差，仅重算受影响的 CLS 后续 MLP、LayerNorm 和分类头；不对 patch 行施加干预。

每个方向都将手工重建的 alpha=0 输出与原 xFormers 路径比较，容差 atol=rtol=2e-5，不通过就终止。原正负顺序也必须与保存排名相符（允许 1e-4 数值平局）。这些检查不能替代完整 top100 排名复现，本诊断不声称完成后者。

缓存使用已有 `.cache/clearclip_token_drop_msls_v1`。d 是原动态像素覆盖率，不是 CLIP 校准概率，也不是旧的二值删 token 掩码。将 20×20 分片常数覆盖图按面积重叠投影至 23×23，保持全图平均覆盖率；这是一种粗分辨率近似，没有恢复原稠密语义。官方图像与缓存均为整图缩放。无新 CLIP 推理或人工标注，也无引擎盖专用识别。

对齐语义与 5 个固定种子（11、29、47、71、101）的全空间排列比较。相同图像在所有候选对中使用相同排列，保留完整数值分布。报告每张图的 soft overlap 和平均绝对差；这些排列不是五次独立模型实验。

## 运行

训练机现有 VPR 环境：

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
bash scripts/run_pairvpr_cls.sh
```

脚本先运行 CPU 工具测试及 torch 数值测试，不跳过缺失依赖。随后仅使用物理 GPU1，局部重提所需 Pair-VPR 特征（没有保存的 dense features 可直接复用），不提取完整数据库。每 10 个查询输出进度。输出目录存在时拒绝覆盖，可通过 shell 脚本第一个参数指定新目录。本版不支持断点恢复；只有 completed.json 存在且哈希吻合才视为完整结果。

下载（Windows PowerShell 或 CMD）：

```text
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/pairvpr_cls_diagnostic_v1 C:/Users/zy/Desktop/MyVPR/doc/
```

## 输出与判读

- contract.json：固定候选、排除查询、缓存/模型/源码哈希与规则。
- queries.jsonl：逐查询原分数、干预分数、正负间隔及三张图的平均覆盖率。
- summary.json：错误、成功两组分别汇总间隔改善/损失、与所有打乱结果的比较。
- mask_overlap.json：空间对照与语义图的重叠。
- completed.json：结果文件哈希和最大零干预误差。

优先看错误组 aligned 是否优于打乱，同时检查成功组受损。不要将 fixed-pair 翻转计为全库纠正，也不要用 GT 参与实际推理门控。只有出现选择性信号才考虑完整 top100 评测；不能凭本次诊断认定语义有效。

本机完成 3 个 NumPy/标准库测试及语法检查。没有安装 torch，torch 测试和真实模型零干预检查由上述训练机命令执行，未在本机验证。
