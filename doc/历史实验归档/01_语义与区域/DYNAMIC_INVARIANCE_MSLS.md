# 现有动态不变性 checkpoint 的完整 MSLS 回归评估

独立脚本 `scripts/eval_dynamic_invariance_msls.py`，不修改原训练源码或哈希契约，不训练、不在线调用 CLIP、不扰动图像。评估 frozen、plain、random、semantic 四个固定模型的原始查询和完整数据库（18871 参考、740 查询）。

复用 MSLS CLIP 缓存中的图像路径、图像 SHA 和 GT 做身份核验，不读取其掩码来修改推理。核对三组 checkpoint、共同训练配置、训练计划和原始 RU 权重；输出每个模型的完整描述子、全库点积分数、逐查询 GT 排名、正负间隔、top20，以及所有模型间的纠正/退化。

先计算 frozen：要求正确数为 675，且全部 740 个 top1 图像 ID、查询路径和正确标记与 `doc/clip_token_drop_eval_v1/report/query_outcomes.json` 中 frozen 结果相同。否则中止，不继续消耗三组模型评估算力。

## 运行

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
unset CUDA_VISIBLE_DEVICES
python -u scripts/eval_dynamic_invariance_msls.py \
  --device cuda:1 \
  --output doc/dynamic_invariance_msls_v1
```

默认 batch-size=4，仅使用物理 GPU1。如果旧基线报告在不同位置，用 `--baseline-reference` 指向对应 query_outcomes.json。输出已存在时拒绝覆盖，不支持断点恢复，重新评估使用新目录。未完成目录没有 complete 标记。

四组描述子与分数约需 4.1 GB 磁盘空间，不需要下载。本机已通过两个 CPU 汇总测试和语法检查，未运行 GPU 模型推理。

Windows 下载：

```text
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_invariance_msls_v1/report C:/Users/zy/Desktop/MyVPR/doc/dynamic_invariance_msls_report_v1
```

report 保存 summary.json、outcomes.json、各模型逐查询记录、provenance.json 和 completed.json。completed 包含轻量文件及训练机大数组 SHA256。分析时先核验轻量文件，再检查 semantic 相对 frozen、plain、random 的纠正与退化，避免仅比较总正确数。

MSLS 已多次用于实验分析，这轮是历史基准回归，不能作为独立验证，也不用于选 epoch 或扫参数。当前小型开发集全满分不能替代本轮评估；本轮即使正向，也仍需未参与开发的数据验证。
