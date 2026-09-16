# 最终轮 checkpoint 只读诊断

两组探索训练均选中 epoch0，不应使用 best 或最终选模 npz 诊断 epoch5 的作用。
新入口加载 independent_last.pt / set_last.pt 的 model（不是 best_model），验证完整
run与缓存hash、原挖掘和GT、checkpoint合同、epoch以及逐组最终轮结果复现。
不修改训练入口、缓存、失败gate或模型，因此原缓存指纹不变。

输出针对训练105查询与完整GSV留出2048查询：

- 当前最强正减最强负的 margin 前后变化，允许最强候选身份改变；
- 固定原最强正/负的改分差，避免将它误当成整个候选集纠错；
- 范围内错误、仍未修复错误、近边界正确、普通正确等分组；
- effective residual 饱和比例 abs(score-base)>=3.96，范围与标准差；
- 两组原始改分相关性，以及逐query去均值后、与排序相关的改分一致性；
- 逐query原始ID、地点、路径、margin、纠错/回退，最终轮小型分数文件。

1e-5仅用于区分数值近零，不是显著性阈值。相关性常数输入输出null。
不根据结果自动延长训练、不选新的模型、不将开发留出当成独立测试。
本机未运行测试/模型；以下在训练机运行，物理GPU1：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_diagnose_candidate_hard_last.py -v
python -u scripts/diagnose_candidate_hard_last.py --output doc/candidate_hard_last_diagnostic_v1
```

仅读取已有cache与小型head，不需要RU_CKPT/官方权重或图像。启动hash核验有磁盘I/O。
完成后Windows首次下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_hard_last_diagnostic_v1 D:/Desktop/MyVPR/doc/
```
