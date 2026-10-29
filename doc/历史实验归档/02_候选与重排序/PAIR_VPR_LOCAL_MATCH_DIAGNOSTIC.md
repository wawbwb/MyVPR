# Pair-VPR 局部对应诊断（不训练）

2026-09-07。距离核查：47 个错误中 12 个在 (25,30] 米、9 个在 (30,100] 米、9 个在 (100,1000] 米、17 个超过 1km。没有距离缺失；7955 条 GT 坐标距离均小于 25m。这不支持将全部错误归因于边界或标注。

固定检查候选可达但错误 top1 超过 1km 的查询 1/34/126/131/163/164/439，并加入成功对照 49/325/565。q34、q126 的展示 GT 有明显视角差异，所以不能仅凭一个 GT 宣称存在充分重叠。脚本检查 global top-100 内全部 GT，加 global/refined top1；不扩大候选、不改排序。

`scripts/diagnose_pairvpr_local_matches.py` 使用官方完整权重严格加载、官方配置与数据预处理、本地 DINO 源码免下载。记录真实双向 Pair-VPR 分数，另画 encoder 23x23 token 的余弦互为最近邻（MNN）。所有匹配保存在 results.json，图中只画余弦最高的 40 条，不设语义阈值。

MNN 不是 decoder 注意力，不是经过几何验证的对应，也不能仅以匹配数量认定地点正确。真实分数按照官方 eval.py 的双向求和、越大越靠前解释，不沿用模型文件中容易误导的注释。来源：https://github.com/csiro-robotics/Pair-VPR/blob/main/pairvpr/eval/eval.py

## 训练机执行

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -u scripts/diagnose_pairvpr_local_matches.py --official-repo /home/wt/workspace/Pair-VPR-official --msls-path datasets/msls-val --audit doc/pairvpr_official_paired_audit_v1 --output doc/pairvpr_local_matches_v1
```

只检查 10 个查询的选定候选，逐查询打印进度；不跑全部 740 查询重排。输出目录必须不存在。仍需官方权重、本地 DINO 源码和既有 pairvpr_speed_local.yaml。HTML 内嵌原图，下载整个输出目录即可。

本机 PowerShell 下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/pairvpr_local_matches_v1 D:\Desktop\MyVPR\doc\
```

## 阅读顺序和边界

先检查 completed.json；再比较每查询错误 top1 与全部候选内 GT 的真实分数、MNN 是否集中在可辨认的建筑结构或仅在天空/植被/道路。成功案例是必要参照。若小批量重算分数与已存排序不一致，先排查版本/数值/预处理，不据此解释模型。

这批样本由 GT 选择，只用于描述性诊断，不能估计重排增益或挑阈值。没有开展新训练，也未按用户要求在本机运行代码测试；需训练机验证接口及运行结果。
