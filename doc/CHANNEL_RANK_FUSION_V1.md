# 原通道独立检索分支互补性验证V1（结果读取前固定）

原版channel在官方特征上有效，但直接插入C-neck后退化。本轮测试让两分支各自在原输入分布下检索，固定合并排名能否保留纠错；只用已有报告，CPU运行，不训练、不提取图像特征。

来源固定seed42原spatial_adapter_repeat的official/channel与original_channel_transfer的innovation1。逐数据集核验图库数量、查询数量、图像顺序和GT摘要；保存源报告SHA，重新用GT复算原报告召回。数据仍是重复开发集。

方法预定为等权Reciprocal Rank Fusion：两份Top20的候选并集最多40张，每张候选分数为所在各列表的1/(60+rank)之和，rank从1开始，未出现贡献0。按分数降序，平分按创新点1的名次优先，再按reference ID确定顺序。返回20条。k固定60，不扫描权重、k、截断长度或置信度阈值。

比较innovation1、原official、原channel、innovation1+official融合、innovation1+channel融合。official融合控制双分支投票的一般效果；后者超过前者才有通道特异证据。两分支实际部署需分别提取创新点1与官方特征；本轮只分析已有排名，不宣称单分支计算成本。

预定探索性叠加收益：channel融合的Pitts R@1严格超过创新点1，MSLS/GSV不低于创新点1。通道特异信号还要求Pitts严格超过official融合，MSLS/GSV不低于official融合。单次满足不证明论文创新或稳定增益。

额外报告GT oracle：任一分支Top1正确的查询数、任一Top20存在GT的可达查询数。这些是潜力上界，不可部署、不得用于融合打分、选参数或训练；实际收益以固定标签无关融合的完整逐查询结果为准。

服务器执行：

```bash
cd /home/wt/workspace/OpenVPRLab
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wt/.conda/envs/VPR/bin/python -m pytest -q tests/test_channel_rank_fusion.py
CUDA_VISIBLE_DEVICES='' /home/wt/.conda/envs/VPR/bin/python scripts/channel_rank_fusion.py
```

输出doc/channel_rank_fusion_v1；不覆盖旧报告。若无信号，本轮停止，不自动扫融合参数。
