# 候选参考图像关系：第一阶段 CPU 可行性审计

本轮不是新的残差训练，也没有运行参考图像间的 Pair-VPR。先判断结构机会和准确预算。
旧训练代码、缓存、gate不改；不读取原图、不调用torch模型、不使用GPU、不自动开启第二阶段。

## 固定规则

在每条查询已有global top20内部，以缓存参考描述子的余弦相似度建立互为top2近邻，
保留相似度最高的10条无向边。排除自身，重复边合并；同分按数据库ID排序，
不依赖候选数组位置。节点和边选择不使用任何地点GT、查询正确性或留出错误ID。
train/dev分别规划，不跨地点隔离边界合并。

边选择后才用GSV place-ID审计：

- 正候选是否有第二个正候选，能否在所选边中相互连接；
- 10边预算截断是否丢弃正候选连接；
- 边连接同地点的比例（按出现次数和去重配对分别统计）；
- **错误地点内部的一致关系**，不能误当成查询正确性的支持；
- 可达错误、当前范围内错误、已正确、不可达等子集的覆盖；
- 两张参考/地点的人工构造约束。place-ID同地点不等于必然存在可见重叠，
  不同ID也不能绝对排除地理邻近或视觉重叠，报告不是边匹配真值保证。

正-正边覆盖只表示结构机会，不能当成预计纠错数或R@1上限。
原top20不可达查询不会因这一步增加正样本。

## 输出与预算

summary.json、per_query.json、reference_mapping.json、contract/completed清单，
train_pair_plan.npz、dev_pair_plan.npz保存全局去重的无向数据库ID对、每query边索引、
offset和余弦分数。所有DB ID必须结合各split的reference_mapping，不能跨split混用。

预算包括精确去重配对数、两倍方向forward数、涉及参考图像数，以及紧凑数组字节数。
提议的未来score缓存为两个float32分数加一个完成bool/配对，即9字节/配对；
**这不是整体磁盘、内存或运行时间预算**，不含模型、dense重算、元数据和文件头。
本轮没有预分配或生成score缓存。下一阶段必须先用少量配对测吞吐与显存，再批准规模。
报告不设自动通过门槛，不因留出GT修改图规则。

## 训练机运行

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -m unittest discover -s tests -p test_audit_candidate_reference_graph.py -v
python -u scripts/audit_candidate_reference_graph.py --output doc/candidate_reference_graph_audit_v1
```

测试通过再运行审计。CPU会校验旧plan/cache清单、逐行GT、规范化特征和参考描述子一致性。
hash读取有磁盘I/O；重复参考向量在RAM保留一份，不写新的特征缓存。
原数据和报告只读，输出新目录；中断后不要覆盖已有完成报告。
本机按用户要求未运行测试/训练。

Windows首次下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_reference_graph_audit_v1 D:/Desktop/MyVPR/doc/
```

第二阶段冻结Pair稀疏配对、分数融合与打乱边对照**尚未实施**；依据本审计先决定是否投入。
文献背景，不代表本脚本复现或证明有效：

- k-reciprocal encoding: https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html
- SuperGlobal: https://openaccess.thecvf.com/content/ICCV2023/html/Shao_Global_Features_are_All_You_Need_for_Image_Retrieval_and_ICCV_2023_paper.html
