# 查询多视图候选互补性：第一阶段

只验证候选覆盖，不训练、不跑 Pair 配对、不宣称 R@1 改善。
复用 candidate_hard_v1 的完整数据库与查询全局描述子、原 top20 配对结果。
新增查询原生 RGB 图上的三个固定裁剪：宽度75%、完整高度、水平位置左/中/右。
裁剪后 bilinear resize 到322，再使用原 ImageNet 归一化和冻结官方 Pair-VPR global 分支。
该跨视野检索是否适配模型是本轮待验证假设，不是已知有效结论。

对照：整图top20、整图top80、整图top20与三个裁剪各top20的去重并集。
并集最多80，不补齐；因此整图top80的候选预算不少于并集。
同分按DB ID；裁剪选择与查询GT、基线是否错误无关，对全部查询一致执行。
输出并集独有补回、整图80独有补回、相对20新增可达、实际候选数量及新增配对预算。
不将并集顺序解释为重排顺序。预测只用于判断是否值得下一阶段配对。

## 训练机

先测试（本机未测试），然后先跑train。train结果不支持互补性时暂停，不自动消耗dev。
train有机会时，再按完全相同固定规则跑dev，不按结果更改裁剪。

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_candidate_multiview_screen.py -v
python -u scripts/candidate_multiview_screen.py --split train --output doc/candidate_multiview_train_v1 --resume
```

中断后原命令即可续跑。已有完整输出则校验后退出。
坏分片/不一致契约报错，不静默忽略；只有完整带hash的分片才复用。
只存紧凑crop描述子与候选ID，不存dense feature或裁剪图片。
预计train裁剪描述子约48 MiB（8192×3×512×4字节），不含文件头、候选和元数据。
验证所有输入完成清单，模型/源码哈希和原始查询图哈希；首张整图描述子复现检查，全部查询原top20复现检查。
不会修改旧缓存。使用物理卡1、逻辑cuda:0；不需要RU_CKPT。

train报告下载（Windows）：

```powershell
scp wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_multiview_train_v1/summary.json D:/Desktop/MyVPR/doc/candidate_multiview_train_summary.json
```

train分析后再决定是否执行：

```bash
python -u scripts/candidate_multiview_screen.py --split dev --output doc/candidate_multiview_dev_v1 --resume
```

dev已反复参与方法设计，不是独立测试；本轮也不是语义方法，不以裁剪结果宣称语义增益。
若只相对top20有收益、但不优于整图top80的覆盖，应优先考虑更简单的整图候选扩展。
若有独有补回也不能自动断言净收益：下一阶段必须实际比较相同Pair模型的纠错与退步。
