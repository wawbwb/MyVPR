# 固定 top44：Pitts 跨数据集验证

GSV dev上top20为1994/2048，top44为2003/2048，纠错11退步2；参数固定，不继续选K。
本轮复用原 `.cache/pitts_pair_admission_v1` 中1024个按路径哈希选定的查询、完整10000张数据库。
这批Pitts30k-val查询以前用于开发与模型选择，因此明确称为跨数据集探索性验证，
**不是独立测试，也不是全7608查询的结果**。不重新选择更有利的查询。

旧冻结Pair基线985/1024（历史参照），程序从原report验证，不强制硬编码通过。
与GSV同一官方模型、预处理、FP32、双向分数之和，固定top20/top44。
原global缓存恢复完整排名，并要求top20逐条一致；多正样本GT从原NPY重新验证。
新增24576配对=49152次定向forward，另1024个旧赢家复现=2048次定向forward。
总51200次定向forward，不含dense重新编码；不存全库dense，64条CPU LRU。
复用旧top20分数，不读取任何学习过的candidate head，也不引入语义或裁剪。

每query原子保存与SHA，--resume恢复相同契约，坏文件报错。
summary与per_query中的query_index/纠错ID为**原始Pitts查询ID**；
pairs/000000.npz文件名仍是固定1024子集位置，query_mapping显式映射两者。
只读旧数据，不修改旧代码及其fingerprint。

## 训练机

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_candidate_top44.py -v
python -m unittest discover -s tests -p test_candidate_top44_pitts.py -v
```

两个测试均通过后：

```bash
python -u scripts/eval_candidate_top44_pitts.py --output doc/candidate_top44_pitts_v1 --resume
```

中断重复原命令。本机按要求未运行测试。
路径可通过 --cache / --dataset / --official-repo 指定，不需要RU_CKPT。
没有缓存时停止，不自动提取新数据库或扩大实验。

Windows下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_top44_pitts_v1 D:/Desktop/MyVPR/doc/
```

观察净纠错、退步、R@1/5/10/20、新增可达转化和时间，不因结果扫描K或逐查询阈值。
若无收益/退步，记录跨数据集差异；若正收益，也不宣称独立泛化已获证明。
