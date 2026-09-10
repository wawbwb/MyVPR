# 17Places：官方 SegVLAD 与 AnyLoc 成对比较

## 2026-09-10 成对结果与案例复查

用户返回 `segvlad_paired_20260910_095818`：AnyLoc R@1–5 正确数为
385/390/392/393/395；SegVLAD 为387/394/395/397/399。
纠错 query index 243、257、291，退步237，净增2（+0.4926个百分点）。
有小幅正收益，不能据4个正确性变化样本宣称稳定语义增益。

`scripts/segvlad_case_review.py` 在 CPU 读取已有结果，核验 masks/matches 哈希、
图像顺序与投票排名，生成四案例可离线浏览的HTML及图片。
显示两方法top1、GT索引偏移、双方最强三组区域对应、窗口边缘参考图。
区域可视化是官方order3邻居mask的并集示意，不是attention、精确token贡献，
也不能仅凭高相似度认定同一地点。不调用模型，不改变排序。

```bash
python -m pytest -q tests/test_segvlad_case_review.py
python -u scripts/segvlad_case_review.py --repo ../Revisit-Anything-official --data-root datasets/segvlad_official/17places_full --paired doc/segvlad_paired_20260910_095818 --output doc/segvlad_case_review_v1
```

下载整个 `doc/segvlad_case_review_v1`（包含images），打开index.html；人工判定
同一物理场景、重复物体、GT边界或不确定。先看四案例，再决定是否投入公平区域消融。
代码和测试未在本机执行。

已观察的 SegVLAD map/order3/PCA1024 结果：406 queries，R@1–5 正确数
387/394/395/397/399。sklearn 降到作者 PCA 的 1.3.0 后，汇总正确数不变。
这只是官方缓存检索运行成功，不是端到端复现或 MSLS 收益证明。

## 新脚本

`scripts/segvlad_paired.py` 固定官方 commit
`628508d4a60c832961a2e2d8efa9eec724c2dc86`，不修改上游源码或覆盖既有结果。

1. 检查 ref/query 各406张；自然排序与四份 HDF5 的 key 顺序一致。
2. 调用官方 get_gt，确认索引±15的 GT（保留边缘越界ID）。
   官方没有使用 ground_truth_new.npy/my_ground_truth_new.npy。报告只记录这两份
   文件的元数据和哈希，不宣称与官方GT等价，也不擅自替换GT。
3. 从已保存的 matches/sims 和官方 masks 重建 SegVLAD 图像排名，必须复现上述正确数。
4. 复用相同 DINO-G HDF5、map/17places 词典，调用官方 aggFt、normalizeFeat、get_recall。
   不重新提取 SAM/DINO，不训练，不额外拟合 PCA。
5. 输出逐查询预测、纠错、退步和净收益。

这是完整方法比较，不是单变量语义消融：AnyLoc 整图49152维且无PCA；SegVLAD
含分割、邻域聚合、PCA1024及投票。因此即使提高，也不能全部归因于语义。
严格 GT 内容核验（与数据集原始标注的关系）仍需单独检查，不等于本脚本的索引一致性核验。

上游依据：
[官方主程序](https://github.com/AnyLoc/Revisit-Anything/blob/628508d4a60c832961a2e2d8efa9eec724c2dc86/place_rec_main.py)、
[GT规则](https://github.com/AnyLoc/Revisit-Anything/blob/628508d4a60c832961a2e2d8efa9eec724c2dc86/gt.py)。

## 训练机命令

逐块粘贴；维持现有 VPR 环境与 sklearn1.3.0。

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -m pytest -q tests/test_segvlad_official.py tests/test_segvlad_paired.py
```

固定使用已复现成功的第二次结果目录，不按文件名自动挑选：

```bash
export SEG_RESULTS='/home/wt/workspace/OpenVPRLab/datasets/segvlad_official/17places_full/17places/out/results/global/exp0_global_SegLoc_VLAD_PCA_o3_17places_10092026_094425'
test -d "$SEG_RESULTS"
export PAIRED_OUT="doc/segvlad_paired_$(date +%Y%m%d_%H%M%S)"
set -o pipefail
python -u scripts/segvlad_paired.py --repo ../Revisit-Anything-official --data-root datasets/segvlad_official/17places_full --seg-results "$SEG_RESULTS" --gpu 1 --output "$PAIRED_OUT" 2>&1 | tee "${PAIRED_OUT}.txt"
```

会先审计再跑基线。只想审计可加 `--audit-only`，但正式运行需换一个输出目录。
出现错误时保留日志，不删除原结果、不自动降级方法。AnyLoc沿用上游插值与聚合，
其显存和时间开销应以实际运行观察为准。

结束后查看：

```bash
cat "$PAIRED_OUT/summary.json"
cat "$PAIRED_OUT/completed.json"
echo "$PAIRED_OUT"
```

下载整个输出目录及同名txt即可；anyloc_descriptors.npz可留在训练机，不是分析必需。
本机只做静态代码检查，未运行这些测试或模型。
