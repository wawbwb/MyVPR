# RU-BoQ候选区域验证：轻量探索

已移除官方DINO-G/SAM-H全量MSLS适配。此方案不是SegVLAD复现。
保留RU-BoQ，复用已有visual-pair评测缓存的local.npy（19611×400×768）、
descriptors.npy和per_query.npz。没有这些文件时停止，不自动生成大缓存。
只处理RU top20涉及的图片，用现有SAM-B分区；不训练、不拟合VLAD或PCA。

每图最多16个区域：20×20占据网格覆盖率>=0.5；4–240个token；
按面积优先，区域IoU>0.3排除。区域向量是现有归一化DINO token均值再归一化。
三组同数量：SAM、普通矩形分块、SAM错位10格。少于3个区域时三组一起弃权。
一对一互为最近邻、cos>=0.6、归一化位移差<=0.2的平移一致性；
最强四组相似度求和除4。此为粗几何启发式，不声称尺度/透视不变。
仅在RU分数差<=0.02、区域支持>=3、区域评分优势>=0.05时提升一个候选。
三组规则完全一致，GT只用于结果统计。阈值是首次探索的固定设置，不是训练校准值。
不用此MSLS结果反复调阈值；需SAM胜过grid及shifted才支持语义分区价值。
不能声称分块面积/形状已完全匹配，也不能由零退步证明安全保证。

新区域数组最大每图约72KiB（三组float16向量），只处理候选相关图像，
通常数百MiB；启动会打印估计。旧local.npy不会复制或重写。
首次核验源文件哈希与RU checkpoint，并抽样重新计算12图特征确认映射。
不代表所有图像逐一重算。完整CPU/GPU耗时取决于涉及图片数，SAM不是零成本。
正常中断可重复同一命令续用原子写入的shards。测试未在本机执行。

## 训练机：同步重写后的分支

先fetch，然后查看状态。若训练机仍在185GB提交4bb7e09，且没有自己的新提交，
可以按明确旧提交条件执行mixed reset；不要对未知分支状态强制回退。

```bash
git -c http.version=HTTP/1.1 fetch origin main
git status --short --branch
git log -1 --oneline
```

若HEAD是4bb7e09：`git reset --mixed 0345f73`，再 `git merge --ff-only origin/main`。
旧185GB三个文件会变为未跟踪文件，不会被新流程调用；如需删除，仅删除
scripts/segvlad_msls.py、tests/test_segvlad_msls.py、doc/SEGVLAD_MSLS.md。
若HEAD本来是0345f73，直接merge即可。不处理其他训练记录、权重或缓存。

## 执行

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -m pytest -q tests/test_region_pair_lite.py
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
ls -lh doc/visual_pair_msls_hard_mix_v2/local.npy doc/visual_pair_msls_hard_mix_v2/descriptors.npy doc/visual_pair_msls_hard_mix_v2/per_query.npz
python -u scripts/region_pair_lite.py --source doc/visual_pair_msls_hard_mix_v2 --checkpoint "$RU_CKPT" --sam-checkpoint .cache/sam/sam_vit_b_01ec64.pth --limit-images 4
```

没有旧缓存先报告，不要启动任何官方185GB流程。如果只有uniform_v2缓存，source改成
doc/visual_pair_msls_uniform_v2，其余相同。

四图通过后删除limit参数运行：

```bash
python -u scripts/region_pair_lite.py --source doc/visual_pair_msls_hard_mix_v2 --checkpoint "$RU_CKPT" --sam-checkpoint .cache/sam/sam_vit_b_01ec64.pth
```

默认cuda:1、doc/region_pair_lite_v1。结果summary.json包含三组纠错/退步ID。
这是候选重排序，不是BoQ单描述子训练改进。先证明信号，再讨论训练融合。
