# RU-BoQ候选区域验证：轻量探索

## 完成后的低成本分数诊断

本次用户返回：RU 675/740；SAM 671（0 纠错、4 退步，query 30/145/228/670）；
grid 673（0/2）；shifted 674（1/2）。本配置筛选未通过，不能据此否定全部语义方法。

`diagnose_region_pair_lite.py` 只读取已有分数、候选、GT 和运行元数据，不加载模型、
不读取大特征缓存、不重新提取 SAM。先复现完整排序和纠错/退步统计，再输出：
可达错误查询中正样本的分数优势；固定门槛的阻挡原因（可重叠）；所有纠错和退步的
20 个候选逐项分数、支持数和门槛情况；以及去掉门槛直接取区域 argmax 的诊断结果。
后者不是推荐的新方法。GT 仅用于事后分析，不进行阈值搜索或重新选择正式配置。

```bash
python -m pytest -q tests/test_diagnose_region_pair_lite.py
python -u scripts/diagnose_region_pair_lite.py --run doc/region_pair_lite_v1 --source doc/visual_pair_msls_hard_mix_v2 --dataset-root datasets/msls-val --output doc/region_pair_lite_diagnostic_v1
```

查看 `summary.json` 与 `changed_correctness_cases.json`；完整逐查询信息见
`per_query.json`，来源哈希见 `provenance.json`。输出目录必须未存在，旧结果不覆盖。
此补丁未在本机执行测试，交由训练机运行。

## 空分片恢复（EOFError）

原续跑逻辑仅检查文件存在，会跳过空或损坏的 NPZ。修复后每次先完整读取并校验所有必需分片，
有效分片保留，损坏分片移入同一输出目录的 `quarantine/`（可恢复），只重算损坏或缺失的图片。
合法的零区域数组代表弃权，不视为损坏。新分片写入后刷新磁盘、校验，再原子替换。
`cache_validation_*.json` 记录复用数量及修复清单；不修改区域参数、匹配规则或评测阈值。
仅允许已知原版脚本哈希的维护性迁移，其他输入、权重、评分代码哈希仍须完全一致。
保留原来的 `doc/region_pair_lite_v1`，不要删除缓存或换输出目录。

训练机验证（未在本机运行测试）：

```bash
python -m pytest -q tests/test_region_pair_lite.py tests/test_region_pair_lite_cache.py
python -u scripts/region_pair_lite.py --source doc/visual_pair_msls_hard_mix_v2 --checkpoint "${RU_CKPT:?请先设置RU_CKPT}" --sam-checkpoint .cache/sam/sam_vit_b_01ec64.pth
```

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
