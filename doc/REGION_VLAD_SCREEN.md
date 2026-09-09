# Region-VLAD：冻结 RU 的分割区域检索预筛

日期：2026-09-09。状态：代码实现，未在本机运行测试；等待训练机验证。

## 定位与来源

借鉴 [Revisit Anything / SegVLAD（ECCV 2024）](https://arxiv.org/abs/2409.18049)
及 [官方代码](https://github.com/AnyLoc/Revisit-Anything)，把区域作为数据库检索单元，
而非把语义类别直接当地点标签或在 RU 特征上加门控。
实现参照官方 `func_vpr.py` 的余弦簇分配、区域残差聚合、簇内归一化、
邻域掩码并集以及按相似度累计图像票数。并集不会对重叠像素重复计数。

**这是适配实验，不是官方 SegVLAD 复现**：使用现有 RU 训练过的 DINO-B 最后层
280×280 / 20×20 特征，不是官方默认 DINO-G layer31 value；使用 SAM-B，
16×16 prompt grid，每图最多16个区域，默认一阶邻域；PCA为256维且不白化。
不能载入官方 DINO-G 词典。词典与 PCA 在256张 GSV train 地点图像上重新拟合，
不使用 MSLS 数据拟合。小区域、重复质心与退化 Delaunay 使用保守 identity 邻域，
不会照搬官方少于4个 mask 时仅访问前两个 mask 的特殊分支。
区域召回票权使用完整 query 搜索结果的全局 min/max，无 GT 参与。

SAM 提供无类别对象区域，不等于显式类别语义。如果最后 SLIC 同样好，
只能支持“区域化检索”，不能声称“语义标签带来提升”。
本实验失败也不能否定官方配置或所有语义路线。

## 四个对照与预算

| 组别 | 区域 | 作用 |
| --- | --- | --- |
| sam | SAM区域，面积优先截取 | 对象区域假设 |
| slic | SLIC超像素，面积优先截取 | 非语义边界对照 |
| grid | 等数量、不重叠矩形 | 无对象先验对照 |
| shifted_sam | SAM SuperSegment 平移半幅并环绕 | 数量、面积、重叠保持不变的错位对照 |

每图四组数量严格一致：min(16, SAM有效数, SLIC有效数)，共享特征、32簇词典、
PCA和每区域50个近邻预算。词典拟合使用归一化 DINO tokens；PCA样本四组等量。
区域面积/覆盖率仍不完全相同，`mask_audit.json` 和结果中的统计会报告差异。
SAM无有效区域时明确报错，绝不偷偷用网格替代。实际图像排名不足20时用-1补齐，
不混入RU结果冒充区域命中。

## 训练机准备（继续使用 VPR 环境）

先检查依赖，不重装 torch、torchvision、xformers：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -c "import torch, sklearn, scipy, skimage; print(torch.__version__)"
python -c "from segment_anything import SamAutomaticMaskGenerator; print('SAM import OK')"
```

如果缺包，先预览安装计划；不要执行会替换现有 torch/CUDA 的安装：

```bash
python -m pip install --dry-run segment-anything "scikit-image==0.24.0"
python -m pip install segment-anything "scikit-image==0.24.0"
python -m pip check
```

SAM-B 权重链接来自 [SAM官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints)。
只下载到缓存目录，不提交权重。已有相同权重则设置 SAM_CKPT 指向它即可。

```bash
mkdir -p .cache/sam
curl -L --fail --retry 3 --connect-timeout 20 -C - -o .cache/sam/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
export SAM_CKPT='.cache/sam/sam_vit_b_01ec64.pth'
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
python -m pytest -q tests/test_region_vlad.py
```

这些单元测试覆盖独立数学实现，不宣称已与整个官方模型做数值一致性复现。

## 分阶段运行

先做小规模拟合和4图工程检查，不启动任何 VPR 训练：

```bash
bash scripts/run_region_vlad_screen.sh fit
bash scripts/run_region_vlad_screen.sh smoke
```

脚本固定物理卡1，内部为 cuda:0。改卡用 `REGION_VLAD_GPU=0 bash ...`。
第一阶段也会显示 GSV 文件发现（模型加载前）、掩码提取、词典拟合、PCA拟合日志；
PCA是CPU计算，其耗时不代表卡住。模型与路径 SHA256、代码和依赖版本写入契约。

确认 smoke 无异常，检查打印的区域数量、覆盖率和 `unique_super_masks`。
若 SAM 平均唯一 SuperSegment 接近1或覆盖接近全图，先反馈输出，不继续全库缓存。
工程通过并不代表方法有效。

```bash
bash scripts/run_region_vlad_screen.sh cache
bash scripts/run_region_vlad_screen.sh eval
```

缓存每张图片原子保存，可中断后重跑 `cache` 自动跳过已完成分片。
不保存全库原生 DINO tokens，四组压缩区域描述子及RU描述子合计通常数GB，
但依地区域数与压缩率变化；建议预留10GB。SAM逐图推理耗时由 smoke 实测，
不承诺整个过程只需几分钟。第一次 fit 不支持中断续跑；其输出完成后不再重复拟合。
升级代码、改特征或改掩码设置后契约不匹配会停下，不能拼接旧缓存。

## 看什么结果

输出 `doc/region_vlad_msls_<时间>/summary.json` 与 `predictions.npz`：

1. `actual_retrieval`：RU必须复现675/740，否则停止比较。四组各自全库区域检索
   R@1/5/10/20、纠错与回退查询ID、net。**这才是实际检索效果**。
2. `equal_budget_union20_ORACLE_ONLY`：RU10+区域10，去重后用RU20补齐20个，
   与RU20比较新增/丢失可达查询。没有重排序，不能将其当最终R@1。
3. 相同区域预算下 SAM 与 SLIC/grid/shifted 的差别；区域面积与唯一邻域数量。
4. 提取时间、检索时间（含分片读取）和数据库描述子字节数。

不自动通过几条任意阈值开始长训练。先看 SAM 是否比对照有稳定差异，
是独立 R@1 提升还是仅提供互补候选，再决定下一阶段。
MSLS已反复用于探索，这次仍是探索性结果；有效后需要另一个未调参测试集。
不做事后按GT选query、挑区域或扫描融合权重；也不修改RU checkpoint。
