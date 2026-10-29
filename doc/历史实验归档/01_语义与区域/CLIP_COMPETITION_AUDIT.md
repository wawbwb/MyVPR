# CLIP 缓存类别竞争审计

脚本 scripts/audit_clip_competition.py 仅依赖 CPU、NumPy、Pillow，复用已下载的旧/新 CLIP 余弦与既有标注。不会加载模型、重新提取特征、训练或检索。

输出 REPORT.md 和逐窗口 report.json。按 external 区域并集及每个独立多边形分别统计：动态/自车/静态严格赢家份额、逐类赢家、三个组的最大余弦、margin、窗口中物体占比、正证据均分。窗口权重按每个区域像素的重叠平均贡献计算，避免重复累计。

类别组并列单列为 tie，不将 argmax 的类别顺序当作胜者。原始余弦不是概率，不能用绝对大小断言模型完全没有车辆表征。多边形之外也可能有动态物体，本报告不计算分割 precision/recall。

校验旧新缓存 SHA、query 路径、图像身份、既有标注 SHA、teacher 身份、类别/窗口配置，以及所用分片能否重建保存的正证据图。原缓存和标注不修改；输出目录存在时要求另设新目录。

训练机复现：

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python -m unittest discover -s tests -p test_audit_clip_competition.py -v
python scripts/audit_clip_competition.py \
  --old-cache .cache/clip_dynamic_msls_v1 \
  --localization doc/clip_dynamic_localization_v2 \
  --annotations doc/dynamic_mechanism_cases_v1/annotations_validated.json \
  --output doc/clip_competition_audit_v1
```

本机已用下载缓存完成运行，因此无需为了分析再到训练机运行一遍。

## 本轮发现

份额均是区域像素加权的窗口贡献份额，不是物体识别率。

- q295 大巴 polygon_0：大窗口动态胜出 72.0%，小窗口仅 3.4%；小窗口静态组占 89.1%，其中 wall 占 43.4%。这与小窗口失去整车上下文相容，但尚不能独立证明因果。
- q664 external 并集：小窗口动态胜出 0%，静态占 91.4%，road surface 占 86.2%。仍受道路相关窗口分数主导。
- q498 external 并集：大窗口 car hood 胜出 96.4%；小窗口自车组仍占 38.8%，静态占 40.6%。固定的自车类别竞争可能同时压掉外部车辆，不能直接删除 ego 类别后就假定问题解决。
- q130、q344、q728 均有部分窗口仍由动态类别胜出，说明不是所有车辆都没有动态响应。

这些证据支持先解决 crop 分类到区域定位的转换和 ego/外部车辆混淆问题；尚不支持通过新一轮抑制系数扫描或扩大数据库掩码计算解决问题。若继续开发，应预先固定一个定位机制与空间对照，并保留当前案例用于回归诊断，不把六例当训练集或独立泛化评测。

CPU 测试覆盖重叠加权与实际区域评分一致、并列/自车竞争、空区域和非法数值拒绝，3 项通过。
