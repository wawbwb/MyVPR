# CLIP 抑制后仍失败的查询分析

scripts/analyze_clip_failures.py 仅用 CPU 分析已经保存的分数、掩码和原图，不加载 CLIP/RU，不训练，不提取特征，不要求人工标注。

选择规则预先固定为：全部基线 R@1 错误且掩码非零的查询，本轮共 53 个。按 query 编号展示，不按收益挑选。每个查询固定两个数据库图像：基线最佳 GT 和基线错误 top1。比较三个实验中这两个相同图像的分数，避免干预后最佳候选变化导致解释混乱。固定差值不是干预后的最佳正负差值；后者及最佳 GT 排名另行保存。

本机已完成全部 53 个查询的分数分析，结果位于 doc/clip_failure_cases_local_v1。2 项测试通过，覆盖候选排名改变时仍保持固定比较对象，以及非法输入拒绝。完整案例原图在训练机，本机只能复用上一轮已有的部分预览。

训练机生成全部案例图（CPU）：

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python scripts/analyze_clip_failures.py \
  --evaluation doc/clip_external_screen_v1 \
  --cache .cache/clip_external_positive_v1 \
  --dataset-root datasets/msls-val \
  --output doc/clip_failure_cases_v1
```

Windows 下载（目标路径末尾不要加反斜杠）：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/clip_failure_cases_v1 "C:\Users\zy\Desktop\MyVPR\doc"
```

输出 index.html 展示 query 原图和掩码、固定 GT、固定错误图像及分数表；report.json 保留全部数字。带 dataset-root 时应生成 159 张图片（53×3）。输出存在时拒绝覆盖，请勿成功后重复运行。没有 dataset-root 时仍分析全部分数，但图片缺失会明确标出。

## 已获得的分数结论

| 53 个失败查询 | CLIP 定位 | 空间打乱 |
| --- | ---: | ---: |
| 固定正确匹配分数上升 / 下降 | 27 / 26 | 33 / 20 |
| 固定错误匹配分数上升 / 下降 | 20 / 33 | 25 / 28 |
| 固定正负差值改善 / 恶化 | 28 / 25 | 28 / 25 |
| 固定正确匹配超过原错误第一名 | 0 | 1 |

CLIP 确实压低了部分错误匹配，但同时改变了正确匹配；没有实现足以反转固定比较对象顺序的优势。不能把正负分数变化当作某个具体像素的因果贡献，也不能仅从余弦分数变化断定真实地点信息被删除。

## 已有图片的初步观察

人工查看本机已有预览中的 q1、q34、q49、q98、q237、q242，作为案例观察，不是完整 53 个查询的视觉审阅。

- q34、q98 的车辆附近存在明显评分响应，说明并非所有失败都因为 CLIP 完全没有命中车辆；但热图窗口也覆盖邻近道路或其他区域。
- q49 的评分同时延伸到建筑立面和道路；仅看车辆是否变暗不足以确认抑制具有选择性。
- q1、q34、q237 的查询与给定 GT 在视角或场景构图上差异明显，错误候选也有相似的街景结构。不能据此确定错误由动态物体造成，也不据此更改 GT。

下一步用完整案例页检查：车辆响应是否明确、掩码是否同时覆盖建筑等静态区域、正确/错误参考图像之间是否有更明显的视角或外观差异。无需圈图或填写标签；完整原图供后续分析，不作为新增人工标注任务。
