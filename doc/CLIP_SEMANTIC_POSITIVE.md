# CLIP Semantic Positive：同地点语义分歧困难正样本

## 方法

旧的 semantic-alias 实验已经表明，把 CLIP 语义相似的异地点图像继续推远并不能改善
MSLS。本阶段不再处理异地点负样本，而是在每个已知地点的 4 张图像内部构造 6 个无向
正样本对：用冻结 CLIP 选择余弦相似度最低的 `top-k` 对，并在学生的 VPR 描述符空间中
最小化：

```text
L_positive = mean(1 - cosine(vpr_i, vpr_j))
L_total = L_vpr + warmup * lambda_positive * L_positive
```

CLIP 只在训练期选择关系，不是学生的拟合目标。CLIP 使用无 RandAugment 的确定性 clean
view，学生仍使用原 A2 的增强图像；二者来自同一批采样图像并保持顺序一致。验证和推理
不运行 CLIP，网络结构仍是 ResNet-50 + MixVPR。

四组实验使用完全相同的候选池、损失、预算和 clean-view 教师路径，仅选择策略不同：

| `selection` | 作用 |
| --- | --- |
| `clip` | 选择同地点内 CLIP 相似度最低的图像对，主实验 |
| `random` | 用不消耗全局 RNG 的确定性哈希选择图像对 |
| `shuffled` | 打乱 batch 内 CLIP pair score 与真实图像对的对应关系 |
| `student` | 选择当前 VPR 描述符相似度最低的同地点对 |

初始设置固定为 `lambda=0.05`、`positive_topk=1`、100 places × 4 images、20 epoch。
旧的 global/region/attention distillation 和 semantic alias 均显式关闭。首轮不要扫描
lambda、top-k 或其他超参数，也不要使用 `--compile`。

## TensorBoard 指标

训练日志应包含：

- `loss_semantic_positive`：未乘 lambda 的正样本辅助损失；
- `effective_lambda_positive`：经过 1500 step warmup 后的有效权重；
- `semantic_positive_selected_clip_sim` / `semantic_positive_selected_clip_disagreement`：
  被选 pair 的 CLIP 相似度及分歧度；
- `semantic_positive_all_clip_sim`：全部同地点候选 pair 的平均 CLIP 相似度；
- `semantic_positive_selected_student_sim` / `semantic_positive_all_student_sim`：
  被选 pair 与全部候选在 VPR 空间的平均相似度；
- `semantic_positive_student_hardness_gap`：所选 pair 相对全部候选的学生困难程度；
- `semantic_positive_student_hard_overlap_frac`：所选 pair 与 student-hard pair 的重合率；
- `semantic_positive_selected_pair_count`、`semantic_positive_candidate_pair_count`、
  `semantic_positive_valid_place_frac`、`semantic_positive_valid_view_frac`、
  `semantic_positive_view_coverage_frac`：候选和覆盖诊断；
- `semantic_positive_selected_year_gap`、`semantic_positive_selected_month_gap`、
  `semantic_positive_selected_heading_gap_deg`：可用元数据上的条件差异诊断；
- `semantic_positive_year_pair_frac`、`semantic_positive_month_pair_frac`、
  `semantic_positive_heading_pair_frac`：各项元数据的有效 pair 比例；
  `semantic_positive_metadata_pair_frac` 表示三项同时有效；
- `semantic_positive_{random,shuffled,student}_control`：确认日志与配置身份。

主实验必须满足 `selected_clip_sim < all_clip_sim`，否则 CLIP disagreement 没有真正
参与选样。`student` control 的 `student_hard_overlap_frac` 应接近 1。若 `clip` 与
`student` 的 overlap 长期接近 1 且结果相同，则 CLIP 没有提供超出学生自身困难挖掘的
新信息。

## 运行命令

从项目根目录先运行完整回归测试：

```bash
pytest -q tests/test_semantic_positive.py tests/test_semantic_alias.py \
  tests/test_semantic_reliability.py tests/test_phase_c_attention.py
```

建议先分别进行单 batch 冒烟测试：

```bash
for mode in random shuffled student clip; do
  python run.py --train --dev \
    --config "config/mixvpr_distill_semantic_positive_${mode}.yaml" \
    --seed 42 --devices 0 --precision 32-true
done
```

冒烟测试全部通过后，按相同 seed 完成四组 20 epoch 实验：

```bash
python run.py --train --config config/mixvpr_distill_semantic_positive_random.yaml \
  --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_positive_shuffled.yaml \
  --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_positive_student.yaml \
  --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_positive_clip.yaml \
  --seed 42 --devices 0
```

配置默认已经是 20 epoch。若服务器通过 `CUDA_VISIBLE_DEVICES` 只暴露一张物理显卡，
命令仍使用逻辑设备 `--devices 0`。

clean teacher view 会增加主机预取内存和约 241 MB 的单 batch GPU 输入。若 DataLoader
因主机内存不足被系统终止，四组命令统一追加 `--num_workers 4`（仍不足时用 2）；不要只
修改某一组，也不要首先改变 batch size。

## 判读与停止条件

统一使用 0--19 epoch 中最高的 MSLS R@1，并同时报告该 checkpoint 对应的 Pitts30k
R@1；不要分别挑选两个数据集的最佳 epoch。历史 A2 结果可以用于方向判断，但四组新方法
必须使用完全相同的代码版本、seed 和预算相互比较。

只有同时满足以下条件，才进入 40 epoch 和多 seed：

1. `clip` 高于 `random`、`shuffled` 和 `student` 三个 control；
2. 相对同预算 A2，MSLS R@1 有清晰的约 `+0.3 pp` 改善；
3. 对应 checkpoint 的 Pitts30k R@1 下降不超过 `0.2 pp`；
4. TensorBoard 证明 CLIP 确实选择了更低相似度的同地点 pair，且有效地点率和 pair 数
   正常。

若 `clip` 不优于任一 control，或仅 random/student 有提升，立即停止该路线，不扫描
lambda/top-k。此时最多只能说明额外正样本加权有效，不能声称 CLIP 语义信息有效。
若 seed 42 通过，再从头以 40 epoch 运行 seed `1、2、42`；当前训练入口不应把筛选阶段
的 20 epoch checkpoint 当作无偏的 40 epoch 实验续跑。
