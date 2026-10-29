# CLIP Semantic Alias：实验说明

## 目标与假设

现有实验表明，把 CLIP 全局特征、空间 attention 或 patch reliability
直接作为学生的拟合目标，不能稳定改善 VPR。这个实验不再要求 VPR
描述子模仿 CLIP，而是利用 CLIP 找出“语义很像、地点却不同”的负样本：

| 方法 | MSLS 最佳 R@1 | 后 20 epoch MSLS R@1 均值 | 后 20 epoch Pitts R@1 均值 |
| --- | ---: | ---: | ---: |
| A2 MixVPR | 87.84 | 87.24 | 93.18 |
| C0 spatial gate，无 CLIP | 87.57 | 87.20 | 93.35 |
| CLIP attention KL，λ=0.01 | 86.89 | 86.61 | 93.64 |
| VPR semantic reliability | 86.49 | 86.23 | 93.35 |

最新 reliability 相对 C0 在后 20 epoch 的 MSLS R@1 每轮都更低，平均低
0.96；Pitts 基本不变。这说明它不是偶发崩溃，而是 CLIP 空间目标把模型
稳定地拉向了不利于 MSLS 泛化的解。因此不再跑 positive-only、KL 权重或
temperature 扫描。

- CLIP 负责训练期的关系发现；
- 地点标签负责判断正负关系；
- MixVPR 描述子仍由原生 VPR loss 和 semantic-alias 辅助损失优化；
- 推理期移除 CLIP，模型结构与 A2 baseline 完全相同。

主实验在 batch 内选择 CLIP 相似度最高的、不同地点且任意采样视图间的
最小地理距离不小于 50 m 的负地点。random control 从同一个有效候选池
按对称 label-pair hash 选择相同数量的负地点；shuffled control 对 CLIP
相似度矩阵同步置换行列，保留分数分布与对称近邻图、只破坏其与真实地点
的对应。除 `selection` 外三者设置完全一致。

对应配置：

- `config/mixvpr_distill_semantic_alias.yaml`：`selection: clip`；
- `config/mixvpr_distill_semantic_alias_random.yaml`：`selection: random`；
- `config/mixvpr_distill_semantic_alias_shuffled.yaml`：`selection: shuffled`。

两份配置均沿用 A2 的 ResNet-50、MixVPR、MultiSimilarityLoss、
100 places × 4 images、AdamW 与学习率计划。所有旧蒸馏权重为 0，且
`spatial_attn.enabled=false`。

## 初始超参数

| 参数 | 初始值 | 含义 |
| --- | ---: | --- |
| `lambda` | 0.05 | semantic-alias 辅助损失权重 |
| `negative_topk` | 1 | 每个地点选择的负地点数 |
| `min_geo_distance_m` | 50.0 | 排除数据异常造成的邻近 false negative |
| `student_margin` | 0.2 | 学生描述子需要满足的负样本间隔 |
| `loss_temperature` | 0.05 | 平滑 margin loss 的温度 |

首轮不要同时扫描这些参数。先回答两个问题：辅助关系损失本身是否有效，
以及 CLIP 选择是否显著优于 random 与 shuffled 对照。GSV-Cities 的地点
本身大多按约 100--130 m 网格分离，因此 50 m 过滤只是安全检查，不作为
方法贡献；候选 pair 比例正常应接近 1。

## 运行前检查

1. 确认训练 metadata 能为每张图像提供经纬度；若缺失，程序应明确报错，
   不能悄悄跳过 50 m 过滤。
2. 确认 `clip`、`random` 和 `shuffled` 使用相同的不同地点、地理距离有效 mask，以及
   相同的 top-k 数量。
3. TensorBoard/日志应出现 semantic-alias loss、有效候选比例、被选负样本
   的 CLIP 相似度、地理距离和学生相似度。主实验所选的真实 CLIP 相似度
   应明显高于 random/shuffled control；否则 mining 没有实际选择作用。
4. 该分支包含动态 batch mining，不使用 `--compile`。

## 命令草案

从项目根目录执行。拉取代码后先跑单元测试，再做单 batch 冒烟测试：

```bash
pytest -q tests/test_semantic_alias.py tests/test_semantic_reliability.py \
  tests/test_phase_c_attention.py
python run.py --dev --config config/mixvpr_distill_semantic_alias.yaml --devices 0
python run.py --dev --config config/mixvpr_distill_semantic_alias_random.yaml --devices 0
python run.py --dev --config config/mixvpr_distill_semantic_alias_shuffled.yaml --devices 0
```

### 20 epoch、seed 42 筛选

为了比较相同训练预算，建议同时保留/重跑 A2 的前 20 epoch：

```bash
python run.py --train --config config/mixvpr_resnet50.yaml \
  --max_epochs 20 --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias_random.yaml \
  --max_epochs 20 --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias_shuffled.yaml \
  --max_epochs 20 --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias.yaml \
  --max_epochs 20 --seed 42 --devices 0
```

先运行 random 和 shuffled，再运行 CLIP 主实验，可以避免只在看到主实验
结果后才决定是否补对照。所有组都使用相同 seed、batch size、epoch 数与验证集。

`--devices 0` 只是命令示例；若服务器目标卡不是 0，请三组一起改成相同的
设备编号。

### 通过筛选后跑满 40 epoch

配置默认是 40 epoch，完整实验无需覆盖 `max_epochs`：

```bash
python run.py --train --config config/mixvpr_distill_semantic_alias_random.yaml \
  --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias_shuffled.yaml \
  --seed 42 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias.yaml \
  --seed 42 --devices 0
```

只有 seed 42 达标后，再运行至少两个额外 seed：

```bash
python run.py --train --config config/mixvpr_distill_semantic_alias_random.yaml \
  --seed 1 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias_shuffled.yaml \
  --seed 1 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias.yaml \
  --seed 1 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias_random.yaml \
  --seed 2 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias_shuffled.yaml \
  --seed 2 --devices 0
python run.py --train --config config/mixvpr_distill_semantic_alias.yaml \
  --seed 2 --devices 0
```

## 20 epoch 判读与停止条件

使用 0--19 epoch 中的最高验证 R@1，不用最后一个 epoch 代替最佳值。进入
40 epoch 的条件满足以下任一项：

1. CLIP alias 相对同预算 A2 的 MSLS R@1 至少提高 0.3，且 Pitts30k
   下降不超过 0.2；
2. MSLS night/season 等跨条件子集至少提高 2.0，MSLS overall 下降不超过
   0.2，且 Pitts30k 下降不超过 0.2。

同时，CLIP alias 必须优于 random 和 shuffled control；若差异落在单 seed 波动内，
只能说明额外关系损失可能有效，不能证明 CLIP 语义有效。出现下列任一情况
时停止，不继续做 lambda/temperature 扫描：

- CLIP、random 和 shuffled 均未超过 A2；
- CLIP 不优于 random/shuffled，或任一 control 反而更好；
- MSLS 的改善伴随 Pitts30k 超过 0.2 的退化；
- 有效候选率过低、地理过滤未生效，或主实验所选 CLIP 相似度与随机选择
  没有明显差异。

建议按下面的表记录结果。Pitts30k 应填写“MSLS 最佳 checkpoint 对应的
Pitts30k”，不要从另一个 epoch 单独挑 Pitts30k 峰值。历史 A2 的 40 epoch
MSLS 最佳 R@1 为 87.84（epoch 22），对应 Pitts30k R@1 为 93.14；该数值只用于方向判断，20
epoch 筛选仍应与相同预算的 baseline 比较。

| 实验 | seed | 预算 | MSLS 最佳 R@1 / epoch | 同 checkpoint Pitts R@1 | alias loss | 有效候选率 | 被选 CLIP 相似度 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| A2 baseline | 42 | 20 | 待填 | 待填 | -- | -- | -- |
| random alias | 42 | 20 | 待填 | 待填 | 待填 | 待填 | 待填 |
| shuffled CLIP | 42 | 20 | 待填 | 待填 | 待填 | 待填 | 待填 |
| CLIP alias | 42 | 20 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 40 epoch 与论文实验门槛

跑满后至少报告三个 seed 的均值和标准差。可作为后续论文主线的最低条件：

- MSLS overall 相对 A2 的三 seed 均值提高至少 0.5；或跨条件子集提高至少
  2.0，同时 overall 不下降超过 0.2；
- Pitts30k 的三 seed 均值下降不超过 0.2；
- CLIP alias 稳定优于 random 和 shuffled control，而非仅某个 seed 的最佳 checkpoint；
- mining 统计证明 CLIP 选择了更强的语义碰撞负样本，且学生训练后确实
  拉开了这些负样本的 descriptor margin。

若只在 MSLS 单一 seed 上提高、control 同样提高，结论应写成
“关系辅助损失的收益”，不能写成“CLIP 语义信息带来收益”。若 CLIP
选择稳定优于 random 和 shuffled，才支持“CLIP 语义碰撞用于训练期地点判别”的创新点。
