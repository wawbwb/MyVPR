# Query-conditioned Semantic BoQ 实验说明

更新日期：2026-08-27

## 1. 要检验的假设

已有实验否定的是“所有 BoQ query 共用同一张语义显著图或类别负掩码”，并没有否定语义本身。BoQ 的 learned query 作用不同，因此本实验让每个 query 根据自身状态学习独立的类别偏好，再把该偏好作为 cross-attention logit 的小幅、有正有负的偏置。

该方法不是检索后的 reranking。语义直接参与全局 VPR 描述子的生成；推理时保留轻量学生语义头，不需要 SegFormer、标签或缓存。

## 2. 模型路径

训练和推理都使用：

```text
image -> frozen DINOv2 -> frozen trained RU gate -> patch features
                                                |-> semantic student head -> p(class | patch)
BoQ query state -> query-to-class projection -> class preference
class preference x patch class probabilities -> query-specific attention bias
patch features + query-specific bias -> frozen BoQ -> VPR descriptor
```

对第 `q` 个 query 和第 `n` 个 patch，偏置为：

```text
bias(q,n) = scale * normalize(sum_c preference(q,c) * p(c|n))
```

类别偏好先按类别中心化，空间分数再按 patch 中心化，并限制在 `[-0.2, 0.2]`。它既可以抑制，也可以增强某一 query 对某类区域的读取，而不是预设“天空/道路/车辆一定无用”。

query-to-class projection 为零初始化。第一次更新前，代码显式返回历史 RU-BoQ 的无 mask 结果，同时用零值 straight-through 项保留新 adapter 的梯度，因此 warm start 的描述子与 RU checkpoint 严格一致。

## 3. 冻结范围与监督

以下部分从最佳 repeatability+uniqueness（RU）checkpoint 严格加载并冻结：

- DINOv2 backbone；
- 原始 BoQ projection、queries、attention 和输出层；
- 已训练的 RU semantic-region gate。

只训练：

- `aggregator.semantic_head.*`；
- 每层 `aggregator.boqs.*.semantic_query_proj.*`。

学生语义头的输入从 RU 特征上 `detach`，所以分割辅助损失不会改变 DINO/RU。训练 loss 为原 VPR loss 加 confidence-weighted ADE20K patch cross entropy。只有 confidence 不低于 `0.5` 的 patch 进入语义监督。

SegFormer 只离线运行一次。缓存保存 20x20 hard label 和 top-1 confidence；对于约 53 万张 GSV-Cities 图像，两个 uint8 主数组约占 404 MiB，另有少量索引和 manifest。

## 4. 四个必要对照

| 配置 | 语义目标 | 目的 |
| --- | --- | --- |
| `boq_dinov2_query_semantic_architecture_only.yaml` | 无辅助标签 | 测量新结构和 VPR loss 自身的收益 |
| `boq_dinov2_query_semantic_aligned.yaml` | 本图 ADE20K patch 标签 | 主实验 |
| `boq_dinov2_query_semantic_shuffled.yaml` | 同 city、不同 place 的固定 donor | 排除标签分布/正则化收益 |
| `boq_dinov2_query_semantic_random.yaml` | 本图 label-confidence 对在空间上稳定置乱 | 排除每图类别直方图收益 |

四组都从同一 RU checkpoint、同一 seed 42 开始，冻结同一组参数，训练 10 epochs。除 `mode`、cache 使用和语义 loss 权重外，模型与优化设置一致。

## 5. 生成 ADE20K patch cache

首次在现有 VPR 环境安装依赖：

```bash
pip install transformers==4.44.2
```

生成缓存：

```bash
python scripts/cache_gsv_patch_semantics.py \
  --dataset-root datasets/gsv_cities \
  --output .cache/ade20k_patch_labels/segformer_b0_ade20k_grid20 \
  --device cuda:1 \
  --batch-size 16 \
  --num-workers 8 \
  --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
  --revision 489d5cd81a0b59fab9b7ea758d3548ebe99677da \
  --grid-size 20 20 \
  --target-image-size 280 280 \
  --eligible-min-views 4 \
  --flush-every 10 \
  --amp
```

中断后使用完全相同的命令并在末尾增加 `--resume`。完成后必须看到 `manifest.json` 中 `"complete": true`。这里固定的是 Hugging Face 模型仓库当前最新提交，而不是会漂移的 `main`；manifest 还会记录实际解析到的完整 commit 和三个数组的 SHA256。`summary.json` 会给出类别分布及 confidence 0.5/0.6/0.7 的覆盖率；若 coverage@0.5 接近 0，应先停止而不是启动监督实验。

## 6. 训练命令

先选择现有 RU 最佳 checkpoint：

```bash
RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
sha256sum "$RU_CKPT"
```

该文件的预期 SHA256 为
`38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e`。
四个配置已固定此值；路径指向其他 epoch、残缺副本或不同权重时，训练会在加载前直接停止。

先运行结构对照和 aligned。每次启动都会严格检查 checkpoint 必须包含 backbone、原 BoQ 和 RU gate；不会把缺失权重静默忽略。

```bash
python run.py --config config/boq_dinov2_query_semantic_architecture_only.yaml \
  --train --init-checkpoint "$RU_CKPT"

python run.py --config config/boq_dinov2_query_semantic_aligned.yaml \
  --train --init-checkpoint "$RU_CKPT"
```

若 aligned 不优于 architecture-only，或 MSLS overall 相对冻结 RU 下降超过 `0.2 pp`，则已经不可能通过判据，可直接停止。若 overall 未达到 `+0.3 pp` 但仍在允许范围内，应先做困难条件评估；只有困难条件达到 `+1.0 pp` 才值得继续。通过其中一道初筛后，再运行两个归因对照：

```bash
python run.py --config config/boq_dinov2_query_semantic_shuffled.yaml \
  --train --init-checkpoint "$RU_CKPT"

python run.py --config config/boq_dinov2_query_semantic_random.yaml \
  --train --init-checkpoint "$RU_CKPT"
```

TensorBoard 重点检查：

- `loss_query_semantic` 是否下降；
- `query_semantic_accuracy` 是否明显高于随机水平；
- `query_semantic_valid_frac` 是否不是接近 0；
- `query_semantic_entropy_norm` 和 `query_semantic_logit_std` 是否没有塌缩；
- `query_semantic_bias_std`、`query_semantic_query_std` 和
  `query_semantic_adapter_rms` 是否从 0 稳定增长，确认语义确实进入 BoQ；
- 四组 MSLS-val/Pitts30k 的 R@1，而不是只比较训练 loss。

## 7. 判定规则

语义归因成立必须同时满足：

1. aligned 优于 architecture-only；
2. aligned 优于 shuffled 和 random；
3. aligned 相对冻结 RU 的 MSLS-val overall 至少 `+0.3 pp`，或者在足够大的困难条件集至少 `+1.0 pp` 且 overall 下降不超过 `0.2 pp`。

若 aligned 与 shuffled/random 相同或更差，停止该路线，不扫描 `lambda` 或 bias scale。若单 seed 通过，再跑三个 seed 和完整 condition full-db 评估；之后才考虑 40 epochs 或 SALAD 复现。

条件评估可直接读取新 checkpoint；评估入口会恢复 RU gate，BoQ 会从 checkpoint 配置自动构建并使用学生语义分支：

```bash
python scripts/eval_condition_robustness.py \
  --my-ckpt '<query-semantic checkpoint>' \
  --origin-ckpt "$RU_CKPT" \
  --msls-path datasets/msls-val \
  --device cuda:1 \
  --image-size 280 280
```

## 8. 同步与训练前检查

开发机提交并推送后，训练机执行：

```bash
cd ~/workspace/OpenVPRLab
git status --short --branch
git pull --ff-only origin main
conda activate VPR
pip install transformers==4.44.2
python -m pytest -q \
  tests/test_query_semantic_boq.py \
  tests/test_query_semantic_cache.py \
  tests/test_query_semantic_configs.py \
  tests/test_condition_eval_loader.py \
  tests/test_dynamic_category_prior.py
```

测试通过后先运行 architecture-only 的 `--dev` smoke。它仍需要 RU checkpoint，但不需要 ADE20K cache：

```bash
test -d ~/.cache/torch/hub/facebookresearch_dinov2_main
python run.py \
  --config config/boq_dinov2_query_semantic_architecture_only.yaml \
  --train --dev --init-checkpoint "$RU_CKPT"
```

`--dev` 也可能占用一个 TensorBoard `version_*` 目录；正式 architecture-only 训练通常会进入下一个 version，比较结果时不要把 smoke 目录当成正式 run。

缓存完成后，再用 aligned 跑一次真实缓存/metadata/辅助损失 smoke：

```bash
python run.py \
  --config config/boq_dinov2_query_semantic_aligned.yaml \
  --train --dev --init-checkpoint "$RU_CKPT"
```

启动正式训练前确认输出明确包含 RU checkpoint 的 SHA256 前缀、载入 tensor 数，以及“Frozen RU base”。监督组还会在构建模型前逐一核验 `labels.npy`、`confidence.npy`、`shuffled_indices.npy` 的完整 SHA256。任何 provenance、missing key、cache grid/commit/hash 不匹配都应直接停止，不要绕过检查。
