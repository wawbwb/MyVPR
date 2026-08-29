# Crop-CLS Local Semantic FiLM-BoQ（SemVPR-lite）

## 1. 当前状态

代码、配置与测试已经实现，seed-42 的 500-step 预飞也已完成；正式 retrieval 训练尚未运行，因此本方案目前仍没有检索正/负结论。原始审计因 wrong-place 探索项未达到人为阈值而输出 `FAIL`，但实现、空间对齐、梯度和描述子干预检查均通过，具体解释见第 6.1 节。它是参考 SemVPR 核心机制设计的低成本因果筛选，不是 SemVPR 官方实现或完整复现。

本路线不再使用已经失败的 ADE20K hard class bias、CLIP raw patch attention/affinity、逐图单位方差化、固定动态类别负先验或 BoQ attention-logit bias。它验证的是两个此前尚未测试的要素：连续局部语义表征，以及 BoQ 前的 patch×channel 特征调制。

## 2. 固定架构

```text
280×280 photometric student image
        |
frozen DINO blocks 0..9
        |
20×20×768 patch tokens
        +--> 768→128→512 --> 2×2 region mean --> cosine(clean crop CLIP CLS)
        |
        +--> zero-init 128→768 --> 0.1*tanh(delta) --> channel-wise FiLM
                                                        |
                                          DINO blocks 10..11
                                                        |
                                                 pretrained RU gate
                                                        |
                                                       BoQ
```

- 教师固定为 OpenCLIP `ViT-B-16/openai`，只在训练监督模式使用，不进入 checkpoint 或推理。
- batch 固定 P=40、K=4。每个 place 使用 sampled K 中的 view 0；所有 place 在同一步使用 `global_step % 4` 指定的同一象限。
- clean 280×280 teacher image 先切成 140×140 象限，随后由已有 `CLIPTeacherEncoder` resize 到 224。
- FiLM 的 `128→768` 权重和 bias 严格零初始化，调制范围是 `1 ± 0.1`。
- 512D projection 只计算选中的 P=40 个 view；FiLM 仍作用于 P×K=160 个 student view。
- 推理实际执行的额外路径仅约 19.75 万参数的 bottleneck/FiLM，跳过 512D 蒸馏 projection，并完全移除 CLIP；checkpoint 仍保留该小型 projection 权重以便严格恢复。
- 从已审计 RU checkpoint 严格 warm-start；前 10 个 DINO block 冻结，最后 2 个 block、RU gate、BoQ 和 FiLM 联合训练。

损失固定为：

```text
L = L_MultiSimilarity + warmup(step, 500) * 0.05 * L_crop_cosine
```

## 3. 因果对照

| 配置 | 唯一语义变量 | 教师调用 |
| --- | --- | --- |
| `architecture_only` | `lambda_crop=0`，只让 VPR loss 训练相同的推理 FiLM 路径 | 否 |
| `aligned` | 正确 crop CLS 对齐正确 student 象限 | 是 |
| `wrong_region` | 同一 crop CLS 配给同图真正的对角象限（TL↔BR、TR↔BL） | 是 |
| `wrong_place` | 同一象限，但 teacher embedding 在 P 个不同地点间稳定轮换 | 是 |

architecture-only 保留相同的推理架构、数据 I/O、最后两层 DINO/RU gate/BoQ/FiLM 训练范围；纯教师监督用的 `128→512` projection 被跳过并冻结，因为它不参与描述子生成。

## 4. 已实现的防错检查

- config 强制 280×280、P=40、K=4、seed 42、full GSV-Cities、photometric augmentation 和单 GPU；
- config 强制 RU gate、last-two-block 插入点、`768→128→512`、`alpha=0.1`、500-step warmup；
- RU checkpoint 必须同时通过 SHA256、训练 provenance 和逐 key 严格检查；只允许缺少 `backbone.crop_semantic_film.*`；
- step 0 自动用 FiLM enabled/bypass 两条 fp32 路径比较 descriptor，最大误差超过 `1e-6` 立即终止；
- wrong-place 在运行时检查 P 个 donor 确实是不同地点；
- teacher chunk 固定 20，降低 RTX 3090 峰值显存；
- TensorBoard 记录 cosine 因果间隔、channel/projection gradient RMS、modulation RMS 和 descriptor drift；
- `scripts/eval_condition_robustness.py` 会从 checkpoint 的 `backbone.params` 重建并严格加载 FiLM，评测时不会调用 CLIP。

## 5. 配置与核心文件

- `src/models/crop_semantic_film.py`：FiLM、crop-CLS target、严格 RU warm-start；
- `src/models/backbones/dinov2.py`：在最后两个 DINO block 前插入 FiLM；
- `src/core/vpr_framework.py`：crop loss、零起点验证和 TensorBoard 诊断；
- `src/dataloaders/train/gsv_cities.py`、`src/core/vpr_datamodule.py`：每 place 一张 clean teacher view；
- `config/boq_dinov2_crop_semantic_film_preflight.yaml`：500-step 非正式预飞；
- `config/boq_dinov2_crop_semantic_film_{architecture_only,aligned,wrong_region,wrong_place}.yaml`：四个 5-epoch 配置；
- `scripts/audit_crop_semantic_preflight.py`：预飞自动 PASS/FAIL；
- `tests/test_crop_semantic_{film,dinov2,configs,dataset}.py`：离线定向测试。

## 6. 推荐执行顺序

先运行四组定向测试和 `--dev`。随后运行 500-step aligned 预飞；该 run 禁用 checkpoint 且通常不会执行 epoch-end recall，它只验证实现、梯度和因果信号，不能写入结果表。

预飞只有同时满足以下条件才通过：

1. 两个连续记录的因果间隔指标都至少到达 optimizer step 490，证明 500-step run 没有中途退出；
2. `crop_film_zero_start_max_abs_error <= 1e-6`；
3. 最后五个记录点的 `aligned_minus_wrong_region >= 0.05`；
4. 最后五个记录点的 `aligned_minus_wrong_place >= 0.05`；
5. channel-scale gradient、semantic-projection gradient、modulation RMS 和 descriptor drift 全部出现非零值。

通过后先运行 5-epoch architecture-only，再运行 aligned。若 aligned 的 overall 增益尚未达到 0.3 pp，但相对 RU 和 architecture-only 的下降都不超过 0.2 pp，则先做低成本 condition evaluation；只有 overall 达到门槛，或 season 达到下述门槛时，才依次运行 wrong-region 和 wrong-place。最终只有 aligned 同时胜过 RU、architecture-only 和两个错误语义对照，并满足以下任一条件，才进入三个 seed：

- MSLS overall R@1 至少 `+0.3 pp`；或
- season R@1 至少 `+1.0 pp`，且 overall 不低于 RU 超过 `0.2 pp`。

失败时只能否定这套 BoQ 适配的 sparse crop-CLS SemVPR-lite，不能否定完整 SemVPR 的 dense LSA teacher 与原论文 aggregation。

### 6.1 实际预飞结果与正式训练前的协议修订

500-step run 完整到达 step 499，原始审计结果必须保留为 `Verdict: FAIL`，不能事后改写为 PASS：

| 检查 | 数值 | 原始判定 |
| --- | ---: | --- |
| zero-start descriptor max error | 0.000000 | PASS |
| aligned − wrong-region，尾五点均值 | 0.05352 | PASS |
| aligned − wrong-place，尾五点均值 | 0.02755 | FAIL（阈值 0.05） |
| aligned − wrong-place，最后值/全程最大值 | 0.03837 / 0.03837 | 正向且仍在增长 |
| channel/projection gradient | 非零 | PASS |
| modulation RMS，最后值 | 0.00934 | PASS |
| descriptor drift RMS，最后记录值 | 0.000366 | PASS |

终端中的 `batch_acc=0.81` 只对应第 500 step 附近的单个训练 batch，不是 MSLS validation R@1，不能用于判断方案是否提高检索。

原审计把 `aligned − wrong-place >= 0.05` 作为致命门槛，是本项目额外增加的地点特异性检查，不是 SemVPR 的必要训练条件。SemVPR 论文 Eq. (10) 的 LSA 只最小化同图 region feature 与对应 crop CLS 的正余弦距离，没有跨地点负样本项；道路、天空、植被和普通建筑等连续语义本来就可能跨地点相似。因此该项未过线不能证明实现错误，也不能单独否定局部语义蒸馏。

为避免隐瞒原预注册失败，同时避免用不匹配的方法判据产生假停止，现于任何正式 retrieval 结果产生前固定如下修订：

1. 保留原 JSON 的字面 `FAIL`，解释为“机制检查通过，wrong-place 地点特异性探索项未达门槛”；
2. 不修改 `lambda=0.05`、FiLM `alpha=0.1`、warmup 或训练范围，也不重复 500-step run；
3. 只运行预先存在的 5-epoch architecture-only 与 aligned 两组，用 matched MSLS retrieval 直接判断；
4. 两组完成后先看 overall；若 aligned 未达到 `+0.3 pp` 但下降不超过 `0.2 pp`，先评估 season，只有 season 达到 `+1.0 pp` 才继续，否则停止；
5. 最终语义归因门槛不变：aligned 仍必须胜过 RU、architecture-only 和两个错误语义训练对照，并达到 overall/season 的预定增益。

证据为 `doc/crop_semantic_film_runs/preflight_500steps.txt` 与 `preflight_audit.json`。

## 7. 训练机命令

以下命令均从仓库根目录运行。不要设置 `CUDA_VISIBLE_DEVICES=1`，否则配置中的 `devices: [1]` 和 `cuda:1` 会被重新编号。

### 7.1 测试与 RU 核验

```bash
conda activate VPR

python -m pytest -q \
  tests/test_crop_semantic_film.py \
  tests/test_crop_semantic_dinov2.py \
  tests/test_crop_semantic_dataset.py \
  tests/test_crop_semantic_configs.py \
  tests/test_condition_eval_loader.py \
  tests/test_query_semantic_cache.py \
  tests/test_query_semantic_configs.py

python -c "from importlib.metadata import version; v=version('open_clip_torch'); print(v); assert v=='2.26.1'"
test -d ~/.cache/torch/hub/facebookresearch_dinov2_main
export HF_ENDPOINT=https://hf-mirror.com

RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
EXPECTED_RU_SHA='38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e'
test -f "$RU_CKPT"
ACTUAL_RU_SHA="$(sha256sum "$RU_CKPT" | cut -d' ' -f1)"
printf 'RU SHA256: %s\n' "$ACTUAL_RU_SHA"
test "$ACTUAL_RU_SHA" = "$EXPECTED_RU_SHA"
```

### 7.2 真实 batch smoke 与固定 500-step 预飞

```bash
python run.py \
  --config config/boq_dinov2_crop_semantic_film_preflight.yaml \
  --train --dev --init-checkpoint "$RU_CKPT"
```

输出必须包含严格 RU/SHA 载入、frozen crop-CLS teacher 和 `zero-start descriptor max_abs_error <= 1e-6`。

```bash
set -o pipefail
mkdir -p doc/crop_semantic_film_runs

python run.py \
  --config config/boq_dinov2_crop_semantic_film_preflight.yaml \
  --train --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/crop_semantic_film_runs/preflight_500steps.txt

python scripts/audit_crop_semantic_preflight.py \
  --logdir logs/dinov2_vitb14/BoQ_crop_semantic_film_aligned_preflight \
  --output doc/crop_semantic_film_runs/preflight_audit.json
```

审计脚本会自动选择父目录下最新的 event file；FAIL 时写出 JSON 并以状态 2 退出。预飞不保存 checkpoint，也不能当作 recall 结果。

### 7.3 PASS 后的正式 5-epoch 顺序

```bash
python run.py \
  --config config/boq_dinov2_crop_semantic_film_architecture_only.yaml \
  --train --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/crop_semantic_film_runs/architecture_only_5ep.txt

python run.py \
  --config config/boq_dinov2_crop_semantic_film_aligned.yaml \
  --train --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/crop_semantic_film_runs/aligned_5ep.txt
```

只有 aligned 的 best MSLS R@1 胜过 architecture-only，才运行：

```bash
python run.py \
  --config config/boq_dinov2_crop_semantic_film_wrong_region.yaml \
  --train --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/crop_semantic_film_runs/wrong_region_5ep.txt

python run.py \
  --config config/boq_dinov2_crop_semantic_film_wrong_place.yaml \
  --train --init-checkpoint "$RU_CKPT" \
  2>&1 | tee doc/crop_semantic_film_runs/wrong_place_5ep.txt
```

### 7.4 条件评测

跨 `version_*` 选择 architecture-only 与 aligned 的最高 R@1 checkpoint 后运行：

```bash
ARCH_ROOT='logs/dinov2_vitb14/BoQ_crop_semantic_film_architecture_only'
ALIGNED_ROOT='logs/dinov2_vitb14/BoQ_crop_semantic_film_aligned'
ARCH_CKPT="$(find "$ARCH_ROOT" -type f -name '*.ckpt' -printf '%p\n' | sort -t'[' -k2,2nr | head -n 1)"
ALIGNED_CKPT="$(find "$ALIGNED_ROOT" -type f -name '*.ckpt' -printf '%p\n' | sort -t'[' -k2,2nr | head -n 1)"
test -f "$ARCH_CKPT" && test -f "$ALIGNED_CKPT"

python scripts/eval_condition_robustness.py \
  --my-ckpt "$ALIGNED_CKPT" --my-name 'Aligned Crop-FiLM' \
  --origin-ckpt "$RU_CKPT" --origin-name 'RU Baseline' \
  --msls-path datasets/msls-val --batch-size 100 --num-workers 8 \
  --device cuda:1 --image-size 280 280 \
  2>&1 | tee doc/crop_semantic_film_runs/aligned_vs_ru_conditions.txt

python scripts/eval_condition_robustness.py \
  --my-ckpt "$ALIGNED_CKPT" --my-name 'Aligned Crop-FiLM' \
  --origin-ckpt "$ARCH_CKPT" --origin-name 'Architecture Only' \
  --msls-path datasets/msls-val --batch-size 100 --num-workers 8 \
  --device cuda:1 --image-size 280 280 \
  2>&1 | tee doc/crop_semantic_film_runs/aligned_vs_arch_conditions.txt
```
