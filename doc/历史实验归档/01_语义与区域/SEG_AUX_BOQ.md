# SegVPR-inspired shared-backbone semantic auxiliary training

> 2026-09-09：V1/V2已完成分析并归档，当前路线停止扩展。
> V2 aligned最佳R@1=91.22%，最终90.68%，对照最终90.54%，冻结RU91.22%。
> 初期梯度失衡缓解，但整体R@1增益未获支持。详见[结果归档](SEG_AUX_RESULTS_ARCHIVE.md)。
> 下文配置、训练与同步命令仅保留作历史复现，不表示仍待启动新实验。

## V2：分割头预热与权重渐增（独立新实验）

配置：config/boq_dinov2_seg_aux_warm.yaml。复用原始RU与grid20标签，不接着v1权重训练。
aligned先用独立优化器训练分割头500步，骨干eval/no_grad，不执行VPR更新；
记录head_warmup.csv，保存prewarmed_head.pt，并检查检索参数未变化。
然后重置数据随机种子和优化器，开始完整5轮联合训练。语义权重在联合训练前1000步
从0.00002线性增加至0.02，骨干与BoQ学习率等其他配置保持不变。
vpr_only不预热无用的分割头，直接做相同5轮VPR微调；因此对照匹配的是检索网络
更新预算，不是总算力。若以后跑shuffled，预热及联合阶段都使用其固定donor标签。
500步预热不保证头已充分收敛，不能把它描述成“已消除初始化问题”。

smoke包含2步头预热+2步联合更新，正式模式为500+7815步（若数据仍为1563batch/epoch）。
从新输出目录启动，保留v1；先跑vpr_only和aligned，不自动开始shuffled。
联合阶段中断仍用同一配置加--resume和原last.ckpt恢复，跳过已完成的头预热；
若头预热中断，尚无完整训练checkpoint，请换新目录重跑预热。
新指标gradient_probe_step是实际梯度测量step，可用于排除日志重复沿用。
effective_seg_weight记录当前权重。V1配置缺省无预热/无ramp，行为保留。

```bash
python -m pytest -q tests/test_seg_aux.py tests/test_query_semantic_cache.py
python -u scripts/train_seg_aux.py --config config/boq_dinov2_seg_aux_warm.yaml --mode aligned --init-checkpoint "$RU_CKPT" --device 0 --smoke-test --output logs/seg_aux/smoke_warm_v2
python -u scripts/train_seg_aux.py --config config/boq_dinov2_seg_aux_warm.yaml --mode vpr_only --init-checkpoint "$RU_CKPT" --device 0 --output logs/seg_aux/vpr_only_warm_v2
python -u scripts/train_seg_aux.py --config config/boq_dinov2_seg_aux_warm.yaml --mode aligned --init-checkpoint "$RU_CKPT" --device 0 --output logs/seg_aux/aligned_warm_v2
```

本机未运行测试，需训练机执行。不能根据一次恢复运行和少量查询变化声称有显著增益。

状态：训练机实验已完成，结果已归档；本机未运行测试。不是原版 SegVPR 复现。

## 假设与边界

借鉴 SegVPR 的共享编码器联合学习，而不是照搬其多尺度注意力和域对抗。
语义 CE 直接更新 DINOv2 最后两个 block；不使用 CLIP、FiLM、新 attention bias
或语义检索后处理。分割头输入是检索所使用的 raw DINO patch features，没有 detach。
RU gate 参数冻结，但 gate 运算仍允许 VPR 梯度传回骨干。BoQ 可训练。
推理仍是 DINO → RU gate → BoQ，不执行分割头。

使用既有 grid20 ADE20K 标签与置信度缓存，不需要生成 grid70 或新教师。
标签是伪标签，不是 SegVPR 合成数据的真值，因此不能继承其效果结论。
仅 photometric 增强，禁止旋转/裁剪/翻转造成缓存错位。

三组共用一个配置，通过 --mode 选择：

| mode | 训练目标 |
| --- | --- |
| vpr_only | 原 VPR loss，相同解冻与训练预算 |
| aligned | VPR + 0.02 × confidence-weighted segmentation CE |
| shuffled | 同上，使用缓存内同城异地点固定 donor 的标签及置信度 |

三组都加载同一缓存、实例化同一分割头，并在开始训练前重置 RNG。
vpr_only 的头冻结且不参与前向。全部为 P40/K4、280×280、seed42、5完整epochs，
骨干/BoQ lr=2e-6，分割头 lr=1e-4，无 warmup、无提前停止。
lambda=0.02 是待验证的起点，不是已证实最优值。禁止根据 MSLS 反复挑权重。
FP32 避免将 AMP 数值失败与路线效果混淆，显存/速度需由训练机 smoke 确认。

## 安全检查与诊断

- 校验 RU checkpoint SHA256、来源配置及三个缓存数组 SHA256。
- 数据加载器继续检查缓存版本、图像索引、CSV 和异地点 donor 对应关系。
- 更新前检查 RU forward 等价，并在正式训练前复现 MSLS 675/740。
- 每100步在最后一个 block 的一个共享参数上记录语义/VPR梯度范数、
  加权范数比及余弦。它是局部 probe，不是整个网络的梯度统计。
- 第一批语义梯度为零时停止；非有限损失/梯度时停止，不写成功状态。
- 每个 epoch 都保存 checkpoint，额外保存 last.ckpt；输出目录不得已存在。
- smoke 只跑2个batch，不写 checkpoint，不判断准确率或路线成功。
- completed.json 只代表跑完，不代表实验有效。

TensorBoard 重点看 seg_loss、query_semantic_accuracy、query_semantic_valid_frac、
seg_shared_grad_norm、weighted_seg_vpr_grad_ratio、seg_vpr_grad_cosine 与 msls-val/R1。
CE降低不等于VPR提升；长期梯度相反是负迁移的线索，不是单独的终止定理。

## 训练机操作（仓库根目录，已有 VPR 环境）

```bash
conda activate VPR
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
python -m pytest -q tests/test_seg_aux.py tests/test_query_semantic_cache.py
python -u scripts/train_seg_aux.py --mode aligned --init-checkpoint "$RU_CKPT" --device 1 --smoke-test --output logs/seg_aux/smoke_aligned_v1
```

smoke通过后分别执行（不要复制终端提示符，也不要把相邻命令粘成一行）：

```bash
python -u scripts/train_seg_aux.py --mode vpr_only --init-checkpoint "$RU_CKPT" --device 1 --output logs/seg_aux/vpr_only_v1
python -u scripts/train_seg_aux.py --mode aligned --init-checkpoint "$RU_CKPT" --device 1 --output logs/seg_aux/aligned_v1
python -u scripts/train_seg_aux.py --mode shuffled --init-checkpoint "$RU_CKPT" --device 1 --output logs/seg_aux/shuffled_v1
```

device为进程可见GPU索引；若设置CUDA_VISIBLE_DEVICES=1，只能传 --device 0。
新一轮运行改输出目录名，不删除旧结果。不要将 --smoke-test 留在正式命令中。

```bash
tensorboard --logdir logs/seg_aux --host 127.0.0.1 --port 6006
```

每组 config.yaml、contract.json、csv/version_0/metrics.csv、completed.json 和 checkpoint
都保留。三组下载齐全再比较；主比较固定第5epoch（last.ckpt），最佳epoch仅作次要结果，
不可拿不同训练预算的单点宣布优劣。aligned需胜vpr_only和shuffled，且检查是否胜冻结RU。
若只有shuffled提升，不能声称语义对应关系有效。短训无收益仅归档当前配置未获支持。

现有评测脚本按保存配置严格恢复 backbone/BoQ/RU，忽略训练专用 seg_head：

```bash
python -u scripts/eval_condition_robustness.py --my-ckpt logs/seg_aux/aligned_v1/checkpoints/last.ckpt --origin-ckpt logs/seg_aux/vpr_only_v1/checkpoints/last.ckpt --my-name SegAux-aligned --origin-name matched-VPR-only --msls-path datasets/msls-val --device cuda:1 --image-size 280 280
```

条件评测也需补 shuffled 和冻结 RU。不是只看 aligned 最好 checkpoint。

## 同步（本机 PowerShell）

### 中断恢复

使用原 output 下的 checkpoints/last.ckpt，通过 --resume 恢复模型、优化器、
学习率调度器及 epoch/global_step，而不是把它当作 --init-checkpoint。
--init-checkpoint 仍指向最初的 RU。总预算仍为5轮，不是额外5轮。
仅支持保存完整的 epoch 边界；中断轮次未保存的批次需要重跑。
原 contract 和训练配置必须一致。旧 CSV/TensorBoard 保留，新日志自动使用新 version。
旧版未保存数据加载 worker 的随机状态，恢复后的采样/增强不能保证与不中断运行逐步一致，
因此恢复记录单独保留，分析时应披露。

```bash
python -u scripts/train_seg_aux.py --mode aligned --init-checkpoint "$RU_CKPT" --device 0 --output logs/seg_aux/aligned_v1 --resume logs/seg_aux/aligned_v1/checkpoints/last.ckpt
```

只提交本路线五个文件；不包括下载的 SegVPR-main、历史未提交归档或删除：

```powershell
git add src/models/seg_aux.py scripts/train_seg_aux.py config/boq_dinov2_seg_aux.yaml tests/test_seg_aux.py doc/SEG_AUX_BOQ.md
git diff --cached --stat
git commit -m "Add SegVPR-inspired shared-backbone semantic auxiliary screen"
git push origin main
```

检查暂存清单，不要把之前暂存的无关文件一起提交。训练机 git pull --ff-only origin main。
若网络失败，使用 bundle（本机）：

```powershell
git bundle create ..\seg_aux.bundle main
scp ..\seg_aux.bundle wt@10.16.48.202:/tmp/seg_aux.bundle
```

训练机：

```bash
git bundle verify /tmp/seg_aux.bundle
git fetch /tmp/seg_aux.bundle refs/heads/main:refs/remotes/scp/seg_aux
git merge --ff-only refs/remotes/scp/seg_aux
```

若未跟踪历史文件阻挡合并，先核对/备份冲突文件，不运行 git clean 或 reset --hard。
