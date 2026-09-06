# 视觉优先成对验证：阶段 V 运行命令

V1 已完成，MSLS 净收益 0。下一轮使用独立的 [V2 同城困难采样运行手册](VISUAL_PAIR_HARD_SCREEN.md)，复用 V1 token，不覆盖下列历史结果。

状态：初版代码已实现；按用户要求未在本机运行测试。不是 Pair-VPR/R²Former 的完整复现。语义阶段 S 尚未实现，须先检查阶段 V 的净收益。

实现为原生 20×20 DINO mutual-NN edge set、共享 MLP/set pooling、小型 pair head 与 RU 分数残差融合。不读取 LSA teacher，不复用旧 14×14 缓存，不使用未经定义的 attention proxy。

## 1. 同步（本机 PowerShell）

以下提交只包含本轮代码和归档文档，不使用 `git add .`，避免收进无关文件。若归档文件已另行提交，重复 add 不会重写提交。

```powershell
cd D:\Desktop\MyVPR
git add src/visual_pair_verifier.py scripts/visual_pair_pipeline.py tests/test_visual_pair_verifier.py doc/VISUAL_PAIR_RUNBOOK.md doc/VISUAL_FIRST_SEMANTIC_PAIR_VERIFICATION.md doc/CC_LSA_RESULTS_ARCHIVE.md doc/CC_LSA_GATE_A_RUNBOOK.md doc/CC_LSA_PAIR_VPR_EXPERIMENT.md doc/SEMANTIC_VPR_EXPERIMENT_SUMMARY.md
git commit -m "Archive CC-LSA and add visual pair verifier screen"
git push origin main
```

训练机 Git 网络可用时：`git pull --ff-only origin main`。若上传失败，在本机执行以下 bundle 命令，然后训练机 verify/fetch/merge；不发送压缩包、不删除未跟踪训练结果。若 merge 报覆盖冲突，先处理明确列出的路径再继续。

```powershell
git bundle create D:\Desktop\visual_pair_v1.bundle main
scp D:\Desktop\visual_pair_v1.bundle wt@10.16.48.202:/tmp/visual_pair_v1.bundle
```

```bash
cd ~/workspace/OpenVPRLab
git bundle verify /tmp/visual_pair_v1.bundle
git fetch /tmp/visual_pair_v1.bundle refs/heads/main:refs/remotes/scp/visual_pair_v1
git merge --ff-only refs/remotes/scp/visual_pair_v1
```

## 2. 训练机：先测试，再缓存

逐块执行。所有输出目录必须不存在，失败后不要覆盖旧目录，改用新后缀。native token cache 预计约 15 GiB（24,000 图），MSLS 另约 12 GiB；还需要候选特征及工作空间。精确大小依赖实际描述子维数；脚本检查可用空间。缓存无断点续跑。

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"
python -m pytest -q tests/test_visual_pair_verifier.py
```

```bash
python -u scripts/visual_pair_pipeline.py cache --dataset-root datasets/gsv_cities --checkpoint "$RU_CKPT" --output .cache/visual_pair/gsv_v1 --device cuda:1 --batch-size 16 --num-workers 8 --train-places 10000 --holdout-places 1000 --top-k 20
```

每地点固定两张不同图，一张 query、一张 DB；训练/选模/校准地点分别 10,000/1,000/1,000。所有 split 的 DB 相互隔离，包含跨城候选。训练集注入漏召回正样本，以便学习；选模、校准、MSLS 绝不注入。候选范围为 top-20，不与之前 CC-LSA top-100 的可达计数混用。

该 holdout 仅对新 head 未见，不能声称冻结 RU 未见 GSV。此设置是低成本首筛，不是最终跨域论文验证。

## 3. Smoke 与正式三轮训练

```bash
python -u scripts/visual_pair_pipeline.py train --cache .cache/visual_pair/gsv_v1 --output logs/visual_pair/smoke_v1 --device cuda:1 --smoke-test
```

```bash
python -u scripts/visual_pair_pipeline.py train --cache .cache/visual_pair/gsv_v1 --output logs/visual_pair/seed42_v1 --device cuda:1 --epochs 3 --seed 42
```

FP32 训练，实时 tqdm。按 selection split 的全查询命中选 epoch；再仅在 calibration split 从 `[0,0.01,0.03,0.1]` 选择 alpha，平局选择更小 alpha。`run.json` 的 `NO_GAIN` 表示未找到净收益，不能算成功；`CANDIDATE` 也只是下一步评测资格，不是已证明有效。暂不扩展语义训练。

## 4. 冻结参数后的 MSLS 探索性验证

仅在审查 GSV 输出后运行；不根据 MSLS 扫 alpha 或换最佳 epoch。

```bash
python -u scripts/visual_pair_pipeline.py eval --run logs/visual_pair/seed42_v1 --checkpoint "$RU_CKPT" --dataset-root datasets/msls-val --output doc/visual_pair_msls_seed42_v1 --device cuda:1 --batch-size 16 --num-workers 8
```

输出 `summary.json` 与 `per_query.npz`，包含原始排名、残差、正样本标签以及净纠错。要求 RU 复现 675/740，否则报错，不解释模型增益。首轮无证据便停止；小正收益需多 seed 和独立 benchmark 再确认。该初版不自动启动语义分支或下一轮扫参。
