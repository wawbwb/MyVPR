# V2：同城双向候选与困难采样首筛

状态：代码已提供，未在本机测试。保留 V1 脚本和缓存，不修改其 SHA 或结果。无语义训练。

## 为什么做这一轮

V1 的 MSLS 为 675/740，纠错/破坏均 0；40 个可达 RU 错误中，最佳 RU 正样本仅 5 个得到正向修正优势。43.42% 的残差处于 tanh 饱和区。V1 的 GSV selection/calibration 基线均达 99%，因此先检查困难样本训练是否有帮助，不同时改模型、tanh 或扩大 alpha。

本轮复用 V1 两视图缓存，两个方向都作为 query，数据库使用同 split、同城的全部缓存视图，排除 query 自身。较原来一图 DB 的设定，数据库包含每地点两视图。仅去掉跨城候选不保证任务更难；必须阅读实际 audit 中的错误和 hard-query 数。不同 place ID 按 GSV 标注视为负样本，地理相邻地点可能存在标签噪声，不能当几何真值。

困难定义在运行前固定：原 RU 错误，或最佳正样本与最佳负样本余弦差小于 0.03。训练漏召回正样本仍注入；select/calibrate/MSLS 永不注入。同城图像数不足 top-k+1 的城市跳过并写入 `omitted_small_cities`，不得隐去这一评测范围变化。新缓存保存 query/candidate 行号，便于回查。

两组均从零初始化同结构 head，每 epoch 同样 N 次更新：

- `uniform`：新缓存全部查询随机遍历一次。
- `hard_mix`：困难和容易查询各抽取约 N/2 次，有放回；并非每 epoch 看遍全部正确查询。两组总更新数相同，差异是采样分布。

不只对困难子集计分：选模、校准主指标仍是全部保留查询的净命中，另报 hard subset。训练与推理参数沿用 V1（3 epoch，训练 alpha=0.1，校准 alpha=[0,.01,.03,.1]）；不使用 MSLS 选择模式或扫权重。V1 与 V2 的 GSV 数据库不同，不能直接比较百分比作为模型提升。

## 同步

本机 PowerShell，逐行执行：

```powershell
cd D:\Desktop\MyVPR
git add scripts/visual_pair_hard_screen.py tests/test_visual_pair_hard_screen.py doc/VISUAL_PAIR_HARD_SCREEN.md doc/VISUAL_PAIR_RUNBOOK.md
git commit -m "Add same-city hard-pair visual screen with matched sampling control"
git push origin main
```

训练机网络正常则 `git pull --ff-only origin main`。网络失败才走 bundle：

```powershell
git bundle create D:\Desktop\visual_pair_hard_v2.bundle main
scp D:\Desktop\visual_pair_hard_v2.bundle wt@10.16.48.202:/tmp/visual_pair_hard_v2.bundle
```

```bash
cd ~/workspace/OpenVPRLab
git bundle verify /tmp/visual_pair_hard_v2.bundle
git fetch /tmp/visual_pair_hard_v2.bundle refs/heads/main:refs/remotes/scp/visual_pair_hard_v2
git merge --ff-only refs/remotes/scp/visual_pair_hard_v2
```

不删除旧结果解决 merge 冲突，若有冲突先审查明确路径。

## 训练机命令

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -m pytest -q tests/test_visual_pair_verifier.py tests/test_visual_pair_hard_screen.py
```

重建候选不重新提特征；会校验旧 token/descriptor SHA，期间可能没有进度条。输出需新目录。

```bash
python -u scripts/visual_pair_hard_screen.py remine --source .cache/visual_pair/gsv_v1 --output .cache/visual_pair/gsv_hard_v2 --device cuda:1 --batch-size 16 --top-k 20
```

```bash
python -u scripts/visual_pair_hard_screen.py train --cache .cache/visual_pair/gsv_hard_v2 --output logs/visual_pair/hard_smoke_v2 --device cuda:1 --mode hard_mix --smoke-test
```

```bash
python -u scripts/visual_pair_hard_screen.py train --cache .cache/visual_pair/gsv_hard_v2 --output logs/visual_pair/uniform_v2 --device cuda:1 --mode uniform --epochs 3 --seed 42
```

```bash
python -u scripts/visual_pair_hard_screen.py train --cache .cache/visual_pair/gsv_hard_v2 --output logs/visual_pair/hard_mix_v2 --device cuda:1 --mode hard_mix --epochs 3 --seed 42
```

到此先发送新缓存 `manifest.json` 和两组 `run.json`。若仍只有极少 GSV 错误或两组均 NO_GAIN，就停止，不自动进入语义实验。硬采样没有稳定优于 uniform 时不能归因为困难挖掘有效。

## 审查后才运行 MSLS

两组一起评估，禁止看 MSLS 后挑组当预注册赢家。评测沿用 V1 的新目录输出方式，会分别提取 MSLS native token（每组约 12 GiB，注意磁盘空间）。

```bash
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
python -u scripts/visual_pair_hard_screen.py eval --run logs/visual_pair/uniform_v2 --checkpoint "$RU_CKPT" --dataset-root datasets/msls-val --output doc/visual_pair_msls_uniform_v2 --device cuda:1
```

```bash
python -u scripts/visual_pair_hard_screen.py eval --run logs/visual_pair/hard_mix_v2 --checkpoint "$RU_CKPT" --dataset-root datasets/msls-val --output doc/visual_pair_msls_hard_mix_v2 --device cuda:1
```

MSLS 仍只作为探索性证据。小幅净收益不能替代多 seed 与独立 benchmark 确认。
