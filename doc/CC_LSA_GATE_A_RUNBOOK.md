# CC-LSA Pair-VPR Gate A 运行手册

本文只覆盖 `Candidate-Conditioned LSA Semantic Pair-VPR` 的 dense LSA teacher
与 Gate A 充分性审计。Gate A 不训练 RGB pair classifier，也不修改冻结的 RU、
DINOv2 或 BoQ。

完整预注册定义见 `doc/CC_LSA_PAIR_VPR_EXPERIMENT.md`。本手册中的配置、随机种子、
阈值、top-K、top-L 和对照组不得根据 MSLS-val 结果调整。

## 0. 本次交付的简短操作命令

最终代码按用户要求交由训练机测试；本机未执行本次最终版测试。下面的显式同步清单
同时包含代码和 AG-SLRD 紧凑归档，不包含 checkpoint、训练缓存或旧实验的大文件。

本机 PowerShell：

```powershell
cd D:\Desktop\MyVPR
$SyncArchive = Join-Path $env:TEMP 'cc_lsa_gate_a_code.tgz'
tar -czf $SyncArchive -T scripts/cc_lsa_sync_files.txt
if ($LASTEXITCODE -ne 0) { throw '打包失败' }
scp $SyncArchive wt@10.16.48.202:/tmp/cc_lsa_gate_a_code.tgz
if ($LASTEXITCODE -ne 0) { throw '上传失败' }
```

训练机安装（先备份已存在文件，再覆盖清单内文件）：

```bash
cd ~/workspace/OpenVPRLab
CC_LSA_STAGE="$(mktemp -d /tmp/cc_lsa_sync.XXXXXX)"
tar -xzf /tmp/cc_lsa_gate_a_code.tgz -C "$CC_LSA_STAGE"
sed -i 's/\r$//' "$CC_LSA_STAGE/scripts/install_cc_lsa_bundle.sh"
bash "$CC_LSA_STAGE/scripts/install_cc_lsa_bundle.sh" "$CC_LSA_STAGE" "$PWD"
```

旧文件备份目录由安装脚本最后打印，位于 `.cache/cc_lsa_sync_backup.*`。
这一步通过 SCP 同步工作区，不会推进训练机的 Git HEAD。今后恢复 Git 同步时，先检查
工作区差异，避免覆盖这些修改。

训练机测试：

```bash
conda activate VPR
python -m pytest -q tests/test_cc_lsa_*.py
```

测试全部通过后，完整运行：

```bash
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
bash scripts/run_cc_lsa_gate_a.sh all
```

`all` 包含六个阶段，教师合同失败自动终止。希望逐阶段运行时，依次使用
`targets`、`smoke`、`teacher`、`calibrate`、`features`、`audit`。

## 1. 固定路径与终止条件

默认产物如下：

| 阶段 | 默认路径 |
| --- | --- |
| GSV crop-CLS target | `.cache/cc_lsa/gsv_crop_cls_v1` |
| dense LSA teacher | `logs/cc_lsa/teacher_seed42/final.pt` |
| teacher 合同 | `logs/cc_lsa/teacher_seed42/lsa_teacher_contract.json` |
| GSV Q95 calibration | `.cache/cc_lsa/gsv_calibration_v1` |
| calibration scratch | `/tmp/cc_lsa_calibration_v1` |
| 新生成的 RU descriptors | `.cache/cc_lsa/ru_msls.npy` |
| MSLS top-100/local feature cache | `.cache/cc_lsa/msls_top100_v1` |
| Gate A 报告 | `doc/cc_lsa_gate_a` |
| 控制台日志 | `doc/cc_lsa_runs` |

实现文件分工：

| 路径 | 职责 |
| --- | --- |
| `src/models/cc_lsa.py` | 可训练 dense LSA encoder、2×2 region pooling、checkpoint 合同 |
| `src/dataloaders/train/cc_lsa.py` | 固定 GSV one-view-per-place 划分与 clean 280×280 输入变换 |
| `src/cc_lsa_features.py` | 冻结 RU descriptor 与 pre-gate/pre-BoQ DINO local token 的同次前向提取 |
| `src/cc_lsa_gate_a.py` | 确定性 top-100、MNN、placebo、rerank 与 Gate A 判据 |
| `src/cc_lsa_config.py` | Gate A 预注册配置的严格校验 |
| `scripts/cache_gsv_cc_lsa_targets.py` | 冻结 crop-CLS target cache |
| `scripts/train_cc_lsa_teacher.py` | 5-epoch LSA 训练与 GSV holdout teacher 合同 |
| `scripts/calibrate_cc_lsa_gate_a.py` | GSV-only semantic Q95 calibration |
| `scripts/cache_cc_lsa_msls_features.py` | MSLS 固定候选与局部特征 cache |
| `scripts/audit_cc_lsa_gate_a.py` | aligned/controls 的最终 Gate A 审计 |
| `scripts/run_cc_lsa_gate_a.sh` | 分阶段 fail-fast 运行封装 |

必须遵守以下终止条件：

- teacher 合同不是 `PASS`：停止，不运行 calibration、MSLS feature cache 或 Gate A；
- Gate A 不是 `PASS`：路线结论为 `FAIL`，停止，不实现或训练 pair classifier；
- Gate A `PASS`：也只表示 MSLS-val 开发集上的 pair-level 信息充分，先审查报告并冻结
  Gate B 协议，不能直接宣称最终语义 VPR 成功；
- `teacher` 和 `audit` 都可能在科学判据失败时正常写完文件，因此不能只看 shell
  退出码，必须读取各自 JSON 中的 `verdict`。

## 2. 同步代码

### 2.1 GitHub 可连接时

本机 PowerShell 在项目根目录提交时只添加本路线的文件，避免把本地大缓存或训练日志
误提交：

```powershell
cd D:\Desktop\MyVPR

git status --short
git add -- `
  config/cc_lsa_gate_a.yaml `
  config/cc_lsa_teacher.yaml `
  scripts/audit_cc_lsa_gate_a.py `
  scripts/cache_cc_lsa_msls_features.py `
  scripts/cache_gsv_cc_lsa_targets.py `
  scripts/calibrate_cc_lsa_gate_a.py `
  scripts/extract_ag_slrd_msls_descriptors.py `
  scripts/run_cc_lsa_gate_a.sh `
  scripts/train_cc_lsa_teacher.py `
  src/cc_lsa_config.py `
  src/cc_lsa_features.py `
  src/cc_lsa_gate_a.py `
  src/dataloaders/train/cc_lsa.py `
  src/models/cc_lsa.py `
  tests/test_cc_lsa_gate_a.py `
  tests/test_cc_lsa_audit.py `
  tests/test_cc_lsa_configs.py `
  tests/test_cc_lsa_features.py `
  tests/test_cc_lsa_teacher.py `
  tests/test_cc_lsa_targets.py `
  doc/AG_SLRD_BOQ_EXPERIMENT.md `
  doc/CC_LSA_PAIR_VPR_EXPERIMENT.md `
  doc/CC_LSA_GATE_A_RUNBOOK.md `
  doc/SEMANTIC_VPR_EXPERIMENT_SUMMARY.md `
  doc/ag_slrd_phase0_audit/per_query.csv `
  doc/ag_slrd_phase0_audit/summary.csv `
  doc/ag_slrd_phase0_audit/summary.json `
  doc/ag_slrd_phase0_audit/verdict.txt `
  doc/ag_slrd_runs/README.md `
  doc/ag_slrd_runs/phase0_audit.txt `
  doc/ag_slrd_runs/teacher_epoch_summary.csv

git diff --cached --check
git diff --cached --stat
git commit -m "Implement CC-LSA Pair-VPR Gate A"
git push origin main
```

上面的归档清单保留 AG-SLRD 的结构化审计、逐 query 结果和 epoch 汇总，但不提交
`teacher_*10ep.txt`、`teacher_aligned_failed_amp_step6371.txt` 等重复 tqdm 原始日志；
这些日志约 4.4 MB，结论已由 compact CSV/JSON 与总结文档覆盖。

如果本机还有本次实现新增的 `tests/test_cc_lsa_*.py`，先用 `git status --short`
确认后将它们显式加入；不要使用 `git add .`。

训练机执行：

```bash
cd ~/workspace/OpenVPRLab
git status --short --branch
git pull --ff-only origin main
git rev-parse --short HEAD
```

如果 `git pull` 报“未跟踪文件将被覆盖”，不要运行 `git clean`，改用下一节的
staging + backup 同步方法。

### 2.2 GitHub 不可连接时：PowerShell + SCP

本机 PowerShell 创建只含本路线文件的压缩包：

```powershell
cd D:\Desktop\MyVPR

$SyncArchive = Join-Path $env:TEMP "cc_lsa_gate_a_code.tgz"
tar -czf $SyncArchive `
  config/cc_lsa_gate_a.yaml `
  config/cc_lsa_teacher.yaml `
  scripts/audit_cc_lsa_gate_a.py `
  scripts/cache_cc_lsa_msls_features.py `
  scripts/cache_gsv_cc_lsa_targets.py `
  scripts/calibrate_cc_lsa_gate_a.py `
  scripts/extract_ag_slrd_msls_descriptors.py `
  scripts/run_cc_lsa_gate_a.sh `
  scripts/train_cc_lsa_teacher.py `
  src/cc_lsa_config.py `
  src/cc_lsa_features.py `
  src/cc_lsa_gate_a.py `
  src/dataloaders/train/cc_lsa.py `
  src/models/cc_lsa.py `
  tests/test_cc_lsa_gate_a.py `
  tests/test_cc_lsa_audit.py `
  tests/test_cc_lsa_configs.py `
  tests/test_cc_lsa_features.py `
  tests/test_cc_lsa_teacher.py `
  tests/test_cc_lsa_targets.py `
  doc/AG_SLRD_BOQ_EXPERIMENT.md `
  doc/CC_LSA_PAIR_VPR_EXPERIMENT.md `
  doc/CC_LSA_GATE_A_RUNBOOK.md `
  doc/SEMANTIC_VPR_EXPERIMENT_SUMMARY.md `
  doc/ag_slrd_phase0_audit/per_query.csv `
  doc/ag_slrd_phase0_audit/summary.csv `
  doc/ag_slrd_phase0_audit/summary.json `
  doc/ag_slrd_phase0_audit/verdict.txt `
  doc/ag_slrd_runs/README.md `
  doc/ag_slrd_runs/phase0_audit.txt `
  doc/ag_slrd_runs/teacher_epoch_summary.csv

if ($LASTEXITCODE -ne 0) { throw "tar failed" }
tar -tzf $SyncArchive
scp $SyncArchive wt@10.16.48.202:/tmp/cc_lsa_gate_a_code.tgz
if ($LASTEXITCODE -ne 0) { throw "scp failed" }
```

训练机先解压到 staging，备份所有将被覆盖的已有文件，再复制。该流程不会删除训练结果
或未跟踪文件：

```bash
cd ~/workspace/OpenVPRLab

export CC_LSA_SYNC_ARCHIVE=/tmp/cc_lsa_gate_a_code.tgz
export CC_LSA_SYNC_STAMP="$(date +%Y%m%d_%H%M%S)"
export CC_LSA_SYNC_STAGE="/tmp/cc_lsa_gate_a_stage_${CC_LSA_SYNC_STAMP}"
export CC_LSA_SYNC_BACKUP="/tmp/cc_lsa_gate_a_backup_${CC_LSA_SYNC_STAMP}"

test -f "$CC_LSA_SYNC_ARCHIVE"
tar -tzf "$CC_LSA_SYNC_ARCHIVE"

if tar -tzf "$CC_LSA_SYNC_ARCHIVE" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
  echo "Unsafe path found in archive"
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$CC_LSA_SYNC_STAGE" "$CC_LSA_SYNC_BACKUP"
tar -xzf "$CC_LSA_SYNC_ARCHIVE" -C "$CC_LSA_SYNC_STAGE"

while IFS= read -r CC_LSA_SYNC_FILE; do
  case "$CC_LSA_SYNC_FILE" in
    */) continue ;;
  esac
  if [ -f "$CC_LSA_SYNC_FILE" ]; then
    mkdir -p "$CC_LSA_SYNC_BACKUP/$(dirname "$CC_LSA_SYNC_FILE")"
    cp -a -- "$CC_LSA_SYNC_FILE" "$CC_LSA_SYNC_BACKUP/$CC_LSA_SYNC_FILE"
  fi
done < <(tar -tzf "$CC_LSA_SYNC_ARCHIVE")

cp -a "$CC_LSA_SYNC_STAGE"/. "$PWD"/
chmod +x scripts/run_cc_lsa_gate_a.sh

echo "Backup: $CC_LSA_SYNC_BACKUP"
git status --short
```

如果复制后发现代码不对，可将已备份的旧文件恢复回来：

```bash
cd ~/workspace/OpenVPRLab
cp -a "$CC_LSA_SYNC_BACKUP"/. "$PWD"/
```

恢复命令只恢复同步前已经存在的文件，不会删除这次新引入的文件。

## 3. 训练机预检

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR

export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'

test -f "$RU_CKPT"
python -c 'import open_clip, numpy, torch, yaml; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("open_clip imported")'
nvidia-smi -i 1
df -h . /tmp
```

建议项目所在文件系统至少保留 20 GiB，可供 `/tmp` 使用的空间至少保留 3 GiB。
MSLS local feature cache 是主要占用项；不要把 `.cache/cc_lsa` 提交到 Git。

运行新增测试：

```bash
python -m pytest -q tests/test_cc_lsa_*.py
```

测试不全通过时停止，不生成昂贵缓存。

## 4. 推荐：使用 fail-fast 封装脚本

脚本语法为：

```bash
bash scripts/run_cc_lsa_gate_a.sh <stage>
```

`stage` 可取：

- `targets`：生成或验证 GSV crop-CLS target cache；
- `smoke`：只做一个有限梯度更新，不写 checkpoint；
- `teacher`：固定 seed 42 训练 5 epochs，并检查 teacher 合同；
- `calibrate`：只用 GSV holdout 校准 LSA/raw-CLIP 的 Q95；
- `features`：固定 RU top-100，并缓存 MSLS 所需局部特征；
- `audit`：运行 aligned 与全部 placebo，强制校验数组 SHA256；
- `all`：依次执行 targets、smoke、teacher、calibrate、features、audit，教师合同失败立即停止。

前三个阶段不需要 `RU_CKPT`；从 `calibrate` 开始必须先导出它。完整运行：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR

export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
test -f "$RU_CKPT"

set -o pipefail
bash scripts/run_cc_lsa_gate_a.sh all
```

若要逐阶段观察，使用：

```bash
bash scripts/run_cc_lsa_gate_a.sh targets
bash scripts/run_cc_lsa_gate_a.sh smoke
bash scripts/run_cc_lsa_gate_a.sh teacher
bash scripts/run_cc_lsa_gate_a.sh calibrate
bash scripts/run_cc_lsa_gate_a.sh features
bash scripts/run_cc_lsa_gate_a.sh audit
```

缓存签名一致且已完整时，`targets`/`features` 会复用；不完整且签名一致时会续跑。
`teacher`、`calibrate` 和 `audit` 是一次性冻结产物，已有输出不应覆盖或换参数重跑。

## 5. 手动逐阶段命令

只有在需要定位封装脚本问题时才使用本节。以下命令与固定配置对应。

### 5.1 GSV crop-CLS targets

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
mkdir -p doc/cc_lsa_runs
export CC_LSA_RUN_TAG="$(date +%Y%m%d_%H%M%S)"
set -o pipefail

python -u scripts/cache_gsv_cc_lsa_targets.py \
  --dataset-root datasets/gsv_cities \
  --output .cache/cc_lsa/gsv_crop_cls_v1 \
  --device cuda:1 \
  --batch-size 16 \
  --num-workers 8 \
  --teacher-chunk-size 32 \
  --model-name ViT-B-16 \
  --pretrained openai \
  --hf-mirror https://hf-mirror.com \
  --image-size 280 280 \
  --seed 42 \
  --minimum-views 4 \
  --holdout-modulus 10 \
  --flush-every 50 \
  2>&1 | tee "doc/cc_lsa_runs/${CC_LSA_RUN_TAG}_targets.txt"
```

target cache 可按 manifest 的 cursor 续跑；完整 cache 会直接报告已存在。

### 5.2 teacher smoke test

```bash
python -u scripts/train_cc_lsa_teacher.py \
  --config config/cc_lsa_teacher.yaml \
  --smoke-test \
  2>&1 | tee "doc/cc_lsa_runs/${CC_LSA_RUN_TAG}_teacher_smoke.txt"
```

预期末行包含 `SMOKE TEST PASS`，且不写 checkpoint。

### 5.3 teacher 固定 5-epoch 训练与合同

正式训练前应确认输出目录不存在：

```bash
test ! -e logs/cc_lsa/teacher_seed42 || {
  echo "Teacher output already exists; inspect it instead of overwriting"
  return 1 2>/dev/null || exit 1
}

python -u scripts/train_cc_lsa_teacher.py \
  --config config/cc_lsa_teacher.yaml \
  2>&1 | tee "doc/cc_lsa_runs/${CC_LSA_RUN_TAG}_teacher_5ep.txt"
```

训练结束后必须显式检查合同；不能只看命令是否报错：

```bash
python -c 'import json; from pathlib import Path; p=Path("logs/cc_lsa/teacher_seed42/lsa_teacher_contract.json"); d=json.loads(p.read_text()); print(json.dumps(d, indent=2)); assert d["verdict"] == "PASS", "LSA teacher contract failed; stop Gate A"'
test -f logs/cc_lsa/teacher_seed42/final.pt
```

### 5.4 RU descriptors

旧的 `.cache/ag_slrd_descriptors/ru.npy` sidecar 没有本路线强制要求的 MSLS index
哈希，不能复用。必须用更新后的 `extract_ag_slrd_msls_descriptors.py` 新生成一份；
其 sidecar 同时绑定 checkpoint SHA256、数据库/查询计数、图像顺序和 MSLS index。

生成一份与当前 `$RU_CKPT` 明确绑定的新 descriptor cache：

```bash
export RU_DESCRIPTORS=.cache/cc_lsa/ru_msls.npy

if [ ! -f "$RU_DESCRIPTORS" ] || [ ! -f "${RU_DESCRIPTORS}.json" ]; then
  test ! -e "$RU_DESCRIPTORS"
  test ! -e "${RU_DESCRIPTORS}.json"
  python -u scripts/extract_ag_slrd_msls_descriptors.py ru \
    --checkpoint "$RU_CKPT" \
    --msls-path datasets/msls-val \
    --image-size 280 280 \
    --output "$RU_DESCRIPTORS" \
    --device cuda:1 \
    --batch-size 32 \
    --num-workers 8 \
    2>&1 | tee "doc/cc_lsa_runs/${CC_LSA_RUN_TAG}_ru_descriptors.txt"
fi
```

不要手工复制旧 `.npy` 而漏掉相邻的 `.npy.json` provenance。

### 5.5 仅用 GSV 校准 Q95

```bash
test ! -e .cache/cc_lsa/gsv_calibration_v1
test ! -e /tmp/cc_lsa_calibration_v1

python -u scripts/calibrate_cc_lsa_gate_a.py \
  --config config/cc_lsa_gate_a.yaml \
  --dataset-root datasets/gsv_cities \
  --ru-checkpoint "$RU_CKPT" \
  --lsa-checkpoint logs/cc_lsa/teacher_seed42/final.pt \
  --output .cache/cc_lsa/gsv_calibration_v1 \
  --scratch-dir /tmp/cc_lsa_calibration_v1 \
  --device cuda:1 \
  --batch-size 16 \
  --num-workers 8 \
  2>&1 | tee "doc/cc_lsa_runs/${CC_LSA_RUN_TAG}_calibration.txt"
```

该阶段只读取固定 GSV holdout，不读取 MSLS label 或 retrieval score 来调阈值。
保留 calibration 输出和 scratch，至少到 Gate A 完成并归档之后。

### 5.6 固定 top-100 并缓存 MSLS 局部特征

```bash
python -u scripts/cache_cc_lsa_msls_features.py \
  --config config/cc_lsa_gate_a.yaml \
  --ru-checkpoint "$RU_CKPT" \
  --ru-descriptors "$RU_DESCRIPTORS" \
  --lsa-checkpoint logs/cc_lsa/teacher_seed42/final.pt \
  --msls-path datasets/msls-val \
  --output .cache/cc_lsa/msls_top100_v1 \
  --device cuda:1 \
  --batch-size 16 \
  --num-workers 8 \
  --flush-every 20 \
  2>&1 | tee "doc/cc_lsa_runs/${CC_LSA_RUN_TAG}_msls_features.txt"
```

该脚本必须打印 RU 精确复现 `675/740`；否则立即终止。feature cache 支持在签名
一致时续跑，不要为了重试删除完整 cache。

### 5.7 Gate A 审计

```bash
test ! -e doc/cc_lsa_gate_a || {
  echo "Gate-A output already exists; inspect it instead of overwriting"
  return 1 2>/dev/null || exit 1
}

python -u scripts/audit_cc_lsa_gate_a.py \
  --config config/cc_lsa_gate_a.yaml \
  --feature-cache .cache/cc_lsa/msls_top100_v1 \
  --calibration .cache/cc_lsa/gsv_calibration_v1 \
  --msls-path datasets/msls-val \
  --output doc/cc_lsa_gate_a \
  --device cuda:1 \
  2>&1 | tee "doc/cc_lsa_runs/${CC_LSA_RUN_TAG}_gate_a.txt"

cat doc/cc_lsa_gate_a/verdict.txt
python -c 'import json; from pathlib import Path; d=json.loads(Path("doc/cc_lsa_gate_a/summary.json").read_text()); print(json.dumps(d["verdict"], indent=2)); assert d["verdict"]["verdict"] == "PASS", "Gate A failed; pair classifier must remain NOT IMPLEMENTED"'
```

审计固定运行以下组别：`aligned_lsa`、`dino_full`、`raw_clip`、
`token_permutation`、`wrong_place`，以及 100 个 count/weight-matched seeds。
审计脚本强制验证 feature/calibration 数组 SHA256，不提供跳过完整性检查的正式模式。

## 6. 结果审查与下载

重点检查：

- `logs/cc_lsa/teacher_seed42/lsa_teacher_contract.json`；
- `.cache/cc_lsa/gsv_calibration_v1/calibration.json`；
- `doc/cc_lsa_gate_a/verdict.txt`；
- `doc/cc_lsa_gate_a/summary.json`；
- `doc/cc_lsa_gate_a/summary.csv`；
- `doc/cc_lsa_gate_a/per_query.csv`；
- `doc/cc_lsa_gate_a/random_seed_summary.csv`。

在本机 PowerShell 下载小型报告，不下载 `.cache` 中的大型局部特征：

```powershell
cd D:\Desktop\MyVPR
scp -r wt@10.16.48.202:~/workspace/OpenVPRLab/doc/cc_lsa_gate_a doc/
scp -r wt@10.16.48.202:~/workspace/OpenVPRLab/doc/cc_lsa_runs doc/
scp wt@10.16.48.202:~/workspace/OpenVPRLab/logs/cc_lsa/teacher_seed42/lsa_teacher_contract.json doc/cc_lsa_gate_a/
scp wt@10.16.48.202:~/workspace/OpenVPRLab/logs/cc_lsa/teacher_seed42/run.json doc/cc_lsa_gate_a/teacher_run.json
```

最终解释只允许二选一：

```text
CC-LSA Pair-VPR Gate A: PASS
Pair classifier: NOT YET IMPLEMENTED
```

或：

```text
CC-LSA Pair-VPR Gate A: FAIL
Pair classifier: NOT IMPLEMENTED
```
