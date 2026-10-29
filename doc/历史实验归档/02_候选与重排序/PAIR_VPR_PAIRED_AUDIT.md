# 官方 Pair-VPR：逐查询审计

本轮已知结果（用户控制台）：global 639/740=86.35%，refined 693/740=93.65%；净增加54，实际纠错/破坏尚未知。RU 675/740，比refined少18；不是同模型公平消融。top-100覆盖率97.84%对应724/740，若本轮严格复现，则剩余47个错误中应有31个候选内错误和16个候选外错误，需新保存的逐查询排名验证。

原始官方输出没有保存逐查询预测，因此必须重跑一次（此前低内存模式约一小时）。本脚本仅拦截最终结果报告函数来保存预测；不改变模型、输入几何、候选或评分。沿用已成功的本地DINO源码初始化与严格完整checkpoint加载。保留官方打印，但自己的摘要只报告K<=100。

脚本：`scripts/audit_official_pairvpr.py`。按用户要求未在本机运行测试；训练机先preflight。原官方仓库不改动，使用现有VPR环境，不安装新依赖。

## 同步

本机PowerShell：

```powershell
cd D:\Desktop\MyVPR
git add scripts/audit_official_pairvpr.py doc/PAIR_VPR_PAIRED_AUDIT.md doc/PAIR_VPR_OFFICIAL_REPRODUCTION.md
git commit -m "Add prediction-only audit for official Pair-VPR"
git push origin main
```

训练机：

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
```

如果Git网络失败，本机制作bundle（不发压缩包）：

```powershell
git bundle create D:\Desktop\pairvpr_audit.bundle main
scp D:\Desktop\pairvpr_audit.bundle wt@10.16.48.202:/tmp/pairvpr_audit.bundle
```

训练机：

```bash
cd ~/workspace/OpenVPRLab
git bundle verify /tmp/pairvpr_audit.bundle
git fetch /tmp/pairvpr_audit.bundle refs/heads/main:refs/remotes/scp/pairvpr_audit
git merge --ff-only refs/remotes/scp/pairvpr_audit
```

只针对上述文件提交。当前其他归档文档变更和用户原有删除未自动加入提交。若merge有未跟踪文件冲突，先处理明确路径，不使用git clean。

## 运行

必须从OpenVPRLab运行，不是Pair-VPR-official。每条命令单独复制。V1的per_query保存了RU候选；脚本读取它的原RU排名，不读取学习残差，所以不会将V1分类头当RU。

```bash
conda activate VPR
cd ~/workspace/OpenVPRLab
export CUDA_VISIBLE_DEVICES=1
```

先校验（不提取特征、不创建输出目录）：

```bash
python -u scripts/audit_official_pairvpr.py --official-repo /home/wt/workspace/Pair-VPR-official --msls-path datasets/msls-val --ru-audit doc/visual_pair_msls_seed42_v1 --output doc/pairvpr_official_paired_audit_v1 --preflight-only
```

看到PREFLIGHT PASS后正式运行：

```bash
set -o pipefail
mkdir -p doc/pairvpr_official_runs
```

```bash
python -u scripts/audit_official_pairvpr.py --official-repo /home/wt/workspace/Pair-VPR-official --msls-path datasets/msls-val --ru-audit doc/visual_pair_msls_seed42_v1 --output doc/pairvpr_official_paired_audit_v1 2>&1 | tee "doc/pairvpr_official_runs/paired_$(date +%Y%m%d_%H%M%S).txt"
```

输出目录必须不存在。不自动删除上次失败输出或重复运行。错误时先发日志，不关闭严格加载/索引校验。

## 输出与判读

- `provenance.json`：开始时记录的环境、官方Git SHA及diff、实际源码SHA、权重、配置、数据索引、RU证据。其complete=false表示开始记录，最终完成标记另存。
- `predictions.npz`：global/refined/RU三套候选索引；无巨大特征缓存，可离线分析。
- `per_query.jsonl`：各查询路径、三路top-1、正确性和top-100可达性。
- `summary.json`：global→refined、RU→refined的四格表，纠错/破坏/净收益，候选内外剩余错误。
- `completed.json`：官方main正常返回且已记录一份审计后才写入。

完成后发summary.json和completed.json即可。global→refined是真正同系统重排对照；RU→refined仅评估跨系统互补性。若结果未复现639/693，不立刻视为新收益，先核对provenance和数值误差。

下一步按证据决定：若错误主要不可达，才研究召回；若候选内错误仍多，再人工查看其中有足够重叠的查询，判断语义能否提供区分信息。不因“候选内存在31个错误”直接推断语义可修复，不开启新训练。
