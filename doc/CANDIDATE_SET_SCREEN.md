# 候选集合条件判别：首轮可行性试验

实现日期：2026-09-15。对应 NONSEMANTIC_VPR_RESEARCH_DIRECTIONS.md 第一条。该版本只验证冻结官方 Pair-VPR 表示是否支持候选集合建模，不代表完整新方法，更不预先声称论文创新。无 CLIP、语义分割或人工区域标注。

## 本轮内容

1. 严格加载已审计的官方 Pair-VPR ViT-B 权重、配置、Python 源文件；读取本地 DINO hub 源，不下载模型。
2. 从 GSV 原始 Dataframes 构建新计划。每地点至少三个不同 panoid，按年月/路径排序，最早一张为 query，最后两张为参考。只是不同全景观测，不保证跨季节或不同日期。
3. 以固定城市名称哈希选择两个各有至少1024有效地点的城市，分别用于 dev/test；训练使用其余城市中固定哈希选出的至多4096地点。训练1024查询、dev/test各512查询。dev/test数据库使用该城市所有符合规则地点的两张参考，不能缩成512地点。生成计划后不因结果改变城市或挑困难查询。
4. GSV地点ID作为本次诊断GT。两城市与本次适配训练隔离，但可能已被官方预训练及项目旧模型看过，不能称独立未见预训练数据。不同地点ID的近邻可能存在地理/视野歧义；本轮不改标签，不当作官方MSLS结果。
5. 用官方global检索固定top20，不注入GT；每候选缓存双向配对分类头输入各768维、双向分数和512维参考global描述子。保留所有查询，候选不可达仍计错；只有训练损失跳过无正或无负候选的查询。
6. 缓存dev后先检查任务区分度：至少10例正确候选可达但原pair排错，才提取训练缓存并训练。该固定门槛是预算准入，不是显著性门槛。若失败，保存报告正常停止，不降低门槛追求通过。

## 五个共享输入、预算与初始化的辅助头

| 模式 | 跨候选交互 | 视觉密度校正 | 重复一致性损失 |
|---|---|---|---|
| independent | 无，attention限制在自身 | 无 | 无 |
| set | 普通attention | 无 | 无 |
| density | 有 | 有 | 无 |
| consistency | 普通attention | 无 | 有 |
| competition | 有 | 有 | 有 |

所有模式均接收双向配对CLS和参考global向量。密度由参考global余弦相似度核计算，温度固定0.02；attention key bias减去log密度。这只是视觉冗余近似，不是推理时已知真实地点分组。普通set也可以利用同样的global信息。

所有模式结构参数形状相同、seed42、AdamW lr1e-4/weight_decay0.001、5轮、batch8、宽度128。independent限制交互后Q/K部分没有有效跨候选梯度，因此参数总量一致不等于有效容量完全一致。官方模型始终冻结，不是从头训练的弱pair替代品。

输出是原双向pair分数 + 4*tanh(增量)，增量头零初始化，初始精确保留原评分。训练使用多正样本列表损失；所有组都接受相同的复制5个候选增广，增广损失只在原始唯一候选上计算，避免复制次数改变正样本概率质量。consistency/competition额外约束复制前后原候选分布一致。评价包括候选乱序检查、重复global第一名5次的压力测试。

这不是完整的“局部竞争证据”模型：输入是配对CLS，并未提供全部patch对应。候选上下文、密度降权和一致性都不是凭命名即可成立的新贡献。若表现无增益，结论限于本次冻结表示与小头配置。成功后还需证明独立机制及跨基准收益。

## 选模与结果

每组按dev R@1选择5轮中最优checkpoint，同分取最早。test仅在选模结束后运行，所有组都报告；不按test筛掉差组。512查询是适配诊断测试，不作为最终论文泛化证据。首轮不重新提取MSLS特征；若开发任务非饱和且集合模块有额外收益，再另行锁定标准基准回归和独立测试协议。

输出目录 doc/candidate_set_report_v1 包含 admission_时间戳/decision.json；通过准入并跑完后还有 evaluation/，含summary、逐query分数、纠错退化ID、训练日志、所选epoch、乱序/重复候选诊断及文件哈希。所有行的query编号对应 plan 中各自 split 的 queries 顺序，不是MSLS查询编号。

## 资源与恢复

物理 GPU1，脚本设置 CUDA_VISIBLE_DEVICES=1，程序内部逻辑cuda:0。不要再传cuda:1。复用 /home/wt/workspace/Pair-VPR-official、其已审计权重和项目 doc/pairvpr_official_paired_audit_v1/provenance.json。

不保存全库稠密token。RAM中最多保留64张图的稠密特征，未命中时重新编码，节省磁盘但增加GPU时间。五组复用同一配对缓存，主要开销是双向decoder计算，不是小头训练。默认共2048查询×20候选×2方向；实际耗时取决于硬件和城市库大小，不承诺短时完成。配对证据约数百MB，加global缓存、检查点和报告，建议预留5GB磁盘；这是估计，不包括已有模型/数据。

cache阶段按global128图、pair1查询原子保存，并校验SHA；重复运行验证并复用已完成分片。中断在尚未写校验和的分片会重新生成。校验不通过的分片不会悄悄跳过。train --resume复用完整模式，当前未完成模式从seed42重新开始，并非精确逐批恢复。evaluate要求新输出目录，完整流程完成后不要重复evaluate覆盖结果。

## 命令

训练机：

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
mkdir -p logs
set -o pipefail
bash scripts/run_candidate_set_screen.sh 2>&1 | tee logs/candidate_set_screen_v1.log
```

本机PowerShell下载轻量报告：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_set_report_v1 C:/Users/zy/Desktop/MyVPR/doc/
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_set_plan_v1 C:/Users/zy/Desktop/MyVPR/doc/
```

本机没有torch，已执行CPU协议测试、语法/CLI检查。服务器preflight必须执行全部9项测试且没有skip，并检查真实官方权重加载、双向接口所需CLS形状和评分重建。GPU实际行为仍待服务器验证。
