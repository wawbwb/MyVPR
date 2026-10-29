# VPR 实验交接：截至 2026-09-15

## 1. 给接手 AI 的第一条说明

用户希望研究一个能够提升视觉地点识别（VPR）、有论文创新潜力的方向。
大量语义辅助实验未建立稳定收益后，用户已明确决定停止语义提升路线。
当前转向非语义的候选集合条件重排，但最新五组首筛也没有证明集合交互有收益。
请先读本文件，再读取引用的原始报告和实现，不要从历史文档中的旧“下一步”重新开始。

**当前停点：实验已完成、结果已下载并分析；尚未实施下一步 GSV 训练监督审计。**
最新建议是用服务器现有缓存做 CPU 审计，先确认训练是否有足够的可达排序错误，
不是立即增加 epoch、改权重或开启另一轮训练。该建议仍是待办，不是已经确认的失败原因。

本交接是已有实验的整理，不是新的文献检索或端到端复现。
全文路径除平台绝对路径外，均相对仓库根目录。

## 2. 用户目标、偏好与执行约定

- 最终目标是可靠的 VPR 提升和可验证的论文贡献，不是仅得到掩码、漂亮案例或压力测试改善。
- 早期目标是 CLIP 自动抑制外部动态类别。用户不希望持续手工标注，也明确要求不纠缠自车引擎盖。
  该语义目标后来已被用户主动放弃，不应擅自重新启动。
- 新增较多代码时，直接在本机项目对应目录实现、测试、Git 提交推送；用户在训练机拉取运行。
- 本机是 Windows 工作副本，不是训练机。不在本机训练，不为运行实验安装 PyTorch。
- 用户负责 SSH 登录、运行 GPU 作业与 scp 下载。此前并没有由 AI 持续管理的 SSH 会话。
- 给用户可复制的同步、运行和下载命令。不要要求手工拼接大量代码。
- 不因单次 +1 或优于 random/shuffled 就声称成功；不把具体配置失败升级为理论否定。
- 保留已有缓存与历史结果。不要通过改 contract/hash、覆盖目录或改 GT 绕过实验协议。

## 3. 机器、仓库、运行环境

| 项目 | 位置/状态 |
|---|---|
| Windows 项目 | `C:\Users\zy\Desktop\MyVPR` |
| 训练机 | `wt@10.16.48.202`，Linux 主机 msi-MS-7D28 |
| 训练机项目 | `/home/wt/workspace/OpenVPRLab` |
| Conda | `VPR`，服务器日志为 Python 3.11 |
| Git origin | `https://github.com/wawbwb/MyVPR.git` |
| 当前分支 | `main` |
| 最新已推送提交 | `91f0aa5`：GSV 训练 / Pitts 开发五组候选头 |
| 物理 GPU | 当前脚本固定 `CUDA_VISIBLE_DEVICES=1`，进程内使用 `cuda:0` |
| Pair-VPR 官方代码 | `/home/wt/workspace/Pair-VPR-official` |
| 本机代理 | HTTP 端口 7897，Git 推送需要代理 |

本机可用 Python：
`C:/Users/zy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`。
已用于 NumPy/JSON/PIL 审计，没有 PyTorch。PowerShell 输出中文时可先
`sys.stdout.reconfigure(encoding='utf-8')`。Git Bash 位于 `D:/Git/bin/bash.exe`，可检查 shell 语法。

服务器已确认的数据目录：`datasets/gsv_cities`、`datasets/msls-val`、
`datasets/pitts30k-val`、`datasets/segvlad_official`、`datasets/cache`。
MSLS 只确认有 cph/sf 验证城市，没有发现官方训练城市；不能默认 MSLS train 已存在。
Pitts 目录约 716MB，具有 `pitts30k_val_dbImages.npy`、`pitts30k_val_qImages.npy`、
`pitts30k_val_gt_25m.npy`，图像分布于编号目录和 `queries_real`。

工作区并不干净：存在用户原有的 `scripts/audit_dynamic_coverage.py`、
`tests/test_audit_dynamic_coverage.py` 修改，`doc/ag_slrd_runs` 三个文件删除，以及大量未跟踪下载结果。
不要 `git add .`、清理未跟踪文件或恢复这些修改。仅提交自己本轮需要的文件。
本交接文件和最新 ANALYSIS 文档本次仅保存本机，未推送；下载结果也不保证在远端 Git 中。

## 4. 先读哪些文件

1. `doc/candidate_set_pitts_report_v1/ANALYSIS.md`：最新实验结论。
2. `doc/CANDIDATE_SET_PITTS.md`：最新训练协议和运行说明。
3. `scripts/candidate_set_pitts.py`、`src/models/candidate_set.py`：真实实现。
4. `doc/pitts_pair_admission_report_v1/ANALYSIS.md`：Pitts 冻结基线与准入。
5. `doc/candidate_set_report_v1/ANALYSIS.md`：GSV Barcelona 开发任务为何停止。
6. `doc/NONSEMANTIC_VPR_RESEARCH_DIRECTIONS.md`：三条非语义研究方向及创新边界。
7. `doc/SEMANTIC_VPR_EXPERIMENT_SUMMARY.md`：截至 9 月 11 日的完整旧路线档案。

注意第 7 项顶部仍有“重新开放动态不变性”等历史决定。用户后来的停止语义决定优先。
9 月 12–15 日的实验需要结合下列补充报告，旧总归档不是当前完整状态。

## 5. 历史路线：已知事实与不能丢失的正证据

主要旧评测为 MSLS-val 740 查询；除特别说明，下表正确数分母均为 740。
不同骨干、训练预算、数据协议不可当作纯消融。RU 指后续使用的视觉
repeatability + uniqueness 基座，基于 DINOv2-BoQ。

| 路线 | 关键结果 | 解释 |
|---|---|---|
| 视觉参考基线 | MixVPR 87.84%；原 BoQ 91.22%；SALAD 91.08%；RU 675/740 | 不能混用为同预算匹配基线 |
| CLIP 全局 CLS 蒸馏 | 86.49–87.30%，低于 MixVPR | 未通过 |
| CLIP attention 蒸馏 | 语义组 86.35–86.89%，无语义 gate 87.57% | 未通过 |
| patch reliability | 最佳 86.49% | 未通过 |
| CLIP 难负/正样本选择 | 难负 aligned87.43%、random88.11%；正样本 aligned87.03%、random87.57% | random 优势不等于语义优势 |
| RU + CLIP affinity 平滑 | full89.86%，RU/shuffled91.22% | 语义传播未带来收益，保留 RU |
| DeepLab 动态负先验 | aligned674，RU675，random674，shuffled671 | 具体 β=0.5 晚期 attention 干预无净提升 |
| 类别 query bias | aligned、architecture-only 最佳均675 | 无收益 |
| Crop-CLS FiLM | aligned673，architecture669，wrong-region670，wrong-place674 | 未建立正确语义对应关系的增益 |
| Residual CLIP Fusion | 多组均675；推理残差倍数0.5–4未提高R@1 | 未继续 VLAQ Phase B |
| 语义遮挡 RSCD | aligned673、no-mask673、uniform669 | 只比更破坏的控制好 |
| 语义布局 teacher | MSLS344，仅补 RU2 错，shuffled补4错 | 可学习不等于有效互补 |
| dense teacher/手工 pair | 探索重排377，纠错2退化300 | 首筛失败 |
| 共享骨干分割辅助 | V2 aligned671，VPR-only670，RU675 | 无整体收益 |
| 自研轻量视觉 pair V1/V2 | V2 校准选择 α=0，回退 RU675 | 不能声称启用新头且无损 |
| 官方 Pair-VPR | 自身 global639 → refined693；纠错63、退化9 | 明确的视觉方法级正证据 |
| Pair/RU 候选并集 | oracle724→725，仅新增1条可达 | 不等于实际R@1提升 |
| RU + SAM Region-VLAD | SAM382，grid488，RU675 | 适配失败，不否定官方方法 |
| 官方 SegVLAD / 17Places | AnyLoc385→SegVLAD387，分母406 | 正证据，但不是MSLS、不是纯mask消融 |
| SegVLAD 投票封顶 | 事后389/406 | 探索性结果，不是独立验证 |
| Region-Pair-Lite | SAM671，grid673，shifted674，RU675 | 当前区域均值/粗几何配置失败 |

Pair-VPR 相对 RU 的 693−675=18 属跨系统比较；自身 global→pair 的 +54 才是其配对模块前后比较。
MSLS Pair 最终47错误中31条在top100可达、16不可达。31是 oracle空间，不能当作可实现增益。
高容量官方 SegVLAD 全量 MSLS 迁移因资源/用户决定取消，没有完成结果，不应列为失败。
原 SemVPR/SegVPR 完整复现、场景文字身份等仅讨论的路线也不能列为已失败。

## 6. 语义动态路线的后续复查（9 月 11 日以后）

旧动态负先验目录只有 summary.csv、paired_comparisons.csv、run.json；
run.json 明确 `descriptors_retained:false`，没有该轮逐查询 outcomes。
其他实验目录里的 query_outcomes/descriptors 不能冒充这轮结果。
旧掩码仍在服务器：
`.cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz`。

CPU mask 审计的 MSLS740 查询覆盖分布：零64，(0,5%)345，[5,15%)235，
[15,30%)89，[30,100%]7。条件子集为 night55、winter2summer586、summer2winter402，
它们与标准查询并非四个独立数据集，也不能简单合并分母。
人工标注过少量机制案例，`doc/annotations.json`；用户不愿持续标注，也不希望继续纠缠引擎盖。

| 后续实验 | 结果与意义 | 详细报告目录（均在 doc/） |
|---|---|---|
| CLIP自动动态首筛 | late aligned674；input aligned674；baseline675；无净收益 | clip_dynamic_screen_v1 |
| 类别竞争/定位诊断 | 部分外部车辆定位受 ego 竞争影响；修正定位不自动意味着检索改善 | clip_dynamic_localization_v2、clip_competition_audit_server_v1 |
| 聚焦外部动态输入抑制 | baseline675，aligned674（0修复1退化），shuffle674（1修复2退化） | clip_external_screen_v1 |
| 稠密CLIP + token删除训练 | frozen675，none668，aligned671，shuffled674 | clip_token_drop_report_v1 |
| 训练×推理九格交叉 | 668–674，无一超过675 | clip_token_cross_report_v1 |
| Pair最后层CLS动态诊断 | 31可达错误和693成功；固定正负顺序无翻转；未形成新R@1 | pairvpr_cls_diagnostic_v1 |
| 动态不变性训练完整MSLS | 冻结/普通/随机/语义四组均675，正确集合相同 | dynamic_invariance_msls_report_v1 |
| 合成挑战V2 | 对象挑战768次：未抑制676、CLIP677、准确位置675；CLIP救7伤6 | dynamic_challenge_report_v2 |
| 静态证据独立分支 | RU675，全图673，随机558，静态618；静态修复5退化62 | static_evidence_report_v1 |

上述各目录中的 ANALYSIS.md 记录核验范围与局限；另有 dynamic_mechanism_probe_v1、
clip_failure_cases_v1、dynamic_invariance_report_v1、dynamic_challenge_report_v1 等中间诊断。
合成挑战768次来自128查询重复使用，不是768个独立查询；准确位置净−1说明定位正确仍不保证输入衰减有益。
静态优于随机60条不等于优于完整RU；这轮是硬保留静态token分支，不是所有软增强方式的检验。
没有证明语义普遍无效，但用户已决定不再投入此方向。

## 7. 转向非语义：候选集合条件重排

此前列出三条研究方向：
1. 候选集合条件下的判别证据与近重复鲁棒性（已选择并首筛）。
2. 地理地点关系与视觉可匹配关系的分离监督（未实现）。
3. 将有效 Pair-VPR 排序蒸馏到单图描述子，研究可迁移/不可表达的关系（未实现）。

相关工作线索及原始链接已在 NONSEMANTIC_VPR_RESEARCH_DIRECTIONS.md：
Pair-VPR、SuperGlobal、VLAD-BuFF、CliqueMining、SAGE、VOP、DistilVPR 等。
它是前次选题检索记录，接手 AI 如要提出新论文创新仍需重新核查原文与新颖性。
仅增加 self-attention、密度权重、KL 或改 teacher 并不足以构成论文贡献。

当前实现只是“冻结双向 pair CLS + 小集合头”的可行性筛选，
不是完整的局部竞争证据网络，也没有微调 Pair-VPR 主干。

### 7.1 首个 GSV 开发任务未通过准入

原固定计划：GSV train1024查询/8192参考/4096地点；
dev Barcelona512查询/2848参考；test WashingtonDC512查询/2810参考。
城市按预设哈希规则选择，本次小头训练城市隔离；不保证官方基础模型未见过这些城市。
GT 使用地点ID，不是官方地理半径评测。

Barcelona：global468/512 → Pair499/512；Top20可达503，只有4条可达错误，9条不可达。
少于预设10条可达错误门槛，停止了当时的训练和test。
这是开发任务近饱和，不是集合模型实验失败。没有事后降低门槛或把WashingtonDC改成dev。
GSV plan继续用作后来的训练来源。

### 7.2 Pitts 新开发准入通过

读取现有官方NPY索引，完整数据库10000图；从7608查询中预先按路径哈希选择1024条。
选择key为 `pitts-admission-v1:` + query path。多正样本GT使用原数据库索引。
这不是全量Pitts30k-val成绩。

| 指标 | 结果 |
|---|---:|
| global正确 | 935/1024（91.3086%） |
| 双向Pair Top20正确 | 985/1024（96.1914%） |
| Pair相对global | 修复54、退化4，净+50 |
| Top20可达 | 1015/1024 |
| Pair剩余错误 | 39：30可达、9不可达 |

30条通过10条准入门槛，但未证明错误来自候选公共纹理或密度。
头每候选残差限于±4，正负差最多改变不到8；30条错误中22条原gap<8、8条gap≥8。
22也只是当前残差参数化的理论可修复上限，不能据此事后扩大残差。

Pitts已在历史实验中使用，本轮又用于选模。其余6584条也不能自动称为独立test，
因为可能共享地点、全景视角和历史选模信息。MSLS同样已被反复查看。

## 8. 最新五组实现和训练协议

主入口 `scripts/candidate_set_pitts.py`：check / preflight / train / report。
启动脚本 `scripts/run_candidate_set_pitts.sh`。
底层复用 `scripts/candidate_set_screen.py`、`scripts/pitts_pair_admission.py`、
`src/candidate_set_utils.py`、`src/models/candidate_set.py`。

冻结模型：官方 Pair-VPR，DINOv2 reg B/14；图像322×322，
全局描述子512维、dense token529×768。
每查询全局Top20候选；每候选缓存两个方向的最终pair CLS，拼为1536维，
base为官方正反向pair logits之和。当前头输入CLS和候选参考的512维全局描述子，
不是仅拟合标量分数。CLS与归一化参考描述子拼接为2048维，再编码至128维。

官方权重SHA256：
`18e7b95ba57d4d578fbf0a06a1846fc2dfb976c74da56bfd7d6a68afbcea8423`。
配置 `pairvpr/configs/pairvpr_speed_local.yaml`。
合同还保存官方源码、配置、DINO源码等身份；GSV/Pitts冻结身份必须一致。

| mode | 跨候选attention | 视觉密度校正 | 重复一致性损失 |
|---|---|---|---|
| independent | 只允许对角，无跨候选 | 无 | 无 |
| set | 有 | 无 | 无 |
| density | 有 | 有 | 无 |
| consistency | 有 | 无 | 有 |
| competition | 有 | 有 | 有 |

各组参数形状相同、初始化种子相同，但independent的有效跨候选梯度不同。
密度为参考全局余弦的指数核求和，temperature=0.02，attention key侧减log density；
推理不读取GPS、GT或真实地点分组。
输出 `base + 4*tanh(head)`，head零初始化，初始与冻结分数一致。

只用GSV训练，Pitts不进入梯度。固定seed42、5epoch、batch8、AdamW lr1e-4、
weight_decay0.001、梯度裁剪1。多正样本list loss为
logsumexp(all candidates)−logsumexp(positive candidates)，只训练同时有正负的查询。
所有组均重复随机5候选增强，损失为原始与增强后原K唯一候选list loss各半；
consistency/competition额外加权重1的对称KL（归一化仅覆盖原K唯一候选）。
五组同预算；每组选择Pitts R@1最高epoch，平局选最早；epoch0不参与选模，冻结模型单列。
全1024条Pitts查询均参与报告，不剔除不可达或原正确查询。

## 9. 最新结果：已完成、下载和核验

| 方法 | 正确数 /1024 | R@1 | 修复 | 退化 | 净增 |
|---|---:|---:|---:|---:|---:|
| frozen | 985 | 96.1914% | 0 | 0 | 0 |
| independent | 986 | 96.2891% | 1 | 0 | +1 |
| set | 985 | 96.1914% | 1 | 1 | 0 |
| density | 985 | 96.1914% | 1 | 1 | 0 |
| consistency | 985 | 96.1914% | 1 | 1 | 0 |
| competition | 985 | 96.1914% | 1 | 1 | 0 |

所有方法R@5/10/20与冻结相同。所有组选择epoch1；independent五轮为
986/979/983/982/983，set为985/979/981/981/982。
GSV1024条中1008条同时有正负可进入损失，16条不可达；每轮126批。
set平均训练loss约0.0131→0.0001，但开发表现没有同步改善。
这与拟合未迁移一致，不能直接断言是域差异、过拟合或CLS信息不足中的哪一个。

唯一修复为样本行92（原Pitts query5817，queries_real/008/008186_pitch1_yaw10.jpg）：
冻结最佳正−最佳负margin=-0.005579，independent后+0.015383。
集合组新增退化为样本行543（原query3021，queries_real/006/006006_pitch2_yaw10.jpg）：
冻结margin=+0.019409，set后−0.002831。都是近分数边界变化。
不要把报告的样本行当成原始query索引或MSLS编号。

压力测试：
- 候选倒序最大误差≤1.91e-6，等变性检查通过。
- 重复全局第一候选5次：set唯一候选Top1改变3条；density/competition改变0条。
- 最大重复分数变化set约0.03248，density/competition约9.4e-5。
- density与competition训练一致性损失均值约3.35e-14和3.27e-14，几乎为零；
  所选模型最大分数差约4.58e-5，Top1没有差异。

因此：密度校正的抗重复机制在测试中起效，但没有转化为正常VPR提升；
加入密度校正后，当前精确复制增强的一致性目标几乎冗余。
单seed、开发选模后+1不构成可靠收益，也不支持集合创新。

验证范围：下载报告所有已列文件SHA、代码指纹、scores侧车SHA通过；
与原Pitts准入逐项对齐frozen分数、GT标签、候选ID和原queryID；
summary、消融配对比较、selection指标重新计算全部一致。
没有在本机重新做GPU推理，也没有下载全量GSV/Pitts特征缓存。
此前本机代码测试为9项CPU通过、6项torch跳过；服务器脚本预检不允许跳过。
现有报告不是完整服务器preflight日志，不能把其测试过程说成本机已经复现。

## 10. 文件与缓存清单：继续工作所需

服务器当前重要路径：

| 路径 | 内容 |
|---|---|
| `doc/candidate_set_plan_v1` | 原GSV划分及completed/contract |
| `.cache/candidate_set_v1/dev` | Barcelona冻结global/pair缓存 |
| `.cache/candidate_set_v1/train` | 本次已用于训练的GSV冻结缓存 |
| `.cache/pitts_pair_admission_v1` | Pitts完整冻结global/pair缓存及report |
| `doc/candidate_set_pitts_train_v1` | 五组checkpoint、训练日志、selection、gate、contract、completed |
| `doc/candidate_set_pitts_report_v1` | 已下载的轻量开发报告 |

旧GSVtest缓存未由本轮脚本提取，是否另行存在需现场核验。
本机主要有plan与轻量报告，不能默认本机存在服务器缓存、数据图像或权重。

每查询 `pairs/000000.npz`，字段：
- evidence：(20,1536)，base：(20,)，db_vectors：(20,512)；
- candidates：(20,)原数据库索引，global_scores：(20,)；labels：(20,) bool。
- `.sha.json`侧车校验；global分片含vectors和图像hash。

GSV缓存contract：code、official、plan_sha256、split、ndb、queries、topk、scope。
Pitts缓存contract：code、official、topk、index（database、queries、query_indices、gt、原NPY哈希）。
两个缓存schema不同，不能重命名目录冒充或手动改hash。
complete/verify逻辑在上述脚本。代码指纹按LF规范化，不受Windows CRLF转换影响。
修改旧生成脚本或模型会使已有缓存指纹不匹配；CPU审计优先新增独立脚本，保留旧依赖。

最新轻量报告包含：summary.json、scores.npz、stress.json、ablation_comparisons.json、
query_mapping.json、training_logs.json、training_selection.json、training_contract.json、
gate.json、contract.json、completed.json。本机额外添加ANALYSIS.md，不属于服务器完成清单。

## 11. 建议接着做的工作（尚未实施）

首先做GSV训练监督CPU审计，读取 `.cache/candidate_set_v1/train`，不重训、不提特征。
对所有1024条训练查询而非只挑错误统计：

1. 重新验证缓存、plan、地点标签及候选索引，计算冻结global和pair R@1/R@5/R@20。
2. 分别统计不可达、全正、可达且正确、可达但排错；1008有效不能替代困难错误数。
3. 计算最佳正−最佳负margin、正样本数量、最佳正rank、冻结多正list loss分布。
4. 统计loss在最难1%/5%/10%查询中的集中程度，保留逐查询身份以便追溯。
5. 可用已有Pitts缓存做同口径分布对照，但不得用Pitts错误选择训练样本或优化超参。

若训练也近饱和，先解决训练监督覆盖；若困难监督充分，再考虑冻结CLS可判别性诊断。
目前没有证据支持直接解冻大模型、加更多模块或延长训练。
如果接手AI提出不同方向，应说明新假设与已失败方案的差异、所需新证据及停止标准。

不要：选择Pitts30条错误专项训练；把其余6584查询直接冒充独立测试；
只报告超过global935而忽略Pair985；把抗重复改善当成R@1改善；
用饱和开发数据无限扫超参；将所有失败归咎于数据集小或语义天然无效。
后续真正获得候选方法后仍需独立评测、多种子、匹配预算对照和地点/序列相关性处理。

## 12. 常用命令与注意事项

以下是已完成实验的复用命令，不是要求用户现在重跑：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
git pull --ff-only origin main
bash scripts/run_candidate_set_pitts.sh
```

同一版本/合同完成后会核验并跳过；未完成训练组从固定种子重新训练整组5epoch，
不是优化器逐step断点续训。失败先读错误，不能直接删除completed或强行覆盖。

Windows下载（避免带引号且以反斜杠结尾的目标路径）：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_set_pitts_report_v1 C:/Users/zy/Desktop/MyVPR/doc/
```

本机推送约定：

```powershell
git -c http.proxy=http://127.0.0.1:7897 -c http.sslBackend=openssl push origin main
```

相关提交：723092f（GSV集合首筛）、de7ed43（Pitts准入）、91f0aa5（最新五组训练）。
新审计脚本尚不存在，没有新的运行命令可假称已经就绪。

## 13. 可直接给接手 AI 的启动提示

> 请先阅读本交接和最新ANALYSIS，核对当前文件、Git状态和报告。用户已经停止CLIP语义路线。
> 最新GSV训练/Pitts开发五组实验为frozen985、independent986、其余四组985，尚未验证集合收益。
> 下一步优先用已有GSV训练缓存做CPU监督审计，识别冻结pair可达错误数量与loss集中程度，
> 先不要训练或重新提取特征。若需要代码，在Windows项目中新增独立脚本，保留旧缓存代码指纹，
> 完成测试后只提交相关文件并推送，再给用户训练机同步/运行/下载命令。
> 不把“有效1008查询”误当“1008条困难错误”，不把开发集+1当论文成功。
