# 实验产物与 Git 管理

## 2026-10-29二次整理（覆盖下面的现状描述）

初次操作未充分归并文档，且当前未找到对应回收站记录，不能保证上次产物可由回收站恢复。此次不再依赖回收站：完整创新点2证据逐文件SHA256核验后保存为ZIP，才删除展开副本及内容相同的旧传输包。原Git历史只进行禁用裁剪/禁用reflog过期的压缩；HEAD、refs、reflog和暂存索引核对不变。

根目录186份说明已实际归类搬移，指标和方法不改；Markdown相对链接相应调整。19份被代码源校验引用的协议保留原位置与原字节。原创新点1完整归档、创新点2交接包及原源码/配置/测试/权重保留；两篇PDF移至参考论文目录。日常入口为[文档主页](README.md)。原路径到新路径的对应见历史归档中的二次整理计划和回执。旧记录中的“原根目录说明保留原位”“均可从回收站恢复”不再代表当前状态。

## 2026-10-29整理更新（以此节为当前本机状态）

新增统一入口[项目实验总览](项目实验总览.md)，按语义/区域、候选/重排序、结构/训练三类组织原说明与最终指标。用户授权清理后，本次清单内旧报告与重复传输包先提取为[精简历史归档](历史实验归档/README.md)，再移入Windows回收站。原始逐查询路径可能不再存在；分类JSON不替代完整预测、GT或原seal。

创新点1、创新点2全部代码、完整报告、前序跨分辨率证据、专利材料和交接包原样保留；所有原源码/配置/测试、根目录说明、参考PDF、人工标注、模型权重、加载provenance和候选计划保留。logs/.cache及训练机不清理，不改写Git历史。具体数量及保护哈希复核以[执行回执](历史实验归档/清理回执.json)为准。

三类精简指标JSON和清理记录属于经过审查的轻量档案，加入可追踪白名单；原运行目录继续忽略。本次不自动git add/commit/push，用户原暂存状态不重置。下文2026-09旧记录是历史状态，不表示被本次清单清理的本机原报告目前仍存在。

2026-09-17：代码、配置、测试和 doc 顶层 Markdown 继续版本管理。
运行生成的图片、HTML、逐查询预测、NPZ、原始日志、运行配置快照与下载报告默认只存本地/训练机，
不再反复 git add。需要共享的结论写入 Markdown，必要的精简证据单独审查后纳入版本管理。

## 本次操作边界

- 87个已追踪产物从索引移除，合计18,743,496字节（约17.88 MiB）。最初保留原文件；随后经用户要求，已全部移入Windows回收站，原路径剩余0个。
- 不删除训练权重、数据集、缓存、清单外下载报告或实验代码；不改写历史，不运行git clean/gc/filter-repo。
- 保留完整 doc/pairvpr_official_paired_audit_v1：当前模型加载器依赖其中provenance.json，保留全目录避免破坏清单。
- 用户原有3个ag_slrd_runs删除和2个dynamic_coverage代码改动不在本次清理范围。
- doc根目录 Markdown 继续可见；新下载的报告目录和非Markdown文件默认忽略。
- 最新候选实验结论见 [候选结果归档](历史实验归档/02_候选与重排序/CANDIDATE_RESULTS_ARCHIVE.md)，更早路线见 [总归档](SEMANTIC_VPR_EXPERIMENT_SUMMARY.md)。

## 同步与恢复注意

**git rm --cached只在本机保留文件，不保证其他机器拉取删除提交时保留其已追踪副本。**
本次不自动推送或拉取删除变更。若后续提交并同步，应先在训练机将清单中的现存文件复制到仓库外，
再快进合并，最后按原路径恢复（届时被忽略）。不要直接同步删除后再假设训练机结果还在。
仅从Git新克隆的工作区不再拥有这些报告；旧文档中的原始证据链接是本地/训练机产物路径，
网页可能无法打开。需要复查时从训练机拷贝完整报告，不能伪造缺失的完成清单或只用汇总重建逐查询证据。
较早版本的原始文件也仍能从Git历史提取；本次不会缩小已有.git历史，主要减少后续追踪与状态噪音。
2026-09-21更新：清单内87文件已移入回收站，未永久删除或清空回收站，可在Windows回收站还原。
回收站仍占用磁盘空间；没有清理其他下载目录、空目录或训练机。删除变更仍未提交或同步。

## 本次移出追踪的精确清单

- `doc/C0.txt`
- `doc/C3_0.01.txt`
- `doc/C3_0.05.txt`
- `doc/ag_slrd_phase0_audit/per_query.csv`
- `doc/ag_slrd_phase0_audit/summary.csv`
- `doc/ag_slrd_phase0_audit/summary.json`
- `doc/ag_slrd_phase0_audit/verdict.txt`
- `doc/boq_baseline_dynamic_attention_audit/attention_balanced_random.jpg`
- `doc/boq_baseline_dynamic_attention_audit/fc_energy_slot_weights.csv`
- `doc/boq_baseline_dynamic_attention_audit/head_summary.csv`
- `doc/boq_baseline_dynamic_attention_audit/mean_position_attention.jpg`
- `doc/boq_baseline_dynamic_attention_audit/query_slot_summary.csv`
- `doc/boq_baseline_dynamic_attention_audit/run.json`
- `doc/boq_baseline_dynamic_attention_audit/summary.csv`
- `doc/boq_dinov2.txt`
- `doc/boq_dinov2_photometric.txt`
- `doc/boq_dinov2_query_semantic_aligned.txt`
- `doc/boq_dinov2_query_semantic_architecture_only.txt`
- `doc/boq_dinov2_query_semantic_condition_eval.txt`
- `doc/boq_dinov2_semantic_region_full.txt`
- `doc/boq_dinov2_semantic_region_repeatability_only.txt`
- `doc/boq_dinov2_semantic_region_repeatability_uniqueness_only.txt`
- `doc/boq_dinov2_semantic_region_semantic_only.txt`
- `doc/boq_dinov2_semantic_region_shuffled.txt`
- `doc/crop_semantic_film_runs/aligned_5ep.txt`
- `doc/crop_semantic_film_runs/architecture_only_5ep.txt`
- `doc/crop_semantic_film_runs/preflight_500steps.txt`
- `doc/crop_semantic_film_runs/preflight_audit.json`
- `doc/dynamic_category_mask_audit_full_db_condition_union/mask_montage.jpg`
- `doc/dynamic_category_mask_audit_full_db_condition_union/run.json`
- `doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries/paired_comparisons.csv`
- `doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries/run.json`
- `doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries/summary.csv`
- `doc/loss_semantic_region.png`
- `doc/loss_vpr.png`
- `doc/mixvpr_distill_semantic_alias.txt`
- `doc/mixvpr_distill_semantic_alias_random.txt`
- `doc/mixvpr_distill_semantic_alias_shuffled.txt`
- `doc/mixvpr_distill_semantic_positive_${mode}.txt`
- `doc/mixvpr_distill_vpr_semantic_reliability.txt`
- `doc/msls-val-R1.png`
- `doc/pairvpr_candidate_union_v1/candidates.npz`
- `doc/pairvpr_candidate_union_v1/completed.json`
- `doc/pairvpr_candidate_union_v1/per_query.json`
- `doc/pairvpr_candidate_union_v1/summary.json`
- `doc/pairvpr_local_matches_v1/completed.json`
- `doc/pairvpr_local_matches_v1/results.json`
- `doc/pairvpr_match_geometry_v1/completed.json`
- `doc/pairvpr_match_geometry_v1/pairs.json`
- `doc/pairvpr_match_geometry_v1/summary.json`
- `doc/pairvpr_metadata_audit_v1/per_query.csv`
- `doc/pairvpr_metadata_audit_v1/summary.json`
- `doc/region_target_abs_mean.png`
- `doc/region_target_std.png`
- `doc/rscd_runs/formal_checkpoint_inventory.txt`
- `doc/salad_dinov2.txt`
- `doc/seg_aux_aligned_v1/completed.json`
- `doc/seg_aux_aligned_v1/config.yaml`
- `doc/seg_aux_aligned_v1/contract.json`
- `doc/seg_aux_aligned_v1/csv/version_0/hparams.yaml`
- `doc/seg_aux_aligned_v1/csv/version_0/metrics.csv`
- `doc/seg_aux_aligned_v1/csv/version_1/hparams.yaml`
- `doc/seg_aux_aligned_v1/csv/version_1/metrics.csv`
- `doc/seg_aux_aligned_v1/resume_8be6806909784270b9b14db3e90abc21.json`
- `doc/seg_aux_warm_v2/aligned/completed.json`
- `doc/seg_aux_warm_v2/aligned/config.yaml`
- `doc/seg_aux_warm_v2/aligned/contract.json`
- `doc/seg_aux_warm_v2/aligned/csv/version_0/hparams.yaml`
- `doc/seg_aux_warm_v2/aligned/csv/version_0/metrics.csv`
- `doc/seg_aux_warm_v2/aligned/head_warmup.csv`
- `doc/seg_aux_warm_v2/vpr_only/completed.json`
- `doc/seg_aux_warm_v2/vpr_only/config.yaml`
- `doc/seg_aux_warm_v2/vpr_only/contract.json`
- `doc/seg_aux_warm_v2/vpr_only/csv/version_0/hparams.yaml`
- `doc/seg_aux_warm_v2/vpr_only/csv/version_0/metrics.csv`
- `doc/seg_aux_warm_v2/vpr_only/csv/version_1/hparams.yaml`
- `doc/seg_aux_warm_v2/vpr_only/csv/version_1/metrics.csv`
- `doc/seg_aux_warm_v2/vpr_only/resume_2ddb453290a5426bafb4cc11b0c7f606.json`
- `doc/semantic_alias_tb_raw.csv`
- `doc/semantic_alias_tb_summary.csv`
- `doc/semantic_gate_max_delta.png`
- `doc/semantic_gate_std.png`
- `doc/semantic_region_counterfactual_batch0/aligned_mask_montage.png`
- `doc/semantic_region_counterfactual_batch0/counterfactual_run.json`
- `doc/semantic_region_counterfactual_batch0/counterfactual_summary.csv`
- `doc/semantic_region_delta_batch0_clean/run.json`
- `doc/semantic_region_delta_batch0_clean/summary.csv`
