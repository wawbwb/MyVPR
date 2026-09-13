# CLIP 静态证据分支试验

目标是检验静态区域是否提供原 RU 缺失的检索证据。本次不做融合，也不把 oracle union 当作已实现的提升。

三个辅助 BoQ 分支从同一 RU 聚合器初始化：全图、等面积随机平移区域、CLIP 静态区域。冻结 DINO 和 RU gate，仅训练聚合器。原 RU 独立保留。此实现复用 token gather，区别在于显式选择地标类、独立训练辅助描述子和检验互补性；不是一种全新的聚合结构。

CLIP 候选为 building、wall、bridge、traffic sign，与其余原有类别竞争，像素分数差大于 0.02。20×20 网格中覆盖率至少 0.75 的 token 被保留。少于 16 个则该图回退全图，随机分支同步回退。随机对照按图像路径固定二维周期平移，保持数量，可能仍与静态区域重叠；报告实际重叠率。无需人工标注，不专门处理自车引擎盖。类别分数不等于校准概率或地点判别力；墙面和交通标识也不保证独特性。

训练复用已固定的 256 个 GSV 训练地点，每地点按路径排序取 4 图。冻结 RU 选择相似地点组成 P=16、K=4 的共同批次，选入地点坐标两两至少相距 100 米，减少近邻误负例。两轮各 64 批，AdamW lr=1e-5，weight_decay=0.001，检索 MultiSimilarity 损失。所有分支共享批次及预算，记录无有效挖掘比例，不能预先保证训练有充分监督。

冻结特征缓存只生成一次。DINO 和原 3×3 投影已有空间混合，保留 token 不代表完全隔离其他区域。MSLS 查询及数据库均应用对应分支；必须复现原 RU 675/740 及全部 top1 ID。报告单分支正确数、相对 RU 的纠错与退化、静态独有纠错、双方均有静态区域的独有纠错。MSLS 已反复用于探索，不能视作独立最终测试；本轮只是固定小预算筛查。

在训练机项目根目录运行：

```bash
unset CUDA_VISIBLE_DEVICES
bash scripts/run_static_evidence.sh
```

需要原 RU checkpoint、原 ClearCLIP GSV/MSLS 缓存、dynamic_invariance_plan_v1、dynamic_invariance_msls_v1/report/frozen_outcomes.json 和原数据集。使用物理 GPU1。需重新生成显式静态 CLIP 缓存并训练三个聚合器，不是纯 CPU 审计。建议预留至少 10GB 磁盘。服务器 preflight 检查原描述子一致性及实际反向传播；本机不安装 torch。

训练支持从已保存的整轮 checkpoint 恢复。prepare 和 evaluate 需新目录，中断后保留未完成目录用于排查，不会自动覆盖。完整输出在 doc/static_evidence_eval_v1；只下载其 report 子目录，包含训练日志和掩码统计，无需下载大描述子及 checkpoint。

只有静态分支相对全图与随机分支产生可信的额外纠错，才考虑后续在独立验证集选择融合策略。若仅 oracle union 增加或随机分支同样改善，不能据此声称 CLIP 静态语义有效。
