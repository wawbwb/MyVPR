# SLGD：训练期局部匹配知识 → 单阶段全局检索

## 当前阶段

实现了视觉阶段的局部教师与同结构 A/B 对照，不把它称为已经实现的完整语义模型。只有视觉压缩有效，才增加 CLIP 可靠性监督 C 与打乱语义 D。当前不下载新数据，不在开发 PC 测试，不重新安装训练环境。

此前 VPR-guided Semantic v1 三组均完成 5 轮（7815 步）。固定最后一轮：

| 方法 | MSLS R@1 / 正确数 | Pitts30k-val R@1 / 正确数 |
| --- | --- | --- |
| 原始 RU | 91.2162% / 675 | 94.1115% / 7160 |
| vpr_only | 90.5405% / 670 | 94.1115% / 7160 |
| plain semantic | 90.8108% / 672 | 94.1246% / 7161 |
| guided semantic | 90.8108% / 672 | 94.0983% / 7159 |

没有观察到 guided 比 plain 的新增收益。这是本次适配实验的结果，不是 SegVPR 或语义路线整体失效的证据。

## 论文依据与边界

- [SelaVPR（ICLR 2024）](https://arxiv.org/abs/2402.14505)：局部特征的互为最近邻匹配与局部训练目标可提供全局向量之外的判别信息；但原方法仍需候选重排序。
- [SALAD（CVPR 2024）](https://arxiv.org/abs/2311.15937)：用带 dustbin 的最优传输，将 patch 特征聚合成固定槽位描述子。本实现复用仓库 SALAD 的 Sinkhorn 代码。
- [SelaVPR++](https://github.com/Lu-Feng/SelaVPRplusplus)：已经包含单分支聚合方案，因此“DINO + 适配器 + SALAD”本身不是新颖性证据。
- [StructVPR（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_StructVPR_Distill_Structural_Knowledge_With_Weighting_Samples_for_Visual_Place_CVPR_2023_paper.html)：教师知识应经过可靠性筛选，而不是全部强行注入。
- [CLIP](https://arxiv.org/abs/2103.00020)：图文级训练不能直接提供完美的 patch 语义/空间对应标签。后续语义教师需要真实同区域输入、可靠性与打乱对照。

待检验的研究假设是：**局部匹配的正负样本关系，经可靠性筛选后，可以约束共享槽位聚合器，使部分跨图判别能力压缩进单图固定描述子。** 不是声称全局向量可以无损替代任意跨图匹配，也不声称这一机制尚未被其他论文研究。

## 实际实现

1. 用训练好的 RU checkpoint 作为 DINOv2 与 BoQ 初始化。首轮冻结 DINO 与 RU gate；BoQ 和新局部分支可训练。
2. 局部教师：原生 patch 网格池化成 10×10；LayerNorm → 256 → 128 的 projector；对互为最近邻匹配的平均余弦分数施加正负间隔损失。匹配索引不反传，匹配余弦值反传。至少 3 个互为最近邻才有效。
3. 教师先训练 512 步，然后冻结。不是在学生训练中不断变化的自引用目标，不利用 MSLS 或 Pitts 选教师。
4. 学生：保留 RU 的全局向量 g，新局部分支以 16 个共享 OT 槽位聚合为 l（2048 维）；两分支各自单位归一化，再拼接：

   `d = [sqrt(0.8) * normalize(g), sqrt(0.2) * normalize(l)]`

   保留原始 g 的维度，而非随机压成 256 维，以免首轮同时测试两种压缩。A/B 最终维度严格相同。初始完整 d **不等于 RU**，因为局部支路占 20%；初始 A/B 必须逐项相等。
5. 推理不运行 MNN、教师、CLIP、训练诊断 token 或候选网络，只计算单图 d 和数据库点积；并未测量实际加速。

| 组别 | 训练目标 | 网络与初始化 |
| --- | --- | --- |
| teacher | 局部正负间隔损失 | 独立小 projector，DINO/RU 冻结 |
| A / retrieval | MultiSimilarity 检索损失 | 相同 RU、训练后教师 projector 初始化、OT 头 |
| B / local_distill | 检索 + 0.1×局部损失 + 关系 KD | 与 A 完全相同 |

KD 是正负分数差的 Bernoulli KL，教师差值 stop-gradient，只在教师正样本分数高于负样本且匹配数量有效时使用。温度 0.07、乘温度平方，覆盖率与每 64 步局部支路梯度冲突单独记录。B 是联合目标，不足以单独归因“KD 有效”；若通过，还需局部损失-only 消融。

## 划分与门控（运行前固定）

- 仅 GSV-Cities：按稳定地点 ID hash 划分，8192 个训练地点、1024 个留出地点，地点不交叉。历史 GSV 上已有探索，因此不能称独立最终测试。
- 训练 P16×K4、280×280、512 步；每地点每阶段同样取图/增强，A/B 同样 batch 与优化器时钟。只保存固定最后一步，不按开发集挑 checkpoint。
- 留出集每地点第一个采样视角作为查询，另外三个视角进图库（1024 查询 / 3072 图库）；完整检索在整个图库上评估。
- 局部诊断的正样本取冻结 RU 最相似的同地点视角，负样本取冻结 RU 在图库中最难的异地点图。该定义不保证空间重叠、也不保证异地点绝无视觉/GPS 重叠。
- teacher gate：有效比例 ≥95%、正负局部分数准确率 ≥65%、比原始 DINO 局部匹配高 ≥1 pp、平均正负差 >0。没过则暂停学生训练，不自动换阈值。
- student gate：B 比 A 净增 ≥5 个留出正确查询，局部分数准确率比 A 高 ≥1 pp，且 B 不低于冻结 RU 的留出正确数。过关仍只是单种子可行性，不是稳定收益。
- smoke：每组 4 步、32 个留出地点，只验证有限梯度、A/B 初始相同、checkpoint 往返、描述子归一化等；不要求教师收益。

这一步不跑 MSLS/Pitts，不扫描 beta，不启动 CLIP 或全量训练。后续须做多个种子/独立划分确认，并增加局部-only 消融后，才能进入可靠语义监督。

## 训练机操作

入口：`bash scripts/run_slgd_screen.sh all`。它在已有 VPR 环境与物理 GPU1 上顺序执行：remote pytest → 三组 smoke → smoke gate → teacher pilot → teacher gate → A/B pilot → student gate。共享实验锁占用时退出，不抢占其他任务。

结果在 `logs/slgd_v1/`，`progress.json` 标记阶段，`completed.json` 只表示该组执行完且文件 hash 已记录；**不表示效果通过**。gate 文件单独给出 PASS/FAIL。没有全量训练入口。

重复入口会对已完成组校验 hash 后跳过，对未完成组从 `last.pt` 恢复。每 16 步保存一次，因此最多重复 15 步；确定性每地点采样和 RNG/Adam 状态恢复使重算沿用原计划，但不宣称跨软件/硬件位级等价。

```bash
cd ~/workspace/OpenVPRLab
tail -n 25 logs/slgd_v1/queue_*.log
find logs/slgd_v1 -maxdepth 2 -name progress.json -print -exec cat {} \;
```

仅代码、测试和本说明纳入 Git；权重/运行日志/大型特征不提交。

## v1 实际执行结果与协议缺陷（2026-10-11）

17 项单元测试与三组 4 步 smoke 通过；A/B 初始参数与检索结果相同。教师完成 512 步，checkpoint 往返与结果 hash 校验通过。局部正负分数准确率由 98.6328% 提升到 99.3164%，原始 DINO 为 99.4141%；平均正负分数差由 0.05671 提升到 0.14582。上述准确率不是检索 R@1。

冻结 RU 在原留出集已经正确 1022/1024，只剩 2 个错误。原始 DINO 的局部准确率剩余上限只有 0.5859 pp，原定“超过原始 DINO ≥1 pp”无法达成，因此这是验证集/门控设计缺陷，不能据此否定 SLGD 的学生压缩假设。

同步时不正确的换行处理将服务器启动脚本 `gate teacher` 截成 `gate teache`，队列因此退出；A/B pilot 没有启动。现已改用 Git `eol=lf` 与上传字节检查，不再执行该 sed 转换。v1 结果和原门槛全部保留，不追认其通过。

后续先运行 [SLGD_HARD_AUDIT.md](SLGD_HARD_AUDIT.md) 的独立 v2 大图库诊断，复用现有教师，**不重复训练、不自动启动 A/B 或 CLIP**。v1 入口仅用于历史复核。
