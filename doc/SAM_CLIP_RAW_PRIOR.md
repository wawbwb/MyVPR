# SAM 原始片段语义先验：扩展前降权的候选区域检索实验

状态：独立代码已准备；未在本机执行 Python、测试或模型；未启动训练机任务。旧 CV-Slot 教师 FAIL 保留，不再沿用它的训练门控。本实验不训练网络，不修改创新点1、官方 BoQ、RU 或旧 SAM 实验文件。

用户2026-10-20已授权推送和运行。增加 `scripts/launch_sam_clip_prior.py`，在训练机脱离SSH运行“CPU测试 → 四图smoke → 全缓存 → 评测”，任何程序错误停止，不根据语义命中率偷偷改阈值。运行状态见 `.cache/sam_clip_prior_launcher_v1/progress.json`，各阶段独立日志在同目录。启动前检查GPU1占用，运行前后核对创新点1七个保护文件哈希；没有梯度更新。

## 1. 假设与参考

借鉴 [Revisit Anything / SegVLAD（ECCV 2024）](https://github.com/AnyLoc/Revisit-Anything)：SAM 分区、邻域 SuperSegment、DINO 特征聚合用于检索。新增假设不是“CLIP 特征更适合地点匹配”，而是：在邻域合并前识别潜在动态类别，限制其在所有超片段内的聚合贡献，可能减少不稳定对象造成的区域误匹配。

[CLIP](https://github.com/openai/CLIP) 用图文余弦相似度判断预设类别。本实现用原版冻结 OpenAI ViT-B/16 的图像 CLS 和文本编码器分类物体裁剪，不使用先前的 SCLIP patch 关系教师，不拟合 CLIP 特征，也不改 BoQ 注意力。

这是新组合的可检验假设，不代表组合本身已经具有论文或专利新颖性，不是官方 SegVLAD 的复现。

## 2. 固定流水线

```text
同一张图片 → SAM-B 原始片段 → 框裁剪、背景置均值裁剪 → 冻结 CLIP 类别分数
                            ↓ 类别共识、低置信度弃权
                    原始片段风险 r_i
                            ↓ 重叠处取最大风险
                    像素可靠度 w(p)=1−0.5 max_i[r_i M_i(p)]
                            ↓
原始片段固定 Delaunay 邻接 → 一阶超片段 U_j（不删除节点、不按语义改邻接）
                            ↓ U_j(p) × w(p)
              像素覆盖率平均到 DINO 20×20 token 网格
                            ↓
              加权 VLAD → 共享 GSV-only PCA → L2
                            ↓ 保留区域可靠度，不能被 L2 抵消
          固定 RU top20 → 对称区域匹配 → 固定小残差排序
```

前置降权指**像素贡献的先验在扩展前形成**，不是只把一个中心片段的最终向量乘小。

- 汽车中心、建筑邻域：只降低汽车像素贡献，保留建筑，整个超片段不删除。
- 建筑中心、汽车邻域：汽车像素在建筑超片段内仍降权，不能因更换中心重新进入。
- 重叠 SAM 掩码：同一个像素只聚合一次；风险取最大，避免建筑大掩码稀释汽车掩码的风险，但也会传播错误的高置信分类，必须检查案例。
- 未覆盖、类别不确定、两种裁剪不一致：不降权；每个像素至少保留 0.5。

类别包含外部 car/truck/bus/motorcycle/bicycle/pedestrian，以及建筑、路面、天空等静态竞争项、自车 hood/dashboard/wipers 和 unknown 竞争项。两种裁剪必须预测同一种潜在动态类别，top-class softmax 均≥0.6，动态/非动态最大余弦差均≥0.02，才启用风险；分数取两者较小值。文本两个模板等权。原始余弦和缩放系数均保存。

**这些 softmax 是固定 prompt 集下的相对分数，不是校准置信度，更不是运动概率。** 单帧 CLIP 不能判断车辆是否在移动或以后是否离开；树、天气、施工等其他变化也未解决。

SAM 设置与旧区域实验相近：280 方图，points_per_side=16，IoU≥0.88，stability≥0.95，不做 crop layers。保留面积 196 像素至整图80%的原始片段，不做旧 lite 实验的 IoU 去重；预算64片段，超过则明确停止、不默默截断。小于196像素及大于80%区域不属于本实验处理对象。SAM 本身会漏检、切分物体或输出重叠区域。

## 3. 为什么需要区域可靠度

VLAD 使用 `sum_t m_jt (x_t−c_assignment(t))`，之后簇内归一化和整体归一化。如果全部 token 同乘一个常数，常数会被消去；因此不能只给最终向量乘权重后再次 L2。

本方案同时保存 `rho_j = sum(U_j*w)/sum(U_j)`，使用

`s_jk = max(cos(z_j,z_k),0) * sqrt(rho_j*rho_k)`。

图对评分为两个方向的区域最大相似度的平均，再取两方向均值；分母是区域数，不重新除可靠度之和。中心降权对照只使用 `rho_j=1−0.5r_center`，不改内部聚合。

最后统一 `score = frozen_RU_cosine + 0.02 * region_score`，稳定排序保留原 RU 并列顺序；不做 GT 搜索、不更换候选。不代表这个固定尺度已最优，也不是安全性保证。无区域 query 分支弃权，保持原 RU；候选无区域时区域分为0。

## 4. 必须一起报告的对照

| 变体 | 区域内部 | 区域可靠度 | 回答的问题 |
|---|---|---|---|
| ru | 不用区域 | 不用 | 冻结675/740原基线 |
| sam | 原始超片段 | 1 | 无语义的同结构基线 |
| center_only | 原始超片段 | 中心风险 | 仅中心降权是否会漏掉邻域动态贡献 |
| pre | 原始像素风险加权 | 超片段像素平均可靠度 | 新假设 |
| shuffled | 同图打乱片段风险后重新构图内权重 | 同规则 | 类别与位置对应是否有用 |
| uniform_mass | 同图平均像素风险 | 同规则 | 是否只是整图惩罚/一般正则化效果 |

所有变体使用同一 SAM 节点、邻接、候选、VLAD、PCA、评分及排序系数。片段风险打乱保留片段分数多重集，但片段大小/重叠不同，**不保证像素降权面积一致**；缓存记录两者像素风险均值。uniform_mass 匹配正确语义的整图像素质量，不匹配各超片段质量。不能把这些对照描述成完全等效的因果实验。

新 baseline 的 token membership 使用**像素覆盖率**而非旧 region_vlad 的 ANY-pixel，所有新变体一致。因此旧 SAM 数值不能拿来当这轮的直接对照；必须看这轮 `sam`。共享 PCA 沿用旧 GSV训练图像无监督拟合，不对 MSLS 拟合或为各臂重拟合。它未针对降权分布优化，是本轮有意固定的限制。

## 5. 成本、适用范围及运行

仅复用 `doc/visual_pair_msls_hard_mix_v2` 的 `local.npy / descriptors.npy / per_query.npz / summary.json`，和 `.cache/region_vlad/fit_gsv_v1/model.npz` 及 manifest。缺失即报错，不触发 DINO 全库提取、模型下载或185GB流程。

当前适配的是**旧 RU 280px 缓存**，不是官方322权重或创新点1增强模型。先验证区域先验的独立信号，不能据此声称已经改善当前最强 BoQ 单阶段系统。该实验推理仍需要 SAM+CLIP，并且是 top20 候选重排，不是训练时教师、单描述子网络创新。

查询与 top20 涉及的数据库图像都处理，SAM+CLIP各做一次，所有变体共享；不重新提取DINO。缓存保存压缩原始掩码、类别余弦、风险、各臂紧凑区域描述子和可靠度。依实际片段数，可能几GiB；启动打印64片段预算下未压缩上界，不承诺800MiB。时间从四图 smoke 观测，不承诺分钟级。只允许物理GPU1、单进程分配上限20%；不改环境、不杀其他任务。

仅在训练机运行以下命令，**本次未执行**。先CPU测试，再4图缓存，无需任何训练：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
export RU_CKPT='logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_sam_clip_prior.py
python -u scripts/sam_clip_prior.py cache --checkpoint "${RU_CKPT:?}" --output .cache/sam_clip_prior_v1 --limit-images 4
```

查看 `coverage.json` 的片段命中数、两种裁剪类别一致率和像素降权质量，确认没有全为零或异常的大面积惩罚。该文件只是描述统计，不新增“必须通过才允许训练”的教师门槛；本方案本来就没有训练。需要人工查看具体物体时，分片保留 `path / masks_packbits / cosines / risks` 可追溯原图。四图只能检查程序，不能评价总体语义质量。继续同一个缓存（参数不变，不覆盖旧实验）：

```bash
python -u scripts/sam_clip_prior.py cache --checkpoint "${RU_CKPT:?}" --output .cache/sam_clip_prior_v1 --resume
python -u scripts/sam_clip_prior.py eval --cache .cache/sam_clip_prior_v1 --output doc/sam_clip_prior_eval_v1
```

缓存使用进程锁、输入/权重/实现哈希、逐图内容哈希、原子写入。续跑验证已存在分片；损坏分片显式停止，不默默重用或删除。完整缓存有 completed seal，4图 smoke没有。只有完整缓存能评测。不会覆盖旧 SAM/CLIP/创新点1结果。

## 6. 判据和边界

报告全部740query的R@1/5/10、纠错ID、退步ID、排序改变和候选可达错误数；GT只在固定预测后读取。`pre`首先应比本轮`sam`净纠错更多，并优于`shuffled`、`uniform_mass`；与`center_only`的差异才支持“扩展前处理”的必要性。不能只看比 RU 多一例或改变了很多分数。

MSLS已经反复参与开发，因此这里只是机制探索，不能作为独立泛化证据。若获得信号，冻结prompt/权重/阈值/超参数后再迁移 Pitts30k-test 或其他已持有数据；若没有信号，区分识别未命中、动态占比低、固定候选不可达、区域匹配无效，不自动下结论“所有语义方法无效”，也不按MSLS结果反复调阈值。

DINO自注意力中已混入的上下文影响不会被聚合掩码清除。背景置均值只用于CLIP分类视图；本实验从不把遮挡图重新送给DINO，也不声称实现背景补全。
