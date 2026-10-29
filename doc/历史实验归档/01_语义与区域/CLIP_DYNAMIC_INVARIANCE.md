# CLIP 动态外观不变性：第一轮小规模试验

这是一项新机制的试验，不是已经有效的方法。CLIP 自动选择训练扰动区域，模型学习减弱对区域内外观的依赖；推理保留原图和全部 token，不需要 CLIP 或掩码。

## 冻结的试验设计

- 原始 RU 初始化；仅在 DINO 最后两层的 qkv 上加入 rank=4、alpha=4 的零初始化 LoRA。冻结原始编码器、RU gate、BoQ，所有模块始终 eval，避免 dropout 干扰梯度重算。
- 三组：plain（相同参数量的普通适配训练）、random（平移区域扰动＋一致性）、semantic（CLIP 区域扰动＋一致性）。另评估冻结 RU。plain 同样有原图保持损失，准确含义是“带保持约束的普通适配对照”。
- 256 个训练地点，P=4、K=4，每轮 64 步，2 轮共 128 步/组。AdamW lr=1e-4、weight_decay=.001，梯度裁剪 1，固定种子 42。开发结果不选 epoch、不调整参数。
- 三组使用相同地点顺序、视图采样、初始化、微批大小、优化步数。三路 student forward/replay 预算一致；plain 的两路增强输入等于原图。
- 损失为原图 MultiSimilarity 检索损失＋原图描述子对冻结 RU 的平方距离＋（两增强视图与原图及彼此之间的平方距离均值）。两个附加项系数均为 1；plain 关闭一致性项。保留检索目标以防止恒定描述子坍塌。这些超参数是预先固定的试验设定，不是验证所得最优值。

## 自动扰动与对照

复用 `.cache/clearclip_token_drop_train_v1`，无需新 CLIP 推理。缓存是原 ClearCLIP 规则输出的 20×20 动态像素覆盖率，不是逐像素真值或校准概率。选取覆盖率至少 .9 的格子，最近邻展开至 280×280；总面积超过 30% 时不扰动。粗网格仍可能包含静态背景，本试验不能消除这种定位误差。

仅在支持区域内施加通道增益 .65–1.35、偏移 ±.12 和标准差 .025 的小幅像素噪声，生成两个独立版本。区域外像素逐位保持不变，不移动或删除物体、不修复被遮挡背景。它检验颜色和纹理依赖，不能单独证明对车辆消失、移动等遮挡变化的鲁棒性。

random 对整个区域作不越界的非零整数格平移，形状、面积与语义组完全相同。无法平移时两组同时跳过。平移后可能仍与动态区域重叠，plan 报告平均重叠比例；不宣称它完全没有语义。两组复用相同外观扰动随机规则。

prepare 自动生成 12 个预览（原图、语义扰动、随机扰动及两张掩码叠图），无需人工标注。

## 划分与判读边界

由缓存图像路径中的城市名构建地点级划分，按固定哈希选择一个城市作为本次适配的开发城市，在其他城市中固定抽取训练地点。开发使用该城市 256 地点、每地点 4 张图；每地点固定 3 张数据库图、1 张查询。GT 为原 GSV 地点标签。

该城市可能被基础 RU 预训练使用过，开发集合不是标准 VPR 测试集，也不能代表从未见过的地理位置。其小型数据库同样可能高估检索表现。它只用于先检查当前适配是否值得扩展，不能据此发表或宣称标准基准增益。尚未提供真正独立测试集，因此本轮不自动重跑反复使用的 MSLS-val，不用它选择方案。

开发报告对四个模型统一比较：原图 query/原图 DB、语义扰动 query/原图 DB、平移扰动 query/原图 DB；扰动种子与训练 epoch 分离。另在实际有扰动的图像上报告描述子平方变化，避免大量空掩码稀释。

继续的必要证据是：semantic 在真实语义扰动下比 plain/random 更稳定，同时原图检索不因适配明显退化。仅减小描述子变化不足以继续；开发库若饱和，也不能当作已证明有效。有信号后再设计完整 MSLS 历史对照与独立基准测试。

## 显存与验证

microbatch=2。先无梯度计算整批描述子及原始教师，再对完整地点批计算检索/一致性损失，并通过相同输入重算前向传播回传描述子梯度。每个微批检查重算输出一致性；模型保持确定性 eval，但 LoRA 仍可反向传播。

预检包含 4 个 CPU 测试、2 个 torch 测试（身份映射、教师关闭恢复、完整图与重算梯度一致），以及真实 RU 的零初始化、非空语义扰动、梯度检查。任一失败就退出，不跳过缺失依赖。每轮检查冻结参数版本和梯度，保存 adapter 与优化器。

本机仅运行 4 个 CPU 测试和所有新 Python 文件的语法检查；本机未安装 torch，未运行 GPU 或完整训练。

## 同步与运行

训练机现有 VPR 环境，使用物理 GPU1。不要设置 CUDA_VISIBLE_DEVICES=1 后再使用 cuda:1，否则设备编号会改变。

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
unset CUDA_VISIBLE_DEVICES
bash scripts/run_dynamic_invariance.sh
```

复用既有 CLIP 训练缓存和原 RU checkpoint。整个流程依次 prepare → preflight → 三组训练 → 开发评估。每 8 批、每 64 张开发图打印进度；不自动安装依赖。三组每个 epoch 保存 checkpoint；从 epoch 边界恢复，不从中断批次恢复。

输出目录：

- `doc/dynamic_invariance_plan_v1`：划分、区域覆盖统计和预览。
- `doc/dynamic_invariance_train_v1/{plain,random,semantic}`：训练日志和小型 LoRA checkpoint。
- `doc/dynamic_invariance_dev_v1/report`：轻量分析报告，包含四模型三条件的逐查询结果。

开发阶段保留约 600 MiB 描述子数组在训练机，无需下载。下载轻量报告、计划和训练日志即可；此训练集仅 256 地点，结果不稳定风险需结合日志分析，不作为最终方法胜负判定。

Windows PowerShell 或 CMD：

```text
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_invariance_dev_v1/report C:/Users/zy/Desktop/MyVPR/doc/dynamic_invariance_report_v1
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_invariance_plan_v1 C:/Users/zy/Desktop/MyVPR/doc/
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_invariance_train_v1 C:/Users/zy/Desktop/MyVPR/doc/
```

开发输出已存在时拒绝覆盖，重新评估需手工指定新的 --output。训练失败且已有 last.pt 时，shell 脚本自动从匹配 contract 的最近完整 epoch 恢复。若在第一个 checkpoint 前失败，可选择新输出目录或先检查错误；不要删除既有结果来掩盖问题。
