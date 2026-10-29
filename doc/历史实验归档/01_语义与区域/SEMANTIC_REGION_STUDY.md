# 创新点2候选：语义引导的空间区域聚合 SSRP-BoQ

状态：实现及运行协议。服务器已通过Python/Bash语法和依赖/权重文件预检，尚未构造真实模型、进行模型测试、教师缓存、smoke或训练；不表示语义收益已经成立。先检验独立贡献，暂不叠加创新点1。

## 1. 依据与研究问题

研究问题：能否用训练期局部语义来组织DINOv2视觉细节，而不把视觉特征替换为CLIP特征，最终仍用单个全局描述子进行单阶段检索？

- [BoQ，CVPR 2024](https://github.com/amaralibey/Bag-of-Queries)：官方DINOv2-B/14、12288维权重，322×322输入。不是历史RU，也不包含创新点1。
- [Revisit Anything，ECCV 2024](https://github.com/AnyLoc/Revisit-Anything)：借鉴区域定义与视觉描述分工。不使用SAM，不复现其邻域VLAD、PCA、投票或区域匹配。
- [SCLIP，ECCV 2024](https://arxiv.org/abs/2312.01597)：采用末层CSA的QQ/KK相关注意力方程。自有最小OpenCLIP适配实现，无MMSeg、滑窗和PAMR；不是官方分割成绩复现。

待验证的设计是语义类别×重叠空间基的软分配、区域视觉池化和上下文回写；不是简单更换SAM，也不主张该组合已经具有新颖性。

## 2. 结构与教师

冻结官方DINO取得与作者前向一致的末层patch图X，不额外施加最终LN。学生分配器采用LN、1×1降维、深度可分离3×3卷积和18类预测，得到每位置类别概率p。

在归一化坐标(.25,.25)、(.25,.75)、(.75,.25)、(.75,.75)放置四个标准差0.30的高斯空间基b，在每位置归一化。类别概率与空间基相乘形成72个软槽：

    a[k,s,i] = p[k,i] * b[s,i]
    r[k,s] = sum_i(a[k,s,i] * X[i]) / sum_i(a[k,s,i])
    C[i] = sum_k,s(a[k,s,i] * r[k,s])

每个patch在全部槽上的权重和为1，不删除任何类别。四个空间槽只是保留粗空间布局，不保证分开全部独立物体。区域描述子不另存为检索数据库。

上下文C与原特征X的差经768→128→768映射，生成有界残差：

    X_new = X + 0.10 * RMS_channel(X) * tanh(U * GELU(D * (C-X)))

U零初始化，预期初始描述子复现官方BoQ，仍需真实smoke验证。322输入的23×23=529个patch网格保持不变；编码器CLS不输入BoQ，不用72个槽直接替换原patch。

推理：RGB→冻结DINO→学生分配/池化/回写→BoQ→12288维单位描述子。没有CLIP、文本、SAM、区域配对或重排。

教师为现存OpenAI ViT-B/16，SHA256：

`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`

末块采用 `[softmax(QQ^T/sqrt(d)) + softmax(KK^T/sqrt(d))]V`，保留残差和MLP。两个softmax相加，不除以2。由OpenCLIP正确加载本地OpenAI JIT权重，不下载代码或权重。

固定18类：建筑、窗、门、墙、屋顶、道路、人行道、交通标志、路灯、树、草、天空、水、桥、围栏、车辆、行人和其他场景内容。每类三个固定文本模板，先归一化每个文本向量，再均值并归一化。224整图输入生成14×14概率，logit scale=40，缓存float16概率；训练时插值到23×23并重新归一化。插值不增加真实语义分辨率。

指导损失为归一化熵置信度加权KL，系数0.05。不让DINO向量拟合CLIP向量，也不预设动态类别必须抑制。

## 3. 匹配对照

| 组别 | 结构 | 训练目标 |
| --- | --- | --- |
| A official | 官方DINO+BoQ，仅评测 | 不更新 |
| B visual | 学生区域聚合+BoQ | 地点MultiSimilarity |
| C aligned | 与B相同 | 地点损失+正确教师空间概率 |
| D shuffled | 与C相同 | 地点损失+空间错位教师概率 |

D按图像路径和seed生成固定无不动点置换，在教师原生14×14网格打乱位置，保留每图的每类概率分布与置信度分布，再做与C相同的插值/归一化。B也读取、验证同一缓存并计算诊断，但语义损失系数为0。B的槽由地点监督学到，不要求具有固定语义含义。

三组初始化、容量、图像、批次、步数、优化器和精度一致。冻结完整DINO，解冻新模块(lr1e-4)和BoQ(lr1e-5)；AdamW、无weight decay、梯度裁剪1、FP32。MultiSimilarity按经纬度排除25米内不同place-ID负样本，三组相同。

主要比较C-B和C-D，不能只看C-A；B-A包含结构变化和继续训练，不是结构独立贡献。允许BoQ适配是一项设计选择，不是已证明的历史失败原因。

## 4. 数据、预算及评测

直接读取GSV城市CSV，不使用历史RU挖掘缓存。按city+place-ID/seed哈希划分8192训练地点与1024适配holdout地点，每地点按路径哈希固定选择4张不同图，记录图像SHA256。

五个完整适配epoch，每epoch遍历训练地点一次，P=16、K=4，共2560步/组。这不是完整GSV重训；仍为单seed候选筛查，不据此宣称稳定收益。检查点固定最后一步，不挑最好epoch/seed。

只缓存32768张训练图的18×14×14 float16概率，未压缩约220.5MiB，加清单/分片记录；不存全库DINO dense特征，不做185GB缓存。教师与学生均用完整图像，分别缩放224和322；没有随机crop、flip或颜色增强。

开发评测包含：

- GSV adaptation holdout：1024张参考、3072张查询，精确place-ID真值。仅与本轮适配地点不重叠，不保证官方预训练未见；不同地点ID可能地理重叠，是匹配诊断协议。
- MSLS-val：已曝光开发集，完整数据库。
- Pitts30k-val：已曝光开发集，不是test。

Pitts30k-test不进入开发评测CLI。待方案与检查点锁定后再另行进行最终测试，不用于选提示词、权重或epoch。若出现候选收益，先做匹配seed复核，再考虑与创新点1组合。

## 5. 代码与运行清单

- 核心结构：`src/models/semantic_region_pool.py`
- 离线教师：`src/models/sclip_region_teacher.py`
- 协议/原子写入/校验：`src/semantic_region_protocol.py`
- 入口：`scripts/semantic_region_study.py`
- 配置：`config/semantic_region_study.yaml`
- 包装：`scripts/run_semantic_region_study.sh`
- CPU单测：`tests/test_semantic_region_study.py`

以下是后续操作清单，本次只上传代码，不执行缓存、smoke或训练。

```bash
cd /home/wt/workspace/OpenVPRLab
conda activate VPR

# CPU/文件检查；不构造模型、不更新权重。
bash scripts/run_semantic_region_study.sh preflight

# 可选训练机CPU单测；本机不运行。
CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests/test_semantic_region_study.py

# 计划为CPU；缓存仅执行冻结教师推理。
bash scripts/run_semantic_region_study.sh plan
bash scripts/run_semantic_region_study.sh cache

# smoke有真实但被丢弃的4步更新，必须显式授权。
for ARM in visual aligned shuffled; do
  bash scripts/run_semantic_region_study.sh smoke --arm "$ARM" --allow-training
done

# 验证三组smoke后，均重新从官方初始化，不复用smoke权重。
for ARM in visual aligned shuffled; do
  bash scripts/run_semantic_region_study.sh train --arm "$ARM" --allow-training
done

# 不自动触发评测。
for ARM in official visual aligned shuffled; do
  bash scripts/run_semantic_region_study.sh eval --arm "$ARM"
done
bash scripts/run_semantic_region_study.sh summary
```

包装脚本在Python启动前暴露物理GPU1，对应配置逻辑cuda:0；默认不涉及GPU0。直接用Python入口时须在新进程启动前设置CUDA_VISIBLE_DEVICES=1，不在Python进程中修改可见卡。可在新进程前用SRP_PHYSICAL_GPU指定其他卡。

已存在阶段须显式加`--resume`。示例：

```bash
bash scripts/run_semantic_region_study.sh train --arm aligned --allow-training --resume
```

配置/代码/CSV/图像/官方权重/教师权重/缓存均有身份检查，改变后不能接着旧run跑。每16步及结束原子保存优化器、模型、Python/NumPy/CPU/CUDA随机状态；意外中断最多重放15个已运行但未保存的步骤。数据读取使用私有loader RNG，避免恢复时额外消耗模型RNG。已完成阶段验证后返回，不追加训练。

## 6. 当前边界

代码上传不等于实验通过。仍需在训练机验证教师加载、伪语义质量、零起点、真实梯度和显存；概率有限/缓存完整不等于分割准确。三组机械检查通过也不等于语义收益。

不修改旧模型、结果及专利文档。若C没有胜过B/D，只能说本次教师、分组和训练配置未建立语义贡献，不能否定全部语义路线。
