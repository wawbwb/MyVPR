# 自动CLIP动态区域抑制首筛

状态：代码已实现，CPU测试通过，训练机GPU全流程待运行。没有新增性能结论。
不再要求人工圈图。旧多边形仅做事后定位质量检查，不拟合分数、不选阈值、不调整prompt。

## 固定方案与范围

冻结项目现有OpenCLIP ViT-B-16/openai（open_clip_torch==2.26.1），输入沿用ImageNet归一化280×280。
每个query取16个重叠112×112窗口：20×20网格上起点0、4、8、12，窗口宽高8格。
各窗口通过现有CLIP预处理缩放至224，取经过全局投影与L2归一化的CLS向量，计算文本余弦相似度。
不把原始patch token直接当作分割logits。总计740×16=11840个CLIP crop前向。

类别固定为：

- 外部动态：街道上的car、truck、bus、motorcycle、bicycle、pedestrian。
- 自车竞争项：从挡风玻璃看见的hood、车内dashboard、windshield wipers。
- 静态竞争项：building facade、road surface、sidewalk、wall、tree、sky、bridge、street sign。

文本模板统一为`a photo of a {category}`。外部分数为
`sigmoid((max_external_cosine - max(max_ego_cosine, max_static_cosine)) / 0.05)`。
重叠窗口分数平均回填为20×20图；温度0.05固定，不按图标准化、不设二值阈值。
自车竞争分数用相同形式另存作诊断，本轮抑制仅使用外部分数。

这是未校准的语义倾向，不是运动概率、置信度或动态像素真值。两类相等时输出0.5而非0，
因此必须结合对照检查是否只是普遍降权。CLIP也不能凭单帧证明车辆跨时消失。
大窗口会混合物体与背景，自车文本能否真正分离自车需要检查，不视为已解决。

## 8组受控评测

baseline、zero_bias、aligned_late、shuffle_late、rowmean_late、aligned_input、shuffle_input、rowmean_input。

- late：对RU的BoQ施加`-0.5 × mask`。
- input：`normalised_image × (1 - 0.5 × upsampled_mask)`，向固定归一化均值RGB软混合；不是背景补全。
- shuffle：按query路径哈希和种子42，每行独立打乱mask值，逐行面积和所有数值不变。
- rowmean：每行替换为该行均值，保留纵向分布与整图平均强度，移除横向语义位置差异。

shuffle会改变空间形状，rowmean会改变分数方差，均不是完美控制。三组必须共同报告，
不能从某一组偶然更好推出语义因果收益。late与input的0.5作用于不同机制，非等效干预强度。

仅query单侧干预，搜索原始完整18871参考库，复用已有RU描述子。
不重新训练、不分割、不重新提取数据库；不等价于旧query/database双侧干预。
重新提取740个query的原图与三种软遮挡版本，约2960次RU骨干前向，另加3张固定DB哨兵核验。
CLIP和RU分进程运行，不同时保留GPU模型。模型与~885MiB原库张量占用GPU显存；
输出全库分数及query描述子约0.7GiB，另有JSON、图片和CLIP缓存，建议留至少2GiB空闲磁盘。
实际训练机耗时尚未测量，不承诺分钟级完成。

## 训练机命令

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python -m unittest discover -s tests -p test_clip_dynamic_screen.py -v

python scripts/clip_dynamic_screen.py cache \
  --device cuda:1 \
  --output .cache/clip_dynamic_msls_v1

python scripts/clip_dynamic_screen.py evaluate \
  --cache .cache/clip_dynamic_msls_v1 \
  --source doc/visual_pair_msls_hard_mix_v2 \
  --device cuda:1 \
  --output doc/clip_dynamic_screen_v1
```

使用项目已有CLIP权重缓存；若缺权重，现有加载器可能尝试下载并使用配置的镜像。
无需DeepLab掩码、SAM或人工标注。原动态实验run.json仍用于验证RU checkpoint和MSLS清单身份。

cache阶段可原命令续跑，验证已有分片的图像SHA和从余弦分数重建的mask；损坏分片明确报错，不默默跳过。
已存在cache但合同变动时要求新目录。evaluate要求新输出目录，不支持自动续跑；失败目录保留逐query JSON与数组，
没有completed.json不算完整实验，修复后用新目录。不会自动重提完整数据库来修复缓存。

## 来源与结果

CLIP缓存保存原始类别余弦、external/ego评分图、权重和代码哈希、每张query图像SHA。
读取原RU缓存时检查checkpoint、所有描述子范数、740行候选分数；新提取740个query与3个DB哨兵必须复现缓存，
完整baseline top1必须复现缓存且为675/740，否则停止。不会把其他路线的head输出混入基线。
这不是独立重提整个数据库的复现，旧DB没有逐图内容哈希的限制仍存在。

缓存index.html按分数均值和路径哈希固定抽样，无需填写。
localization_diagnostics.json可读取既有annotations_validated.json，报告区域内外平均分数；
不存在文件时记录available=false，不要求补标。旧区域粗糙、小样本、只标部分物体，不用于总体IoU结论。

评测保存scores.npy（740×8×18871）、descriptors.npy（740×8×12288）、query_outcomes.csv/json，
summary.json明确记录变体轴序、纠错ID、退步ID及aligned对两个控制的配对结果。
partial_queries仅是故障排查记录，不是自动恢复承诺。
index.html固定展示12/65个基线错误：query自动热图、原top1、baseline分数最高GT。
按原基线错误再路径哈希抽样，不按新方法纠错结果挑案例；不需要画区域。

主要判断：净纠错、全部误伤、正确空间位置是否优于两个对照。基线65错误与675正确分别报告，
不能只报告错误子集的救回比例。分差变化不等于R@1增益。
这仍是反复用于开发的MSLS首筛；任何收益需冻结设计后独立验证，不宣称已完成泛化验证。

## 本机下载

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/.cache/clip_dynamic_msls_v1 C:\Users\zy\Desktop\MyVPR\doc\
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/clip_dynamic_screen_v1 C:\Users\zy\Desktop\MyVPR\doc\
```

开发验证范围：CPU评分公式、竞争类别、空间对照、排序、分片完整性与既有标注只读诊断测试。
本机没有运行CLIP/RU GPU前向；运行时数值合同在训练机验证。
