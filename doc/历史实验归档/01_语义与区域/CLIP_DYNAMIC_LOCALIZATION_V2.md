# CLIP自动定位第二阶段：固定2×2诊断

本轮只改进并诊断自动定位，不重新训练或运行RU检索，不要求新增人工标注。
旧CLIP全量首筛无净收益；本轮不通过反复调权重从MSLS选择最优检索结果。

## 四组固定比较

| 输出名 | 窗口 | 评分 |
| --- | --- | --- |
| coarse_sigmoid | 112×112，16个，旧缓存 | 原sigmoid |
| coarse_positive | 同上，无新增前向 | 正证据映射 |
| fine_sigmoid | 56×56，81个，新CLIP前向 | 原sigmoid |
| fine_positive | 同上，复用余弦 | 正证据映射 |

图片预处理仍为280×280，CLIP内部将crop缩放为224。小窗口宽高为整图20%，步长28像素，
9×9完整覆盖；旧窗口宽高40%，步长56像素。小窗口可能减轻背景混合，也可能因缺少上下文而更差，不能预先认定有效。
模型、权重、文本类别和模板、温度0.05均保持上一版。所有新crop类别余弦保留。

令m为外部动态最大余弦减去自车/静态最大余弦。
旧评分为p=sigmoid(m/0.05)，新评分为max(2p−1,0)，分别在crop上计算，再平均回填重叠窗口。
m≤0的窗口因此不贡献抑制；等于0不再意味着0.5。
这是固定类别竞争规则，不是校准的运动概率，也不保证消除所有静态误标。
新映射同时降低整体强度；不能仅因为分数更低、零值更多就判定位成功或预测检索会提升。
若位置同时被其他正分窗口覆盖，最终仍可有正分数；不以该分数充当像素语义真值。

## 运行

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python -m unittest discover -s tests -p test_clip_dynamic_localization.py -v
python scripts/clip_dynamic_localization.py \
  --old-cache .cache/clip_dynamic_msls_v1 \
  --device cuda:1 \
  --output doc/clip_dynamic_localization_v2
```

新CLIP工作量为740×81=59940个crop，按16个crop分批。是上一轮CLIP crop数的约5倍，
不等于墙钟耗时严格5倍；本机未测GPU时间。没有数据库特征提取、RU前向、训练或新检索结果。
原缓存余弦仅做CPU重映射，原文件不覆盖。

支持原命令续跑，已有分片先校验图像身份、cosine形状和有限值。代码/输入/旧缓存/标注变化时要求新目录。
损坏分片明确报错，保留文件，不自动删除。权重与文本向量SHA必须和旧CLIP一致。
完成后重跑只验证缓存，不重新提取。默认已有标注仅用于报告；不存在时不要求补标。

## 读报告

- index.html：12个按路径哈希固定选样，加入已有纳入标注的6例并去重；四种评分用同一原图与相同色阶显示。
- summary.json：分数分布、零值比例、全零query、横向方差，以及每组既有外部/自车多边形内外平均分数。
- maps.npz：四组external及对应ego图，外加query路径；全零结果明确保留，不当作成功定位。
- shards：81个小crop的原始余弦，可复核评分映射。
- contract/completed：完整来源、代码哈希与无检索标志。

检查时结合：物体/背景选择性、遗漏程度、宽窗口与小窗口差异、类别竞争误判。
多边形少且粗，区域外可能有未标动态物体，内外分差不是正式分割准确率；
四组评分尺度不同，绝对差异不能直接充当算法优劣排名。
只有定位诊断支持继续时，才另行固定新的检索配置；本脚本不会自动启动旧评测。

本机PowerShell下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/clip_dynamic_localization_v2 C:\Users\zy\Desktop\MyVPR\doc\
```

验证：CPU测试涵盖旧映射精确复现、平局/静态竞争时零证据、细窗口空间范围、ego竞争、全零统计和坏数据拒绝。
本机未运行GPU模型；训练机将验证与旧权重的一致性。
