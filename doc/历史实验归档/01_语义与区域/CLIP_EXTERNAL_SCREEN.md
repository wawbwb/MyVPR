# CLIP 外部动态类别检索实验

目标：验证 CLIP 自动定位的明显动态类别经过抑制后，是否提高 VPR。无需人工标注，不以解决自车引擎盖识别作为前置条件。

固定配置：复用旧 CLIP 16 个 112×112 窗口余弦；动态类别为汽车、卡车、公交车、摩托车、自行车、行人。动态最大余弦仅与静态最大余弦比较，完全忽略 ego 余弦。每个窗口分数为 max(2*sigmoid((dynamic-static)/0.05)-1,0)，重叠平均回填 20×20 网格。没有新增提示词、权重或阈值搜索。

三组固定比较：baseline、aligned_input、shuffle_input。后两组将标准化输入乘以 (1−0.5×mask)，即向归一化均值颜色软混合；不是背景修复。shuffle 使用原有固定种子逐行打乱，精确保留每一行分数分布，但可能仍与动态物体重叠，因此是有限的空间对照。

范围：740 个 MSLS-val 查询，对原完整 18,871 张数据库检索。仅改变查询；本轮不回答查询和数据库两侧共同抑制是否有效。CLIP 无新前向，RU 需每个查询运行基线和两种输入干预，共约 2,220 次查询前向；另重算 3 张数据库哨兵检查映射，其余数据库复用缓存。无需训练。

## 训练机执行

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python -m unittest discover -s tests -p test_clip_external_screen.py -v
python scripts/clip_external_screen.py prepare \
  --old-cache .cache/clip_dynamic_msls_v1 \
  --output .cache/clip_external_positive_v1
python scripts/clip_external_screen.py evaluate \
  --cache .cache/clip_external_positive_v1 \
  --source doc/visual_pair_msls_hard_mix_v2 \
  --device cuda:1 \
  --output doc/clip_external_screen_v1
```

prepare 是 CPU 缓存重映射；evaluate 才运行 RU。两个阶段都拒绝覆盖已有目录；成功后无需重复执行。如中断，保留已有目录，用新的输出目录重跑 evaluate（partial_queries 用于排查，不提供续跑）。

## 读取结果

summary.json 提供三组正确数、相对基线的纠正/退化 query、aligned 与 shuffle 的独有正确 query。query_outcomes.json/csv 保留逐查询排名与 margin。scores.npy 和 descriptors.npy 合计约 277 MB，保留供核验；provenance.json 记录输入身份和参数；index.html 是固定抽取的基线错误预览。

首先比较 aligned 与基线的净变化，再看是否优于 shuffle。相对基线提升而与 shuffle 接近，不足以归因于语义定位。相同 MSLS 被多次用于开发，正结果仍需要独立数据验证；负结果仅否定本轮固定实现的收益，不代表总体思路无效。

不需要下载 CLIP 缓存或新增标注。Windows PowerShell 下载评测输出：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/clip_external_screen_v1 "C:\Users\zy\Desktop\MyVPR\doc\"
```

## 本机验证

CPU 测试检查 ego 余弦完全不影响输出、动态与静态平局不抑制、静态胜出不抑制、单窗口空间支持、打乱分布/确定性以及非法余弦拒绝。旧缓存完整运行 prepare 成功。Windows 检查旧代码身份时允许仅 CRLF/LF 换行差异，其他源码变化仍拒绝。未在本机运行 GPU 检索，训练机执行时会核验原基线 675/740、查询缓存及数据库哨兵一致性。
