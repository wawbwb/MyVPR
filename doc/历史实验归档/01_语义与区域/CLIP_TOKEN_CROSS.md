# 固定 checkpoint 的训练×推理交叉评测

目标：分开观察训练方式与推理剔除规则的影响，不训练新模型、不重新计算 CLIP。

| 训练模型 | 推理全保留 | 推理语义剔除 | 推理空间打乱 |
| --- | --- | --- | --- |
| none | 旧结果668 | 新计算 | 新计算 |
| aligned | 新计算 | 旧结果671 | 新计算 |
| shuffled | 新计算 | 新计算 | 旧结果674 |

另外保留冻结 RU 的675基线。每个格子的查询和整个数据库必须使用同一checkpoint、同一推理规则。

新脚本 scripts/eval_clip_token_cross.py 不修改旧训练/缓存代码，因此不会使已有缓存合同失效。校验原评测全部8个数组的SHA、checkpoint SHA、缓存和训练合同后，从原始分数重新生成旧对角线结果。新增6组全库描述子；共享一次DINO/RU特征提取，再经过各自BoQ与推理规则。另用5个固定图像位置核验四条原路径的描述子一致性。

## 运行

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python -m unittest discover -s tests -p test_clip_token_cross.py -v
python scripts/eval_clip_token_cross.py --device cuda:1 \
  --source doc/clip_token_drop_eval_v1 \
  --cache .cache/clearclip_token_drop_msls_v1 \
  --runs doc/clip_token_drop_train_v1 \
  --output doc/clip_token_cross_v1
```

需要训练机保留原评测大文件与三个last.pt。新增GPU推理及全库检索，不是纯CPU操作；新增数组约6.1GB，建议至少预留7GB。batch-size默认4。输出存在时拒绝覆盖；当前不支持中断续跑，若中断需保留现场并换新输出目录重跑。

Windows下载轻量报告：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/clip_token_cross_v1/report "C:\Users\zy\Desktop\MyVPR\doc\clip_token_cross_report_v1"
```

## 解释规则

- 同一行比较：固定模型参数，仅改变推理规则，观察推理剔除的影响。
- 同一列比较：固定推理规则，比较不同训练方式形成的模型。
- 每个格子都记录相对本行全保留的纠正和退化，不把另一个训练模型当作同checkpoint基线。
- 如果语义训练模型在全保留时较好、启用语义剔除时下降，支持其推理剔除造成性能损失的解释，但推理分布变化也是其中一部分。
- 如果空间打乱训练模型在全保留时仍较好，与训练期正则化解释相容，不能据单seed结果证明普遍规律。
- 不从9格中事后挑最高值作为独立验证的新方法结果。此次是机制分析，未消除训练×推理交互，也不构成像素因果归因。

本机2项汇总测试与语法检查通过；PyTorch/GPU交叉推理未在本机执行，继续使用训练机原VPR环境。原始大文件不修改。
