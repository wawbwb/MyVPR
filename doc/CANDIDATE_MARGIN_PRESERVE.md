# Independent margin 保持单因素对照

2026-09-16，基于最终轮诊断提出的新探索：训练33条范围内错误全部margin改善，
但留出13条只有5条改善、8条变差；两组留出改分去query均值相关性0.990。
不继续增大残差、不叠加set模块，先检验正确查询保持约束。

## 固定设计

- original / preserve 均为同一 independent 模型，分别从相同零残差初始化开始，
  **不是从上一轮epoch5继续训练**。
- 同一105训练查询（35错误、35近边界正确、35普通正确），相同随机顺序、重复增强、
  lr=1e-4、batch8、5epochs、AdamW wd=.001、clip1、±4tanh。每组70次更新。
- 原损失：多正样本列表损失 softplus(logZneg-logZpos)。
- 唯一因素：preserve 加权重1.0的保持损失；original权重0。没有权重扫描。

对每条**训练查询**，冻结模型top1按GT正确时：

    m0 = max_positive(frozen_score) - max_negative(frozen_score)
    m  = max_positive(current_score) - max_negative(current_score)
    Lkeep = relu(stop_gradient(m0) - m)

冻结top1错误的查询Lkeep=0；按整个batch取平均，不按正确查询数重新归一化。
margin增加不处罚，共同分数平移不处罚，允许最强正/负候选身份变化。
权重1只是本轮事先固定的工程选择，不宣称最优。零初始化时保持损失与其梯度为0，
初始第一步可以与原训练一致；约束在正确margin开始下降后起作用。
与“普通正确查询原始排序损失趋近0”不同，线性hinge在margin下降时提供显式梯度。
这也可能与纠错梯度冲突，造成模型不改变，必须检查训练纠错数和留出净收益。

只使用训练标签计算保持损失；不按留出错误编号采样、不使用Pitts。
GSV完整2048留出仍用于epoch选择，包括epoch0，非独立测试。
原failed gate保留，新contract明确探索。旧代码与缓存指纹不变。

## 自动输出

每epoch记录两项loss、总loss、grad norm、正确样本数；train/dev分组纠错回退、
margin变化及残差饱和度。最终同时导出last和selected模型分数与逐query记录，
无需再单独运行一次最终轮诊断。preserve_vs_original.json比较两组相同query。
如果只减少margin下降却没有提升纠错，不能宣称VPR性能改善。
epoch原子checkpoint包含optimizer和RNG，--resume恢复最近完整epoch，中断轮重跑。

## 训练机运行（每条单独执行）

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_candidate_margin_preserve.py -v
python -u scripts/candidate_margin_preserve.py --acknowledge-exploratory --smoke-test
```

smoke复用真实缓存，分别执行两组一次有限梯度更新，不写checkpoint或正式输出。
通过后：

```bash
python -u scripts/candidate_margin_preserve.py --acknowledge-exploratory --output logs/candidate_margin_preserve_v1
```

中断恢复：

```bash
python -u scripts/candidate_margin_preserve.py --acknowledge-exploratory --output logs/candidate_margin_preserve_v1 --resume
```

无需重建plan/cache/mine，无需官方模型、图片、RU_CKPT。启动核验为CPU/磁盘I/O。
GPU使用物理卡1，逻辑cuda:0。旧结果不覆盖。本机按要求未执行测试或模型。

Windows首次下载小报告：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/logs/candidate_margin_preserve_v1/report D:/Desktop/MyVPR/doc/candidate_margin_preserve_report_v1
```

目标目录首次下载前不要已存在，避免多出report嵌套层。report自带独立完成清单，
不下载checkpoint也能核验。先看summary和preserve_vs_original，再看两组history。
