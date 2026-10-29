# 有上限的 margin 保持：最小单因素对照

本轮只实施建议中的第一步，不同时切换纯边界损失或网络。
复用8192训练池挖掘出的同一105查询及2048 GSV开发留出；不提取新图像。

两组都是 independent，从相同零残差初始化开始：

- original：原多正样本列表损失。
- preserve：原损失 + capped保持损失（本轮preserve名称指有上限版本）。

只对冻结top1原本正确的**训练**查询施加：

    Lkeep = relu(min(stop_gradient(m0), 1.0) - m)

m=max正分数-max负分数，支持多正样本；按整个batch平均，错误查询贡献0。
权重固定1.0；cap固定1.0；不使用留出集扫描参数。margin25允许降至1，
margin0.3仍保持0.3。约束是软损失，不是推理时强制不回退的保证。
这是工程假设，不是论文方法复现或已证实提升。

两组均5epochs/70updates，batch8、lr1e-4、AdamW wd.001、gradclip1、±4残差，
相同随机顺序和重复增强。旧失败gate保持失败，新contract标记探索。
完整GSV留出用于选epoch（含epoch0），不能称作独立测试；不使用Pitts。

实现入口是独立薄封装，在进程内有界替换原runner的objective/POLICY/code指纹，
结束或异常时恢复。原runner文件不变，避免破坏已完成缓存与实验指纹。
保留last/selected两套完整分数、margin诊断、逐query记录、训练日志和成对结果。
输出目录logs/candidate_margin_cap_v1，与旧无上限实验完全隔离。

## 训练机运行

同步完成后，每条单独执行；不需要重建plan/cache/mine：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_candidate_margin_cap.py -v
python -u scripts/candidate_margin_cap.py --acknowledge-exploratory --smoke-test
python -u scripts/candidate_margin_cap.py --acknowledge-exploratory --output logs/candidate_margin_cap_v1
```

必须确认测试、smoke通过再运行下一条。smoke执行两组各一步，不写正式checkpoint。
中断恢复：

```bash
python -u scripts/candidate_margin_cap.py --acknowledge-exploratory --output logs/candidate_margin_cap_v1 --resume
```

恢复到最近完整epoch，中断轮重跑。运行使用物理GPU1，逻辑cuda:0。
本机按用户要求未运行测试或训练。

Windows首次下载小报告：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/logs/candidate_margin_cap_v1/report D:/Desktop/MyVPR/doc/candidate_margin_cap_report_v1
```

优先检查原损失组是否复现、capped是否恢复训练纠错同时避免回退、留出净纠错是否
为正。只改善loss或大margin不构成准确率收益；若仍无收益，不自动追加权重扫描。
