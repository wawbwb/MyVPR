# Visual pair V1/V2 停止归档

状态：COMPLETED / NO VALIDATED GAIN / STOP。来源为用户提供的训练机日志与诊断输出，未在本机重跑或独立校验原始数组。保留全部代码、缓存、checkpoint 和逐查询记录。

## V1

3 epochs 完成。GSV selection 三轮均 990/1000，calibration 从 990 提升至 991，选择 alpha=0.01。MSLS 实际启用该修正后仍为 675/740，纠错0、破坏0，top-1 候选变化4次。

65 个 RU 错误中，40 个正确答案在 top-20 内，25 个不可达；候选 oracle 为715/740。40 个可达错误中，按 RU 分数选出的最佳正样本只有5个获得高于错误 top-1 的修正，1个超过原分差但未战胜全部候选。不能将5个视作所有正样本的严格纠错上限。

tanh 残差饱和比例43.42%。关闭局部边明显改变打分，但没有检索收益；该操作为分布外敏感性检查，不是重新训练的公平消融。16个查询原生缓存重算误差0，只证明抽样推理一致。

## V2

复用V1两视图，构建同城双向候选；uniform和hard_mix每epoch均20,000次更新，完成3轮。保持模型与校准权重范围不变。

| 组别 | selection epoch | 纠错 | 破坏 | 净变化 |
| --- | ---: | ---: | ---: | ---: |
| uniform | 1/2/3 | 0 | 0 | 0 |
| hard_mix | 1 | 0 | 0 | 0 |
| hard_mix | 2 | 1 | 4 | -3 |
| hard_mix | 3 | 2 | 7 | -5 |

选模集1972/1982基线正确，仅19个困难查询、8个可达错误；校准集1976/1984基线正确，仅18个困难查询、8个可达错误。双向查询有同地点相关性，增加行数不等于增加独立难例。

uniform的四个校准alpha均无净收益，平局取0。hard_mix在所有非零alpha下均纠错0、破坏1，因此也取0。两组状态均NO_GAIN。

两组MSLS均675/740，R@5=95.1351%、R@10=96.0811%，但此时alpha=0，实际退回RU。因此不能把它们描述成“启用V2修正后跨域无增益”或“新头有效且无损”；正确解释是校准不支持部署修正。

## 停止决定

停止当前轻量 MNN/set-head 的继续训练、alpha扫描与语义分支。已有证据否定当前低成本实现的继续准入，不是否定所有成对检索。困难采样未改善净收益，而且新验证集仍过于容易；不能仅再提高采样比例开新实验。

保留训练机：

- `logs/visual_pair/{seed42_v1,uniform_v2,hard_mix_v2}` 下的权重和run.json。
- `doc/visual_pair_msls_{seed42_v1,uniform_v2,hard_mix_v2}` 下的summary.json及per_query.npz。
- `.cache/visual_pair/{gsv_v1,gsv_hard_v2}`，暂不清理。

下一步为独立的 [官方 Pair-VPR 推理复现](PAIR_VPR_OFFICIAL_REPRODUCTION.md)，不以当前小头冒充论文方法，不启动从头训练。
