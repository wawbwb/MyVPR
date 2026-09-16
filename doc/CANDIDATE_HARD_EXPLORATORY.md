# 35-error GSV 探索性对照（2026-09-16）

原 candidate_hard_mining_v1 的 gate=false 保持不变：训练错误地点 35 < 40。
训练池 8192 查询：7911 正确 / 35 可达错误 / 246 不可达。
留出 2048：1994 正确 / 15 可达错误 / 39 不可达。
范围内可纠错训练33、留出13；训练错误涉及16城市，留出9城市。
105条选择为35错误+35低margin正确+35普通正确；三组初始损失和分别
131.8545、8.04644、0.000858。等数量不等监督强度；普通正确34条margin>8，
不能把它们视为有力的防回退监督。本次保持原选择以避免多因素同时改变。

这是查看挖掘结果后由用户授权的小型探索，不是预登记门槛通过，不能作确认性结论。
原模型、±4残差、5epochs、lr、batch、初始化、增强、选择规则全部保持。
只跑independent/set，epoch0也参与选模。完整GSV留出用于选模，非独立测试集。
只有70次更新/组（ceil(105/8)*5）；若拟合不足，不能据此证明表达能力无效。
不提取新图像、不使用Pitts、不删除缓存、不改原gate和源代码指纹。

入口独立为 scripts/candidate_hard_exploratory.py，严格验证旧代码hash、完成清单、
plan/official身份，并从原缓存逐行重新校验GT、margin及挖掘记录。
只有已审查的35/35/35且唯一失败项为train_error_places的报告可进入。
--acknowledge-failed-gate把知情探索记入新contract，不是通用忽略检查开关。

逐epoch报告：范围内错误、范围阻塞错误、低margin正确、普通正确、完整留出。
corrected/regressed始终为原split query_index，不是子集行号。
最终同时保存选中checkpoint的训练与留出分数，以及set-vs-independent分组比较。
若最佳是epoch0，必须结合history查看训练是否曾拟合而留出回退，不能只看最终相同。
--resume从最近完整epoch恢复，未保存的中断轮重跑。不要并发写同一输出。

## 同步后，在训练机依次执行

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
export CUDA_VISIBLE_DEVICES=1
python -m unittest discover -s tests -p test_candidate_hard_exploratory.py -v
python -m unittest discover -s tests -p test_candidate_set_torch.py -v
python -u scripts/candidate_hard_exploratory.py --acknowledge-failed-gate --output logs/candidate_hard_exploratory_v1
```

不需要重新prepare/cache/mine，不需要RU_CKPT、SAM或官方权重；这里仅读缓存训练小头。
启动核验是CPU/磁盘I/O，随后训练使用物理GPU1（逻辑cuda:0）。

中断恢复：

```bash
python -u scripts/candidate_hard_exploratory.py --acknowledge-failed-gate --output logs/candidate_hard_exploratory_v1 --resume
```

完成后Windows下载（report有独立完成清单，无需checkpoint）：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/logs/candidate_hard_exploratory_v1/report D:/Desktop/MyVPR/doc/candidate_hard_exploratory_report_v1
```

目标目录首次下载前不应已存在，避免scp产生额外嵌套层。本机未执行测试或模型运行。
