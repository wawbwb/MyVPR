# GSV候选头训练监督审计

日期：2026-09-15。新增独立CPU脚本；不训练、不提取特征、不修改旧生成代码或合同。
本机未执行测试或模型，训练机先执行下列测试。

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -m unittest tests.test_audit_candidate_training -v
python -u scripts/audit_candidate_training.py --cache .cache/candidate_set_v1/train --plan doc/candidate_set_plan_v1 --output doc/candidate_training_audit_v1
```

如果tests目录不支持模块导入，可用等价命令：

```bash
python -m unittest discover -s tests -p test_audit_candidate_training.py -v
```

审计会完整核验旧completed清单及代码指纹，再逐分片校验SHA、候选顺序、标签、特征形状。
无需图像、权重或GPU。哈希验证会读取缓存文件，耗时取决于磁盘，非零I/O预算。
只保留逐query的小型分数/身份记录，不复制1536维pair evidence。

输出summary.json、per_query.json、provenance.json、completed.json。复现冻结global和pair的
R@1/5/10/20，报告不可达、全正、可达正确、可达错误；global是全库top20的前缀，pair是
这个候选集内的排序，不能当成全库pair搜索。

逐query记录margin（最佳正−最佳负）、正样本数、最佳正排名与冻结多正list loss。
不可达/全正不进入损失统计，不将不可达误记为高损失负样本。最难1%/5%/10%使用有效查询
数作分母、向上取整，报告损失份额及城市/地点覆盖。另给出loss有效数量(sum loss)^2/sum(loss²)，
该数不是独立样本量或统计显著性。负样本softmax总质量只是logit梯度代理，不是网络参数梯度。

±4*tanh允许单对正负margin变化严格小于8，因此当前错误gap≥8记为范围阻挡；gap<8只代表
理想独立残差在数学范围上允许翻转，不保证共享网络能学到。稳定同分排序沿用候选原序。

判断顺序：训练错误太少或集中→审查训练任务覆盖；训练困难充分→再比较训练后修复情况与
开发迁移。此脚本不加载训练头，不声称已经区分优化失败与跨域失败。不会自动给新门槛或
按Pitts错误挖训练样本。本轮不添加Pitts对照，先获得缺失的GSV监督证据。

Windows下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_training_audit_v1 D:/Desktop/MyVPR/doc/
```

当前五组结果仍为frozen985、independent986、set/density/consistency/competition985（Pitts1024开发查询），
没有证明集合交互提升检索。本审计待运行，不预判失败原因，不重新启动语义路线。
