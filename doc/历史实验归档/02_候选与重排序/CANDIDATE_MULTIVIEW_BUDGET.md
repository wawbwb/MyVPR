# 多视图候选：等预算 CPU 审计

上一轮训练侧 full20/full80/union 可达数为7946/8039/8013，开发侧为2009/2024/2021。
裁剪独有补回train15、dev1，不能证明实际重排收益。
本轮保留所有旧代码、缓存和结果，只读取已保存full80排序，无需全局相似度重算。

五组：full20、固定full44、逐查询fullM、原union、full80。
44来自已观察训练侧平均44.1622四舍五入；不按开发侧结果选择。
M是原union去重后候选数，fullM直接取已有full80的前M项，不使用GT确定M。
这是事后探索性预算对照，并非之前预注册的确认实验。

逐查询等候选预算不等于端到端等耗时。fullM要知道原union大小，因此首先是诊断对照，
不能将其描述为无需裁剪计算即可独立部署的自适应预算算法。
固定full44可以独立执行，但只是近似匹配训练侧平均预算，开发侧不再调K。

校验完成清单/代码/模型/plan/cache身份、原查询顺序、排名前缀、候选ID范围及去重并集，
重新计算上一轮per_query/summary并要求完全一致。旧数据只读，CPU磁盘校验会花一些时间。
输出各组可达数、候选总量、新增配对数，以及union相对三个对照独有补回和遗漏的查询ID。
不调用torch模型、不读图片、不跑配对、不训练，不自动推进下一阶段。
本机按用户要求未运行测试；新增8项单元测试交训练机运行。

## 训练机

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -m unittest discover -s tests -p test_candidate_multiview_budget.py -v
```

通过后：

```bash
python -u scripts/audit_candidate_multiview_budget.py --output doc/candidate_multiview_budget_v1
```

默认读取 doc/candidate_multiview_train_v1、doc/candidate_multiview_dev_v1，
doc/candidate_hard_plan_v1、.cache/candidate_hard_v1。支持显式路径参数。
已有输出拒绝覆盖；中断后使用新输出目录。本阶段不产生大型中间缓存。

Windows下载完整小型报告：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/candidate_multiview_budget_v1 D:/Desktop/MyVPR/doc/
```

若裁剪不优于等预算整图，优先更简单整图扩展；若有互补性，再决定实际Pair重排是否值得，
不得只对已知错误查询评测。候选覆盖始终不等于实际R@1。dev已反复查看，非独立测试。
