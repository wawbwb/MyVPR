# 冻结浅层头：GSV剩余1664查询扩展

保持原先导选定head.pt、2层CLS定义、12个保留候选不变，不调用训练函数。
原dev2048扣除128校准、256原评估，余下1664全部加入，按原索引排序；
不按后排赢家或GT挑样本。先锁定旧完成清单、head、索引和代码哈希，再采集。

先复现原256查询预测及汇总，输出唯一丢失赢家案例的原查询ID、排名、
旧赢家/替代赢家是否正确。相同正确数不自动意味着相同逐查询结果。
新增1664主要报告，与原256描述结果分开；128校准不混入评估。
仍采用原先导门槛：>=99%赢家保留，>=10后排赢家，后排保留>=90%；
这只是可行性门槛，无显著性/泛化/加速保证。逐查询报告相对固定20与44纠错退化。

旧模型和缓存不改；输出.cache/wppr_extension_v1，原子分片支持续跑。
仍完整运行教师收集第二层特征，不是部署早退；1664×44×双向=146432方向前向。
新增CLS约429MiB，dense仅64条LRU。报告doc/wppr_extension_v1，原始缓存不进Git。

训练机：

```bash
python -m pytest -q tests/test_wppr_extension.py tests/test_wppr_pilot.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/wppr_extension.py
```

此扩展共享已历史使用的GSV dev参考库，不是独立测试；Pitts不用于本轮选择。
