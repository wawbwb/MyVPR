# HCEN模块必要性消融

## 已确认的边界

真正原版BoQ基座三seed、固定512步下，跨层对同容量仅L12在Pitts净增17/13/15（平均15±2个，约+0.197个百分点），相对原版基线净增37/31/33（平均约+0.443个百分点）。MSLS跨层对仅L12为-3/0/-2，GSV整体-1/-1/+1。三seed所有模型及汇总完成校验通过。

这支持当前固定样本与预算下Pitts上的跨层正信号，不支持跨域普遍收益，也没有证明Mamba/SASF是必要的。完整模型不能只对比更弱基线，而需要与更简单的跨层方法比较。

## 本轮四种跨层融合

所有组输入均L8/L10/L12、同样128维投影、相同最终零初始化升维层、10%残差界、固定原版DINOv2-BoQ读出；辅助和排序均关闭。

| 组 | 修改 | 注册可训练参数 | 扣除原有零权重辅助头 |
|---|---|---:|---:|
| full（已有） | 完整两级MFM+选择性扫描+SASF | 891268 | 825732 |
| linear | 两级逐位置低/高线性融合，去掉空间、扫描路径 | 594432 | 528896 |
| conv | 只把选择性扫描换成局部2D门控卷积，其他MFM及SASF保留 | 892356 | 826820 |
| no_sasf | 只将两个SASF替换为Identity，其余保持完整 | 817024 | 751488 |

linear保留原低/高线性权重，用`.5*(low+high)+.1*(linear_low+linear_high)`逐级融合。conv保留扫描模块原输入/输出投影，输出仍零初始化；用inner32的3×3 depthwise卷积、32→32线性和SiLU门控替代扫描。conv只比full多1088参数（约0.12%），是近似容量匹配的局部/扫描替换，而不是精确参数/FLOP匹配。

所有新模型先构造原完整颈部，再移除/替换模块，以保留共有参数的原历史初始化。新替换层用独立CPU RNG作用域，不改变共有权重或主RNG。外层LN/投影、升维及原辅助头的初始化必须等于历史full；重建full参数SHA必须复现对应seed封存初始SHA。没有用已训练full权重初始化新组。

原有两辅助头65,536参数均保留以沿用训练循环，但权重0且wd0，不发生有效更新；不是为了凑参数新建dummy层。linear/no_sasf的容量减少会影响解释，不能将full胜出全部归因于独特机制。

## 固定预算及参照

seed42/43/44，每个seed三种新组各8步真实图像smoke；全部机械检查通过后，九种新运行各从原版检查点重新初始化512步。full及仅L12三个seed直接复用已经封存的旧结果，保留其全部代码、合同和模型，不重复训练或挑选最高分。

保持固定训练图像、4视图、8192地点及原RU-mined硬负索引、相同4096锚点日程、FP32、AdamW lr1e-4/wd0/clip1、微批4和预处理。新组末步独立重新提取原版基线，逐查询预测必须与封存的直接forward基线相同；零起点仍严格逐字节匹配。无新数据、模型下载或大特征缓存。

新的机械检查按实际模块区分：linear两级线性融合有梯度；conv两级替代mixer隐层有梯度；no_sasf两级选择性参数有梯度。不能因为conv/linear没有SSM就误判失败，也不能只证明最终升维动了就说替代模块被训练。冻结基座、原辅助头不变、512优化器时钟及保存恢复一致均需校验。

## 分析问题和不能下的结论

- full对linear：复杂方法是否比简单逐位置跨层融合更值得使用？容量不同，若full不胜出，不能继续强调复杂性必要。
- full对conv：近似同容量下，选择性扫描是否比局部卷积提供额外价值？相同步数不是实测算力/速度匹配。
- full对no_sasf：SASF在此配置是否带来增量？参数数量也改变，负结果不能单独证明SASF原创或无效。
- full对仅L12：仍保留已测结构对照，避免把所有新增训练收益算作跨层贡献。

三seed均报告GSVcal/eval/all、MSLS、Pitts的成对纠错/退化、均值及样本标准差。不得只挑Pitts最好组/seed，不得把重复查询当三倍独立样本。不修改预算、残差系数、损失来追分，不自动进入更长训练。完成状态不是模块必要性、原创性或稳定收益PASS。

## 运行及GPU队列

新增`src/models/hcen_ablation.py`、`src/hcen_module_protocol.py`、`scripts/hcen_module_ablation.py`、`scripts/run_hcen_module_ablation.sh`及`tests/test_hcen_module_ablation.py`，仅在训练机现有VPR环境测试。

```bash
cd /home/wt/workspace/OpenVPRLab
bash scripts/run_hcen_module_ablation.sh all
```

卡1当前有其他用户进程占约15GB，卡0也被占用。队列不会kill、重启或干预它们：先CPU回归测试，再在后台每15秒读取卡1状态；连续三次显存占用≤1500MiB且利用率≤5%后才启动smoke。等待期间不初始化或预留CUDA，状态写`logs/hcen_module_ablation_v1/gpu_wait.json`。共享实验锁仍有效。

结果位置`logs/hcen_module_ablation_v1/seed{42,43,44}/{linear,conv,no_sasf}_{smoke,screen}/`，每seed比较`compare_screen/summary.json`，最终`aggregate/summary.json`。同合同支持续跑，不覆盖旧结果。新增九组仍待执行/完成。

2026-10-12执行记录：训练机CPU上141项相关测试全部通过，本机未测试；验证了实际参数数量、各替代模块梯度、共用外层权重一致，以及CPU重建full颈部与三seed历史真实初始参数SHA256相同。后台等待队列PID667769，日志`logs/hcen_module_ablation_v1/run_20261012_174929.log`。启动核查时`gpu_wait.json`为`waiting_for_gpu1`，卡1占用15307MiB，训练进度文件为0；未新增CUDA进程，未操作其他用户任务。真实图像GPU smoke和九组正式训练均尚未开始，资源释放后脚本自动推进。
