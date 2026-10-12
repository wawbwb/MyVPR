# HCEN原版BoQ基座对照

## 为何补这一轮

RU不是官方BoQ模型，而是本项目在DINOv2-BoQ上增加可重复性/独特性监督空间门控后的检查点。三seed无辅助HCEN的跨层相对同参数仅L12组，在Pitts净增10/12/11个、MSLS净增1/3/2个；但GSV evaluation均净少4。低辅助跨层在Pitts的增量5/3/-11不一致。因此先关闭辅助及排序，不增加权重或预算，检验多层输入是否能迁移到真正无RU的BoQ模型。

已完成三seed共12组训练及评测，全部封存校验通过，汇总为`logs/hcen_multiseed_v1/aggregate/summary.json`。收益是固定样本下初始化复核，不是独立测试或Mamba模块必要性证明。三seed GSV evaluation无辅助的共同纠错为4647，共同退化3355/4043/4490/4588/5955；Pitts跨层有20个共同纠错和13个共同退化，既有正信号也有重复代价。

## 原版基座的身份

固定使用历史原版训练的`logs/dinov2_vitb14/BoQ/version_8/checkpoints/epoch(29)_step(46890)_R1[0.9122]_R5[0.9568].ckpt`：相同最佳R@1保存模型中固定该版本最早轮，不在本轮按新结果重选模型。

SHA256：`dba5770cd3351d5354decc2bd385a831107591cfa392a0f0d4e66e95bf0c316d`。

原检查点配置为DINOv2 ViT-B/14，最后2块在原训练可调；BoQ投影384、64 queries、2 layers、row_dim32。GSV训练280、验证280、P40/K4、seed42、40轮，MultiSimilarity。`distillation.enabled=false`，231个保存张量仅属于`backbone.*`和`aggregator.*`，无RU、spatial attention、CLIP或附加颈部。

这是**原论文架构的本项目训练版本**，不是作者发布权重或声称完全复现论文训练设置。不得用RU检查点关闭门控冒充此基座。加载校验固定文件哈希、配置、所有参数键及每个参数值；任何隐藏模块或被遗漏张量都停止。

## 三组与固定预算

| 组 | 内容 | 本轮可训练 |
|---|---|---|
| baseline | 真正原版DINOv2-BoQ | 无，固定参照 |
| last_only | 原版基座+L12/L12/L12的同容量HCEN | 仅891,268个颈部参数 |
| hierarchy | 原版基座+L8/L10/L12的HCEN | 同上 |

本轮seed42：先两组各8步机械检查，再全部重新初始化各512步，固定末步，不做最佳验证轮选择。保持之前的颈部结构、10%特征残差上限、FP32、AdamW lr1e-4/wd0/clip1、microbatch4。原DINO及BoQ全部冻结。两组辅助头都存在以保持原实现和参数量，但损失乘数为0、头权重必须不变；排序损失完全关闭。

为控制训练样本来源，继续使用先前8192个GSV训练地点、固定每地点4图、4096个步长2锚点及其固定硬负地点。**硬负索引由此前RU训练图库挖掘，不重新挖掘；这被明确保留为历史依赖，不可宣称训练流程完全与RU无关。** 只读取索引及地点身份，不读取旧RU描述子作为模型输入；所有实际特征来自新基座。

比较重点是同一原版基座、同样固定日程下`hierarchy vs last_only`，而不只是`hierarchy vs baseline`。若有正信号，再单独预注册原版BoQ挖掘和多seed，不在这轮混入新采样政策。

## 基线优先及评测防错

1. 固定原版BoQ先以其原生直接forward独立完成GSV6144查询/10240图库、MSLS740/18871、Pitts7608/10000评测并封存。零初始化两种HCEN必须逐字节等于同图直接forward。
2. GSV沿用原PIL预处理及25米负例排除/指定正样本协议；MSLS/Pitts沿用张量bicubic280/FP32/microbatch4。不添加PCA、局部拼接或重排序。
3. MSLS历史R@1对应675个正确查询；若不复现，写诊断并停止训练，不能自动改阈值。Pitts和GSV新基线独立测量，不硬套RU的7160/6030。
4. 每组末步评测同图像冻结原版基线和学生，要求完整逐查询baseline预测及outcome与独立封存基线相同；结果变动不能被误当成HCEN收益。
5. 汇总三个组对原版基线的纠错/退化、两种颈部的成对差异、GSVcal/eval分区和三套标准集。smoke只查机械，不按性能决定启动哪个组。

两种基座训练史、原增广和已学到的骨干权重不同，所以此轮回答“HCEN能否在无RU模型基座上工作”，**不是只移除RU门控的因果消融**。所有数据集此前已曝光，仍是探索性短预算，不是稳定泛化或方法创新证明。

## 执行与结果位置

保留所有旧代码、合同和模型，新文件为`src/hcen_plain_protocol.py`、`scripts/hcen_plain_base.py`、`scripts/run_hcen_plain_base.sh`、`tests/test_hcen_plain_base.py`。只在训练机测试，现有VPR环境、物理卡1、共享实验锁；无新增数据集、环境、模型下载或大型特征缓存。

```bash
cd /home/wt/workspace/OpenVPRLab
bash scripts/run_hcen_plain_base.sh all
```

依序：测试→`baseline`三套独立评测→两组8步smoke→匹配机械汇总→两组512步+评测→正式汇总。支持`baseline`/`smoke`/`screen`分阶段；重复同一命令按原合同恢复，不覆盖封存结果，每16步保存Adam/RNG。

输出`logs/hcen_plain_base_v1/baseline/`、`last_only_screen/`、`hierarchy_screen/`和`compare_screen/summary.json`。`PLAIN_BASE_SCREEN_COMPLETE_NOT_STABLE_GAIN_PROOF`仅表示筛查完成。当前新基座收益待实测。

### 原版无门控张量布局检查

初次启动在训练前因逐字节零起点检查中止，尚未优化任何参数。只读诊断发现：原生backbone最终特征是NCHW contiguous，hook捕获及加零残差后数值完全相同但内存布局不同，原版无门控BoQ的CUDA卷积路径因此带来最大9.62e-8描述子差异；在两路读出前恢复原生NCHW contiguous后，实际GPU同批次误差为0。RU门控原先生成新张量，不能把其布局默认当成原版布局。

新PlainHCENVPR仅在读出前canonicalize布局，不新增权重或改变颈部结构；特征捕获输入也canonicalize，两对照同样处理。保留严格逐字节零起点及逐查询基线检查，不放宽阈值。失败启动目录将原样留存，修正后的实验使用新合同重新开始。

2026-10-12执行记录：训练机118项相关测试全部通过，本机未测试；修正后的真实GPU严格零起点检查通过，已进入原版BoQ完整GSV独立评测。后台PID202112，日志`logs/hcen_plain_base_v1/run_20261012_134537.log`。第一次未训练的失败预检原样移至`logs/hcen_plain_base_v1_failed_layout_20261012_134537/`，没有删除任何数据或检查点。基线全部评测、两组smoke和512步训练结果仍待生成。
