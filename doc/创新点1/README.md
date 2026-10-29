# 创新点1：代码与结果归档

论文式说明见[创新点1.md](方法说明.md)。本附件对应2026-10-15完成的官方BoQ权重末层锚定条件补充实验，编号不代表已经证明原创或获得专利。

## 快速入口

| 内容 | 文件 |
|---|---|
| 新模块及A/B/C路由 | [conditional_complement.py](code/src/models/conditional_complement.py) |
| 原渐进卷积结构 | [hcen_ablation.py](code/src/models/hcen_ablation.py)、[hcen_vpr.py](code/src/models/hcen_vpr.py) |
| 原RMS残差公式 | [conv_residual_ablation.py](code/src/models/conv_residual_ablation.py) |
| 官方权重严格加载 | [official_boq.py](code/src/models/official_boq.py) |
| 新模块训练循环 | [训练脚本](code/scripts/conditional_complement.py) |
| 实验协议、成对统计、固定门槛 | [conditional_complement_protocol.py](code/src/conditional_complement_protocol.py) |
| 回归测试 | [test_conditional_complement.py](code/tests/test_conditional_complement.py) |
| 独立归档恢复工具 | [archive_conditional_complement.py](code/scripts/archive_conditional_complement.py) |
| 指标、对照和判定 | [summary.json](results/summary.json) |
| 逐查询R@1正误 | [query_outcomes.json](results/query_outcomes.json) |
| 模型和结果来源 | [provenance.json](results/provenance.json) |

## 结果口径

322输入：官方MSLS694/740、Pitts7264/7608；最终C为695/740、7276/7608，分别+1、+12。B与C的逐查询R@1正误相同；不能宣称跨层有独立收益。280输入C比官方少正确5、2张。完整指标和限制见正文。

## 代码快照与运行

`code/`保持项目相对路径，包含训练合同列明的源码和静态导入项目依赖。`code_manifest.json`区分封存训练来源、补充依赖快照和事后归档工具。需在已有OpenVPRLab工程、VPR环境、GSV/MSLS/Pitts数据与官方DINO/BoQ离线缓存中使用；不包含图像、外部大模型、完整Python环境或全部历史实验结果。

不要把`code/`直接当作安装完成的新项目，也不要覆盖正在执行的旧工程。完整复现时应使用隔离工作副本，并检查合同要求的历史训练缓存、Warm检查点和源码哈希。已训练A/B/C的检查点没有复制到doc，其绝对路径及文件哈希均在`provenance.json`中。

**保留的已知问题：** 训练脚本末尾的旧`aggregate`会因重复`checks`字段报错。该脚本是执行时原样快照，不应为使报告成功而修改封存来源。若原训练已经完成，不要重训；可在工程根目录用独立工具重新导出：

```bash
python -u scripts/archive_conditional_complement.py \
  --source logs/conditional_complement_v1 \
  --reference logs/conv_official_boq_v1 \
  --warm-start logs/conv_official_layers_v1/last_only \
  --output doc/innovation1_export_new
```

输出必须为新的doc子目录，工具只读原实验，重用原比较函数，区分机械检查与收益门槛。该命令不会启动训练或图像提取。本次实际导出源为训练机`doc/innovation1_official_archive_20261015_v2`。

如需从头复现三组，应另设`logs/conditional_complement_replay_<版本>`，依次运行三组smoke、`aggregate --stage smoke`、三组正式训练，最后使用独立归档工具，不调用有已知字段冲突的正式`aggregate`。每组累计口径为继承2048步+新增2048步；smoke8步不继承。本文归档操作没有执行新的训练。

## 完整性与Git边界

下载的代码与机器结果受`archive_manifest.json`保护。清单不包含它自身，也不包含下载后撰写的本README和论文正文；清单中的`complete`表示归档导出完整，不代表模型收益门槛通过。原训练的完成标记另存于`results/original_completions.json`，其中引用的部分大文件仅在训练机保留，不应把它当本地文件清单使用。

本次附件约3.11 MiB，包含94份代码、依赖及环境/许可证文件、8份结果文件和两份清单。2026-10-15下载后，清单所覆盖的103个文件全部通过SHA256逐文件校验，论文与本README的本地链接检查通过；未在本机运行模型测试。

沿用项目现有规则：根目录`doc/创新点1.md`可作为精选文档追踪，附件目录默认仍受`/doc/*`忽略规则保护。本次不修改`.gitignore`，不自动`git add/commit/push`。如将来决定提交附件，先按其体积和许可证单独确认，不上传私有ReID稿件、模型或数据集。
