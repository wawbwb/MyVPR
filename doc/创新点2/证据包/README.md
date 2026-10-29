# 完整证据：无损ZIP存储

2026-10-29整理。为避免500余MiB的展开报告及重复传输件占据项目，改用以下无损ZIP。每一个归档成员均按源文件SHA256和长度核验，核验通过后才删除展开副本；没有删减逐查询预测、对照、源快照或原完成清单。

|ZIP|原展开目录|文件数|
|---|---|---:|
|scale_factorial_v1_server.zip|doc/scale_factorial_v1_server|137|
|scale_agreement_repeat_v1_server.zip|doc/scale_agreement_repeat_v1_server|87|
|scale_agreement_fusion_v1_server.zip|doc/scale_agreement_fusion_v1_server|47|
|scale_reliability_gate_v1_server.zip|doc/scale_reliability_gate_v1_server|84|
|cross_resolution_detail_v1_server.zip|doc/cross_resolution_detail_v1_server|28|
|cross_resolution_detail_v1_partial_server.zip|doc/cross_resolution_detail_v1_partial_server|21|

ZIP中保留了原目录内的相对结构；旧Markdown记载的原本机目录/旧tar包位置是历史位置，以本表为现在的存储对应关系。原训练机路径不变。完整成员SHA256与ZIP摘要见[压缩核验清单](../../历史实验归档/二次整理压缩核验.json)。

例如需要继续本地逐查询审计，可在项目根目录用以下命令恢复六格报告到原位置（不存在时再执行，不覆盖既有目录）：

```powershell
if (Test-Path -LiteralPath 'doc/scale_factorial_v1_server') { throw '目标已存在，停止以免覆盖' }
Expand-Archive -LiteralPath 'doc/创新点2/证据包/scale_factorial_v1_server.zip' -DestinationPath 'doc/scale_factorial_v1_server'
```

日常阅读只需说明目录中的最终结果，不必展开。所有ZIP仅本地保留，不自动推送。六个旧reports.tar.gz已逐成员与这些ZIP核对相同后删除，其内容没有丢失。
