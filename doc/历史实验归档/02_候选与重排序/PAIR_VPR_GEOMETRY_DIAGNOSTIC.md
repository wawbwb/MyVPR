# 已保存 MNN 的空间一致性诊断

输入：`doc/pairvpr_local_matches_v1`，10 个查询、58 对已保存 encoder MNN。输出不改变官方排名，不声称 R@1 增益。

使用 `scripts/audit_pairvpr_match_geometry.py`，仅依赖现有 NumPy 和 Python 标准库，无需 GPU、OpenCV、模型、原图或新环境。按照用户要求未在本机运行测试，需训练机运行验证。

## 固定分析方法

- 坐标是 23x23 网格的 patch 中心，以 patch 为单位。
- 所有 MNN 一视同仁参与拟合，不按余弦或 GT 选点。固定 256 次三点仿射 RANSAC，最多两轮最小二乘细化。
- 同时检查正向、反向映射残差；两者都不超过 1.5 patch（322 像素输入下 21 像素）才计为内点。该阈值仅是预先固定的描述性参数，不是经过验证的定位容差。
- 报告内点数、比例、双端 4x4 分区覆盖率、双端包围框面积。面积大不保证真实对应，数量多也不保证地点正确。
- 每对做 5 次目标坐标排列打乱，再使用同样的拟合流程。保留端点的空间分布与匹配数，破坏对应关系。报告全部对照内点数、均值及真实值减均值；不计算显著性或通过阈值。
- summary 中的主要 GT 仍按原 Pair-VPR 分数选择，同时列出全部 GT 和已选非 GT，不按几何结果挑“最好看”的正样本。成功查询的 top1 和最佳 GT 可能是同一对，应同时检查 all_non_gt。

仿射模型不能覆盖任意三维场景、强透视和视角变化。全局上下文化的 token 也不是精确局部关键点。因此，本脚本只能判断“是否出现值得后续验证的空间一致性信号”，不能证明语义有效或给出完整候选重排效果。

## 操作

本机 PowerShell：

```powershell
scp D:\Desktop\MyVPR\scripts\audit_pairvpr_match_geometry.py wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/scripts/
```

训练机：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python -u scripts/audit_pairvpr_match_geometry.py --input doc/pairvpr_local_matches_v1 --output doc/pairvpr_match_geometry_v1
```

若输出目录已存在，换新目录名；脚本不覆盖。每对打印进度；完成才写 completed.json。异常中断的目录不作为完成结果。

本机下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/pairvpr_match_geometry_v1 D:\Desktop\MyVPR\doc\
```

下一步按同一查询对照错误 top1、原评分最佳 GT、全部候选内 GT 和成功案例。只有多例中 GT 的一致性与双端覆盖都体现互补性，才值得在独立数据上验证。若没有，则停止增加这一套 MNN 派生规则。
