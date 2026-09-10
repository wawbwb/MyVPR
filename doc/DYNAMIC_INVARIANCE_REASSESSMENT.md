# 动态不变性重新评估：先检验问题，再决定干预

日期：2026-09-11。状态：覆盖率分层与固定抽样核验代码已实现，等待训练机结果；未在本机测试。

## 修正旧结论

冻结RU、VOC动态掩码、BoQ末端attention bias=-0.5×patch动态面积的实验未产生总体收益，
只能否定该具体干预的首筛表现。它不是输入端动态删除，也没有训练模型适应动态缺失。
平均动态attention mass接近面积占比，不是动态物体因果影响弱的充分证据。
SAM区域匹配失败同样不是动态抑制失败的直接证据。方向重新开放，旧结果不篡改。

## 第一阶段：现有结果分层与人工核验

不加载torch/SAM/DINO，不重建描述子，不修改旧分数、阈值、预测或GT。
从旧屏幕结果读取query_outcomes.csv，用原GT逐条复核预测正误，匹配summary.csv；
mask-cache与run.json记录的SHA必须一致，query/GT清单哈希、DB顺序、路径均检查。
缺逐query结果明确停止，不能从总R@1倒推，先找原文件而非自动重跑数小时。

主要分层变量仅为query的动态面积占比，固定组：0、(0,5%)、[5%,15%)、[15%,30%)、[30%,100%]。
面积取旧20×20浮点掩码均值，不是原像素真值。所有query保留，报告总体、分组样本量、
五个变体R@1、相对zero-bias纠错/退步，以及aligned对baseline/shuffled/random的成对比较。
少于30例提示样本少；不因超过30例就认定统计充分。不做多重比较后的显著性或置信结论。
standard、night、winter2summer、summer2winter分别报告，重叠查询不当作独立重复。

每组最多4例，按sha256(42:query_path)排序固定抽样，与检索正误无关。
HTML显示query、固定最小GT索引参考图、baseline top1的原图及掩码面积叠加。
只显示汇聚后的20×20掩码，不能据此评估原始分割像素IoU；需人工标记漏车、误伤静态物、
边界混合、可见重叠和疑似动态变化。小物体细节检查仍需原图/原像素mask。
GT表示官方正地点，不保证图像可见重叠或拍摄时间差。

命令（训练机，VPR环境）：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
test -f doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries/query_outcomes.csv && echo "per-query outcomes ready"
python -m pytest -q tests/test_audit_dynamic_coverage.py
python -u scripts/audit_dynamic_coverage.py --run doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries --mask-cache .cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz --dataset-root datasets/msls-val --output doc/dynamic_coverage_audit_v1
```

若旧结果移过位置，使用--run指定实际目录。不更换mask来绕过SHA检查。
输出目录必须不存在。报告只含小图、JSON和HTML，不复制模型或大特征缓存。

本机PowerShell下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_coverage_audit_v1 D:\Desktop\MyVPR\doc\
```

打开index.html，人工结论可填写review_samples.json；模型不会自动把没看到车认定为车消失。
summary.json和per_query.json记录分层及逐query，provenance.json记录来源哈希。

## 未完成的第二、三阶段

1. 真实跨时变化：结合实际时间戳与可见重叠核验同地点参考；仅query与固定GT的面积差不是物体变化估计，视点差和分割误差都可改变面积。当前只报告这一代理量，不据其调参。
2. 受控干预：若有干扰证据，再设计车辆变化与等面积随机扰动的成对对照，区分原图、早期特征干预和后段bias；生成内容不能作真实静态背景。
3. 独立评测：先核对现有数据中的跨时重复路线与车辆变化覆盖，再固定一个未用于调参的数据协议；不默认Pitts更大就更合适，也不把MSLS条件切片当第二数据集。本轮未下载新数据。
4. 训练只有在问题与掩码经验证后另行决定；不自动启动动态不变性训练或beta扫描。

## 如何解释第一阶段

- 高覆盖aligned有净纠错且胜等面积随机/打乱：支持进一步验证条件性收益，不直接宣称普遍增益。
- 只在低覆盖持平，高覆盖退化：检查mask误差、剩余静态证据和干预位置，不解释成“加大权重即可”。
- 高覆盖样本太少：当前数据对假设检验不足，应优先独立的真实动态场景，不先加训练轮数。
- 各组均无收益：仍只是该掩码/该晚期bias的负证据。结合人工核验与变化敏感性，再决定是否值得不同干预。

文献线索（沿用2026-09-11核对结果，不当作本项目收益）：
[分割+移除+补全的VPR研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC7340885/)、
[Dynamics-Invariant Perception Space](https://arxiv.org/abs/2105.07800)。
