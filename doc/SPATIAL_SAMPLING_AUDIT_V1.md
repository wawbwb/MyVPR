# 空间重采样后续：全局位移与局部对应的冻结机制验证

2026-10-25用户授权沿空间采样方向继续探索。V1主候选在Pitts多对1条，GSV/MSLS不变，但三个训练组纠正同一条118，且整图/打乱组全套Top1一致。位移接近每轴幅值上限，因此先检验涨点是否需要内容条件和真实局部对应，不立即增加网络或调参。

## 固定权重与训练集标定

原创新点1及V1 train/last.pt全部冻结，只读校验train/eval封存及源码来源。没有优化器、重新训练或标签拟合。

对原8192训练地点，按SHA256('sampling-audit-v1:'+place字符串)排序选前256地点，每地点固定4图，共1024图。提取V1 global_shift和local_shift预测场，记录每图有符号xy均值、空间标准差及符号比例。各分支在这些训练图上的位移场总均值形成自己的固定二维位移（float64汇总、float32推理）。不从验证集选择方向、幅度或图像子集。

## 八个固定评测版本

|版本|位移来源|检验问题|
|---|---|---|
|innovation1|无新增残差|基线|
|trained_global|V1整图模型|严格复现|
|constant_global|其训练图平均位移，所有图相同|是否需要逐图条件化|
|trained_local|V1局部模型|严格复现|
|constant_local|其训练图平均位移，所有图相同|固定平移能解释多少|
|local_mean|每图局部场的空间均值|去掉局部变化后如何|
|local_centered|局部场减去每图均值|原有全局平移的作用|
|local_centered_shuffled|对上项按图像SHA打乱二维向量位置|局部位移与内容对应|

去均值后若最大轴位移>1，按每图最大轴幅值整体缩放至1；保持零均值。centered与其打乱版严格同数值分布/幅度，二维向量整体移动。原始local与centered幅度可能不同，这是干预的一部分，不能把它们的差值全归因空间对应，也不能据此断言重新训练零均值模块无效。

所有版本使用V1完全相同的warp差值、border边界、每轴±1限制和2.5%真实残差上限。基线、trained_global、trained_local必须在三套验证逐项复现原完整Top20，否则停止。新增干预属于既有权重的事后机制诊断，不当作独立确认或新增训练方法。

## 分析和停止条件

分别报告对创新点1的R@1/5/10/20、逐查询纠错/退化；比较trained_global与constant_global、trained_local与constant_local/local_mean、centered与其打乱版。若常量复现同一涨点，不能继续将其归于内容适配。若均值版解释收益，后续需重新论证局部机制；若centered优于匹配打乱，才值得考虑进一步局部对应训练。不能按八个版本的最高验证分数挑新方法。

没有自动成功判定、验证调参或后续训练。唯一自动完成判定是来源/基线/冻结检查通过。固定方向统计不能从位移范数代推，幅度接近sqrt(2)也不意味着越界（每轴±1）。评测8份描述子数组峰值约8GB RAM，启动前已核实服务器可用内存约55GB，GPU分配20%+2GB余量，使用旧实验锁。

## 执行

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_spatial_sampling_audit.py --allow-evaluation --allow-shared-gpu
cat .cache/spatial_sampling_audit_v1_launcher/progress.json
```

服务器CPU测试→1024训练图标定→三套完整验证。输出logs/spatial_sampling_audit_v1。旧源码/权重不改；状态另记SPATIAL_SAMPLING_AUDIT_V1_STATUS.md。
