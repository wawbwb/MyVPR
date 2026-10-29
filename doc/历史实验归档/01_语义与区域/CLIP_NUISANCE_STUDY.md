# 同地点 CLIP 变化代理引导的对抗解耦实验（WP-CRD）

## 为什么改变方向

2026-10-18完成的CLIP-Key评测：正确引导在GSV留出/MSLS/Pitts-val分别比官方少4、少6、多3个正确查询；打乱组分别持平、持平、多9个。这不是“打乱总有收益”，而是正确引导未证明优越。当前Key模块本就允许正负门控，简单翻转符号没有增加函数族。

本轮将CLIP从推理期空间引导，改为**仅训练期的变化因素观察器**。不对图像打mask、不强制模仿语义、不删除完整CLIP信息，不修改创新点1或已封存实验。

## 依据及不能过度解释之处

- [DANN，JMLR 2016](https://www.jmlr.org/beta/papers/v17/15-239.html)：任务判别与域不变表示可通过梯度反转共同学习。本实验是对连续变化代理的对抗回归，不是原论文的离散域分类复现。
- [Explicit feature disentanglement for VPR，2021](https://journals.sagepub.com/doi/abs/10.1177/17298814211037497)：地点内容与外观因素分离已有先例。因此“对抗解耦”本身不是可直接主张的新颖点；本轮首先验证使用CLIP同地点变化代理的可行性。

同地点跨视图差异既可能含天气、光照、车辆，也可能含视角、可见结构或定位误差。**它只是候选变化代理，绝不称为纯干扰真值。** 降低代理可预测性但损害检索，不能算成功。

## 方法

1. 沿用封存GSV计划：8192训练地点、1024留出地点，每地点4张固定图片。只读复用已有训练图片的连续SCLIP缓存；仅给留出图片补充CLIP池化向量。不新建全库dense缓存。
2. 每图的512维SCLIP patch取空间均值后L2归一化，得到scene向量（不是原始CLIP CLS）。同地点内减去四视图均值。
3. 仅在训练地点拟合32维PCA白化；限制罕见异常值后再次做同地点中心化。留出地点只应用固定变换。不能跨地点/城市混算均值。
4. 学生使用官方DINOv2-BoQ，原始网络全冻结，新增视觉自路由器（复用隔离CLIP-Key模块的visual路径）。推理输入只有RGB，原始visual Value不改。CLIP及变化预测头均不进入推理学生。
5. 描述子固定投影到256维、同地点中心化、按冻结基线的训练集统计缩放。MLP预测32维CLIP变化。中心化旨在避免直接对抗整个地点语义，不代表严格的因果解耦保证。
6. 每步先用detach的学生描述子更新预测头两次，再冻结预测头，通过梯度反转更新学生。地点Multi-Similarity损失仍为主任务，另有统一的轻量官方描述子保持项。

学生梯度目标等价于：

```
min_student L_VPR + 0.1 * mean(1-cos(z,z_official)) - lambda(t) * L_proxy
min_predictor L_proxy(stop_gradient(z))
lambda(t) = 0.02 * min(1, (step+1)/512)
```

源码通过GRL实现负梯度，不能把整个优化器学习率设负。两套AdamW优化器各自保存。保持项、学生结构、批次、初始化和所有预算完全匹配。

## 三组

| 组别 | 变化预测头 | 反向约束学生 |
|---|---|---|
| visual（无解耦） | 学正确代理，仅作监测 | 无，系数0 |
| aligned | 学正确视图对应的代理 | 有 |
| shuffled | 学同一地点内错位视图代理 | 有 |

三组参数量相同，原始初始化相同，预测头均从同一随机初始化开始，不复用预检探针。shuffled以地点ID+seed构造固定无不动点视图置换，保留每地点代理分布与和为0的性质。**不能沿用空间patch打乱：池化后向量不变，无法破坏本轮对应关系。**

FP32，P16×K4，每组固定5epoch/2560步；学生lr=3e-4、预测头lr=1e-3、weight_decay=0、梯度截断1、microbatch2、每16步原子保存。恢复保存优化器、RNG、精确步数及历史；不选择开发集最佳轮次。

## 一次性验证链

1. 服务器CPU测试：GRL方向、预测头梯度方向不反转、detach隔离、原始网络冻结、CLIP-free推理、视图置换、训练专属PCA/归一化、断点契约。
2. 无学生更新的准备：生成原始官方描述子的256维投影和CLIP池化向量。新数组约百余MiB，不存全库密集特征。
3. 机制预检：CPU探针分别拟合正确/打乱代理，固定8轮，留出地点测R²。预先固定正确R²至少0.01且高于打乱至少0.01。若未通过，自动停止，不开始学生训练。这里只判断有没有当前探针可用信号，不证明代理是纯干扰，也不否定全部非线性关系。
4. 三组独立8-step smoke，更新全部丢弃；通过后分别从原始初始化正式训练。
5. 自动评测官方和三组固定最终模型：GSV留出、MSLS-val、Pitts30k-val的R@1/5/10/20与逐查询纠错/退化。Pitts-test保持封存。
6. 从最终各组描述子重新训练同容量、同预算、不同初始化的新探针。探针输入按各自TRAIN统计标准化，避免仅因缩小特征尺度造成“不可预测”。所有组探针都预测**真实**代理，不给shuffled组特殊目标。

不能用对抗损失升高宣布成功，因为预测头可能失效。至少要看到aligned相对visual/shuffled的开发集检索收益，并结合独立探针判断；单种子仍需后续复核。若只降低R²但检索下降，视为过度去除有用信息。

## 运行与目录

```bash
cd /home/wt/workspace/OpenVPRLab
/home/wt/.conda/envs/VPR/bin/python scripts/launch_clip_nuisance.py --allow-training --evaluate
```

物理GPU1，内部逻辑cuda:0，单进程CUDA allocator上限20%；不使用GPU0、不停止其他任务、不改环境。

- 准备：`.cache/clip_nuisance_v1/prepared/`
- 后台状态：`.cache/clip_nuisance_launcher_v1/progress.json`
- 机制预检：`logs/clip_nuisance_v1/preflight_probe/summary.json`
- 三组：`logs/clip_nuisance_v1/{visual,aligned,shuffled}/`
- 最终检索与独立探针：`logs/clip_nuisance_v1/evaluation/summary.json`

旧训练/评测源码、模型、缓存、契约只读使用，均不改写；新实验单独记录代码、运行时、官方权重、计划、代理和参数的SHA。后台阶段任一失败即停止，不自动跳过检查或扩展训练。
