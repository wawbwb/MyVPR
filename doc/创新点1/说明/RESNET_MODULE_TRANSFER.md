# 创新点1：官方ResNet50-BoQ迁移实验（待运行）

2026-10-16。当前授权仅为实现、同步与准备检查，不启动真实数据smoke或正式训练。结果在运行前未知；实验用于检验模块有效性，不预设必须成功。

## 1 研究问题与基座

检验已有单图卷积跨层融合模块是否仅适用于DINOv2，还是也能改善CNN编码器的全局地点描述子。分别检验：整体适配相对原模型的收益，以及真实跨层输入相对等容量末层输入的额外收益。

使用BoQ作者发布的**完整ResNet50+BoQ检查点**，不是ImageNet权重、不是项目ResNet18训练结果，也不加载历史RU或DINO适配参数。

- [官方hub接口](https://github.com/amaralibey/Bag-of-Queries/blob/1a4965ea7dfd9bd0dd846adf7a0e430f68101d12/hubconf.py)：`resnet50`，16384维描述子，BoQ输入1024、投影512、查询64、2块、行维32。
- [官方权重](https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth)：`resnet50_16384.pth`。
- [官方预处理说明](https://github.com/amaralibey/Bag-of-Queries/blob/1a4965ea7dfd9bd0dd846adf7a0e430f68101d12/README.md)：训练尺寸320，推荐ResNet评测384；顺序`ToTensor → bicubic/antialias Resize → ImageNet Normalize`。
- [官方ResNet实现](https://github.com/amaralibey/Bag-of-Queries/blob/1a4965ea7dfd9bd0dd846adf7a0e430f68101d12/src/backbones.py)：裁掉layer4，保留layer1/2/3。离线以`pretrained=False`构建，随后**strict加载完整VPR检查点并逐张量检查**，不另下载ImageNet参数。

源码沿用已核对提交及5份上游文件的固定SHA256。权重首次从作者HTTPS地址取得后记录本地SHA256，后续强校验；没有伪称该本地摘要为作者公布的校验值。

## 2 结构迁移

保留已封存DINO模型及实验源码不变，新建ResNet接口。一次骨干前向取得`layer1/2/3`，即常用C2/C3/C4，通道为256/512/1024。输入320时空间尺寸约80/40/20，输入384时约96/48/24。

浅层先面积池化到末层网格，再分别沿通道重复4次/2次到1024通道，末层不变。该接口没有可训练参数，重复操作保持池化后逐位置RMS；**不宣称无损保留池化前空间细节**。两组都经过相同LN与1024→128投影，只改变输入路由，避免普通的异通道投影导致对照容量不一致。后续若更换对齐接口，应另立实验版本，不能静默修改本合同。

主要模块沿用两级渐进局部卷积融合（无SASF、无Mamba扫描），合并后1024通道升维，通过原RMS-tanh有界残差注入末层，再送入冻结的官方BoQ。保持已有乘法次序`0.25*(0.1*stopgrad(RMS(last))*tanh(raw))`，相对L2上限0.025。

另外提供最新的末层锚定条件低秩补充分支，保留其局部门控、整图通道门控和rank32结构。它在本实验中从**新ResNet零起点**开始，不继承DINO的Warm模型，因此不是把DINO最终权重跨骨干加载。

所有分支推理均为单图→单个16384维单位向量，无语义模型、局部成对匹配或候选重排序。冻结ResNet全部参数及BatchNorm运行统计，冻结BoQ；只有真实参与前向的模块参数进入优化器，旧辅助头冻结且不计算损失。

## 3 固定对照与预算

| 组别 | 输入/结构 | 用途 |
|---|---|---|
| Official | 原始官方ResNet50-BoQ | 复现基线；不训练 |
| conv_last | 原两级卷积模块，末层/末层/末层 | 原模块等容量末层对照 |
| conv_cross | 原两级卷积模块，C2/C3/C4 | 检验原渐进跨层方案迁移 |
| conditional_last | 末层锚路径，补充分支也仅末层 | 条件补充等容量对照 |
| conditional_cross | 末层锚路径，C2/C3条件补充 | 检验最新改进版迁移 |

共4组正式训练，单种子42，各2048步；每组另8步smoke，smoke更新丢弃。原模块两组初始化与参数量相同，条件模块两组初始化与参数量相同；条件版比原版参数更多，不能把二者差值单独归因于跨层。

复用已封存GSV8192训练地点池与2048步日程（512步重复四次），每步至多16地点×4视图，固定epoch99取图。历史RU仅用于既有难负样本索引，没有教师特征、RU权重或语义监督。全组采用320训练与同样的Tensor-first预处理，不混用PIL-first训练变换。

AdamW，学习率1e-4、衰减0、FP32、microbatch2、梯度裁剪1；GPS排除25米内不同ID负对；无辅助/保持损失。第2048步固定评测，不挑最好轮数、尺寸、层组合或系数，不自动增加训练。

主要评测为384的MSLS-val（740查询）及Pitts30k-val（7608查询），次要评测320。Pitts验证集不是作者表中的Pitts30k-test，不能直接将二者分数比较。保存R@1/5/10/20、完整逐查询正误、纠错/退化列表及原始未校正配对p值。

## 4 判读规则

1. 模块组相对Official的R@1差值回答“该完整适配是否提升”。
2. `conv_cross − conv_last`及`conditional_cross − conditional_last`回答“真实跨层输入是否优于同结构末层输入”。
3. 首要正信号门槛是两项384评测都净提升；同时展示所有320退化，不以运行完成代替收益通过。
4. 单种子、曝光验证集、有限训练预算只能支持探索性证据，不能证明所有CNN有效或稳定增益。无提升时也不能倒推出所有ResNet训练方式均无效。

## 5 文件与防误启动

- `src/models/resnet_module_transfer.py`：严格官方加载、原生stage提取、无参对齐、四组路由。
- `src/resnet_transfer_protocol.py`：固定协议与成对汇总。
- `scripts/prepare_resnet_boq.py`：仅下载和校验官方源码/权重，无模型训练。
- `scripts/resnet_module_transfer.py`：CPU预检、基线评测、显式训练、断点恢复、汇总。
- `scripts/run_resnet_module_transfer.sh`：默认`preflight`，不默认训练。
- `tests/test_resnet_module_transfer.py`：仅在训练机执行CPU合成单元测试。

缓存独立位于`.cache/boq_resnet50_official_v1`，结果独立位于`logs/resnet_module_transfer_v1`，每16步原子保存优化器、参数和Torch RNG；自动恢复不绕过原代码/日程/参数/基线身份校验。GPU阶段使用物理GPU1与既有共享锁；不杀其他进程、不重启服务器。

**准备阶段（不训练）**：`bash scripts/run_resnet_module_transfer.sh tests`、`prepare`、`preflight`。预检只使用CPU：严格加载、合成小图零起点一致性、配对容量/初始化、完整冻结检查。它不代替真实数据smoke，不评测召回率，不初始化CUDA。

**以后获得训练授权才运行**：`bash scripts/run_resnet_module_transfer.sh all`，依次基线→4组smoke→门槛检查→4组正式训练→汇总。单独调用Python训练子命令必须显式提供`--allow-training`。本次不执行该命令。

完整复现仍需要原工程、数据、历史索引合同和PyTorch环境；代码不是包含数据与权重的独立安装包。本次不改专利正文，也不把尚未运行的实验写成有效性结果。

## 6 本次交付状态（2026-10-16）

已通过SCP将上述7个新增文件同步到训练机`/home/wt/workspace/OpenVPRLab`，逐文件字节SHA256与本地一致；没有提交混杂的Git工作区，没有覆盖既有封存源码或清理用户文件。

- 训练机CPU单元测试：**23 passed**；Shell语法检查通过。本机未运行代码测试。
- 已取得作者ResNet50完整权重（约91 MiB），本次下载SHA256：`4691d1545db847da2c0ba911f34e6c520a5da63f8b94c6415fe87d0d8800ebaa`。
- CPU预检：10项检查全部通过，四组的合成64×64小图零起点描述子最大绝对误差均为0。完整320/384的GPU零起点检查安排在日后各组smoke开始前，本次没有执行。
- `conv_last/conv_cross`有效可训练参数各885184；`conditional_last/conditional_cross`各1478530。两对初始参数哈希分别完全相同。
- 官方完整张量身份SHA256：`f6a9c88080d0f485fb593ae2300aa4f353a0c08cf3cc4d789e3d346be92bb572`。它与完整权重文件SHA256不是同一概念。
- 服务器预检报告：`logs/resnet_module_transfer_v1/preflight/summary.json`；`verdict=PREFLIGHT_PASS_NO_TRAINING`，`cuda_context_initialized=false`。
- **未启动GPU基线评测、真实图像smoke或正式训练，目前没有迁移效果结果。**
