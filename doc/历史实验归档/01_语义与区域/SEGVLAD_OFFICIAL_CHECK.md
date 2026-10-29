# 下一步：官方 SegVLAD 的有限复现

状态：准备检查脚本已实现，未在本机运行测试、导入上游或运行模型。

## 官方包缺少domain PCA：显式使用map分支

用户下载的包仅列出 `_order3_dinoNV.pkl`、`_order3_map.pkl`、
`_order3_map_dinoNV.pkl`，没有原定domain/indoor的 `_order3.pkl`。
因此提供显式 `--vocab-vlad map`：同时选择 `17places/c_centers.pt` 和
`17places_r_fitted_pca_model_order3_map.pkl`，透传到官方主程序。
不复制/改名PCA，不使用dinoNV，不修改官方源码。默认仍为domain。
map结果是另一套词典配置，不能冒充先前domain配置或未经核对直接对应论文表格。
运行前检查PCA为1024×49152、词典为32×1536；尺寸一致也不单独证明来源完全匹配。

```bash
python -m pytest -q tests/test_segvlad_official.py
python scripts/segvlad_official.py check --repo ../Revisit-Anything-official --data-root datasets/segvlad_official/17places_full --vocab-vlad map --output "doc/segvlad_official_map_check_$(date +%Y%m%d_%H%M%S)"
```

检查通过后（若词典缺失，先反馈，不替换indoor词典）：

```bash
set -o pipefail
python -u scripts/segvlad_official.py run --repo ../Revisit-Anything-official --data-root datasets/segvlad_official/17places_full --vocab-vlad map --gpu 1 --output "doc/segvlad_official_map_run_$(date +%Y%m%d_%H%M%S)" 2>&1 | tee "doc/segvlad_official_map_run_$(date +%Y%m%d_%H%M%S).txt"
```

下面保留原始domain准备流程供溯源；本次选择map时请使用上面的命令。

当前RU适配版的SAM R@1为382/740，网格488/740，RU675/740；SAM候选并集净可达性为0。
停止该适配配置，不继续调权重。现在换用官方配置验证，而不是声称官方方法已经失败。

来源：[官方README](https://github.com/AnyLoc/Revisit-Anything)、
[官方配置](https://github.com/AnyLoc/Revisit-Anything/blob/main/place_rec_global_config.py)。
固定提交 `628508d4a60c832961a2e2d8efa9eec724c2dc86`。

## 为什么先选17Places

作者推荐从17Places开始，并提供 `17places_full.zip`（README标注10.3GB），可跳过
掩码/特征提取和PCA拟合，直接运行核心检索。这样先验证官方链路，不再混入RU训练特征。
本阶段不能验证端到端分割提取，也不能与MSLS的91.22%横向比较。

采用官方默认 `exp0_global_SegLoc_VLAD_PCA_o3`、domain/indoor词典。
上游检索程序使用1024维PCA、DINO-G layer31 value的32簇词典。
不复用我们280×280的SAM缓存或256维PCA，不改官方聚合和排序函数。

## 先执行准备检查

继续使用VPR环境，不执行官方旧环境安装命令，不安装torch1.11。

```bash
cd ~/workspace
conda activate VPR
git clone https://github.com/AnyLoc/Revisit-Anything.git Revisit-Anything-official
git -C Revisit-Anything-official checkout --detach 628508d4a60c832961a2e2d8efa9eec724c2dc86
cd ~/workspace/OpenVPRLab
python scripts/segvlad_official.py check --repo ../Revisit-Anything-official --data-root datasets/segvlad_official --output doc/segvlad_official_check_v1
```

如果仓库目录已经存在，不重复clone，先检查其git状态；不要覆盖用户修改。
脚本报告真实上游import错误和缺少的数据文件；准备未完成不是实验失败。
**先回传preflight.json和imports.txt，再补缺少的依赖。** 不盲目安装整个旧环境。
脚本不修改上游源码、不自动下载数据、不执行GPU reset。

## 官方预处理数据

[作者Box共享目录](https://universityofadelaide.box.com/s/199q2lpvy3psm5qgfagvh25r9c51ey6b)。
本次未能验证Box文件直链是否可用，不提供猜测下载URL。
选择 `17places_full.zip` 而非仅61MB的原始图片包。
这是下载作者数据，不是用压缩包同步本项目代码。
下载前确认训练机至少预留约25GB空间供压缩包和解压数据；实际以解压体积为准。
若Box不可访问，先报告，不替换成RU缓存继续冒充官方复现。

按共享包目录结构放置，使以下路径存在（check将列出准确缺失路径）：

```text
datasets/segvlad_official/17places/ref/
datasets/segvlad_official/17places/query/
datasets/segvlad_official/17places/out/17places_r_masks_320.h5
datasets/segvlad_official/17places/out/17places_q_masks_320.h5
datasets/segvlad_official/17places/out/17places_r_dino_640.h5
datasets/segvlad_official/17places/out/17places_q_dino_640.h5
datasets/segvlad_official/17places/out/17places_r_fitted_pca_model_order3.pkl
```

不需要这一步重新下载SAM-H和DINO-G权重。pickle只允许使用作者可信来源的文件。
缺依赖或PCA维度不一致时先停止，不将1024擅自改成256/512。

## 准备通过后再运行

输出目录必须全新。stdout完整保存；官方结果保存在数据目录的out/results中。

```bash
cd ~/workspace/OpenVPRLab
set -o pipefail
python -u scripts/segvlad_official.py run --repo ../Revisit-Anything-official --data-root datasets/segvlad_official --output doc/segvlad_official_run_v1 --gpu 1 2>&1 | tee doc/segvlad_official_run_v1.txt
```

本包装器只覆盖workdir路径，原始程序运行不做算法重写。若上游依赖与当前环境不兼容，
后续只做显式、可审计的兼容修复，不能在本轮尚未验证时宣称已可完整复现。
跑完后核对数据顺序、GT约定、PCA、召回输出及对应论文表格条件，才决定是否达到复现目标。
