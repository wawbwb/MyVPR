# 官方 Pair-VPR：先做预训练权重推理复现

状态：官方权重推理已完成（用户控制台结果）；global 639/740、refined 693/740，重排净增加54。实际使用原VPR环境（torch 2.10.0+cu128、xformers 0.0.35），不是官方原始环境；本地DINO离线初始化后完整checkpoint严格加载通过。下面保留准备阶段记录，不再建议创建环境。下一步见 [逐查询审计及同步命令](PAIR_VPR_PAIRED_AUDIT.md)。独立于已停止的Visual pair V1/V2，当前不做语义训练。

## 1. 范围

使用 [官方仓库](https://github.com/csiro-robotics/Pair-VPR) 的 ViT-B speed checkpoint，先比较其自身global与reranked结果。官方提供权重，无需先从头训练。官方训练说明要求多卡，故本阶段只复现推理；这是权重评测复现，不是完整训练复现。单独建立环境，不改原VPR环境。

[speed配置](https://github.com/csiro-robotics/Pair-VPR/blob/main/pairvpr/configs/pairvpr_speed.yaml) 为322输入、带register的DINOv2 ViT-B、512维全局描述子、top-100重排；与本项目RU的280输入和不同描述子配置不等价。主对照必须是Pair-VPR自身global vs reranked；RU只列为工程参考，不把系统差异都归因于重排。

## 2. 先准备官方仓库，不启动训练

训练机逐块执行。目标目录必须不存在；已有目录时先检查，不覆盖。

```bash
cd ~/workspace
git clone https://github.com/csiro-robotics/Pair-VPR.git Pair-VPR-official
cd Pair-VPR-official
git rev-parse HEAD
git status --short
nvidia-smi
free -h
```

若训练机Git网络失败，本机PowerShell独立克隆，再制作Git bundle，不用压缩包：

```powershell
git clone https://github.com/csiro-robotics/Pair-VPR.git D:\Desktop\Pair-VPR-official
git -C D:\Desktop\Pair-VPR-official bundle create D:\Desktop\pairvpr_official.bundle --all
scp D:\Desktop\pairvpr_official.bundle wt@10.16.48.202:/tmp/pairvpr_official.bundle
```

```bash
cd ~/workspace
git clone /tmp/pairvpr_official.bundle Pair-VPR-official
cd Pair-VPR-official
git rev-parse HEAD
git status --short
nvidia-smi
free -h
```

记录本次SHA后不再自动pull更新。尚未下载或运行官方代码，不能声称已验证当前训练机兼容性。

## 3. 检查接口后再安装与评测

先把以下输出发回，以核对正式命令，不猜测现有msls-val与官方目录布局兼容：

```bash
cd ~/workspace/Pair-VPR-official
sed -n '1,180p' environment.yml
sed -n '1,200p' scripts/eval_speed.sh
sed -n '1,220p' pairvpr/eval/get_datasets.py
sed -n '1,160p' pairvpr/configs/pairvpr_speed.yaml
```

环境创建应使用官方environment.yml（默认独立环境pairvpr）。核对驱动、内存和数据加载器后再执行，避免污染原环境或反复下载。正式评测使用`CUDA_VISIBLE_DEVICES=1`映射物理GPU1；官方脚本示例的GPU0不能直接照搬。

官方权重入口：[CSIRORobotics/Pair-VPR](https://huggingface.co/CSIRORobotics/Pair-VPR)，文件`pairvpr-vitB.pth`。下载时记录仓库revision及权重SHA256；若走第三方镜像，必须与官方可信元数据核验，不能仅凭下载成功当作同一权重。

本轮网页核对尚未成功读取官方get_datasets.py正文，因此不提前给出可能错误的dsetroot/软链接命令。先核对上面的本地源码，再确定数据适配；不拷贝图像、不覆盖索引。

## 4. 验证清单

- 固定官方代码SHA、权重SHA、环境和配置；审查checkpoint加载的缺失/多余参数，官方入口使用strict=False，不可忽略不匹配。
- 检查标准MSLS确为740 queries/18871 DB、顺序与GT一致；不混入先前night/season子集。
- 初次按官方speed几何/候选参数运行，不改回280，不借用RU权重或旧特征缓存。
- 记录global/reranked两套R@1/5/10、耗时与资源；后续增加逐查询记录时应是仅记录输出的改动，并单独保存补丁。
- 不扫描MSLS权重、不从多个模型中看分选赢家。当前MSLS已经作为开发集反复查看，结果不能充当未见测试集证明。
- 官方推理通过后才讨论资源可行的训练复现和语义消融。若官方基准未复现，先核查版本/数据/权重，不自动发明新的小模块。
