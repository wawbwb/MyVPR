# 稠密 CLIP 动态语义与 BoQ 聚合前 token 剔除

本实现对应新的固定实验，不是旧 Crop-CLS 像素软化或晚期 attention bias 的强度调整。参考 SemVPR 的局部语义与聚合思路、ClearCLIP 的末层 QQ attention-only 构造，以及 Semantic-guided De-attention 的聚合前抑制思路。没有宣称完整复现这些论文或已经取得 VPR 收益。

## 固定实现

- 冻结 OpenCLIP ViT-B/16、OpenAI 权重，依赖原环境 open_clip_torch==2.26.1。独立实现 ClearCLIP 最后一层 query-query 注意力输出，去掉该层 residual/FFN；前面各层保持原计算。位置编码采用官方插值约定。无需安装官方仓库、MMSeg 或额外分割模型。
- 原图先按原 RU 几何转换为 280×280，教师再缩放至 560×560，获得 35×35 类别余弦图。将类别分数插值到 280×280 后比较动态与静态组，再按每个 RU token 对应的 14×14 区域求面积比例。这是固定全图稠密推理适配，不是官方滑窗分割评测复现，也不声称像素精确。
- 动态类别：car、truck、bus、motorcycle、bicycle、person。静态参照：building、road、sidewalk、wall、fence、tree、vegetation、sky、bridge、traffic sign、traffic light、pole、terrain。不使用 ego 类别。
- 固定像素 margin >0.02；token 动态像素比例 ≥0.75 时剔除。如果剩余少于 32/400 个 token，整张图回退全保留。阈值是本轮预先固定的工程假设，未经过独立开发集校准，不是论文给出的最优值，不根据 MSLS 结果自动扫描。
- 在原 BoQ 的 3×3 投影和逐 token LayerNorm 后、首个空间 encoder 前收集保留 token。批内不同长度的填充位置在每一层 encoder 和 cross-attention 都被屏蔽，且每层清零填充值；不是将动态 token 乘零后继续作为普通有效 token。
- DINOv2 和 3×3 投影仍会提前混合信息，因此不声称动态信息被完全移除。原图与冻结 RU gate 保持原计算。
- 全保留走原 aggregator 路径，preflight 校验与完整 RU 输出一致。未修改旧 BoQ、旧教师或历史实验脚本。

## 三组训练和评测

三组从同一个经过 SHA 校验的 RU checkpoint 初始化：none（无剔除）、aligned（语义剔除）、shuffled（每图固定逐行打乱剔除）。shuffle 精确保留每一行剔除数量，但不保证与真实物体完全无重叠。

仅训练 BoQ，包括原投影、encoder、queries、fc；冻结 DINO 和 RU gate。三组均使用同一完整 GSV-Cities、5 epochs、seed42、每批16地点×4图、AdamW lr=1e-5、weight_decay=0.001、梯度范数上限1。损失沿用项目 MultiSimilarityLoss 和 miner。只有 ColorJitter，不做改变掩码坐标的几何增强。每个 epoch 的地点顺序、视图采样与增广随机种子匹配。

不按 MSLS 挑 checkpoint，比较第5轮最终模型。无剔除也进行同预算训练，另重算冻结RU作为675/740基线。评测分别用每组模型重新生成全数据库和查询描述子，查询与数据库规则一致。

## 一键运行

训练机先同步并激活原环境：

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
bash scripts/run_clip_token_drop.sh
```

默认使用 cuda:1。脚本依次执行：

1. preflight：8项单元测试必须全部通过且无跳过；真实 CLIP 权重两图前向；真实 RU 全保留一致性；剔除路径梯度与冻结骨干检查。失败即停止。
2. 新建 GSV 训练语义缓存和 MSLS 全库+查询语义缓存。旧 crop 缓存不能转换成稠密教师特征，因此本步需要新的 CLIP GPU 前向。
3. 按 none、aligned、shuffled 顺序完成训练。
4. 重算四组全库+查询描述子，检索并保存结果。

这次是新教师缓存、正式训练和完整数据库重提取，计算量显著大于前几轮 CPU 审计；未在本机测量训练机耗时。评测四组描述子和分数约4.1GB，另需教师缓存、训练 checkpoint 和原数据空间，建议额外预留至少10GB。训练期缓存 fractions 常驻 CPU 内存，每百万图片约0.8GB，另有路径、哈希等开销。

若只想先验证环境和实际权重执行：

```bash
python scripts/clip_token_drop.py preflight --device cuda:1 \
  --output doc/clip_token_drop_preflight_manual_v1
```

## 分阶段命令

```bash
python scripts/clip_token_drop.py cache --split train --device cuda:1 \
  --output .cache/clearclip_token_drop_train_v1
python scripts/clip_token_drop.py cache --split msls --device cuda:1 \
  --output .cache/clearclip_token_drop_msls_v1

for mode in none aligned shuffled; do
  python scripts/clip_token_drop.py train --mode "$mode" --device cuda:1 \
    --cache .cache/clearclip_token_drop_train_v1 \
    --output "doc/clip_token_drop_train_v1/$mode"
done

python scripts/clip_token_drop.py evaluate --device cuda:1 \
  --cache .cache/clearclip_token_drop_msls_v1 \
  --runs doc/clip_token_drop_train_v1 --output doc/clip_token_drop_eval_v1
```

checkpoint 默认从原动态实验 run.json 定位并核对 SHA，可用 --checkpoint 提供搬迁后的同一文件。数据默认 datasets/gsv_cities 和 datasets/msls-val，可通过 --gsv-root/--msls-root 修改。

## 中断和完整性

缓存每128张存分片；重复命令会核验已有分片和图像身份后继续。完成的缓存会检查合同与分片 SHA；训练、评测读取图片时再次核对图像 SHA。教师视觉/文本权重、提示词、阈值、代码、manifest 均进入合同。

训练每个 epoch 原子保存 last.pt（聚合器、优化器、epoch/step、合同）。原命令加 --resume 从上个完整 epoch 继续；中断的半轮重新执行，不是逐step续跑。一键脚本会对未完成训练自动传入 --resume；成功训练会保留，在评测时验证其合同与 checkpoint SHA。开始第一轮前中断、尚无 last.pt 时，从初始化重新执行。

评测暂不支持分片续跑，中断后保留现场，用新的 --output 重跑评测；一键脚本不会自动删除或覆盖该目录。不要把 smoke-test 产物混入正式评测。

## 下载轻量结果

训练机保留大文件，仅下载 report 子目录即可分析三组差异：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/clip_token_drop_eval_v1/report "C:\Users\zy\Desktop\MyVPR\doc\clip_token_drop_report_v1"
```

报告包含 summary、逐查询 JSON/CSV、provenance 和 completed。主判据是 aligned 同时优于匹配训练 none 和 shuffled，并与冻结 RU 比较；MSLS 已反复用于开发，正结果还需独立数据验证。

## 验证状态

本机3项 NumPy测试通过：阈值/最低保留回退、逐行打乱与无剔除、非法输入拒绝。Python语法和CLI检查通过。

本机没有 PyTorch，安装请求未获授权，因此5项PyTorch测试未在本机执行，不声称通过。训练机preflight强制执行：全保留原路径一致、批量填充与逐图等价及梯度、已剔除投影token不能影响输出、QQ注意力等价构造、全剔除输入拒绝；然后再做真实权重检查。尚未启动真实缓存、训练或评测。

参考：[SemVPR](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Efficient_Visual_Place_Recognition_Through_Multimodal_Semantic_Knowledge_Integration_ICCV_2025_paper.html)、[ClearCLIP论文](https://arxiv.org/html/2407.12442v1)、[ClearCLIP官方实现](https://github.com/mc-lan/ClearCLIP)、[Semantic-guided De-attention](https://ksp.etri.re.kr/ksp/article/file/66898.pdf)。
