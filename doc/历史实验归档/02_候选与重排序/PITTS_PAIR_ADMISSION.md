# Pitts30k-val 官方 Pair-VPR 准入审计

读取现有 pitts30k_val_dbImages.npy、pitts30k_val_qImages.npy、pitts30k_val_gt_25m.npy，保留原索引和所有正样本。先检查索引维度、GT范围、重复路径以及全部索引图片存在性。真实图片解码在提特征时检查。

预算在推理前固定：按query路径哈希选最多1024个查询，对完整参考数据库检索top20。若总查询更多，报告只代表此固定样本，不能写成完整Pitts30k-val成绩。所有被选查询均进入分母，包括top20无正样本的查询。不根据预测挑错例，不注入GT候选。

复用已核验官方Pair-VPR ViT-B、bilinear322和双向pair评分。缓存配对CLS和reference global描述子，方便以后研究，但本轮不训练候选模块。GSV版本脚本与缓存指纹不变。程序读取已有官方配置创建模型，不调用其MSLS数据加载器。

报告同时给global、pair R@1/R@5/R@10/R@20、pair相对global纠错/退化、候选不可达数及候选内排错数。准入仍为至少10例候选内排错；通过也不会自动训练。Pitts30k-val已经参与过历史开发，不能作为新论文独立最终测试。

物理GPU1由shell脚本设置为唯一可见GPU，内部cuda:0。缓存分片支持重跑复用与哈希检查；不存全库dense token，最多64张保存在RAM，超出重编码，因此时间主要花在encoder/decoder前向。默认最多40960次单向pair前向，另有global提取；不承诺固定时长。

训练机运行：

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
mkdir -p logs
set -o pipefail
bash scripts/run_pitts_pair_admission.sh 2>&1 | tee logs/pitts_pair_admission_v1.log
```

本机PowerShell下载轻量报告即可：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/.cache/pitts_pair_admission_v1/report C:/Users/zy/Desktop/MyVPR/doc/pitts_pair_admission_report_v1
```

本机只执行CPU索引测试、Python语法检查及bash语法检查，没有运行PyTorch模型。首次服务器前向检查配对CLS形状及分类分数重建。旧GSV实验无需重跑。
