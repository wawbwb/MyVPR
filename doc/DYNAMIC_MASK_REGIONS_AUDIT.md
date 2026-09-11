# 旧动态掩码空间分布与分类核验

只读CPU审计，不训练、不提取特征、不重新分割、不读取逐查询预测、不修改原掩码。
依赖Python、NumPy和Pillow；测试使用标准库unittest，不要求安装pytest。

## 训练机运行

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python -m unittest discover -s tests -p test_audit_dynamic_mask_regions.py -v
python scripts/audit_dynamic_mask_regions.py --output doc/dynamic_mask_regions_audit_v1
```

默认读取原实验目录、原掩码缓存和datasets/msls-val。脚本要求缓存与原run.json的SHA一致，
检查数据库顺序及query/GT清单哈希；不需要query_outcomes.csv或前一次下载报告。
输出目录必须不存在，重跑时使用新的目录名，不覆盖已有审计。

## 输出与分母

- `summary.json`：每个split按旧覆盖率分箱的空间统计；另有按路径去重的query并集与数据库统计。
- `spatial_per_image.json`：全部缓存图片的逐行均值、三个空间带的掩码面积和底部占全部掩码的比例。
- `query_reference_map.json`：各split的query索引到缓存索引、固定最小GT索引的映射。
- `index.html`：按覆盖率与路径哈希固定抽样的query/GT核验页，每组默认4例。蓝线为图像75%高度。
- `review_samples.json`：带样本身份的未填写标注模板；页面可导出已填的`review_annotations.json`，也可导入继续填写。
- `provenance.json`与`completed.json`：输入/脚本哈希和完成范围；性能审计始终为false。

三个空间带为20×20网格的[0,10)、[10,15)、[15,20)行。各带的面积均除以整图400个patch，
三者相加等于整图覆盖率。`above_bottom_area_fraction_of_image`同样使用整图分母，并非裁图后的覆盖率。
`bottom_share_of_mask`为底部掩码面积/全部掩码面积，零掩码为null。
`bottom_share_ge_half_count`仅表示至少一半掩码集中于底部，不是“自车图像数”。

外部动态物体、自车区域和静态误标均需看图判断，默认unknown。
自车属于car不一定是类别分割错误，但和外部车辆跨时变化是不同研究对象。
20×20边界混合不能用于精确像素IoU。抽样是事后描述性检查，不能推断全部数据的误检率。
固定GT并非外观最佳对应；路径去重也不等于序列或地点独立。
本脚本没有按结果选样、调参、裁掉底部、重算掩码或实施新检索干预。

页面通过JSON导出保存标注，刷新/关闭前务必导出；不自动保存到服务器或原文件。
导入仅接受同一审计ID和相同图片身份的标注，失败时不部分覆盖当前标注。

本机PowerShell下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_mask_regions_audit_v1 C:\Users\zy\Desktop\MyVPR\doc\
```

先阅读空间分布和图片，随后再决定是否需要序列核查、精细人工区域标注或新检索实验。
