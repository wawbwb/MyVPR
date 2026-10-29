# Pair-VPR：结果判读与离线案例审查

2026-09-07。已读取下载的summary/completed/provenance与逐查询记录，并使用PowerShell核验summary和predictions的SHA256与记录一致。未在本机运行Python测试或模型推理。官方源码diff为空，严格加载的权重与原轮一致；provenance中的complete=false是开始记录，completed.json才是完成标记。

## 已确认结果

| 比较 | 两者正确 | 仅参考正确 | 仅重排正确 | 两者错误 | 重排净收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pair-VPR global → refined | 630 | 9 | 63 | 38 | +54 |
| RU → Pair-VPR refined | 667 | 8 | 26 | 39 | +18 |

Pair-VPR自身从639/740提升到693/740。跨系统RU比较不是纯重排消融。global/refined oracle union为702，RU/refined oracle union为701；两个oracle不能直接相加，也不是已实现的融合性能。

最终47个错误分成31个top-100可达、16个不可达。候选内重排最多可额外纠正31个（4.19pp），并非语义可实现的增益。63次纠错伴随9次误伤；下一步优先理解具体误判，而不是改重排权重。

## 当前不做什么

不重新训练，不增加语义，不扫描top-K或按MSLS选择融合规则。不能把GT可达等同于图像有足够重叠，也不能把oracle当实际结果。

## 新增无GPU审查脚本

`scripts/review_pairvpr_cases.py`验证审计哈希和数据索引，生成全部47个剩余错误的页面，索引单列9个global退化与8个RU-only案例，并加12个固定抽样的成功纠错作参照。组之间重叠，不重复计算样本数。每页展示query、refined top3、global top1、RU top1及一个GT候选，原始图像复制到assets，无需模型、联网或重新提特征。不会生成AI图像或改变原照片。

GT参照优先选保存的global排名中最高的正样本；若不在保存列表里，选固定GT并明确标注，不能误称最近/最相似GT。代码未在本机执行测试，训练机运行时先检查证据与文件存在性。

### 同步：本机PowerShell

```powershell
cd D:\Desktop\MyVPR
git add scripts/review_pairvpr_cases.py doc/PAIR_VPR_CASE_REVIEW.md
git commit -m "Add offline Pair-VPR case review and record paired results"
git push origin main
```

### 训练机

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python scripts/review_pairvpr_cases.py --audit doc/pairvpr_official_paired_audit_v1 --msls-path datasets/msls-val --output doc/pairvpr_case_review_v1
```

输出必须是新目录；旧权重/特征/图像不修改。该操作复制有限数量原图，不复制整个数据集。

### 下载：本机PowerShell

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/pairvpr_case_review_v1 D:\Desktop\MyVPR\doc\
```

下载整个目录后，浏览器打开index.html，图片从assets读取。不要只下载HTML；不要把大量assets加入Git。

若Git网络失败，本机：

```powershell
git bundle create D:\Desktop\pairvpr_review.bundle main
scp D:\Desktop\pairvpr_review.bundle wt@10.16.48.202:/tmp/pairvpr_review.bundle
```

训练机：

```bash
cd ~/workspace/OpenVPRLab
git bundle verify /tmp/pairvpr_review.bundle
git fetch /tmp/pairvpr_review.bundle refs/heads/main:refs/remotes/scp/pairvpr_review
git merge --ff-only refs/remotes/scp/pairvpr_review
```

## 人工判断之后才决定研究方向

每个案例检查：query与GT有无可见重叠；错误候选是否存在明确且可靠的类别矛盾；主要问题是否是视角、重复纹理、光照或低重叠；25m地点标签是否与视觉内容存在歧义。模板不自动填写原因，不把top1预测当真值。

若候选内错误多为几何/视野问题，暂停语义假设；若多例确有可检验语义矛盾，才能在独立训练/校准数据上设计对应监督，并保留匹配随机对照。这批MSLS图只能用于描述性错误分析，不转成训练集或独立测试证据。
