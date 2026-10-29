# 固定案例动态干预：准备、核验与运行

## 预先固定的范围

standard查询295、130、664、10、581、344、498、728，共8例。
这些来自已查看的掩码案例，覆盖公交/交通车辆、自车前景、混合覆盖、隧道和路边车辆；
是事后机制案例，不是随机独立测试集。不得根据本轮检索结果挑多边形或更换样本。
固定最小GT用于核验可见静态重叠；若无法确认，显式skip并记录原因。
不自动换GT、注入正候选、修改数据库或声称受控生成了真实车辆消失。

只改query；全部变体搜索同一完整18871参考库。复用visual_pair_msls_hard_mix_v2的原RU描述子，
不使用其重排头、不重提数据库。8例最多约48次骨干前向（各例baseline、GT验证和4个输入干预）；无训练。
FP32、关闭TF32、默认cuda:1。时间还包括约1GB描述子读取/哈希、缓存核验与CPU全库矩阵乘，未在训练机实测耗时。

## 固定11个变体

- baseline：原图、无bias。
- zero_bias：原图、全零bias，用于路径一致性核验。
- old_late：旧掩码、beta=0.5晚期bias，仅query受干预。
- external_aligned_late / external_shift_late：已确认外部区域与其水平半幅循环平移区域，beta=0.5。
- ego_aligned_late / ego_shift_late：已确认自车区域与同样平移对照，beta=0.5。
- external_aligned_input / external_shift_input：在输入端将对应区域替换为归一化均值RGB，再完整提取特征。
- ego_aligned_input / ego_shift_input：同上，使用自车区域。

多边形在归一化坐标绘制，栅格化为280×280二值区域；晚期面积mask用14×14像素块均值变成20×20。
平移固定140像素，保留每行面积、总面积及循环意义下形状，不保证完全避开原物体；记录相交面积。
这个控制不是完美静态区域对照，平移会切断边界或遮到其他物体。输入填充也会引入边缘伪影，
不是背景补全，更不是“仅改变真实车辆”的因果实验。不存在的类别明确absent，变体保留为零干预并报告nonempty_target_n。

## 第一步：训练机CPU准备

```bash
cd ~/workspace/OpenVPRLab
git pull --ff-only origin main
conda activate VPR
python -m unittest discover -s tests -p test_dynamic_mechanism_probe.py -v
python scripts/dynamic_mechanism_probe.py prepare --output doc/dynamic_mechanism_cases_v1
```

本机PowerShell下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_mechanism_cases_v1 C:\Users\zy\Desktop\MyVPR\doc\
```

打开index.html，在query上按external/ego分别逐点绘制多边形并完成。
对每例选择include/skip、静态重叠yes/no/unknown，每类选择reviewed/absent；跳过须写原因。
未确认状态不能运行。只圈可见目标，不把道路、建筑或被遮挡的未知背景包进去。
页面不自动保存，导出annotations.json到本机案例目录。可以导入继续填写。
这一步需要真实区域核验，不可用“底部25%”或旧分割mask自动替代。

## 第二步：上传区域并运行

本机PowerShell：

```powershell
scp C:\Users\zy\Desktop\MyVPR\doc\dynamic_mechanism_cases_v1\annotations.json wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_mechanism_cases_v1/annotations.json
```

训练机：

```bash
cd ~/workspace/OpenVPRLab
conda activate VPR
python scripts/dynamic_mechanism_probe.py run \
  --prepared doc/dynamic_mechanism_cases_v1 \
  --annotations doc/dynamic_mechanism_cases_v1/annotations.json \
  --source doc/visual_pair_msls_hard_mix_v2 \
  --device cuda:1 \
  --output doc/dynamic_mechanism_probe_v1
```

默认checkpoint从原run.json读取；路径移动时可传--checkpoint，但SHA必须一致。
缺缓存或缓存不匹配会停止，不自动重新提取。输入路径与准备阶段必须一致。
输出目录须不存在；失败输出保留，无completed.json不视为完成，修复后使用新目录。

检查原run/mask/清单哈希、描述子维度和范数、740个缓存top20分数、新提取选定query和固定GT描述子的数值一致性。
这些验证提高缓存对应关系可信度，但旧缓存本身没有逐图内容哈希，不是独立重提全部数据库的复现。
源缓存哈希、本次图像哈希、标注和准备代码哈希写入来源记录。

## 结果解释

输出summary.json、query_outcomes.csv/json、scores.npz（全库分数）、descriptors.npz、intervention_masks.npz和来源/完成记录。
逐例查看最佳GT排名、正负分差、描述子变化、R@1纠错和退步；同时比较aligned与shift，及input与late。
先看工程核验，再看个案。baseline已正确时没有纠错空间，不把分差变化当成新的正确查询。
不能用8个事后案例宣布总体R@1提升，不能据此反复调beta、填充色或多边形。
输入端胜过晚期仍可能来自填充伪影；只有与匹配对照、实例核验共同支持时才值得设计独立验证。
此处query-only结果不直接复现旧query与database双侧干预，不把old_late当成旧全库实验的精确重复。

运行结束下载：

```powershell
scp -r wt@10.16.48.202:/home/wt/workspace/OpenVPRLab/doc/dynamic_mechanism_probe_v1 C:\Users\zy\Desktop\MyVPR\doc\
```

开发验证：CPU单元/准备流程测试和页面JavaScript语法检查；本机未运行GPU模型或完整缓存检索，运行时验证会在训练机执行。
