# 03 评测口径

对应 `dadeploy/metrics.py`、`evaluation.py`、`data.py`、`reference.py`。检测任务看 mAP;这里是逐像素的深度,指标和统计方法都不一样。

## 任务指标:先对齐再打分

模型输出的是相对视差,没有绝对尺度,不能直接和真值深度比。`align_disparity_to_depth` 对每张图做一次最小二乘:

```
求 s, t 使  || s · pred + t − 1/gt ||²  最小(只在有效像素上)
对齐后的深度 = 1 / max(s · pred + t, 1/10)
```

之后在有效像素上算三个指标(`depth_metrics`):

- AbsRel = mean(|pred − gt| / gt),越小越好;
- δ1 = max(pred/gt, gt/pred) < 1.25 的像素比例,越大越好;
- RMSE。

有效像素(`eigen_valid_mask`):真值深度在 1 mm 到 10 m 之间,且落在 Eigen 裁剪框 `(45, 471, 41, 601)` 内。

跳过对齐,AbsRel 没有意义。

## 输出偏差:SQNR

`output_drift` 直接比较原始输出和落盘的 PyTorch FP32 输出,不需要真值:

```
SQNR = 10 · log10( Σ ref² / Σ (pred − ref)² )   单位 dB,每 +6 dB 约等于误差减半
```

为什么两个指标都要看:逐图对齐会吸收一部分偏差。一个配置的 AbsRel 达标,不代表它的输出接近 FP32。下游如果直接使用原始输出,要看 SQNR。

量级参考(518 档,测试集 drift 子集):TensorRT FP32 122.4 dB,TF32 68.8 dB,FP16 53.6 dB,INT8 20.3 dB。

## 配对 bootstrap

量化造成的 AbsRel 差异在 0.001 量级,图与图之间的难度差异大得多。直接比两个均值,差异会被淹没。

`paired_bootstrap` 的做法:两个变体在同一批图上逐图相减,对差值重采样 2000 次,取 2.5% 和 97.5% 分位作为 95% 置信区间。

- 区间不含 0,才说"确实变差";
- 区间含 0,只能说"654 张上分辨不出"。

README 表格里带 `*` 的就是区间不含 0 的行。

## 三份数据各管什么

| 划分 | 张数 | 用途 |
|---|---|---|
| calib | 200 | 只用来统计激活范围(校准) |
| val | 100 | 一切决策:敏感度排序、回退选择、NNCF accuracy control |
| test | 654 | 只出最终报告,不在上面挑配置 |
| drift | 109 | test 每 6 张取 1,用来算 SQNR |

- calib 和 val 来自 12 个训练场景,与测试场景不重叠。
- 划分只由 `data/nyu/manifest.json` 定义,代码里一律用 `dadeploy.data.split_names` / `iter_split` 取。
- 接受标准(两栈共用):val 上 AbsRel 相对同分辨率 FP32 上升不超过 0.001(`config.ACCEPT_ABSREL_DELTA`)。

## 参考输出为什么要落盘

`ov` 环境里没有 torch,无法现场算 FP32 参考。所以 `scripts/03_reference.py` 在 `dl` 环境里算好,存到 `artifacts/reference/<res>_<split>.npy`(`[N, H, W]` float32),两个环境共用。旁边的 `.json` 记录样本名顺序,`load_reference` 读取时逐一核对。

参考必须存成 FP32:存成 FP16 引入的误差和要测的 FP16 引擎偏差是同一量级。

## 评测主循环

所有变体的测试集评测都走 `dadeploy.evaluation.evaluate`,每张图是:

```
rgb(480×640) → preprocess → runner.infer → 原始输出 [H, W]
   ├─ 这张图在参考集里:和 FP32 参考比 → SQNR
   └─ postprocess 放回 480×640 → 对齐 → AbsRel / δ1 / RMSE
```

预处理和后处理只有一份实现(`dadeploy/preprocess.py`),四种后端共用。这是对照公平的前提。

## 基准精度

PyTorch FP32 在测试集上的 AbsRel:518 档 0.0617,364 档 0.0684,266 档 0.0696。降分辨率的代价(518 → 364 是 +0.0067)比任何一个量化配方都大。
