# depth-anything-deploy

Depth Anything V2-Small(DINOv2 ViT-S/14 主干 + DPT 头,约 25M 参数)在一台笔记本上的两栈部署:

- **GPU 栈**:TensorRT 10.16,RTX 4060 Laptop(Ada,sm_89)。FP32 / TF32 / FP16 / FP8 / INT8 及两种 INT8 混合精度。
- **CPU 栈**:OpenVINO 2026.4 + NNCF 3.4,i9-14900HX(WSL2 分到 4 个 vCPU)。FP32 / INT8 / SmoothQuant 扫描 / Accuracy Control。

两条栈用同一套评测(NYUv2 官方 654 张测试图)、同一个"可接受"标准,所以结论可以并排比较。
本项目的重点不是"把模型跑起来",而是回答三个问题:

1. **ViT 的 INT8 敏感点在哪,和 CNN 有什么不同?**
2. **哪个杠杆真正提速:降精度,还是降 token 数?**
3. **能不能用便宜的仿真代替真实引擎来做量化决策?**

> 所有表格里的数字都由 `scripts/12_report.py` 从 `results/*.json` 生成,不手写。
> 正文里引用的数字与表格一致;如果重跑后数字变了,以表格为准。
> 数字只代表这台笔记本:GPU 动态频率,CPU 跑在 WSL2 里且无法绑定物理 P 核。方法可以迁移到 Jetson Orin 等端侧平台,数字不能。

---

## 结论速览

| 问题 | 实测回答 |
|---|---|
| GPU 上选哪个配方 | **FP16**。精度与 FP32 在 654 张上无法分辨,推理比 TF32 快约 5.8×(518 档)。INT8 / FP8 相对 FP16 **没有可分辨的提速**,只把引擎从 51 MB 压到约 30 MB |
| 为什么 INT8 不比 FP16 快 | 逐 kernel 剖析(518 档):INT8 让线性层 GEMM 从 1.99 ms 降到 1.30 ms、卷积从 1.11 降到 0.78 ms,但融合 MHA 从 1.55 ms 涨到 2.51 ms,总时间打平。注意力内部的两个 BMM 本来就不量化,原因还没拆清 |
| ViT 的 INT8 头号敏感点 | **残差流**。modelopt 为了让 TRT 融合"GEMM + 残差相加",会把残差流本身也量化;只把残差流留在 FP16 就收回大部分精度。fc2(GELU 输出)激活离群也大,但它的回退在 TRT 上触发异常,见坑 5 |
| CPU 上 INT8 | 推理快 2.0×(518 档)/ 2.25×(364 档),精度损失比 GPU INT8 小一个量级。原因与上一条吻合:NNCF 不在残差相加和 LayerNorm 的输入上插量化节点 |
| SmoothQuant 有没有用 | CPU 上有用:α 从关闭 → 0.5 → 0.75 → 0.95(NNCF 默认),ΔAbsRel 单调变小,速度不变 |
| 降 token 数 | 518 → 364 档(1813 → 884 token)FP16 推理快 1.95×,但 AbsRel 上升 0.0067,比任何一个量化配方的代价都大。降分辨率是 GPU 上唯一真正提速的杠杆,也是最贵的杠杆 |
| 仿真能否代替真实引擎 | **不能**。同一张 INT8 QDQ 图,ORT 仿真与真实 TRT 引擎的输出偏差差几个 dB,逐图排序也对不上。所有回退决策都在真实引擎上做(坑 4) |
| C++ 端到端 | GPU:纯 C++ runtime 端到端 6.57 ms / 152 FPS(518 档 FP16),预处理从 Python 的 3.6 ms 降到 0.87 ms;同一份输入下与 Python 输出逐位相同。CPU:OpenVINO 的 C++ 程序同样逐位相同,但只省下约 3 ms 预处理,不到一帧的 1%,推理与 Python 是同一个库、快慢分不出 |

---

## 环境

| 项 | 版本 |
|---|---|
| 机器 | 笔记本,i9-14900HX + RTX 4060 Laptop 8 GB(驱动 581.80),Windows 11 + WSL2 Ubuntu(内存配额 8 GB,4 vCPU) |
| GPU 环境 `dl` | torch 2.12+cu126、transformers 5.22、onnx 1.21、onnxruntime-gpu 1.24、onnx-graphsurgeon 0.6.1、nvidia-modelopt 0.44、tensorrt-cu12 10.16.1 |
| CPU 环境 `ov` | openvino 2026.4.1、nncf 3.4.0、onnx 1.23 |
| 精确版本 | `requirements-dl.txt` / `requirements-ov.txt`(由流水线开头 `pip freeze` 生成) |

两个 conda 环境严格隔离:modelopt 锁定的 onnx 版本与 NNCF 需要的不同,装进同一个环境会互相打架。

---

## 快速开始

全部在 WSL 中运行,入口是 `Makefile`。

```bash
# 0. 数据(一次性,约 1.5 GB)
#    测试集:NYUv2 官方 654 张(HF 数据集 sayakpaul/nyu_depth_v2 的 val 划分)+ 官方划分索引 splits.mat
mkdir -p data/raw && cd data/raw
python3 ../../tools/parallel_fetch.py https://hf-mirror.com/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000000.tar nyu_val-000000.tar --size 1001553920
python3 ../../tools/parallel_fetch.py https://hf-mirror.com/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000001.tar nyu_val-000001.tar --size 14786560
curl -L -o splits.mat https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
cd ../.. && bash tools/fetch_calib_windows.sh     # 校准集:12 个训练 tar 中段各取 40 MB

# 1. 全流程(模型下载 → 导出 → 量化 → 敏感度 → 构建 → 评测 → 基准 → 剖析 → C++ → OpenVINO → OpenVINO C++ → 报告)
make all

# 单步示例
make engines        # 构建全部 TRT 引擎
make eval-gpu       # 654 张测试集评测
make bench-gpu      # 交错轮次延迟基准
make cpp parity     # GPU 栈的 C++ 程序:构建 + 与 Python 对拍 + 端到端基准
make cpp-ov parity-ov   # CPU 栈的 C++ 程序(OpenVINO),同上
make report         # 用 results/ 回填本 README 的表格
make test           # 单元测试(指标、预处理、变体解析、计时)
```

变体用 `<后端>:<分辨率>:<配方>` 表示,例如 `trt:r518:int8res`、`ov:r364:int8sq050`,所有脚本的 `--variants` 都吃这个格式。

---

## 目录结构

```
dadeploy/              公共库(两个环境都能导入的部分不依赖 tensorrt / openvino)
  config.py            单一事实来源:路径、分辨率档位、评测协议、计时协议、接受标准
  data.py              NYUv2 h5 读取与划分清单
  preprocess.py        与 HF preprocessor 一致的预处理 / 后处理
  metrics.py           视差域尺度-平移对齐、AbsRel / δ1 / RMSE、输出偏差、配对 bootstrap
  evaluation.py        所有变体共用的评测主循环
  groups.py            把 Q/DQ 节点映射到"第几个 block 的哪个子层",供消融与统计使用
  trt_recipes.py       TRT 精度配方表(强类型 / 弱类型、读哪个 ONNX)
  trt_utils.py         引擎构建 + IEngineInspector 剖析"引擎里实际跑的精度"
  runners.py           PyTorch / ORT / TRT / OpenVINO 四种 Runner,同一个 infer 接口
  timing.py            分段计时与交错轮次编排
  variants.py          变体字符串解析与产物路径
scripts/00–13          流水线各步,编号即执行顺序,每个脚本开头的说明写了"为什么这么做"
tools/                 下载工具、fc2 异常复现脚本
cpp/                   纯 C++ runtime:da_infer(TensorRT)、da_infer_ov(OpenVINO),预处理 / 后处理共用 da_common.h
tests/                 单元测试
results/               机器可读结果(README 表格的唯一来源)
artifacts/ data/ logs/ 可重新生成的产物,不进版本库
```

---

## 评测协议

- **测试集**:NYUv2 Eigen 划分 654 张,解包后用官方 `splits.mat` 的 `testNdxs` 逐个核对文件名。
- **对齐**:Depth Anything V2 输出的是相对视差,没有绝对尺度。按 MiDaS / Depth Anything 的标准做法,每张图在**视差域**用最小二乘拟合一个 scale 和 shift,再转回深度。不做这一步,AbsRel 没有意义。
- **指标**:Eigen 裁剪区域内、深度 1 mm–10 m 的像素上算 AbsRel、δ1(阈值 1.25)、RMSE。
- **基准**:同分辨率的 PyTorch FP32。每个变体报 ΔAbsRel 及 95% 配对 bootstrap 置信区间(2000 次重采样)。区间不含 0 才说"确实变差",否则只能说"654 张上分辨不出"。
- **输出偏差**:测试集每 6 张取 1 张(109 张),与落盘的 FP32 原始输出比 SQNR。它不需要真值,衡量"换精度后输出变了多少"。注意:任务指标做过逐图尺度-平移对齐,会掩盖相当一部分偏差,需要原始输出保真度的下游(比如拿深度做度量)应看 SQNR。
- **数据纪律**:三个划分互不相交。calib 200 张只用来统计激活范围;val 100 张用于一切决策(敏感度排序、回退选择、NNCF accuracy control);测试集只用于最终报告。calib 和 val 来自 12 个训练场景,与测试场景不重叠。
- **接受标准**(两栈共用):val 集 AbsRel 相对同分辨率 FP32 上升不超过 0.001。GPU 侧 int8mix 的回退选择和 CPU 侧 NNCF 的 `max_drop` 都用它。

修改 `dadeploy/config.py` 里的任一协议参数都会使 `results/` 失效,需要整套重跑。

### 计时协议

- 所有变体一次性加载,跑 5 轮,每轮随机打乱顺序;每段先预热再计时(GPU 20 + 100 次,CPU 3 + 15 次)。这样每个变体的样本都均匀分布在整个时间窗里,笔记本发热降频的漂移被平摊。
- 分段:预处理(CPU)/ H2D / 推理 / D2H / 后处理(CPU)/ 整段墙钟。报 P50 和 P90,不报均值。
- 各轮中位数的相对极差超过 10% 的,表里标 `~`,读数时应视为不稳定。
- 输入固定为一张真实图片的预处理结果,排除图片差异。

---

## 结果

### GPU:TensorRT,分辨率 × 精度

"推理"是 GPU 纯推理(cudaEvent),"端到端"含 Python 预处理与后处理。`*` 表示 ΔAbsRel 的置信区间不含 0。

<!-- RESULTS:GPU:BEGIN -->
| 分辨率 | 配方 | AbsRel | Δ AbsRel vs FP32 [95% CI] | δ1 | SQNR dB | 推理 P50 ms | 端到端 P50 ms | FPS | 产物 MB |
|---|---|---|---|---|---|---|---|---|---|
| 518×686 (1813 tok) | PyTorch fp32 | 0.0617 | 基准 | 0.9557 | — | — | — | — | — |
| 518×686 (1813 tok) | fp32 | 0.0617 | -0.0000 [-0.0000, -0.0000] | 0.9557 | 122.4 | 30.55 | 35.17 | 28.4 | 99 |
| 518×686 (1813 tok) | tf32 | 0.0617 | +0.0000 [-0.0000, +0.0000] | 0.9557 | 68.8 | 29.08 | 33.62 | 29.7 | 99 |
| 518×686 (1813 tok) | fp16 | 0.0617 | +0.0000 [-0.0000, +0.0000] | 0.9557 | 53.6 | ~5.05 | ~9.88 | 101.2 | 51 |
| 518×686 (1813 tok) | fp8 | 0.0633 | +0.0016* [+0.0013, +0.0020] | 0.9550 | 26.1 | 4.88 | 9.86 | 101.4 | 28 |
| 518×686 (1813 tok) | int8 | 0.0668 | +0.0051* [+0.0038, +0.0062] | 0.9510 | 20.3 | 5.23 | 10.04 | 99.6 | 30 |
| 518×686 (1813 tok) | int8res | 0.0629 | +0.0012* [+0.0007, +0.0017] | 0.9547 | 25.3 | ~5.23 | ~10.04 | 99.6 | 33 |
| 518×686 (1813 tok) | int8mix | 0.0630 | +0.0013* [+0.0008, +0.0018] | 0.9549 | 25.3 | 5.21 | 10.04 | 99.6 | 31 |
| 364×476 (884 tok) | PyTorch fp32 | 0.0684 | 基准 | 0.9501 | — | — | — | — | — |
| 364×476 (884 tok) | fp32 | 0.0684 | -0.0000 [-0.0000, +0.0000] | 0.9501 | 122.1 | 9.36 | 11.81 | 84.7 | 97 |
| 364×476 (884 tok) | tf32 | 0.0684 | +0.0000 [-0.0000, +0.0000] | 0.9501 | 69.3 | 8.04 | 10.50 | 95.3 | 97 |
| 364×476 (884 tok) | fp16 | 0.0684 | -0.0000 [-0.0000, -0.0000] | 0.9501 | 52.9 | 2.59 | 5.10 | 196.1 | 50 |
| 364×476 (884 tok) | fp8 | 0.0698 | +0.0014* [+0.0010, +0.0018] | 0.9483 | 26.0 | 2.50 | 4.95 | 201.8 | 27 |
| 364×476 (884 tok) | int8 | 0.0734 | +0.0050* [+0.0040, +0.0060] | 0.9439 | 20.9 | 2.57 | 4.95 | 201.8 | 29 |
| 364×476 (884 tok) | int8res | 0.0700 | +0.0016* [+0.0011, +0.0021] | 0.9481 | 24.1 | 2.58 | 5.10 | 196.0 | 28 |
| 364×476 (884 tok) | int8mix | 0.0704 | +0.0020* [+0.0013, +0.0026] | 0.9476 | 23.0 | 2.59 | 5.04 | 198.3 | 28 |
| 266×350 (475 tok) | PyTorch fp32 | 0.0696 | 基准 | 0.9491 | — | — | — | — | — |
| 266×350 (475 tok) | fp16 | 0.0695 | -0.0000 [-0.0000, +0.0000] | 0.9491 | 50.9 | ~1.59 | ~3.03 | 329.6 | 49 |
<!-- RESULTS:GPU:END -->

配方说明:

| 配方 | 构建方式 |
|---|---|
| fp32 | 弱类型,关闭 TF32,数值基准 |
| tf32 | 弱类型,TRT 默认(FP32 张量 + TF32 tensor core) |
| fp16 | 强类型,modelopt AutoCast 把全图转成 FP16,输入输出保持 FP32 |
| fp8 | 强类型,modelopt FP8(E4M3)PTQ,max 校准,其余 FP16 |
| int8 | 强类型,modelopt 默认 INT8 PTQ(权重 per-channel、激活 per-tensor、熵校准),会量化残差流,其余 FP16 |
| int8res | 在 int8 图上把**全部**残差流分组的 Q/DQ 拆掉(规则化配置) |
| int8mix | 在 int8 图上按 val 集真实引擎消融逐块选择回退(数据驱动配置) |

三个 INT8 配方共用同一份校准,只拆 Q/DQ 不重新校准,所以它们之间的差异只来自"哪些位置量化"。

### CPU:OpenVINO,分辨率 × 精度

SmoothQuant 扫描只在 364 档上做(CPU 评测一张约 1 秒,全量扫两档太慢),518 档只跑结论配方。

<!-- RESULTS:CPU:BEGIN -->
| 分辨率 | 配方 | AbsRel | Δ AbsRel vs FP32 [95% CI] | δ1 | SQNR dB | 推理 P50 ms | 端到端 P50 ms | FPS | 产物 MB |
|---|---|---|---|---|---|---|---|---|---|
| 518×686 (1813 tok) | fp32 | 0.0617 | +0.0000 [-0.0000, +0.0000] | 0.9557 | 119.9 | 670.91 | 674.87 | 1.5 | 95 |
| 518×686 (1813 tok) | int8 | 0.0622 | +0.0005* [+0.0001, +0.0009] | 0.9558 | 31.9 | 335.68 | 339.91 | 2.9 | 27 |
| 518×686 (1813 tok) | int8ac | 0.0622 | +0.0005* [+0.0001, +0.0009] | 0.9558 | 31.9 | 335.24 | 339.49 | 2.9 | 27 |
| 364×476 (884 tok) | fp32 | 0.0684 | -0.0000 [-0.0000, +0.0000] | 0.9501 | 120.8 | 265.83 | 268.28 | 3.7 | 94 |
| 364×476 (884 tok) | int8 | 0.0686 | +0.0001 [-0.0002, +0.0005] | 0.9503 | 31.6 | 118.14 | 120.37 | 8.3 | 26 |
| 364×476 (884 tok) | int8sqoff | 0.0692 | +0.0008* [+0.0004, +0.0011] | 0.9495 | 29.2 | 116.94 | 119.20 | 8.4 | 26 |
| 364×476 (884 tok) | int8sq050 | 0.0690 | +0.0006* [+0.0003, +0.0009] | 0.9498 | 31.4 | 116.45 | 118.64 | 8.4 | 26 |
| 364×476 (884 tok) | int8sq075 | 0.0689 | +0.0004* [+0.0001, +0.0008] | 0.9499 | 30.9 | 116.87 | 119.10 | 8.4 | 26 |
| 364×476 (884 tok) | int8ac | 0.0686 | +0.0001 [-0.0002, +0.0005] | 0.9503 | 31.6 | 117.89 | 120.08 | 8.3 | 26 |
<!-- RESULTS:CPU:END -->

| 配方 | 做法 |
|---|---|
| fp32 | `ov.convert_model` + `save_model(compress_to_fp16=False)`(默认会把权重压成 FP16,见坑 6) |
| int8 | `nncf.quantize(model_type=TRANSFORMER, preset=MIXED)`,NNCF 默认 SmoothQuant α=0.95 |
| int8sqoff / int8sq050 / int8sq075 | 同上,α 分别为关闭 / 0.5 / 0.75 |
| int8ac | `nncf.quantize_with_accuracy_control`,val 集 1 − AbsRel 为指标,`max_drop` = 接受标准 |

没有 FP16 配方:14900HX 没有 AVX512-FP16 / AMX,FP16 权重仍要解压到 FP32 计算,不会更快。

### 引擎里实际跑的精度

"我以为它是 INT8"与"它真的是 INT8"之间可能有差距。下表来自 `IEngineInspector` 逐层导出的 tactic 名(按 `f16f16` / `tf32` / `i8i8` / `e4m3` 等标记统计计算层)。

<!-- RESULTS:KERNELS:BEGIN -->
| 引擎 | 计算层实际 kernel 精度 | 融合 MHA | kernel 数 | 构建 s |
|---|---|---|---|---|
| trt:r518:fp32 | fp32 90, unknown 13 | 0 | 189 | 20 |
| trt:r518:tf32 | tf32 53, fp32 39, unknown 11 | 0 | 193 | 16 |
| trt:r518:fp16 | fp16 76 | 12 | 155 | 6 |
| trt:r518:fp8 | fp8 48 | 12 | 179 | 16 |
| trt:r518:int8 | int8 76 | 12 | 160 | 7 |
| trt:r518:int8res | int8 76 | 12 | 160 | 6 |
| trt:r518:int8mix | int8 76 | 12 | 160 | 6 |
| trt:r364:fp32 | fp32 83, unknown 20 | 0 | 186 | 15 |
| trt:r364:tf32 | tf32 53, fp32 31, unknown 19 | 0 | 184 | 14 |
| trt:r364:fp16 | fp16 77 | 12 | 150 | 4 |
| trt:r364:fp8 | fp8 48 | 12 | 176 | 15 |
| trt:r364:int8 | int8 77 | 12 | 175 | 5 |
| trt:r364:int8res | int8 77 | 12 | 175 | 5 |
| trt:r364:int8mix | int8 77 | 12 | 175 | 5 |
| trt:r266:fp16 | fp16 78 | 12 | 151 | 65 |
<!-- RESULTS:KERNELS:END -->

读法:fp32 / tf32 引擎没有融合 MHA,注意力被拆成多个 GEMM;fp16 / int8 / fp8 引擎都有 12 个融合 MHA kernel(每个 block 一个)。

### 逐 kernel 耗时归因(518 档)

IProfiler 会在 kernel 之间插同步,总时间比正常执行偏大,这里只看**占比**;绝对延迟以基准表为准。

<!-- RESULTS:PROFILE:BEGIN -->
| 引擎 | attention_mha | attention_gemm | linear_gemm | kgen | conv | reformat | other |
|---|---|---|---|---|---|---|---|
| trt:r518:fp16 | 29.1% | 0.0% | 37.3% | 6.8% | 20.8% | 4.0% | 2.0% |
| trt:r518:int8 | 46.8% | 0.0% | 24.2% | 7.9% | 14.6% | 4.5% | 2.0% |
| trt:r518:int8res | 46.7% | 0.0% | 25.7% | 7.0% | 14.7% | 4.1% | 1.8% |
| trt:r518:fp8 | 24.9% | 0.0% | 29.7% | 17.3% | 1.5% | 0.0% | 26.6% |
<!-- RESULTS:PROFILE:END -->

### INT8 敏感度:在真实引擎上做类型级消融

每一行是一个实验:在同一张 INT8 图上只把某一类子层(12 个 block 的同名位置)拆回 FP16,强类型构建引擎,在 val 100 张上评测。

<!-- RESULTS:SENSITIVITY:BEGIN -->
**518×686 (1813 tok)**,FP32 val AbsRel = 0.0607,接受线 ≤ 0.0617:

| 配置(val 100 张,真实引擎) | AbsRel | SQNR dB |
|---|---|---|
| 全部 INT8 | 0.0638 | 21.5 |
| 全部 FP16(拆掉全部 Q/DQ) | 0.0607 | 54.1 |
| 仅 residual 回退 FP16 | 0.0616 | 26.4 |
| 仅 patch_embed 回退 FP16 | 0.0634 | 21.2 |
| 仅 head 回退 FP16 | 0.0635 | 21.5 |
| 仅 fc1 回退 FP16 | 0.0635 | 21.8 |
| 仅 neck.fusion 回退 FP16 | 0.0635 | 21.5 |
| 仅 neck.reassemble 回退 FP16 | 0.0638 | 21.5 |
| 仅 qkv 回退 FP16 | 0.0641 | 21.3 |
| 仅 o_proj 回退 FP16 | 0.0646 | 21.0 |
| 仅 fc2 回退 FP16 | 0.2081 | 4.4 |
| 最终 int8mix(10 组 FP16) | 0.0617 | 26.1 |

**364×476 (884 tok)**,FP32 val AbsRel = 0.0696,接受线 ≤ 0.0706:

| 配置(val 100 张,真实引擎) | AbsRel | SQNR dB |
|---|---|---|
| 全部 INT8 | 0.0736 | 21.3 |
| 全部 FP16(拆掉全部 Q/DQ) | 0.0696 | 53.7 |
| 仅 residual 回退 FP16 | 0.0706 | 24.1 |
| 仅 fc1 回退 FP16 | 0.0723 | 21.8 |
| 仅 patch_embed 回退 FP16 | 0.0723 | 21.3 |
| 仅 head 回退 FP16 | 0.0731 | 22.1 |
| 仅 o_proj 回退 FP16 | 0.0731 | 21.5 |
| 仅 qkv 回退 FP16 | 0.0733 | 21.6 |
| 仅 neck.reassemble 回退 FP16 | 0.0735 | 21.3 |
| 仅 neck.fusion 回退 FP16 | 0.0735 | 21.3 |
| 仅 fc2 回退 FP16 | 0.2172 | 3.5 |
| 最终 int8mix(5 组 FP16) | 0.0698 | 23.7 |
<!-- RESULTS:SENSITIVITY:END -->

流程:类型级留一法排序 → 按改善从大到小贪心回退,直到达标 → 在已回退的类型里,按激活统计"从好量化到难量化"的顺序逐个 block 试着改回 INT8,仍达标就保留。两档都只选中了残差流这一类,518 档最终 10 个 block 留 FP16,364 档 5 个。

### 激活统计:为什么残差流难量化(518 档,16 张)

在 PyTorch FP32 模型上挂 hook,对 modelopt 实际插 Q/DQ 的每类张量统计。"通道离群比"= 各通道 absmax 的最大值 / 中位数;"单张量 SQNR"= 只量化这一个张量时的 SQNR。

<!-- RESULTS:ACTIVATION:BEGIN -->
| 量化点 | 通道离群比 中位 | 通道离群比 最大 | 单张量 INT8 SQNR(max 校准) | SQNR(99.99% 截断) |
|---|---|---|---|---|
| q/k/v 输入(norm1 输出) | 4.7 | 7.4 | 30.9 | 32.5 |
| o_proj 输入 | 3.4 | 57.1 | 33.4 | 34.6 |
| fc1 输入(norm2 输出) | 4.4 | 12.6 | 30.5 | 33.7 |
| fc2 输入(GELU 输出) | 7.6 | 17.7 | 23.3 | 26.3 |
| 残差流(注意力相加后) | 6.9 | 18.6 | 26.4 | 30.2 |
| 残差流(block 输出) | 11.1 | 21.6 | 22.9 | 27.9 |
<!-- RESULTS:ACTIVATION:END -->

block 输出处的残差流通道离群比中位数最高、单张量 SQNR 最低,而且离群集中在固定的几个通道上。per-tensor 激活量化只有一个 scale,要么为这几个通道牺牲其余所有通道的分辨率,要么截断它们。这正是 SmoothQuant 要"把难度从激活挪到权重"的对象,也解释了为什么残差流回退收益最大。

### C++ runtime

两个程序,预处理和后处理是同一份代码(`cpp/da_common.h`),只有推理一段不同:

- `cpp/da_infer.cpp`(GPU,TensorRT):读图 → bicubic 缩放 → 归一化查表直接写进锁页内存 → H2D → 推理 → D2H → 双线性放回原尺寸。计时 500 帧。
- `cpp/da_infer_ov.cpp`(CPU,OpenVINO):同样的预处理直接写进推理请求的输入张量 → 推理 → 双线性放回原尺寸,没有 H2D / D2H。编译配置与 Python 的 `OvRunner` 相同(LATENCY、4 线程、FP32 精度提示)。计时 100 帧。

对拍分三层:预处理(两边 OpenCV 版本不同,bicubic 定点实现允许 ±1 灰度级)、同一份输入下的推理输出(应逐位相同,隔离预处理差异)、端到端 AbsRel。

<!-- RESULTS:CPP:BEGIN -->
| 变体 | 预处理 | H2D | 推理 | D2H | 后处理 | 总计 ms | FPS | 同输入输出最大差 | 预处理最大差(灰度级) | AbsRel 差 |
|---|---|---|---|---|---|---|---|---|---|---|
| trt:r518:fp16 | 0.87 | 0.35 | 5.00 | 0.13 | 0.12 | 6.57 | 152.1 | 0.0e+00 | 1.02 | -2.2e-05 |
| ov:r518:fp32 | 0.90 | — | 637.39 | — | 0.13 | 638.52 | 1.6 | 0.0e+00 | 1.02 | -3.2e-05 |
| ov:r518:int8 | 0.90 | — | 314.35 | — | 0.13 | 315.37 | 3.2 | 0.0e+00 | 1.02 | -4.9e-05 |
<!-- RESULTS:CPP:END -->

与 Python 端到端的差距几乎全在预处理:Python 版是 numpy 整图浮点运算(转 float32 → 归一化 → HWC 转 CHW,每步都生成一个新数组),再从可分页内存拷到 GPU;C++ 版用 256 项查找表,在一次遍历里完成归一化和 CHW 重排,直接写进锁页缓冲。

CPU 栈上 C++ 能带来的只有预处理这一段:518 档从 Python 的约 3.9 ms 降到 0.90 ms,而一帧要 315–640 ms,省下的不到 1%。推理调用的是同一个 OpenVINO 库、同一份 IR、同样的配置,同输入下输出逐位相同,耗时也就没有理由不同:表里的 637 / 314 ms 比 CPU 结果表(11 月 8 日测)的 671 / 336 ms 低约 5%,但 C++ 基准当晚在同一次开机里重测 Python 得到 663 / 311 ms,轮间极差 11–15%(被标为不稳定),C++ 自己重跑一次是 655 / 345 ms —— 这 5% 是机器状态的漂移,不是 C++ 的收益。C++ 基准是单进程连续计时,没有交错轮次,所以表里 CPU 两行的推理耗时只能读到这个精度。

INT8 那一行的端到端输出最大差(1.32)比 FP32(0.30)大:C++ 与 Python 的输入相差至多 1 个灰度级,量化模型把这点差异放大得更多。逐图 AbsRel 的差最大 6e-4,5 张图的平均差 -4.9e-5,都在接受标准 0.001 之内。

---

## 关键结论展开

### 1. GPU 上 FP16 是甜点,INT8 / FP8 买到的是体积不是速度

518 档 FP16 推理 5.05 ms,INT8 5.23 ms、FP8 4.88 ms,三者在交错基准的噪声范围内(FP16 那一行本身就被标了不稳定)。精度上 FP16 与 FP32 分辨不出,INT8 和 FP8 都可分辨地变差。

剖析给出了原因:INT8 确实让线性层 GEMM 和卷积变快了(合计省约 1 ms),但同一个引擎里融合 MHA 反而慢了约 1 ms,总时间打平。FP8 引擎里 kgen 和"其他"两类的耗时明显上升(合计从约 0.5 ms 涨到约 2.5 ms),吃掉了 GEMM 的收益;"其他"具体是哪些 kernel 还没逐个归类。

这和 dcn-trt10 项目里"INT8 不比 FP16 快"的结论是同一类现象:4060 的 FP16 tensor core 已经很强,低精度的计算收益会被格式转换与非 GEMM 部分抵消。在这张卡上,量化的实际价值是把引擎从约 51 MB 压到约 30 MB。

### 2. ViT 和 CNN 的敏感点不一样

YOLOv8 上量化的头号敏感点是检测头 DFL 解码链里的单个算子(`model.22/mul`),回退一个节点就收回大部分精度。ViT 上不是某个算子,而是**一条贯穿 12 个 block 的数据通路**——残差流。它的幅值随深度累积、离群集中在固定通道,而 modelopt 的默认配置为了算子融合把它也量化了。

两条栈正好互为对照:NNCF 生成的 IR 里,FakeQuantize 只喂给卷积、MatMul、注意力、Reshape 和上采样,不碰残差相加与 LayerNorm 的输入,CPU INT8 的精度损失也就小一个量级。

### 3. 数据驱动的回退不比规则更好

int8mix 是在 val 集上逐块选出来的,int8res 只是"残差流全部 FP16"这条规则。到了测试集上,int8mix 并没有更好(518 档两者几乎一样,364 档 int8res 略好),而且两者都没有达到"FP32 + 0.001"的接受线。逐块回收那一步的差异接近 val 集的噪声,它在 val 上省下的几个 FP16 block,在测试集上没有兑现。交付时应选 int8res:同样的精度,规则更简单,也不依赖某一份 val 集。

### 4. CPU 上 INT8 是真加速,Accuracy Control 什么都没做

CPU 上 INT8 推理快 2.0–2.25×,和 OpenVINO 前几个 YOLO 项目的 2.5× 量级一致(那是另一台台式机,数字不可直接比)。364 档 INT8 的 ΔAbsRel 置信区间包含 0。

`int8ac` 和 `int8` 的结果完全相同:默认配方在 val 集上已经满足 `max_drop`,accuracy control 没有回退任何层。这本身是一个有用的结论——在 CPU 栈上,NNCF 的默认 Transformer 配方已经够用。

---

## 踩过的坑

### 坑 1:位置编码插值会留在图里

DINOv2 预训练时位置编码是 37×37 网格,推理分辨率不同时,forward 里每次都要对它做 bicubic 插值。动态形状导出会把这次插值连同一整串 Shape / Gather / Concat 留在图里,每帧重算一个常数;端侧编译器(RKNN 等)还常常不支持 cubic Resize。

做法:固定输入形状,用 onnx-graphsurgeon 迭代常量折叠到不动点(19 轮),节点数 1114 → 518,Resize 6 → 5,与 PyTorch 的最大绝对误差约 5e-6。没用 onnxsim:它会把 3D MatMul + Add 改写成 Reshape → 2D Gemm → Reshape,破坏 Transformer 的标准子图形态,对 TRT 注意力融合和 modelopt 的模式匹配都不友好。

### 坑 2:弱类型的 FP16 flag 不保证 FP16

弱类型网络里,`BuilderFlag.FP16` 只是"允许"用 FP16,每层用什么精度由 builder 按计时结果挑。笔记本 GPU 计时抖动大,同一个 ONNX、同一组 flag 连续构建 3 次(`make determinism`),有 1 次退回了 TF32 + FP32 混合、没有融合 MHA,引擎 89.9 MB;另外 2 次是全 FP16 + 12 个融合 MHA,49.8 MB。

"FP16 引擎"本身都不是一个确定的东西,对照实验不能接受。所以除 fp32 / tf32 外,所有参与比较的引擎都用**强类型**构建:精度完全由 ONNX 里的张量类型(FP16 initializer / Cast / Q-DQ)决定,构建可复现。

### 坑 3:INT8 图里残差流也被量化了

modelopt 为了让 TRT 把"GEMM + 残差相加"融成一个 kernel,会在残差 Add 和 LayerNorm 的输入上插 Q/DQ。这对 CNN 的 Conv + Add 是合理的,对 ViT 的残差流就是精度灾难,见上面的敏感度表。int8res 配方就是专门把这些 Q/DQ 拆掉。

### 坑 4:ORT 假量化不能代替真实引擎

最初的敏感度排序打算用 ORT 跑 QDQ 图(便宜、不用构建引擎)。实测不行:

- 第一轮 ORT 静默回退到了 CPU EP,FP16 QDQ 图的仿真 SQNR 只有约 2.7 dB,结果整轮作废;`OrtRunner` 现在拒绝回退,CUDA EP 不可用直接报错。
- 即使在 CUDA EP 上,INT8 图的仿真偏差(val 前 8 张,`results/quantize.json`)约 15 dB,而真实 TRT 引擎约 21 dB;逐图的偏差排序也基本不相关。TRT 把 Q/DQ 融进 kernel,取整与累加精度和 ORT 逐算子执行不同。

既然仿真排不准,决策只能在真实引擎上做。为了控制成本,消融粒度从 64 个分组提到 9 个子层类型,每档 23 次构建 + 评测,约 5–7 分钟。

### 坑 5:INT8 fc1 + FP16 fc2 的组合在 TRT 上失真(根因未证实)

只拆掉 fc2 激活的 Q/DQ 时,TRT 给 fc1 选了 `i8f32`(INT8 进、FP32 直出)→ GELU → FP16 fc2,输出 SQNR 从约 21 dB 跌到约 3.5 dB;ORT CUDA EP 执行同一张图是正常的,其他拆法也正常。`tools/probe_fc2_anomaly.py` 不用 timing cache 可以稳定复现:

<!-- RESULTS:ANOMALY:BEGIN -->
| 拆掉的 Q/DQ | ORT CUDA SQNR dB | TensorRT SQNR dB |
|---|---|---|
| fc2_activation_and_weight | 18.0 | 3.5 |
| fc2_activation_only | 17.1 | 3.9 |
| fc2_weight_only | 17.1 | 21.4 |
| fc1_and_fc2 | 17.0 | 22.7 |
<!-- RESULTS:ANOMALY:END -->

这是从外部黑盒观察到的,只能说"疑似 TensorRT 10.16 的 Myelin 代码生成缺陷",没有证实。它是"每个候选配置都必须在真实引擎上验证"的又一个理由:如果只看 ORT,会把一个在部署时输出崩坏的配置当成合格。

### 坑 6:OpenVINO 的 save_model 默认把权重压成 FP16

`ov.save_model` 默认 `compress_to_fp16=True`,"FP32 基准"会悄悄带上权重舍入误差。FP32 配方必须显式关掉。

### 坑 7:AutoCast 的参考推理会撑崩 WSL

modelopt AutoCast 可以先跑一遍参考推理、按真实幅值决定哪些节点留 FP32。它要记录全部中间张量:364 档 4 张图就吃 7.6 GB,518 档直接撑崩了 8 GB 配额的 WSL。在 364 档上跑过一次:所有幅值都低于 FP16 上限,没有节点需要留 FP32(518 档激活统计也显示 absmax ≤ 55)。所以默认关掉这一步,FP16 是否可用由引擎实测 SQNR 验证。

### 坑 8:NYUv2 数据下载

HF 镜像单连接限速约 90 KB/s 且会中途断流。`tools/parallel_fetch.py` 把文件切块、多线程 Range 请求、逐块断点续传,速度到约 2.7 MB/s。校准集不下载 3 GB 的完整训练 tar,而是从 12 个 tar 的正中间各取 40 MB 字节窗口,扫描 tar 头恢复完整成员,共下载约 1.5 GB 而不是 36 GB;取中段而不是头部,是为了让 12 个窗口覆盖 12 个不同场景。

---

## 已知边界

- **数字只代表这台机器**。GPU 是笔记本 RTX 4060(动态频率);CPU 在 WSL2 里只有 4 个 vCPU,无法绑定物理 P 核,Windows 可能把 vCPU 调度到 E 核上,CPU 数字的不确定度比 GPU 大。不要当成 Jetson 或任何端侧板子的数字引用。
- **只有 batch = 1、静态形状**。三档分辨率各自一个引擎,没有做动态形状 profile。
- **端到端计时不含取流与显示**。输入是内存里的一张图。
- **fc2 异常的根因未证实**,见坑 5。
- **没有 RKNN 栈**。手头没有 RK3588 板子;拿到板子后计划作为第三栈补上(固定形状 + 常量折叠已经为此铺好路,见坑 1)。

---

## 许可

- 模型权重:Depth Anything V2-**Small** 为 Apache-2.0。Base / Large 为 CC-BY-NC-4.0,不可商用,本项目不使用。
- 数据:NYU Depth V2,仅用于评测,不随仓库分发。
- 本仓库代码:MIT,见仓库根目录的 [LICENSE](../LICENSE)。
