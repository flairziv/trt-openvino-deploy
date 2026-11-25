# 04 TensorRT 构建

对应 `dadeploy/trt_recipes.py`、`dadeploy/trt_utils.py`、`scripts/05_quantize_modelopt.py`、`scripts/07_build_trt.py`。

做 YOLO 时常见的流程是:FP32 的 ONNX 交给 builder,设 `FP16` / `INT8` flag,挂一个 calibrator。这个项目反过来:**精度写在 ONNX 里,引擎以强类型构建**,builder 不再做精度选择。

## 为什么不用 flag

弱类型网络里,`BuilderFlag.FP16` 只是"允许"用 FP16,每层实际用什么由 builder 按计时结果挑。`make determinism` 用同一份 ONNX、同一组 flag 在 364 档连续构建 3 次(`results/engines.json` 里带 `#` 的条目):

| 次 | 引擎大小 | 计算层精度 | 融合注意力 |
|---|---|---|---|
| 0 | 89.9 MB | fp16 29 / tf32 48 / fp32 24 | 0 |
| 1 | 49.8 MB | fp16 77 | 12 |
| 2 | 49.8 MB | fp16 77 | 12 |

"FP16 引擎"本身不确定,对照实验就没法做。强类型构建下,同一份 ONNX 得到的精度是确定的。

## 配方表

`trt_recipes.TRT_RECIPES` 决定两件事:从哪份 ONNX 构建,用强类型还是弱类型。

| 配方 | 读哪份 ONNX | 构建方式 | 用途 |
|---|---|---|---|
| fp32 | 基础 FP32 图 | 弱类型,关闭 TF32 | 数值基准 |
| tf32 | 基础 FP32 图 | 弱类型,TRT 默认 | 速度基准 |
| fp16 | `.fp16.onnx` | 强类型 | 对比项 |
| int8 | `.int8.onnx` | 强类型 | 对比项 |
| int8res | `.int8res.onnx` | 强类型 | 残差流全部留 FP16 |
| int8mix | `.int8mix.onnx` | 强类型 | 按 val 消融逐块选择 |
| fp8 | `.fp8.onnx` | 强类型 | 对比项 |
| fp16w | 基础 FP32 图 | 弱类型 + FP16 flag | 只为复现上面的不确定性 |

## 精度怎么写进 ONNX

都在 `scripts/05_quantize_modelopt.py`,由 nvidia-modelopt 完成。

**FP16**(`run_autocast`):`convert_to_mixed_precision(..., low_precision_type="fp16", keep_io_types=True)`。权重存成 FP16,图的首尾各加一个 Cast,输入输出仍是 FP32,所以 Runner 和 C++ 端不用改。

**INT8 / FP8**(`run_modelopt`;最后一行在源码里是 `**params`,这里按 `QUANT_MODES` 展开):

```python
quantize(
    str(src),
    calibration_data={INPUT_NAME: calib},     # calib 划分 200 张
    calibration_eps=["cuda:0", "cpu"],
    high_precision_dtype="fp16",              # 没被 Q/DQ 覆盖的部分转成 FP16
    output_path=str(dst),
    quantize_mode="int8", calibration_method="entropy",   # fp8 时是 "fp8" / "max"
)
```

- 校准在这一步用 ONNX Runtime 完成,scale 写进 Q/DQ(QuantizeLinear / DequantizeLinear)节点。TensorRT 那边不挂 calibrator。
- 这一步只"写下量化方案":权重在文件里仍是浮点,真正的 INT8 kernel 由 TensorRT 构建时把 Q/DQ 融进算子后产生。
- 没被 Q/DQ 覆盖的部分是 FP16,所以"INT8 引擎"实际是 INT8 + FP16 混合。
- 调用前必须先 `preload_ort_cuda()`,否则校准会话静默退回 CPU。

产物(518 档):

| 图 | 节点数 | Q/DQ 对数 | 说明 |
|---|---|---|---|
| `.onnx` | 518 | 0 | FP32 |
| `.fp16.onnx` | 519 | 0 | 多 2 个 Cast |
| `.int8.onnx` | 1017 | 249 | 校准 314 秒 |
| `.fp8.onnx` | 1083 | 282 | 校准 64 秒;注意力里的矩阵乘也量化(96 个 MatMul 全覆盖) |

Q/DQ 具体放在哪,见 [05](05_int8_sensitivity.md)。

## 构建

`trt_utils.build_engine` 的核心:

```python
flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if recipe.strongly_typed else 0
network = builder.create_network(flags)
parser.parse_from_file(str(onnx_path))
cfg = _make_config(builder, recipe, cache)        # 强类型时不设任何精度 flag
serialized = builder.build_serialized_network(network, cfg)
```

`_make_config` 里值得注意的几项:

- 工作区上限 3 GB(8 GB 的显存要和显示共用);
- `ProfilingVerbosity.DETAILED`:后面看逐层信息和做剖析都需要它;
- timing cache:06 和 07 共用 `artifacts/engines/timing.cache`,重建更快,tactic 选择也更稳定;
- 强类型网络不能再设精度 flag,builder 会报错。

用 trtexec 的话对应 `--stronglyTyped`;同时给 `--fp16` 之类的开关,它会关掉并告警。

引擎和 timing cache 都绑定 GPU 型号和 TensorRT 版本,换机器必须重建。

## 看引擎里实际跑的是什么

构建完,`inspect_layers` 用 `IEngineInspector` 导出每层的类型和 tactic 名,存成引擎旁边的 `.layers.json`。`classify_tactic` 靠 tactic 名里的标记判断精度:

| tactic 名里的标记 | 精度 |
|---|---|
| `f16f16` | FP16 |
| `i8i8`、`i8f32`、`i8i32` | INT8(后两个是 INT8 进、高精度出) |
| `e4m3` | FP8 |
| `tf32` | TF32 |
| `f32f32` | FP32 |

一个 block 的 36 个 ONNX 节点,在 FP16 引擎里是 7 个 kernel(在 `da2s_r518.fp16.layers.json` 里搜 `layer.0`):

| kernel | 内容 |
|---|---|
| kgen | LayerNorm(并入了前一个 block 的 ×γ 和残差相加) |
| gemm | q / k / v 三个线性层合成一个 |
| kgen `_gemm_mha_v2` | 融合注意力:两个矩阵乘加 softmax |
| gemm | o_proj |
| kgen | ×γ、残差相加、norm2 |
| fusion `Fc` | fc1(GELU 没有单独的 kernel) |
| gemm | fc2 |

fp32 / tf32 引擎没有融合注意力,注意力被拆成多个 GEMM,这是它们比 FP16 慢近 6 倍的主要原因之一(518 档:29.08 ms 对 5.05 ms)。

README 的"引擎里实际跑的精度"表就是这份统计。它少数了 5 层,见 [09](09_open_questions.md)。

## 结果一览(518 档,测试集)

| 配方 | Δ AbsRel | SQNR dB | 推理 P50 ms | 引擎 MB |
|---|---|---|---|---|
| fp32 | 0 | 122.4 | 30.55 | 99 |
| tf32 | 0 | 68.8 | 29.08 | 99 |
| fp16 | 0 | 53.6 | 5.05 | 51 |
| fp8 | +0.0016 | 26.1 | 4.88 | 28 |
| int8 | +0.0051 | 20.3 | 5.23 | 30 |
| int8res | +0.0012 | 25.3 | 5.23 | 33 |
| int8mix | +0.0013 | 25.3 | 5.21 | 31 |

FP16 是甜点:精度与 FP32 分辨不出,速度和 INT8 / FP8 在噪声范围内相同。量化在这张卡上买到的是体积,原因见 [07](07_inference_timing.md) 的剖析。
