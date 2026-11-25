# 06 OpenVINO 对照

对应 `scripts/08_quantize_openvino.py`、`dadeploy/runners.py` 的 `OvRunner`。在 `ov` 环境里运行。

入口还是做 YOLO 时用的 `nncf.quantize`,ViT 特有的是 `model_type=TRANSFORMER` 和 SmoothQuant。这一栈的价值在于和 GPU 栈互为对照:同一个模型,量化点放的位置不同,精度损失差一个量级。

## 转换

```python
fp32_model = ov.convert_model(str(src))                    # 直接读 02 导出的 ONNX
ov.save_model(model, str(xml), compress_to_fp16=False)
```

`compress_to_fp16` 默认是 True,会把浮点权重舍入成 FP16。不关掉,"FP32 基准"就带上了舍入误差(README 坑 6)。

OpenVINO 读入后,图有两处变化(看 `artifacts/openvino/da2s_r518.fp32.xml`):

- 每个 block 的注意力被合成一个 `ScaledDotProductAttention` 算子(12 个);
- LayerNorm 表示成 `MVN`(25 个),GELU 是单个 `Gelu` 算子。

## 量化

`build_recipe` 里的调用:

```python
common = {"preset": nncf.QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER, "subset_size": CALIB_NUM}
nncf.quantize(fp32_model, calib_dataset(res_name), advanced_parameters=advanced_params(SQ_ALPHA[recipe]), **common)
```

- `model_type=TRANSFORMER`:按 Transformer 的模板放量化点,并打开 SmoothQuant。
- `preset=MIXED`:权重对称、激活非对称。GELU / ReLU 的输出偏正,非对称量化少浪费一半量程。
- `subset_size` 显式给 200:默认 300 大于 calib 的张数。
- 校准数据同样只来自 calib 划分。

## 量化点放在哪

518 档的 INT8 IR 里有 110 个 FakeQuantize。它们接在:

| 被量化输入的算子 | 该类算子总数 | 其中输入带 FakeQuantize 的 |
|---|---|---|
| MatMul(6 个线性层 × 12) | 72 | 72 |
| ScaledDotProductAttention | 12 | 12 |
| Convolution | 31 | 27 |
| Interpolate | 5 | 5 |
| Add(含残差相加) | 164 | 0 |
| MVN(LayerNorm) | 25 | 0 |
| Gelu | 12 | 0 |

和 modelopt 对比:

| 每个 block 里的位置 | modelopt INT8(TensorRT) | NNCF INT8(OpenVINO) |
|---|---|---|
| 6 个线性层的权重 | 量化 | 量化 |
| 线性层的输入 | 量化 | 量化,前面加 SmoothQuant 乘子 |
| 注意力里的矩阵乘 | 不量化 | 注意力算子的输入被量化 |
| 残差流 | 量化,每个 block 4 处 | 不量化 |

最后一行就是两栈精度差距的来源:GPU 全 INT8 的 Δ AbsRel 是 +0.0051,CPU INT8 是 +0.0005(518 档,测试集)。GPU 侧把残差流留回 FP16(int8res)后降到 +0.0012,方向一致。

## SmoothQuant

04 的统计显示离群集中在固定通道。SmoothQuant 给每个通道一个系数 s:激活除以 s,下一层权重的对应列乘以 s。数学上等价,但把离群从"只能共用一个 scale 的激活"挪到了"可以按通道量化的权重"。

在 IR 里能看到它:每个线性层前面多一个名字带 `nncf_smooth_quant` 的 Multiply 节点。

α 控制挪多少。`SQ_ALPHA` 定义了扫描,只在 364 档上做:

| 配方 | α | Δ AbsRel(测试集) | 推理 P50 ms |
|---|---|---|---|
| int8sqoff | 关闭 | +0.0008 | 116.94 |
| int8sq050 | 0.5 | +0.0006 | 116.45 |
| int8sq075 | 0.75 | +0.0004 | 116.87 |
| int8 | 0.95(NNCF 默认) | +0.0001 | 118.14 |

α 越大精度越好,速度不变。

## Accuracy Control

`int8ac` 用 `nncf.quantize_with_accuracy_control`,验证函数是 `make_validation_fn`:

- NNCF 约定指标越大越好,所以返回 1 − AbsRel;
- `max_drop = ACCEPT_ABSREL_DELTA`,`drop_type=ABSOLUTE`,和 GPU 侧 int8mix 用同一个接受标准;
- 验证数据是 val 划分。

结果:`int8ac` 和 `int8` 完全相同。默认配方在 val 上已经达标(518 档 0.0616,目标 ≤ 0.0617),没有回退任何层。

## 为什么没有 FP16 配方

这颗 CPU(i9-14900HX)没有 AVX512-FP16 和 AMX。FP16 权重仍要解压成 FP32 计算,不会更快。`OvRunner` 还显式设了 `INFERENCE_PRECISION_HINT = f32`,避免在别的 CPU 上被自动降精度。

## 结果一览(测试集)

| 分辨率 | 配方 | Δ AbsRel | 推理 P50 ms | IR MB |
|---|---|---|---|---|
| 518 档 | fp32 | 0 | 670.91 | 95 |
| 518 档 | int8 | +0.0005 | 335.68 | 27 |
| 364 档 | fp32 | 0 | 265.83 | 94 |
| 364 档 | int8 | +0.0001 | 118.14 | 26 |

INT8 在 CPU 上是真加速:518 档 2.0 倍,364 档 2.25 倍。这和 GPU 上"INT8 不比 FP16 快"形成对照,因为 CPU 上的对手是 FP32。

CPU 数字的不确定度比 GPU 大:WSL2 只有 4 个 vCPU,无法绑定物理 P 核。
