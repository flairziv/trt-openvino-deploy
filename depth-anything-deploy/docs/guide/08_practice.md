# 08 练习路线

把仓库当答案册,自己从零把每一环重写一遍:每步一个不超过百行的独立脚本,写完和仓库里现成的产物、数字对答案。

顺序是先推理、后转换、最后量化。有了能跑的推理代码,后面每一步才有办法验证。

## 约定

- 全部用 364 档(`r364`),引擎构建和评测都快。
- 脚本放在新建的 `practice/` 目录,产物也写在那里。
- 不要直接跑仓库的 05–13:除了 `09 --limit`,它们都会改写 `results/`。
- 可以直接调用的仓库函数:`dadeploy.data.iter_split`(取样本)、`dadeploy.metrics.evaluate_prediction` 和 `output_drift`(打分)、`dadeploy.reference.load_reference`(取 FP32 参考)。它们是纯 numpy,重写学不到部署的东西。
- 第 1、3、4、5、6 步在 `dl` 环境,第 2、7 步在 `ov` 环境。

从 Windows 侧进 WSL 运行:

```bash
wsl.exe -e bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && cd /mnt/e/code/VscodeProjects/depth-anything-deploy && conda run --no-capture-output -n dl python practice/step1_trt_infer.py'
```

## 七步

| 步 | 自己写什么 | 要学会的调用 | 对答案 |
|---|---|---|---|
| 1 | 用现成的 `da2s_r364.fp16.engine` 跑通一张图,含预处理 | `deserialize_cuda_engine`、`set_tensor_address`、`execute_async_v3` | 输出 `[364, 476]`;和 FP32 参考比 SQNR 约 53 dB |
| 2 | 用现成的 `da2s_r364.fp32.xml` 跑通同一张图 | `Core().compile_model`、`create_infer_request` | SQNR 约 120 dB |
| 3 | 导出固定形状的 ONNX 并常量折叠 | `torch.onnx.export`、graphsurgeon 的 `fold_constants` | 节点 1114 → 518;与 PyTorch 最大差约 7e-6 |
| 4 | 转 FP16 并强类型构建引擎 | modelopt 的 `convert_to_mixed_precision`、`STRONGLY_TYPED` | 引擎约 50 MB,77 层 FP16,12 个融合注意力 |
| 5 | 插 Q/DQ,构建 INT8 引擎,在 val 上评测 | modelopt 的 `quantize` | 249 对 Q/DQ;val AbsRel 0.0736,SQNR 21.3 dB |
| 6 | 自己写函数拆掉残差流上的 Q/DQ,再构建评测 | 用 `onnx` 改节点连线 | val AbsRel 0.0706,SQNR 24.1 dB |
| 7 | NNCF 量化并评测 | `nncf.quantize`、`model_type=TRANSFORMER` | 110 个 FakeQuantize;val AbsRel 0.0700 |

## 每步的要点和答案位置

**第 1 步:TensorRT 推理。** 预处理也自己写,写完和 `dadeploy.preprocess.preprocess` 的输出逐元素比较,应该完全相等。最容易悄悄出错的是通道顺序和归一化。参考输出在 `artifacts/reference/r364_val.npy`(`[100, 364, 476]`),顺序和 `iter_split("val")` 一致。答案:`runners.TrtRunner`。

**第 2 步:OpenVINO 推理。** 十几行。注意 `INFERENCE_PRECISION_HINT`。答案:`runners.OvRunner`。

**第 3 步:导出。** 先不折叠,看一眼节点数(1114)和 Resize 数(6);再写折叠循环,看它几轮收敛(19)。数值核对在 CPU 上做。答案:`02_export_onnx.py` 的 `fold_to_fixed_point` 和 `export_one`。

**第 4 步:FP16 + 强类型。** 构建时不要设任何精度 flag。构建完用 `IEngineInspector` 导出逐层信息,自己数 tactic 名带 `f16f16` 的层和 `_gemm_mha` 的个数。不带 timing cache 时构建约 100 秒。答案:`05_quantize_modelopt.py` 的 `run_autocast`、`trt_utils.build_engine`。

**第 5 步:显式 INT8。** 校准数据用 calib 划分 200 张,预处理成一个 `[200, 3, 364, 476]` 的数组。调用前先 `runners.preload_ort_cuda()`。校准约 180 秒。写完数一下图里的 QuantizeLinear 个数。答案:`05_quantize_modelopt.py` 的 `run_modelopt`。

**第 6 步:手拆 Q/DQ。** 最值得花时间的一步。先自己找出"喂给残差相加和 LayerNorm 的 DQ",把它们的消费者改接到 Q 的输入,再删掉 Q 和 DQ。亲手写一遍才会明白:显式量化下,回退一层就是改图。答案:`groups.find_qdq_pairs` 和 `bypass_pairs`;对照产物是 `da2s_r364.int8res.onnx`。

**第 7 步:NNCF。** 先跑默认配置,再把 SmoothQuant 关掉(α = −1)跑一次,比较 val AbsRel(0.0700 对 0.0703)。量化约 156 秒。答案:`08_quantize_openvino.py` 的 `build_recipe`。

## 选做

- **C++**:`cpp/trt_engine.h` 和第 1 步是同一套调用。写完用 `13_cpp_parity.py` 的三层思路对拍。
- **校准截断**:[09](09_open_questions.md) 里记了一个没验证的问题,适合拿来练手:把残差流 Q/DQ 的 scale 改小(截断到 99.99% 分位),看 val AbsRel 怎么变。

## 换平台时哪些能带走

| 平台 | 量化工具 | 说明 |
|---|---|---|
| NVIDIA(含 Jetson) | modelopt + TensorRT 显式量化 | Q/DQ 图在 PC 上生成,引擎必须在目标板上构建。FP8 需要 Ada 一代的 GPU。Jetson 上的 TensorRT 版本跟着 JetPack 走,用之前先查是否支持强类型 |
| Intel(CPU、核显、NPU) | NNCF + OpenVINO | ARM 板子用不了 |
| ARM / NPU 板子(RK3588 等) | 厂商工具链(如 RKNN) | 量化方式是它自己的;本项目没有这一栈 |

能带走的是方法:固定形状导出、校准和决策的数据分开、在真实目标上做敏感度消融、用量化点的位置控制混合精度。API 每个平台都要重学。
