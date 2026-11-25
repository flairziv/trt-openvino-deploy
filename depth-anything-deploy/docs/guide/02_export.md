# 02 导出 ONNX

对应 `scripts/02_export_onnx.py`。和导出 YOLO 相比,多出两件事:位置编码插值要折叠掉,节点名要保住。

## 做了什么

```python
torch.onnx.export(
    model, (dummy,), str(raw_path),
    input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
    opset_version=ONNX_OPSET, dynamo=False,
)
```

- `model` 是 `DepthOnly` 薄包装,只返回深度张量。
- 不设 `dynamic_axes`:batch = 1、形状固定。三档分辨率各导一份。
- opset 17:从这一版起有原生的 LayerNormalization 算子。
- `dynamo=False`:用 TorchScript 导出器,原因见下文"节点名是接口"。

之后三步:常量折叠 → `onnx.checker` → 在 CPU 上和 PyTorch 对一次数值。

## 位置编码插值

DINOv2 预训练时位置编码是 37×37 的网格。推理分辨率不同,forward 里就要对它做一次 bicubic 插值。原样导出的话,这次插值连同一串 Shape / Gather / Concat 形状计算都留在图里,每帧重算一个常数。

固定输入形状后,这串计算的输入全是常量,可以整段折叠掉。`fold_to_fixed_point` 的做法:

```python
for it in range(1, _MAX_FOLD_ITERS + 1):
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)
    before = len(graph.nodes)
    graph.fold_constants(fold_shapes=True, size_threshold=None).cleanup().toposort()
    model = gs.export_onnx(graph)
    if len(graph.nodes) == before:
        return model, it
```

- 要迭代:折掉一个 Shape,下游的 Gather / Concat 才变成常量,下一轮才能折。
- `size_threshold=None`:折叠出的位置编码有 2 MiB 左右,默认阈值会拒绝。

结果(三档相同):

| 指标 | 折叠前 | 折叠后 |
|---|---|---|
| 节点数 | 1114 | 518 |
| Resize | 6 | 5(剩下的是 DPT 的双线性上采样) |
| 迭代轮数 | — | 19 |
| 与 PyTorch 最大绝对误差 | — | 5.2e-6(518 档) |

## 为什么不用 onnxsim

做 YOLO 时 `onnxsim` 几乎是默认步骤,这里不能用。脚本 docstring 和 README 坑 1 记录的原因:它会把 3D 的 MatMul + Add 改写成 Reshape → 2D Gemm → Reshape,破坏 TensorRT 的注意力融合和 modelopt 的模式匹配。graphsurgeon 只做"形状推断 → 折叠 → 清理",不改算子形态。

这一条没有在本轮重新验证,是项目当时的记录。

## 节点名是接口

TorchScript 导出器给节点起的名字带模块路径:

```
/model/backbone/encoder/layer.0/attention/q_proj/MatMul
/model/backbone/encoder/layer.0/mlp/fc2/MatMul
/model/backbone/encoder/layer.0/Add_1
/model/neck/fusion_stage/layers.3/Resize
```

`dadeploy/groups.py` 用正则把这些名字映射到"第几个 block 的哪个子层",敏感度分析(06)和 fc2 探针都靠它。换导出器(比如 `dynamo=True`)名字就变了,`find_qdq_pairs` 会直接抛错。

## 产物与核对

- `artifacts/onnx/da2s_<res>.onnx`:FP32,约 95 MB。
- `results/export.json`:每档的节点数、Resize 数、算子直方图、数值误差。

折叠后的算子直方图(518 档):Add 119、MatMul 96、Mul 72、Reshape 53、Transpose 53、Conv 31、LayerNormalization 28、Relu 16、Softmax 12、Div 12、Erf 12、Resize 5、Slice 4、ConvTranspose 2,另有 Cast / Concat / Squeeze 各 1。

96 个 MatMul = 12 个 block ×(6 个线性层 + 2 个注意力矩阵乘)。
