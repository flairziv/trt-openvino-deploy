# 07 推理、计时与 C++

对应 `dadeploy/preprocess.py`、`dadeploy/runners.py`、`dadeploy/timing.py`、`scripts/10_benchmark.py`、`scripts/11_profile_trt.py`、`scripts/13_cpp_parity.py`、`cpp/`。

## 一帧的流程

```
rgb(HWC uint8) → preprocess → [1, 3, H, W] float32 → runner.infer → [H, W] float32 → postprocess → 原图尺寸
```

**预处理**(`preprocess.preprocess`),和 YOLO 的三处不同:

- 直接 bicubic 缩放到目标尺寸,不做 letterbox;
- 除以 255 后还要减 ImageNet 均值、除以标准差;
- 在 uint8 上先缩放、再转浮点(内存带宽是 float32 的 1/4)。

**后处理**(`preprocess.postprocess`):一次双线性缩放放回原图尺寸。没有解码,没有 NMS。

## Runner

四种后端实现同一个接口(`runners.Runner`):

- `infer(x)`:收 `[1, 3, H, W]` float32,返回 `[H, W]` float32;
- `infer_timed(x)`:另返回分段耗时。

`_check_input` 会检查形状和 dtype:静态形状的引擎收到错误输入不会报错,只会算出错误结果。

各后端的库都在构造函数里才导入,所以 `ov` 环境也能导入这个模块。

### TrtRunner

构造时:反序列化引擎 → 建 context → 给输入和输出各分配一块显存和一块锁页内存 → `set_tensor_address` 绑定一次。缓冲都用 torch 管理。

每帧(`_run`,去掉计时打点):

```python
self._h_in.numpy()[...] = x                             # 写进锁页内存
with torch.cuda.stream(self._stream):
    self._d_in.copy_(self._h_in, non_blocking=True)     # H2D
    self._context.execute_async_v3(self._stream.cuda_stream)
    self._h_out.copy_(self._d_out, non_blocking=True)   # D2H
self._stream.synchronize()                              # 全链路只同步这一次
```

- 锁页内存让异步拷贝真正异步;
- 三段都排在同一条 stream 上,CPU 只在最后等一次;
- 四个 `cuda.Event` 打点得到 H2D / 推理 / D2H 的分段耗时,打点本身不引入同步;
- FP16 和 INT8 引擎的输入输出也是 FP32,调用方不用区分;
- `close` 显式释放:敏感度分析在一个进程里连续加载几十个引擎,不能等垃圾回收。

### OvRunner

构造函数的核心(config 在源码里分行写,这里并成一行):

```python
config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads, "INFERENCE_PRECISION_HINT": "f32"}
self._compiled = core.compile_model(str(xml_path), "CPU", config)
self._request = self._compiled.create_infer_request()
```

每帧 `self._request.infer({0: x})`,线程数固定为 4。

### OrtRunner

只用于核对导出和做仿真。请求的首选 EP 没生效就直接报错,不允许静默退回 CPU。

## 计时协议

笔记本上有两个噪声源:发热降频(先测的占便宜)和偶发慢帧。`scripts/10_benchmark.py` 的对策:

- 所有变体一次加载,跑 5 轮,每轮用 `timing.interleaved_schedule` 随机打乱顺序;
- 每段先预热再计时:GPU 20 + 100 次,CPU 3 + 15 次;
- 报 P50 和 P90,不报均值;
- 各轮中位数的相对极差超过 10% 的标 `~`;
- 输入固定为一张真实图片的预处理结果。

一次只测一个后端。`06_sensitivity.py` 里也有延迟数字,但只用于流程内部参考,不进任何表。

分段结果(P50,ms):

| 变体 | 预处理 | H2D | 推理 | D2H | 后处理 | 端到端 |
|---|---|---|---|---|---|---|
| trt:r518:fp16(Python) | 3.60 | 0.36 | 5.05 | 0.14 | 0.15 | 9.88 |
| trt:r364:fp16(Python) | 1.78 | 0.18 | 2.59 | 0.08 | 0.11 | 5.10 |
| trt:r518:fp16(C++) | 0.87 | 0.35 | 5.00 | 0.13 | 0.12 | 6.57 |
| ov:r518:int8(Python) | 3.98 | — | 335.68 | — | 0.17 | 339.91 |

Python 端到端里,预处理占了三分之一以上。

## 逐 kernel 剖析

`scripts/11_profile_trt.py` 用 `trt.IProfiler` 收集每个 kernel 的耗时,再用 `.layers.json` 里的类型和 tactic 名归类(`categorize`)。profiler 会在 kernel 之间插同步,总时间偏大,所以只看分布;绝对延迟以基准表为准。

518 档,单位 ms:

| 类别 | FP16 | INT8 | FP8 |
|---|---|---|---|
| 线性层 GEMM | 1.99 | 1.30 | 1.71 |
| 融合注意力 | 1.55 | 2.51 | 1.43 |
| 卷积 | 1.11 | 0.78 | 0.09 |
| kgen(LayerNorm 等) | 0.36 | 0.42 | 0.99 |
| reformat | 0.21 | 0.24 | — |
| 其他 | 0.11 | 0.11 | 1.53 |
| 合计 | 5.33 | 5.36 | 5.74 |

这张表回答了"INT8 为什么不比 FP16 快":

- INT8 让线性层和卷积合计省了约 1 ms;
- 同一个引擎里融合注意力多花了约 1 ms,总时间打平。

引擎 dump 里能看到,INT8 引擎的注意力 kernel 输入是 FP16、输出是 INT8,后面每个 block 还多一个 `MoveReshTran` kernel。它为什么慢,项目没有拆清。

FP8 一列的"卷积"很小、"其他"很大,是归类造成的,见 [09](09_open_questions.md)。

## C++ runtime

`cpp/trt_engine.h` 是一个纯头文件的 RAII 封装,`cpp/da_infer.cpp` 是端到端程序。调用顺序和 `TrtRunner` 相同,换成 C++ API:

| Python(TrtRunner) | C++(TrtEngine) |
|---|---|
| `torch.empty(...).pin_memory()` | `cudaMallocHost` |
| `set_tensor_address` | `setTensorAddress` |
| `copy_(non_blocking=True)` | `cudaMemcpyAsync` |
| `execute_async_v3` | `enqueueV3` |
| `stream.synchronize()` | `cudaStreamSynchronize` |
| `cuda.Event` | `cudaEventRecord` / `cudaEventElapsedTime` |

`DiscoverIo` 假设引擎只有一个输入、一个输出,且都是 FP32;不满足就抛异常。

预处理比 Python 快 4 倍的原因(`Preprocess`):输入只有 256 种取值,归一化预先算成每通道一张 256 项的查找表;一次遍历里同时完成归一化和 HWC → CHW,直接写进引擎的锁页输入缓冲。

构建需要 TensorRT OSS 的头文件(pip 轮子不带),路径由 Makefile 的 `TRT_INCLUDE_DIR` 给出。

## 对拍

`scripts/13_cpp_parity.py` 分三层核对 C++ 和 Python(`trt:r518:fp16`,5 张图):

| 层 | 比什么 | 结果 |
|---|---|---|
| 预处理 | C++ 的输入张量 vs Python `preprocess` | 最大差 1.02 个灰度级(两边 OpenCV 版本不同) |
| 推理 | 同一份输入(C++ dump 出来的)在 Python 再跑一次 | 最大差 0,逐位相同 |
| 端到端 | C++ 输出 vs Python 全流程 → AbsRel | 平均差 −2.2e-5 |

第二层把预处理差异隔离开,只验证 C++ 的拷贝和推理没有写错。

归一化常量在 C++ 里另有一份(`kMean` / `kStd`)。改 `preprocess.py` 或 `da_infer.cpp` 任何一边,都要 `make cpp parity` 重新对拍。
