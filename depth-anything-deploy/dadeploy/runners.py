"""四种推理后端的统一 Runner。

统一契约(所有后端相同,评测 / 计时脚本只面向这个接口编程):

    infer(x)        x: [1, 3, H, W] float32(preprocess 的输出) → [H, W] float32 相对视差
    infer_timed(x)  同上,另返回分段耗时 dict(毫秒),各后端分段不同:
                      trt  : h2d / compute / d2h(CUDA event,在同一 stream 上打点)
                      torch: compute(CUDA event,含 H2D/D2H 的同步拷贝不计入)
                      ov / ort: compute(墙钟)

依赖隔离:各后端的第三方库都在构造函数里**惰性导入**,
因此 `ov` 环境(无 torch / tensorrt)也能 import 本模块并只用 OvRunner。

TensorRT Runner 的设计要点(承接 dcn-trt10 项目的 Nsight 结论):
    - 输入输出 host 缓冲一次性分配为**锁页内存**(pinned)。dcn 项目里每帧 new 可分页内存 +
      同步 cudaMemcpy,D2H 只有 ~3 GB/s,反超推理成为后处理第一大块;这里从一开始就规避。
    - H2D → enqueueV3 → D2H 全部排在同一条 stream 上,只在最后 synchronize 一次。
"""

from __future__ import annotations

import abc
import time
from pathlib import Path

import numpy as np

from .config import CPU_THREADS, INPUT_NAME, Resolution


class Runner(abc.ABC):
    """推理后端抽象基类。

    Attributes:
        key: 变体字符串(用于日志与结果)。
        res: 输入分辨率档位。
    """

    # 构造函数:记录变体字符串与分辨率档位
    def __init__(self, key: str, res: Resolution) -> None:
        self.key = key
        self.res = res

    @abc.abstractmethod
    def infer(self, x: np.ndarray) -> np.ndarray:
        """单帧推理,返回 [H, W] float32。"""

    def infer_timed(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """带分段计时的推理。默认实现只记墙钟。"""
        # 墙钟计时起点
        t0 = time.perf_counter()
        # 执行推理
        out = self.infer(x)
        # 返回输出与耗时(毫秒)
        return out, {"compute": (time.perf_counter() - t0) * 1e3}

    def close(self) -> None:
        """释放后端资源(可选覆盖)。"""

    # 支持 with 语句:进入时返回自身
    def __enter__(self) -> "Runner":
        return self

    # 退出 with 语句时释放资源
    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_input(self, x: np.ndarray) -> None:
        """形状或 dtype 不符直接报错:静态形状引擎收到错误输入不会报错,只会算出错误结果。"""
        # 期望的输入形状
        expect = (1, 3, self.res.height, self.res.width)
        # 形状或类型不符时报错
        if x.shape != expect or x.dtype != np.float32:
            raise ValueError(f"{self.key}: expect input {expect} float32, got {x.shape} {x.dtype}")


# ---------------------------------------------------------------------------
# PyTorch(参考基准)
# ---------------------------------------------------------------------------
class TorchRunner(Runner):
    """PyTorch FP32 参考实现(GPU,TF32 关闭)。"""

    # 构造函数:加载 PyTorch 模型到指定设备
    def __init__(self, key: str, res: Resolution, device: str = "cuda") -> None:
        super().__init__(key, res)
        # torch 只在这里导入,没有 torch 的环境也能导入本模块
        import torch

        from .model import disable_tf32, load_model

        # 关闭 TF32,保证参考是严格 FP32
        disable_tf32()
        # 保存 torch 模块供其他方法使用
        self._torch = torch
        # 运行设备
        self._device = device
        # 加载模型
        self._model = load_model(device)

    # 单帧推理
    def infer(self, x: np.ndarray) -> np.ndarray:
        # 校验输入
        self._check_input(x)
        torch = self._torch
        # 关闭梯度记录
        with torch.inference_mode():
            # 输入转成张量并移到设备上,执行前向
            out = self._model(torch.from_numpy(x).to(self._device))
        # 去掉 batch 维,转 float32 搬回 CPU 再转成 numpy
        return out[0].float().cpu().numpy()

    # 带计时的推理,耗时取自 CUDA event
    def infer_timed(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        # 校验输入
        self._check_input(x)
        torch = self._torch
        # 输入先搬到设备上,不计入耗时
        xt = torch.from_numpy(x).to(self._device)
        # 前后两个计时事件
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            # 前向前后各记录一次事件
            start.record()
            out = self._model(xt)
            end.record()
        # 等待 GPU 执行完,事件时间才有效
        torch.cuda.synchronize()
        # 返回输出与两个事件之间的耗时(毫秒)
        return out[0].float().cpu().numpy(), {"compute": start.elapsed_time(end)}


# ---------------------------------------------------------------------------
# ONNX Runtime(导出核对 / QDQ 假量化仿真)
# ---------------------------------------------------------------------------
# ORT 需要的 CUDA 动态库是否已预加载
_ORT_DLLS_PRELOADED = False


def preload_ort_cuda() -> None:
    """让 ORT CUDA EP 找到 pip 安装的 CUDA / cuDNN 动态库。

    坑:dl 环境的 cuDNN 9 来自 pip 包 nvidia-cudnn-cu12(torch 的依赖),不在 LD_LIBRARY_PATH 里。
    ORT 加载 CUDA EP 时 dlopen("libcudnn.so.9") 失败后**只打一行错误日志就静默回退到 CPU EP**,
    结果所有"GPU 仿真"实际跑在 CPU 上,慢一个数量级还不报错。ORT ≥ 1.19 提供 preload_dlls()
    从 nvidia-* pip 包预加载这些库;modelopt 校准内部新建的 ORT 会话也受益,须在其之前调用。
    """
    # 声明要修改模块级标志
    global _ORT_DLLS_PRELOADED
    # 已经预加载过则直接返回
    if _ORT_DLLS_PRELOADED:
        return
    import onnxruntime as ort

    # 新版 ORT 提供 preload_dlls,用它预加载 CUDA 和 cuDNN
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
    # 置位,避免重复加载
    _ORT_DLLS_PRELOADED = True


class OrtRunner(Runner):
    """ONNX Runtime Runner。

    用途有二:核对导出图与 PyTorch 等价;在 CUDA EP 上**仿真** QDQ 模型(Q/DQ 按浮点逐算子执行,
    即"假量化"),用于敏感度分析时快速评估"去掉某组 QDQ 之后"的输出偏差,无需每次重建引擎。
    """

    # 构造函数:创建 ORT 会话,模型可以是文件路径或字节串
    def __init__(self, key: str, res: Resolution, onnx_path: Path | bytes,
                 providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")) -> None:
        super().__init__(key, res)
        import onnxruntime as ort

        # 用到 CUDA EP 时先预加载动态库
        if "CUDAExecutionProvider" in providers:
            preload_ort_cuda()
        # 会话选项
        opts = ort.SessionOptions()
        # 只输出错误级别的日志
        opts.log_severity_level = 3
        # 关闭 ORT 自身的 QDQ 融合,保证仿真的是"图上写的"量化行为,而不是 ORT 改写后的版本
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # 模型来源:字节串原样传入,路径转成字符串
        source = onnx_path if isinstance(onnx_path, bytes) else str(onnx_path)
        # 创建推理会话
        self._sess = ort.InferenceSession(source, sess_options=opts, providers=list(providers))
        # 核对首选 EP 确实生效
        if providers[0] not in self._sess.get_providers():
            # 不允许静默回退:请求的首选 EP 没生效就直接报错
            raise RuntimeError(f"{key}: requested {providers[0]} but ORT is using {self._sess.get_providers()}")
        # 记下输入张量名
        self._in = self._sess.get_inputs()[0].name

    # 单帧推理
    def infer(self, x: np.ndarray) -> np.ndarray:
        # 校验输入
        self._check_input(x)
        # 执行推理并取第一个输出
        out = self._sess.run(None, {self._in: x})[0]
        # 转成 float32 并整理成 [H,W]
        return np.asarray(out, dtype=np.float32).reshape(self.res.height, self.res.width)


# ---------------------------------------------------------------------------
# TensorRT
# ---------------------------------------------------------------------------
class TrtRunner(Runner):
    """TensorRT 引擎 Runner(显存 / 锁页内存一次分配,单 stream 异步流水)。"""

    # 构造函数:反序列化引擎并分配输入输出缓冲
    def __init__(self, key: str, res: Resolution, engine_path: Path) -> None:
        super().__init__(key, res)
        import tensorrt as trt
        import torch

        # 引擎文件不存在时提示先构建
        if not engine_path.exists():
            raise FileNotFoundError(f"{engine_path} 不存在,请先运行 scripts/07_build_trt.py")
        # 保存 torch 模块供其他方法使用
        self._torch = torch
        # TensorRT 日志器,只输出警告及以上
        logger = trt.Logger(trt.Logger.WARNING)
        # 创建运行时
        self._runtime = trt.Runtime(logger)
        # 读入引擎文件并反序列化
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        # 反序列化失败时报错
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}(GPU 型号或 TRT 版本变了?须重新构建)")
        # 创建执行上下文
        self._context = self._engine.create_execution_context()

        # 列出引擎的全部输入输出张量名
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        # 其中的输入张量
        inputs = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        # 其中的输出张量
        outputs = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        # 只接受一个名为 INPUT_NAME 的输入和一个输出
        if inputs != [INPUT_NAME] or len(outputs) != 1:
            raise RuntimeError(f"unexpected IO tensors: inputs={inputs} outputs={outputs}")
        # 记下输出张量名
        self._out_name = outputs[0]

        # 为一个张量分配显存缓冲和锁页内存缓冲,并把显存地址绑定到上下文
        def alloc(name: str) -> tuple:
            # 引擎的张量类型换成对应的 torch 类型
            dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[self._engine.get_tensor_dtype(name)]
            # 张量的静态形状
            shape = tuple(self._engine.get_tensor_shape(name))
            # 显存缓冲
            device = torch.empty(shape, dtype=dtype, device="cuda")
            # 锁页内存缓冲
            host = torch.empty(shape, dtype=dtype).pin_memory()
            # 把显存地址交给执行上下文,之后每帧复用
            self._context.set_tensor_address(name, device.data_ptr())
            return device, host

        # 输入的两块缓冲
        self._d_in, self._h_in = alloc(INPUT_NAME)
        # 输出的两块缓冲
        self._d_out, self._h_out = alloc(self._out_name)
        # 专用的 CUDA stream
        self._stream = torch.cuda.Stream()
        # 四个计时事件,依次标记 H2D 前、推理前、推理后、D2H 后
        self._events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    def _run(self, x: np.ndarray, timed: bool) -> np.ndarray:
        """H2D → 推理 → D2H 排在同一 stream,只在末尾同步一次;event 打点不引入额外同步。"""
        # 校验输入
        self._check_input(x)
        torch = self._torch
        self._h_in.numpy()[...] = x                  # 拷进锁页缓冲(FP32→FP16 输入时 numpy 自动转换)
        # 计时事件列表
        ev = self._events
        # 以下操作都排在专用 stream 上
        with torch.cuda.stream(self._stream):
            # 需要计时时记录 H2D 开始
            if timed:
                ev[0].record(self._stream)
            # 异步把输入从锁页内存拷到显存
            self._d_in.copy_(self._h_in, non_blocking=True)
            # 记录 H2D 结束,同时也是推理开始
            if timed:
                ev[1].record(self._stream)
            # 把推理排入 stream,返回 False 表示提交失败
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError(f"{self.key}: execute_async_v3 failed")
            # 记录推理结束
            if timed:
                ev[2].record(self._stream)
            # 异步把输出从显存拷回锁页内存
            self._h_out.copy_(self._d_out, non_blocking=True)
            # 记录 D2H 结束
            if timed:
                ev[3].record(self._stream)
        # 等待 stream 上的全部操作完成
        self._stream.synchronize()
        # 拷贝一份 float32 结果并整理成 [H,W],避免被下一帧覆盖
        return self._h_out.numpy().astype(np.float32, copy=True).reshape(self.res.height, self.res.width)

    # 单帧推理,不计时
    def infer(self, x: np.ndarray) -> np.ndarray:
        return self._run(x, timed=False)

    # 带分段计时的推理
    def infer_timed(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        # 执行并记录事件
        out = self._run(x, timed=True)
        ev = self._events
        # 由相邻事件的时间差得到三段耗时(毫秒)
        return out, {
            "h2d": ev[0].elapsed_time(ev[1]),
            "compute": ev[1].elapsed_time(ev[2]),
            "d2h": ev[2].elapsed_time(ev[3]),
        }

    def set_profiler(self, profiler) -> None:  # noqa: ANN001 —— trt.IProfiler
        """挂逐层 profiler(引擎须以 ProfilingVerbosity.DETAILED 构建)。"""
        # 把 profiler 挂到执行上下文
        self._context.profiler = profiler

    def close(self) -> None:
        """显式释放 context / engine / 缓冲。敏感度分析会在一个进程里连续加载几十个引擎,不能等 GC。"""
        # 逐个删除持有 GPU 资源的属性
        for attr in ("_context", "_engine", "_runtime", "_d_in", "_d_out", "_h_in", "_h_out"):
            if hasattr(self, attr):
                delattr(self, attr)
        # 归还 PyTorch 缓存的显存
        self._torch.cuda.empty_cache()

    def device_memory_mb(self) -> float:
        """引擎执行所需的 device 显存(激活 workspace,不含权重)。"""
        # 字节数换算成 MiB
        return self._engine.device_memory_size_v2 / 2**20


# ---------------------------------------------------------------------------
# OpenVINO(CPU)
# ---------------------------------------------------------------------------
class OvRunner(Runner):
    """OpenVINO CPU Runner。

    线程数固定为 CPU_THREADS(=4):WSL 只分到 4 个 vCPU,恰好也等于 RK3588 大核数,
    按"端侧 4 核预算"呈现数字。PERFORMANCE_HINT=LATENCY 让插件按单流低延迟编排。
    """

    # 构造函数:编译 IR 并创建推理请求
    def __init__(self, key: str, res: Resolution, xml_path: Path, threads: int = CPU_THREADS) -> None:
        super().__init__(key, res)
        import openvino as ov

        # IR 不存在时提示先生成
        if not xml_path.exists():
            raise FileNotFoundError(f"{xml_path} 不存在,请先运行 scripts/08_quantize_openvino.py")
        # OpenVINO 的入口对象
        core = ov.Core()
        # 编译配置:单路低延迟、限定线程数、按 FP32 计算
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": threads,
            "INFERENCE_PRECISION_HINT": "f32",   # 显式写出:避免在支持 BF16/AMX 的 CPU 上被自动降精度
        }
        # 读取 IR 并编译到 CPU
        self._compiled = core.compile_model(str(xml_path), "CPU", config)
        # 创建推理请求,之后每帧复用
        self._request = self._compiled.create_infer_request()

    # 单帧推理
    def infer(self, x: np.ndarray) -> np.ndarray:
        # 校验输入
        self._check_input(x)
        # 按输入序号 0 传入并同步推理
        self._request.infer({0: x})
        # 取第 0 个输出张量的数据
        out = self._request.get_output_tensor(0).data
        # 拷贝成 float32 数组并整理成 [H,W]
        return np.array(out, dtype=np.float32).reshape(self.res.height, self.res.width)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_runner(variant) -> Runner:  # noqa: ANN001 —— 避免循环导入,类型为 variants.Variant
    """按变体创建 Runner。

    Args:
        variant: dadeploy.variants.Variant。

    Returns:
        对应后端的 Runner。

    Raises:
        ValueError: 未知后端。
    """
    # 按后端名创建对应的 Runner
    if variant.backend == "torch":
        return TorchRunner(variant.key, variant.res)
    if variant.backend == "ort":
        return OrtRunner(variant.key, variant.res, variant.onnx_path())
    if variant.backend == "trt":
        return TrtRunner(variant.key, variant.res, variant.engine_path())
    if variant.backend == "ov":
        return OvRunner(variant.key, variant.res, variant.ov_path())
    # 未知后端
    raise ValueError(f"unknown backend {variant.backend}")
