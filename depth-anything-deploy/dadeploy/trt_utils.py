"""TensorRT 构建与引擎剖析工具。

本模块只在 conda env `dl` 中可导入(依赖 tensorrt)。

两条构建路线(本项目实测后的结论,详见 README "坑 2"):

    **弱类型(weakly typed)**:网络张量默认 FP32,BuilderFlag.FP16/INT8 只是"允许"使用低精度,
    具体每层用什么精度由 builder **按计时结果挑**。在笔记本 GPU 上计时抖动大,同一 ONNX、同一组
    flag 两次构建,Myelin 可能一次全挑 FP16 + 融合 MHA,一次退回 TF32 + 拆开的注意力 ——
    "FP16 引擎"本身都不是一个确定的东西,做对照实验不可接受。

    **强类型(strongly typed)**:精度完全由 ONNX 里的张量类型(FP16 initializer / Cast / Q-DQ)决定,
    builder 不再自作主张,构建结果可复现。modelopt 产出的 INT8/FP8 QDQ 模型、AutoCast 产出的
    FP16 模型都走这条路线。

引擎剖析:用 IEngineInspector(需 ProfilingVerbosity.DETAILED 构建)导出每层的 TacticName,
按 kernel 名里的精度标记(f16f16 / tf32 / i8i8 / e4m3 …)统计,得到"引擎里实际跑的精度"——
这是验证"我以为它是 INT8"与"它真的是 INT8"之间差异的唯一可靠手段。
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tensorrt as trt

from .logging_ import get_logger
from .trt_recipes import TrtRecipe

# 模块级 logger
log = get_logger("trt")

# TensorRT 日志器,只输出警告及以上
_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
WORKSPACE_BYTES: int = 3 << 30   # 8 GB 笔记本卡与显示共享显存,不限会在大模型 tactic 搜索时抢满


@dataclass
class BuildReport:
    """一次引擎构建的记录(写入 results/engines.json)。"""

    # 变体字符串
    variant: str
    # 输入 ONNX 的文件名
    onnx: str
    # 引擎文件名
    engine: str
    # 构建耗时(秒)
    build_seconds: float
    # 引擎体积(MiB)
    engine_mb: float
    # 引擎层数
    num_layers: int
    # 各精度的计算层数
    precision: dict[str, int] = field(default_factory=dict)
    # 融合注意力 kernel 数
    fused_mha: int = 0
    # Myelin 子 kernel 数
    myelin_layers: int = 0


def _make_config(builder: trt.Builder, recipe: TrtRecipe, timing_cache: trt.ITimingCache | None) -> trt.IBuilderConfig:
    """按配方生成 builder 配置。强类型网络不设精度 flag,精度只由 ONNX 张量类型决定。"""
    # 新建 builder 配置
    cfg = builder.create_builder_config()
    # 限制 workspace 内存上限
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED   # 引擎剖析需要;不影响运行速度
    # 提供了 timing cache 时挂上,不忽略不匹配的情况
    if timing_cache is not None:
        cfg.set_timing_cache(timing_cache, ignore_mismatch=False)
    # 只有弱类型网络才设置精度 flag
    if not recipe.strongly_typed:
        # 强类型网络禁止设置精度 flag(builder 会报错);弱类型时显式声明允许的精度
        # 不允许 TF32 时清除默认开启的 TF32 flag
        if not recipe.tf32:
            cfg.clear_flag(trt.BuilderFlag.TF32)
        # 允许 FP16 时设置 FP16 flag
        if recipe.fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
    return cfg


def load_timing_cache(builder_config_factory: trt.IBuilderConfig, path: Path) -> trt.ITimingCache:
    """读取(或新建)timing cache。

    timing cache 记录每个 (层, tactic) 的实测耗时,复用后重建引擎快数倍,且 tactic 选择可复现。
    注意:它绑定 GPU 型号 + TRT 版本,换卡/换版本须删除重建。
    """
    # 缓存文件存在则读入,否则用空内容新建
    blob = path.read_bytes() if path.exists() else b""
    # 由字节内容创建 timing cache 对象
    return builder_config_factory.create_timing_cache(blob)


def build_engine(onnx_path: Path, engine_path: Path, recipe: TrtRecipe, variant_key: str,
                 timing_cache_path: Path | None = None) -> BuildReport:
    """从 ONNX 构建并序列化引擎,同时做一次剖析。

    Args:
        onnx_path: 输入 ONNX(基础 FP32 或 QDQ / FP16 版本)。
        engine_path: 输出 .engine 路径。
        recipe: 精度配方。
        variant_key: 变体字符串(只用于日志与报告)。
        timing_cache_path: timing cache 文件;None 表示不使用。

    Returns:
        BuildReport。

    Raises:
        RuntimeError: ONNX 解析失败或构建失败。
    """
    # 创建 builder
    builder = trt.Builder(_TRT_LOGGER)
    # 强类型网络需要在创建时指定标志位
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if recipe.strongly_typed else 0
    # 创建网络定义
    network = builder.create_network(flags)
    # 创建 ONNX 解析器
    parser = trt.OnnxParser(network, _TRT_LOGGER)
    # 解析 ONNX 文件,失败时汇总解析器给出的错误
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")

    # 临时的 builder 配置,只用来创建 timing cache 对象
    probe_cfg = builder.create_builder_config()
    # 给了缓存路径时读取或新建 timing cache
    cache = load_timing_cache(probe_cfg, timing_cache_path) if timing_cache_path else None
    # 按配方生成正式的 builder 配置
    cfg = _make_config(builder, recipe, cache)

    # 构建计时起点
    t0 = time.time()
    # 构建并序列化引擎
    serialized = builder.build_serialized_network(network, cfg)
    # 构建失败时报错
    if serialized is None:
        raise RuntimeError(f"engine build failed for {variant_key}")
    # 构建耗时(秒)
    build_s = time.time() - t0
    blob = bytes(memoryview(serialized))      # IHostMemory 没有 len(),先转成 bytes
    # 输出目录不存在时创建
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # 写出引擎文件
    engine_path.write_bytes(blob)
    # 用到了 timing cache 时把更新后的缓存写回文件
    if cache is not None and timing_cache_path is not None:
        timing_cache_path.write_bytes(memoryview(cfg.get_timing_cache().serialize()))

    # 导出引擎里每一层的信息
    layers = inspect_layers(blob)
    # 统计精度分布与融合注意力个数
    summary = summarize_layers(layers)
    # 汇总成构建记录
    report = BuildReport(
        variant=variant_key,
        onnx=onnx_path.name,
        engine=engine_path.name,
        build_seconds=round(build_s, 1),
        engine_mb=round(len(blob) / 2**20, 2),
        num_layers=len(layers),
        precision=summary["precision"],
        fused_mha=summary["fused_mha"],
        myelin_layers=summary["myelin_layers"],
    )
    # 在引擎旁边保存一份层信息
    (engine_path.with_suffix(".layers.json")).write_text(json.dumps(layers, indent=1))
    # 打印构建摘要
    log.info("built %s in %.0fs, %.1f MB, layers=%d, precision=%s, fused_mha=%d",
             variant_key, build_s, report.engine_mb, report.num_layers, report.precision, report.fused_mha)
    return report


def inspect_layers(engine_bytes: bytes) -> list[dict]:
    """用 IEngineInspector 导出每层信息(Name / LayerType / TacticName 等)。

    Args:
        engine_bytes: 序列化引擎。

    Returns:
        层信息字典列表。构建时若未开启 DETAILED,只能拿到层名,TacticName 为空。
    """
    # 创建运行时
    runtime = trt.Runtime(_TRT_LOGGER)
    # 反序列化引擎
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    # 创建引擎检查器
    inspector = engine.create_engine_inspector()
    # 导出 JSON 格式的引擎信息并解析
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    # 只保留字典形式的层记录
    return [layer for layer in info["Layers"] if isinstance(layer, dict)]


# kernel 名中的精度标记 → 精度类别。顺序有意义:先匹配更具体的模式。
_PRECISION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fp8", re.compile(r"e4m3|fp8|f8f8", re.I)),
    # INT8 kernel 命名有多种:i8i8(int8 进 int8 出)、i8f16 / i8f32(int8 进、高精度出,
    # 例如 QKV 融合 GEMM 输出 FP32 直接喂 FP16 的融合 MHA)、i8i32(int32 累加)
    ("int8", re.compile(r"i8i8|i8f16|i8f32|i8i32|int8|imma|_i8_", re.I)),
    ("fp16", re.compile(r"f16f16|h1688|h16816|fp16|_f16_", re.I)),
    ("bf16", re.compile(r"bf16", re.I)),
    ("tf32", re.compile(r"tf32", re.I)),
    ("fp32", re.compile(r"f32f32|ffma|sgemm", re.I)),
)
# 视为计算层的层类型
_COMPUTE_TYPES = {"gemm", "fusion", "CaskConvolution", "CaskDeconvolution", "CaskGemmConvolution"}


def classify_tactic(tactic: str) -> str | None:
    """从 TacticName 推断 kernel 计算精度;无法判断时返回 None。"""
    # 按顺序逐个模式匹配,返回第一个命中的精度类别
    for label, pattern in _PRECISION_PATTERNS:
        if pattern.search(tactic):
            return label
    # 没有模式命中
    return None


def summarize_layers(layers: list[dict]) -> dict:
    """统计计算层(GEMM / 卷积 / 融合 FC)的精度分布与融合注意力个数。

    Args:
        layers: inspect_layers 的输出。

    Returns:
        {"precision": {精度: 层数}, "fused_mha": 融合 MHA kernel 数, "myelin_layers": Myelin 子 kernel 数}
    """
    # 各精度的计算层计数
    precision: Counter[str] = Counter()
    # 融合注意力 kernel 计数
    fused_mha = 0
    # Myelin 子 kernel 计数
    myelin = 0
    # 逐层统计
    for layer in layers:
        # 取出层名、tactic 名和层类型
        name = str(layer.get("Name", ""))
        tactic = str(layer.get("TacticName", ""))
        ltype = str(layer.get("LayerType", ""))
        # 层名带 _myl 或层类型属于 Myelin 的计一次
        if "_myl" in name or ltype in {"kgen", "gemm", "fusion"}:
            myelin += 1
        # 融合注意力 kernel 只计数,不参与精度统计
        if "_gemm_mha" in tactic or "mha" in name.lower() and "gemm" in name.lower():
            fused_mha += 1
            continue
        # 计算层按 tactic 名归入精度类别,判断不出的记为 unknown
        if ltype in _COMPUTE_TYPES:
            label = classify_tactic(tactic)
            precision[label or "unknown"] += 1
    # 返回三项统计
    return {"precision": dict(precision), "fused_mha": fused_mha, "myelin_layers": myelin}


def report_to_dict(report: BuildReport) -> dict:
    """BuildReport → 可 JSON 序列化的字典。"""
    return asdict(report)
