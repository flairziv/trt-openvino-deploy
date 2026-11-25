"""TensorRT 精度配方表(纯配置,不导入 tensorrt,两个 conda 环境都能读)。

配方决定两件事:
    1. 从哪个 ONNX 构建(基础 FP32 图 / AutoCast FP16 图 / modelopt QDQ 图);
    2. 以强类型还是弱类型网络构建、弱类型时打开哪些精度 flag。

强类型 vs 弱类型的取舍见 trt_utils 模块说明与 README "坑 2"。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrtRecipe:
    """一个 TensorRT 精度配方。

    Attributes:
        name: 配方名,即变体字符串里的精度段(如 "fp16")。
        onnx_tag: 读取哪个 ONNX:"" 表示基础 FP32 ONNX,否则读 da2s_<res>.<onnx_tag>.onnx。
        strongly_typed: 是否以强类型网络构建(精度由 ONNX 决定)。
        tf32: 弱类型时是否允许 TF32(TRT 默认开启,严格 FP32 必须显式关闭)。
        fp16: 弱类型时是否允许 FP16。
        description: 写进结果表的一句话说明。
    """

    name: str
    onnx_tag: str
    strongly_typed: bool
    tf32: bool = False
    fp16: bool = False
    description: str = ""


# 配方名到配方的映射
TRT_RECIPES: dict[str, TrtRecipe] = {
    r.name: r
    for r in (
        TrtRecipe("fp32", "", False, tf32=False, description="严格 FP32(关闭 TF32),数值基准"),
        TrtRecipe("tf32", "", False, tf32=True, description="TRT 默认:FP32 张量 + TF32 tensor core"),
        TrtRecipe("fp16w", "", False, tf32=True, fp16=True,
                  description="弱类型 FP16 flag(精度由计时决定,用于复现非确定性问题)"),
        TrtRecipe("fp16", "fp16", True, description="强类型 FP16(AutoCast,按需保留 FP32 的算子)"),
        TrtRecipe("int8", "int8", True,
                  description="modelopt 默认 INT8 PTQ(线性层 + 卷积 + 残差流),其余 FP16"),
        TrtRecipe("int8res", "int8res", True,
                  description="INT8,残差流全部保持 FP16(规则化配置,不依赖逐块回收)"),
        TrtRecipe("int8mix", "int8mix", True,
                  description="INT8,按 val 集真实引擎消融回退敏感子层到 FP16(06_sensitivity.py)"),
        TrtRecipe("fp8", "fp8", True, description="modelopt FP8(E4M3)QDQ,其余 FP16"),
    )
}
