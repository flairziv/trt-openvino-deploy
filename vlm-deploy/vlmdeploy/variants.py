"""变体注册表:变体名 → 后端 + 模型位置。评测、计时、报告三处都只认这里的名字(单一事实来源)。

命名规则 `<栈>-<语言模型精度>[-<校准数据>]`:
    hf-bf16                         HF transformers 参考(P1 用它复现官方分数,验证评测基座)
    trtllm-bf16                     GPU 栈基线(P2)
    trtllm-{fp8,awq}-{txt,img}      GPU 栈量化(P3):语言模型 FP8(W8A8)/ AWQ-INT4(W4A16),
                                    校准数据为纯文本(txt)或图文(img);视觉塔一律保持 BF16
    trtllm-bf16-vistrt              GPU 栈 + 视觉塔 TensorRT FP16 引擎(P5,只用于方图档位 s*,见 imaging.square_to_budget)
    ov-fp16                         CPU 栈基线(P4):FP16 权重,FP32 计算(本机 CPU 无 FP16 / BF16 计算指令)
    ov-<lm>[-vis<vision>]           CPU 栈压缩(P4):见 scripts/13_compress_cpu.py 的配方表
"""

from __future__ import annotations

from pathlib import Path

from vlmdeploy.config import ARTIFACTS_DIR, MODEL_DIR, OV_DIR, TRTLLM_CKPT_DIR

# GPU 栈量化变体的四个后缀,同时也是量化 checkpoint 的子目录名
GPU_QUANT = ("fp8-txt", "fp8-img", "awq-txt", "awq-img")
VISION_ENGINE_DIR = ARTIFACTS_DIR / "vision_trt"   # scripts/15_vision_trt.py 的产物,vision_g{H}x{W}.engine


def is_gpu(variant: str) -> bool:
    """变体是否跑在 GPU 上(决定用哪个 conda 环境、能否与其他 GPU 任务并行)。"""
    # 前缀是 hf- 或 trtllm- 的变体在 GPU 上运行
    return variant.startswith(("hf-", "trtllm-"))


def trtllm_checkpoint(variant: str) -> Path:
    """TensorRT-LLM 变体对应的模型目录。"""
    # BF16(含 -vistrt:它只换视觉塔,语言模型不变)直接用原始 HF 目录
    if variant in ("trtllm-bf16", "trtllm-bf16-vistrt"):
        return MODEL_DIR
    # 去掉前缀得到量化 checkpoint 的子目录名
    suffix = variant.removeprefix("trtllm-")
    # 不认识的名字直接报错,不悄悄落到某个默认目录
    if suffix not in GPU_QUANT:
        raise ValueError(f"unknown TensorRT-LLM variant {variant!r}")
    # 量化 checkpoint 的目录
    return TRTLLM_CKPT_DIR / suffix


def ov_ir_dir(variant: str) -> Path:
    """OpenVINO 变体对应的 IR 目录(ov-fp16 → openvino/fp16,ov-int4-vis8s → openvino/int4-vis8s)。"""
    # 只接受 ov- 开头的变体
    if not variant.startswith("ov-"):
        raise ValueError(f"not an OpenVINO variant: {variant!r}")
    # 去掉前缀得到 IR 子目录名
    return OV_DIR / variant.removeprefix("ov-")


# 评测与演示都经 make_answerer 加载模型。两个计时脚本要拿逐请求的性能指标,是各自直接构造后端的,
# 只借用上面的路径解析;冒烟脚本也是各自构造。
def make_answerer(variant: str, vision_grid: int | None = None):  # noqa: ANN201 —— 返回 backends.Answerer 子类实例
    """按变体名构造答题后端(重依赖在这里才导入)。

    vision_grid: 只对 -vistrt 变体有意义 —— 本次运行用到的方图网格边长(patch 数,如 s576 → 48)。
        给了就只加载这一个视觉塔引擎:三个静态引擎各约 1.3 GB,连同 TensorRT-LLM 自带的 HF 模型(约 4.4 GB)
        在 8 GB 卡上放不下,实测 worker 初始化时报 "Impossible to fit in any sequence in kvCache"。
    """
    import os

    # backends 模块本身也是轻量的,真正的重依赖在各后端的构造函数里才导入
    from vlmdeploy import backends

    # 用环境变量把"只加载哪个网格"传给 worker 进程;不需要时清掉,免得上一次调用的值残留
    if vision_grid is not None and variant.endswith("-vistrt"):
        os.environ["VLMDEPLOY_VISION_TRT_GRIDS"] = f"{vision_grid}x{vision_grid}"
    else:
        os.environ.pop("VLMDEPLOY_VISION_TRT_GRIDS", None)
    # HF 参考后端
    if variant == "hf-bf16":
        return backends.HfAnswerer(variant)
    # TensorRT-LLM 后端
    if variant.startswith("trtllm-"):
        # 只有 -vistrt 变体使用视觉塔引擎
        engines = VISION_ENGINE_DIR if variant.endswith("-vistrt") else None
        return backends.TrtllmAnswerer(variant, trtllm_checkpoint(variant), vision_engines=engines)
    # OpenVINO 后端
    if variant.startswith("ov-"):
        return backends.OpenVinoAnswerer(variant, ov_ir_dir(variant))
    # 未知的变体名
    raise ValueError(f"unknown variant {variant!r}")
