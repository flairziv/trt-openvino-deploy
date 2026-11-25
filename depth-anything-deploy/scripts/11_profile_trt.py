"""TensorRT 逐层耗时剖析:时间到底花在注意力、MLP、LayerNorm,还是 DPT 头上?

用法(env: dl,GPU 空闲时):
    python scripts/11_profile_trt.py --variants trt:r518:fp16 trt:r518:int8mix trt:r518:fp8

方法:IProfiler 回调逐 kernel 耗时(引擎以 ProfilingVerbosity.DETAILED 构建,Myelin 子图也能拆到单个 kernel),
再用 IEngineInspector 导出的 LayerType / TacticName 把 kernel 归类。

WSL2 下 Nsight Systems 抓不到 GPU kernel(dcn-trt10 项目验证过 cuda_gpu_kern_sum 全部 SKIPPED),
所以逐层归因只能走 TensorRT 自己的 profiler。profiler 会在 kernel 间插同步,总耗时比正常执行偏大,
这里只用**占比**做归因,绝对延迟以 10_benchmark.py 为准。

产物:results/profile_trt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402

from dadeploy.config import RESULTS_DIR  # noqa: E402
from dadeploy.data import iter_split  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.preprocess import preprocess  # noqa: E402
from dadeploy.runners import create_runner  # noqa: E402
from dadeploy.variants import parse_many  # noqa: E402

# 模块级 logger
log = get_logger("profile")

# 预热次数
WARMUP = 20
# 计入统计的迭代次数
ITERS = 50


class LayerProfiler(trt.IProfiler):
    """收集每个 kernel 每次执行的耗时(毫秒)。"""

    # 构造函数:初始化基类并准备耗时容器
    def __init__(self) -> None:
        # 显式初始化 TensorRT 的基类
        trt.IProfiler.__init__(self)
        # kernel 名到各次耗时的映射
        self.times: dict[str, list[float]] = defaultdict(list)

    # TensorRT 每执行完一个 kernel 回调一次,这里记录它的耗时
    def report_layer_time(self, layer_name: str, ms: float) -> None:  # noqa: D102
        self.times[layer_name].append(ms)


def categorize(layer_type: str, name: str, tactic: str) -> str:
    """把 kernel 归到语义类别。

    类别含义:
        attention_mha    融合注意力 kernel(Q·Kᵀ + softmax + ·V 合成一个)
        attention_gemm   未融合时的注意力 BMM(按节点名识别)
        linear_gemm      q/k/v、o_proj、fc1、fc2 线性层,含 fc+GELU 融合
        kgen             Myelin 生成的逐元素 kernel:LayerNorm、GELU、残差相加、LayerScale、reshape……
        conv             补丁嵌入 + DPT 颈部 / 头部卷积
        reformat         精度 / 布局转换(INT8 ↔ 高精度边界的额外开销就在这里)
    """
    # 融合注意力 kernel
    if "_gemm_mha" in tactic or "_gemm_mha" in name:
        return "attention_mha"
    # 矩阵乘类:节点名属于注意力矩阵乘的单列一类,其余是线性层
    if layer_type in {"gemm", "fusion"}:
        return "attention_gemm" if "attention/MatMul" in name else "linear_gemm"
    # Myelin 生成的逐元素 kernel
    if layer_type == "kgen":
        return "kgen"
    # Cask 开头的是卷积类
    if layer_type.startswith("Cask"):
        return "conv"
    # 精度或布局转换
    if "reformat" in layer_type.lower():
        return "reformat"
    # 其余归入 other
    return "other"


def profile_variant(variant, sample_rgb: np.ndarray) -> dict:  # noqa: ANN001 —— variants.Variant
    """剪枝掉预热样本后,对一个引擎做逐 kernel 归因。"""
    # 读取引擎旁边的层信息文件
    layers = json.loads(variant.engine_path().with_suffix(".layers.json").read_text())
    # 层名到层信息的映射
    info = {str(x.get("Name", "")): x for x in layers}
    # 加载引擎
    runner = create_runner(variant)
    # 预处理固定的输入图片
    x = preprocess(sample_rgb, variant.res)
    # 创建 profiler 并挂到执行上下文
    prof = LayerProfiler()
    runner.set_profiler(prof)
    # 预热
    for _ in range(WARMUP):
        runner.infer(x)
    # 清掉预热阶段收集的耗时
    prof.times.clear()
    # 正式采集
    for _ in range(ITERS):
        runner.infer(x)
    runner.infer(x)                 # 再触发一次,确保上一轮的回调已全部上报

    # 每个 kernel 取各次耗时的中位数
    per_layer = {name: float(np.median(v)) for name, v in prof.times.items() if v}
    # 各类别的累计耗时
    by_cat: dict[str, float] = defaultdict(float)
    # 逐个 kernel 归类并累加
    for name, ms in per_layer.items():
        # 取该 kernel 的层信息,没有时为空
        meta = info.get(name, {})
        # 按层类型、名字和 tactic 名归类后累加
        by_cat[categorize(str(meta.get("LayerType", "")), name, str(meta.get("TacticName", "")))] += ms
    # 全部 kernel 的耗时之和
    total = sum(per_layer.values())
    # 耗时最长的 10 个 kernel
    top = sorted(per_layer.items(), key=lambda kv: -kv[1])[:10]
    # 释放引擎
    runner.close()
    # 打印总耗时和各类别占比
    log.info("%s total(profiled)=%.2fms %s", variant.key, total,
             {k: f"{v / total:.1%}" for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])})
    # 返回总耗时、kernel 数、各类别的耗时与占比、最慢的 kernel
    return {
        "profiled_total_ms": total,
        "kernels": len(per_layer),
        "by_category_ms": dict(by_cat),
        "by_category_share": {k: v / total for k, v in by_cat.items()},
        "top_kernels": [{"name": n[:120], "ms": ms} for n, ms in top],
    }


# 入口:逐个变体做逐层归因,结果并入 results/profile_trt.json
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="TensorRT 逐层耗时归因")
    # 变体列表,必填
    parser.add_argument("--variants", nargs="+", required=True)
    # 解析参数
    args = parser.parse_args()

    # 固定用 drift 划分的第一张图作输入
    sample = next(iter_split("drift", limit=1))
    # 结果文件
    path = RESULTS_DIR / "profile_trt.json"
    # 已有结果则读入
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    # 逐个变体剖析
    for variant in parse_many(args.variants):
        # 按变体键写入
        report[variant.key] = profile_variant(variant, sample.rgb)
        # 每完成一个就落盘
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    # 打印结果文件位置
    log.info("wrote %s", path)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
