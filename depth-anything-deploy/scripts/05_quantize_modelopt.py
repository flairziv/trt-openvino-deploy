"""生成 TensorRT 强类型构建所需的派生 ONNX:AutoCast FP16 / modelopt INT8 / FP8 QDQ。

用法(env: dl):
    python scripts/05_quantize_modelopt.py --res r518 --mode fp16
    python scripts/05_quantize_modelopt.py --res r518 --mode int8
    python scripts/05_quantize_modelopt.py --res r518 --mode fp8
    (int8mix 不在这里生成:它由 06_sensitivity.py 在 int8 图上拆掉敏感子层的 Q/DQ 得到,
     保证"交付的图"就是"消融实验里验证过的图",且与 int8 共用同一份校准。)

产物:
    artifacts/onnx/da2s_<res>.<mode>.onnx
    results/quantize.json   每个产物的配置、Q/DQ 数、被量化的算子类型、ORT 仿真偏差(val 前 8 张)

各模式说明:
    fp16       modelopt AutoCast:全图转 FP16,输入输出保持 FP32。默认不跑"按真实幅值决定保留 FP32"的
               参考推理:它要记录全部中间张量,r364 下 4 张图就吃 7.6 GB、r518 直接撑崩 WSL(8 GB 配额)。
               用 --autocast-range-check 在 r364 上跑过一次:幅值全部低于 data_max,没有计算节点需要保留 FP32
               (r518 激活统计也显示 absmax ≤ 55)。FP16 是否可用最终由引擎实测 SQNR 验证。
    int8       modelopt PTQ:MatMul / Conv 权重 per-channel、激活 per-tensor 对称 INT8,熵校准;
               为了让 TRT 融合"GEMM + 残差相加",它还会在残差 Add / LayerNorm 的输入上插 Q/DQ
               (即残差流本身被量化)。INT8 模式下注意力内部的两个 BMM 不量化(实测 0/24),
               TRT 会对它们使用 FP16 融合 MHA kernel。其余部分 FP16。
    fp8        modelopt FP8(E4M3)PTQ,max 校准。Ada(sm_89)起 tensor core 原生支持 FP8;
               FP8 的指数位让它对"少数大值"更宽容,这正是 ViT 激活离群的形态。

校准数据只来自 calib 划分(训练场景 200 张),与测试集、val 集互不相交。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import onnx  # noqa: E402

from dadeploy.config import CALIB_NUM, INPUT_NAME, ONNX_DIR, RESOLUTIONS, RESULTS_DIR  # noqa: E402
from dadeploy.data import iter_split, split_names  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.metrics import output_drift  # noqa: E402
from dadeploy.preprocess import preprocess  # noqa: E402
from dadeploy.reference import load_reference  # noqa: E402
from dadeploy.runners import OrtRunner, preload_ort_cuda  # noqa: E402

# 模块级 logger
log = get_logger("quantize")

# 模式 → modelopt 参数。calibration_method:INT8 用熵校准(KL 最小化选截断阈值,抗长尾),
# FP8 用 max(FP8 动态范围大,截断收益小,max 更稳)。
QUANT_MODES: dict[str, dict] = {
    "int8": {"quantize_mode": "int8", "calibration_method": "entropy", "disable_mha_qdq": False},
    "fp8": {"quantize_mode": "fp8", "calibration_method": "max", "disable_mha_qdq": False},
}
# 全部可选模式:fp16 加上各量化模式
ALL_MODES: tuple[str, ...] = ("fp16", *QUANT_MODES)
AUTOCAST_CALIB_NUM: int = 4           # --autocast-range-check 用:只统计各节点 IO 幅值,几张图足够
SANITY_NUM: int = 8                   # 量化后用 ORT 在 val 前 8 张上做一次仿真偏差自检


def load_calib(res_name: str, num: int) -> np.ndarray:
    """读取 calib 划分前 num 张并预处理成 [num, 3, H, W] float32。"""
    # 取该档位的分辨率
    res = RESOLUTIONS[res_name]
    # 逐张预处理后沿 batch 维拼接
    arr = np.concatenate([preprocess(s.rgb, res) for s in iter_split("calib", limit=num)], axis=0)
    # 打印校准数据的形状和占用内存
    log.info("calib data %s: %s (%.0f MB)", res_name, arr.shape, arr.nbytes / 2**20)
    return arr


def run_autocast(src: Path, dst: Path, res_name: str, range_check: bool) -> dict:
    """AutoCast FP16。

    Args:
        src: 基础 FP32 ONNX。
        dst: 输出路径。
        res_name: 档位名。
        range_check: 是否跑数据驱动的幅值检查(内存开销巨大,只适合 r364 及以下)。

    Returns:
        写入 quantize.json 的信息。
    """
    # modelopt 只在这里导入
    from modelopt.onnx.autocast import convert_to_mixed_precision

    kwargs: dict = {"low_precision_type": "fp16", "keep_io_types": True}   # IO 保持 FP32,Runner / C++ 端不用改
    # 临时目录,用于存放校准数据文件
    with tempfile.TemporaryDirectory() as tmp:
        # 需要幅值检查时提供校准数据
        if range_check:
            # 校准数据文件路径
            npz = Path(tmp) / "calib.npz"
            # 保存校准输入
            np.savez(npz, **{INPUT_NAME: load_calib(res_name, AUTOCAST_CALIB_NUM)})
            # 指定校准数据和执行后端
            kwargs.update(calibration_data=str(npz), providers=["cuda:0", "cpu"])
        else:
            kwargs["data_max"] = np.inf          # data_max=inf 时 AutoCast 跳过参考推理
        # 转成 FP16 混合精度模型
        model = convert_to_mixed_precision(str(src), **kwargs)
    # 保存转换后的模型
    onnx.save(model, str(dst))
    # 返回是否做过幅值检查,以及仍按 FP32 计算的节点
    return {"autocast_range_check": range_check, "fp32_kept_nodes": _fp32_compute_nodes(model)}


def _fp32_compute_nodes(model: onnx.ModelProto) -> list[str]:
    """列出 AutoCast 之后仍以 FP32 计算的节点(其输入来自 FP32 张量)。"""
    from onnx import TensorProto

    # 形状推断,补全各张量的类型信息
    inferred = onnx.shape_inference.infer_shapes(model)
    # 张量名到元素类型的映射,来自中间张量、输入和输出
    dtypes = {vi.name: vi.type.tensor_type.elem_type for vi in
              list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    # 补上 initializer 的类型
    for init in inferred.graph.initializer:
        dtypes[init.name] = init.data_type
    # 仍按 FP32 计算的节点
    kept = []
    # 逐个节点检查
    for node in inferred.graph.node:
        # Cast 和 Constant 不算计算节点
        if node.op_type in {"Cast", "Constant"}:
            continue
        # 有任一输入是 FP32 张量的节点记入列表
        if any(dtypes.get(i) == TensorProto.FLOAT for i in node.input if i):
            kept.append(f"{node.op_type}:{node.name}")
    return kept


def run_modelopt(src: Path, dst: Path, res_name: str, mode: str, calib_num: int) -> dict:
    """modelopt ONNX PTQ。"""
    from modelopt.onnx.quantization import quantize

    # 这一步只"写下量化方案",不改变权重语义:输出是带 Q/DQ 节点的 ONNX,真正的 INT8/FP8 kernel
    # 由 TensorRT 在强类型构建时把 Q/DQ 融合进算子后才产生。因此这里的 scale 决定精度,引擎决定速度。
    # 取该模式的量化参数
    params = dict(QUANT_MODES[mode])
    # 写入记录的信息:校准张数加量化参数
    info: dict = {"calib_num": calib_num, **params}
    # 读取并预处理校准数据
    calib = load_calib(res_name, calib_num)
    preload_ort_cuda()                       # 否则 modelopt 内部的 ORT 校准会话会静默退回 CPU
    # 调用 modelopt 做训练后量化,输出带 Q/DQ 的 ONNX
    quantize(
        str(src),
        calibration_data={INPUT_NAME: calib},
        calibration_eps=["cuda:0", "cpu"],   # 用 ORT CUDA EP 跑校准;默认列表里的 trt EP 会额外建引擎,慢且无必要
        # 未被 Q/DQ 覆盖的部分(LayerNorm、GELU、注意力 BMM 等)的运行精度。
        # 强类型构建下这就是真实精度,所以 "INT8 引擎" 实际是 INT8 + FP16 混合,并非全 INT8。
        high_precision_dtype="fp16",         # 未量化部分转 FP16(强类型构建时即真实运行精度)
        output_path=str(dst),
        log_level="WARNING",
        **params,
    )
    return info


def graph_stats(path: Path) -> dict:
    """统计产物里的 Q/DQ 数量与被量化的算子类型。"""
    # 加载产物
    model = onnx.load(str(path))
    # 张量名到产生它的节点的映射
    producers = {o: n for n in model.graph.node for o in n.output}
    # QuantizeLinear 节点数
    q = sum(n.op_type == "QuantizeLinear" for n in model.graph.node)
    # DequantizeLinear 节点数
    dq = sum(n.op_type == "DequantizeLinear" for n in model.graph.node)
    # 输入来自 DQ 的算子类型及个数
    quantized_ops: dict[str, int] = {}
    # 逐个节点统计
    for node in model.graph.node:
        # 跳过 Q/DQ 节点本身
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        # 有任一输入由 DequantizeLinear 产生的节点算作被量化
        if any(producers.get(i) is not None and producers[i].op_type == "DequantizeLinear" for i in node.input):
            quantized_ops[node.op_type] = quantized_ops.get(node.op_type, 0) + 1
    # 返回 Q/DQ 数、被量化的算子、节点总数和文件大小
    return {"quantize_linear": q, "dequantize_linear": dq, "quantized_consumers": quantized_ops,
            "nodes": len(model.graph.node), "size_mb": round(path.stat().st_size / 2**20, 2)}


def sanity_drift(path: Path, res_name: str) -> dict | None:
    """用 ORT(CUDA EP)在 val 前 SANITY_NUM 张上仿真,报告相对 FP32 参考的偏差。

    这只是"产物能跑、偏差量级正常"的冒烟检查。实测 ORT 假量化与 TensorRT 真实 INT8 的 SQNR
    均值相差约 4 dB、逐图相关性≈0(TRT 会把 Q/DQ 融进 kernel,取整与累加方式不同),
    所以**任何精度结论都以 TensorRT 引擎实测为准**;ORT CPU EP 对 FP16 QDQ 图的仿真更是完全失真
    (同一模型 2.7 dB vs CUDA EP 17 dB),这里强制使用 CUDA EP。
    """
    # 取该档位的分辨率
    res = RESOLUTIONS[res_name]
    # val 集前 SANITY_NUM 张的样本名
    names = split_names("val")[:SANITY_NUM]
    # 读取 FP32 参考并创建 ORT Runner,失败时只告警不中断
    try:
        _, ref = load_reference(res, "val")
        runner = OrtRunner(f"ort-sim:{path.name}", res, path)
    except Exception as exc:  # noqa: BLE001 —— 例如 ORT 不支持某种 Q/DQ 类型,只记录不中断
        log.warning("sanity check skipped for %s: %s", path.name, exc)
        return None
    # 逐图的相对 L1 偏差和 SQNR
    l1, sq = [], []
    # 逐张仿真
    for i, sample in enumerate(iter_split("val", limit=SANITY_NUM)):
        # ORT 输出与对应的 FP32 参考输出比较
        d = output_drift(runner.infer(preprocess(sample.rgb, res)), np.asarray(ref[i]))
        # 记录两项偏差
        l1.append(d.rel_l1)
        sq.append(d.sqnr_db)
    # 返回图片数和两项偏差的均值
    return {"n": len(names), "rel_l1": float(np.mean(l1)), "sqnr_db": float(np.mean(sq))}


# 入口:生成指定档位和模式的派生 ONNX,并记录到 results/quantize.json
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="AutoCast FP16 / modelopt INT8 / FP8 QDQ 导出")
    # 分辨率档位,必填
    parser.add_argument("--res", required=True, choices=list(RESOLUTIONS))
    # 模式,必填
    parser.add_argument("--mode", required=True, choices=ALL_MODES)
    # 校准张数
    parser.add_argument("--calib-num", type=int, default=CALIB_NUM)
    parser.add_argument("--autocast-range-check", action="store_true",
                        help="fp16 模式下跑数据驱动幅值检查(内存开销大,只适合 r364 及以下)")
    # 解析参数
    args = parser.parse_args()

    # 基础 FP32 ONNX
    src = ONNX_DIR / f"da2s_{args.res}.onnx"
    # 输出的派生 ONNX
    dst = ONNX_DIR / f"da2s_{args.res}.{args.mode}.onnx"
    # 基础 ONNX 不存在时提示先导出
    if not src.exists():
        raise FileNotFoundError(f"{src} 不存在,请先运行 scripts/02_export_onnx.py")

    # 计时起点
    t0 = time.time()
    # fp16 走 AutoCast,其余走 modelopt 量化
    if args.mode == "fp16":
        info = run_autocast(src, dst, args.res, args.autocast_range_check)
    else:
        info = run_modelopt(src, dst, args.res, args.mode, args.calib_num)
    # 记录耗时
    info["seconds"] = round(time.time() - t0, 1)
    # 统计产物里的 Q/DQ
    info["graph"] = graph_stats(dst)
    # 用 ORT 仿真做一次偏差自检
    info["ort_sim_drift_val8"] = sanity_drift(dst, args.res)
    # 打印摘要,不含较长的节点列表
    log.info("%s.%s done in %.0fs: %s", args.res, args.mode, info["seconds"],
             {k: v for k, v in info.items() if k not in {"fp32_kept_nodes"}})

    # 量化记录文件
    report_path = RESULTS_DIR / "quantize.json"
    # 已有记录则读入,只更新本次的条目
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    # 以档位名加模式名作键
    report[f"{args.res}.{args.mode}"] = info
    # 写回记录
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
