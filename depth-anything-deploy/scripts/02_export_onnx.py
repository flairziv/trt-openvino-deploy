"""导出固定形状 ONNX,并做常量折叠 + 数值核对。

用法:
    python scripts/02_export_onnx.py                 # 导出全部档位
    python scripts/02_export_onnx.py --res r518      # 只导出某一档

产物:
    artifacts/onnx/da2s_<res>.onnx
    results/export.json   每档的节点数(折叠前/后)、Resize 数、与 PyTorch 的最大绝对误差

为什么要固定形状 + 常量折叠(本项目第一个"编译器视角"的坑):
    DINOv2 预训练时位置编码是 37×37 网格,推理分辨率不同就要在 forward 里对位置编码做一次
    **bicubic 插值**。动态形状导出会把这次插值连同一整串 Shape/Gather/Concat 形状计算留在图里,
    运行时每帧重算一次一个常数;端侧编译器(RKNN 等)还常常不支持 cubic Resize。
    固定输入形状后,这串计算的输入全是常量 → 常量折叠把它整段算掉,
    图里只剩 DPT 头真正需要的 5 个双线性上采样(Resize 6 → 5)。

    折叠用 onnx-graphsurgeon 而不是 onnxsim:onnxsim 会顺带把 3D MatMul+Add 改写成
    Reshape→2D Gemm→Reshape(多出 ~100 个 Reshape),破坏 Transformer 的标准子图形态,
    对 TensorRT 的注意力融合与 modelopt 的量化模式匹配都不友好;且本机 onnxsim 版本
    不认 skipped_optimizers 参数。graphsurgeon 只做"形状推断 → 折叠 → 清理",不改写算子形态。
    每轮折叠会暴露出新的可折叠节点(折掉一个 Shape,下游 Gather/Concat 才变常量),
    所以迭代到不动点为止 —— 与 learning/compiler 第 3 课的 fold_constants 同一个思路。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnx_graphsurgeon as gs  # noqa: E402
import onnxruntime as ort  # noqa: E402

ort.set_default_logger_severity(3)  # 折叠阶段 ORT 会刷大量无害的 UnsqueezeElimination 警告
import torch  # noqa: E402

from dadeploy.config import INPUT_NAME, ONNX_DIR, ONNX_OPSET, OUTPUT_NAME, RESOLUTIONS, RESULTS_DIR  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.model import load_model  # noqa: E402

# 模块级 logger
log = get_logger("export")

# 常量折叠的最大迭代轮数
_MAX_FOLD_ITERS = 50


def fold_to_fixed_point(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """迭代"形状推断 → 常量折叠 → 死节点清理",直到节点数不再下降。

    Args:
        model: 原始导出的 ONNX。

    Returns:
        (折叠后的模型, 实际迭代轮数)。

    Raises:
        RuntimeError: 超过最大轮数仍未收敛(通常意味着图里有环或折叠器异常)。
    """
    # 反复折叠,直到某一轮节点数不再减少
    for it in range(1, _MAX_FOLD_ITERS + 1):
        # 形状推断,让依赖形状的节点可以被折叠
        model = onnx.shape_inference.infer_shapes(model)
        # 导入为 graphsurgeon 图
        graph = gs.import_onnx(model)
        # 记下折叠前的节点数
        before = len(graph.nodes)
        # 折叠常量、清理无用节点并做拓扑排序
        # size_threshold=None:位置编码插值结果约 2 MiB,必须允许折叠成常量
        graph.fold_constants(fold_shapes=True, size_threshold=None).cleanup().toposort()
        # 导回 ONNX 模型
        model = gs.export_onnx(graph)
        # 节点数没有变化说明已到不动点,返回模型和轮数
        if len(graph.nodes) == before:
            return model, it
    # 超过最大轮数仍未收敛
    raise RuntimeError(f"constant folding did not converge in {_MAX_FOLD_ITERS} iterations")


# 统计各算子类型的节点数,按数量从多到少排列
def _op_hist(model: onnx.ModelProto) -> dict[str, int]:
    return dict(collections.Counter(n.op_type for n in model.graph.node).most_common())


def export_one(model: torch.nn.Module, res_name: str) -> dict:
    """导出并折叠单个档位,返回统计信息。

    Args:
        model: CPU 上的 FP32 DepthOnly 模型。
        res_name: 档位名(config.RESOLUTIONS 的键)。

    Returns:
        统计字典(写入 results/export.json)。

    Raises:
        RuntimeError: 折叠不收敛,或与 PyTorch 输出偏差过大。
    """
    # 取该档位的分辨率
    res = RESOLUTIONS[res_name]
    # 折叠前的临时 ONNX 路径
    raw_path = ONNX_DIR / f"da2s_{res_name}.raw.onnx"
    # 最终 ONNX 路径
    out_path = ONNX_DIR / f"da2s_{res_name}.onnx"
    # 按该档位形状生成随机输入,用于导出和数值核对
    dummy = torch.randn(1, 3, res.height, res.width)

    # 导出计时起点
    t0 = time.time()
    # 导出固定形状的 ONNX
    # dynamo=False:沿用 TorchScript 导出器,产出的命名(/model/backbone/encoder/layer.N/...)稳定,
    # 后续按 block 分组做量化敏感度分析依赖这些节点名。
    torch.onnx.export(
        model, (dummy,), str(raw_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        opset_version=ONNX_OPSET, dynamo=False,
    )
    # 读回刚导出的原始图
    raw = onnx.load(str(raw_path))
    # 常量折叠到不动点
    simplified, iters = fold_to_fixed_point(raw)
    # 检查折叠后的模型是否合法
    onnx.checker.check_model(simplified)
    # 保存折叠后的模型
    onnx.save(simplified, str(out_path))
    # 删除临时的原始图
    raw_path.unlink()

    # 数值核对:CPU 上 PyTorch vs ORT,排除 GPU TF32 等干扰,纯粹验证导出/折叠是否等价
    # PyTorch 在 CPU 上的参考输出
    with torch.no_grad():
        ref = model(dummy).numpy()
    # 在 CPU 上用 ORT 加载折叠后的模型
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    # 用同一份输入得到 ORT 的输出
    got = sess.run(None, {INPUT_NAME: dummy.numpy()})[0]
    # 两者的最大绝对误差
    max_abs = float(np.abs(ref - got).max())
    # 误差超过阈值时报错
    if max_abs > 1e-3:
        raise RuntimeError(f"{res_name}: ORT vs torch max abs err {max_abs:.3e} too large")

    # 该档位的统计信息,写入 export.json
    stats = {
        "resolution": f"{res.height}x{res.width}",
        "tokens": res.tokens,
        "nodes_raw": len(raw.graph.node),
        "nodes_folded": len(simplified.graph.node),
        "fold_iterations": iters,
        "resize_raw": sum(n.op_type == "Resize" for n in raw.graph.node),
        "resize_folded": sum(n.op_type == "Resize" for n in simplified.graph.node),
        "ops_folded": _op_hist(simplified),
        "max_abs_err_vs_torch": max_abs,
        "size_mb": round(out_path.stat().st_size / 2**20, 2),
        "export_seconds": round(time.time() - t0, 1),
    }
    # 打印该档位的摘要
    log.info("%s %s tokens=%d nodes %d->%d resize %d->%d maxabs=%.2e",
             res_name, stats["resolution"], res.tokens, stats["nodes_raw"], stats["nodes_folded"],
             stats["resize_raw"], stats["resize_folded"], max_abs)
    return stats


# 入口:按档位导出,并把统计写入 results/export.json
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="导出固定形状 ONNX 并常量折叠")
    # 要导出的档位,默认全部
    parser.add_argument("--res", nargs="*", default=list(RESOLUTIONS), choices=list(RESOLUTIONS))
    # 解析参数
    args = parser.parse_args()

    # 创建输出目录
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 在 CPU 上加载 FP32 模型
    model = load_model("cpu")

    # 导出记录文件
    report_path = RESULTS_DIR / "export.json"
    # 已有记录则读入,只更新本次导出的档位
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    # 逐个档位导出
    for name in args.res:
        report[name] = export_one(model, name)
    # 写回记录
    report_path.write_text(json.dumps(report, indent=2))
    # 打印记录文件位置
    log.info("wrote %s", report_path)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
