"""C++ 运行时与 Python 的一致性核对 + C++ 端到端基准(GPU 栈的 da_infer、CPU 栈的 da_infer_ov 共用本脚本)。

用法(变体的后端决定用哪个 C++ 程序、在哪个 conda 环境里跑):
    # env: dl,需先 `make cpp` 且引擎已构建;GPU 独占
    python scripts/13_cpp_parity.py --variant trt:r518:fp16 --num 5
    # env: ov,需先 `make cpp-ov` 且 IR 已生成(08);CPU 独占
    python scripts/13_cpp_parity.py --variant ov:r518:int8 --num 5

核对三层(任何一层不一致都说明 C++ 流水线与评测口径不是同一回事,C++ 的数字就不能引用):
    1. 预处理:C++ 输入张量 vs Python preprocess()。
       两边 OpenCV 版本不同(系统 4.6 vs pip 4.13),bicubic 定点实现可能有 ±1 灰度差,
       这里报告最大绝对差,并换算成灰度级,不要求逐位相同。
    2. 推理:同一个模型、**同一份输入**(C++ dump 下来的)在 Python 再跑一次,输出应几乎完全相同 ——
       隔离预处理差异,只验证 C++ 的推理调用没有写错(TensorRT:H2D / enqueue / D2H;
       OpenVINO:编译配置、输入输出张量的读写)。
    3. 端到端:C++ 输出 vs Python 全流程输出 → AbsRel 差。这是最终口径。

两个后端的对拍逻辑完全相同,差别只在"怎么调 C++ 程序",集中在 cpp_program() 里:
    trt  cpp/build/da_infer        --engine <.engine>            基准 500 帧(一帧约 7 ms)
    ov   cpp/build-ov/da_infer_ov  --model <.xml> --threads N    基准 100 帧(一帧 0.3–0.7 s)
OpenVINO 的线程数由本脚本从 config.CPU_THREADS 传给 C++ 程序,与 Python 侧 OvRunner 用的是同一个值。

C++ 基准是单进程连续计时(不是 10_benchmark.py 的交错轮次),结果写入 results/cpp_bench.json。
注意它与 Python 基准不在同一次运行里测:两者的推理耗时差几个百分点属于机器状态的漂移,
不能读成"C++ 更快 / 更慢";能稳定归因到 C++ 的只有预处理这一段。

产物:results/cpp_parity.json、results/cpp_bench.json(都按变体名作键,读旧文件 → 更新这一个键 → 写回)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dadeploy.config import CPU_THREADS, IMAGE_STD, NYU_HEIGHT, NYU_WIDTH, REPO_ROOT, RESULTS_DIR  # noqa: E402
from dadeploy.data import iter_split  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.metrics import evaluate_prediction  # noqa: E402
from dadeploy.preprocess import postprocess, preprocess  # noqa: E402
from dadeploy.runners import create_runner  # noqa: E402
from dadeploy.variants import Variant  # noqa: E402

# 模块级 logger
log = get_logger("parity")

# C++ 基准默认的计时帧数,按后端区分:CPU 上单帧耗时长,帧数取得少
BENCH_ITERS: dict[str, int] = {"trt": 500, "ov": 100}


def cpp_program(variant: Variant) -> tuple[Path, list[str], str]:
    """按变体的后端选出对应的 C++ 程序。

    Args:
        variant: 被核对的变体,后端必须是 trt 或 ov。

    Returns:
        (可执行文件路径, 指定模型的命令行参数, 构建它的 make 目标名)。

    Raises:
        ValueError: 后端没有对应的 C++ 程序(torch / ort)。

    """
    # TensorRT 版:可执行文件在 cpp/build 下,模型参数是引擎路径
    if variant.backend == "trt":
        return REPO_ROOT / "cpp" / "build" / "da_infer", ["--engine", str(variant.engine_path())], "cpp"
    # OpenVINO 版:可执行文件在 cpp/build-ov 下,模型参数是 IR 路径和线程数
    if variant.backend == "ov":
        # 线程数取自 config.CPU_THREADS,显式传给 C++ 程序
        return (REPO_ROOT / "cpp" / "build-ov" / "da_infer_ov",
                ["--model", str(variant.ov_path()), "--threads", str(CPU_THREADS)], "cpp-ov")
    # 其余后端没有 C++ 程序
    raise ValueError(f"{variant.key}: 只有 trt / ov 两个后端有 C++ 程序")


def main() -> int:
    """入口:逐张图做三层对拍,再跑一次 C++ 端到端基准,两份结果按变体名并入 results/。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="C++ 与 Python 推理一致性核对 + C++ 端到端基准")
    # 变体,必填
    parser.add_argument("--variant", required=True)
    # 对拍的图片数
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--bench", type=int, default=None, help="基准计时帧数;缺省按后端取 BENCH_ITERS")
    # 解析参数
    args = parser.parse_args()

    # 解析变体
    variant = Variant.parse(args.variant)
    # 选出对应的 C++ 程序
    binary, model_args, make_target = cpp_program(variant)
    # 可执行文件不存在时提示先构建
    if not binary.exists():
        raise FileNotFoundError(f"{binary} 不存在,请先运行 make {make_target}")
    # 基准帧数:命令行给了就用,否则按后端取默认值
    bench_iters = args.bench if args.bench is not None else BENCH_ITERS[variant.backend]
    # 模型输入的高和宽
    h, w = variant.res.height, variant.res.width
    # Python 侧加载同一个模型,作为对拍的参照
    runner = create_runner(variant)

    # 逐图的对拍结果
    rows = []
    # 临时目录:存放交给 C++ 的图片和它导出的张量,用完自动删除
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # 取 drift 划分的前 num 张图逐张对拍
        for sample in iter_split("drift", limit=args.num):
            # 输入图片的临时路径
            png = tmpdir / "in.png"
            # 转成 BGR 后保存,供 C++ 程序读取
            cv2.imwrite(str(png), cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR))      # PNG 无损,两边读到同一份像素
            # C++ 导出的输入张量和输出张量的路径
            in_bin, out_bin = tmpdir / "in.bin", tmpdir / "out.bin"
            # 运行 C++ 程序推理一次并导出张量,失败时抛异常
            subprocess.run([str(binary), *model_args, "--image", str(png), "--out", str(tmpdir / "d.png"),
                            "--dump-input", str(in_bin), "--dump-output", str(out_bin)],
                           check=True, capture_output=True, text=True)
            # 按约定的形状读回导出的裸 float32 数据
            cpp_in = np.fromfile(in_bin, dtype=np.float32).reshape(1, 3, h, w)
            cpp_out = np.fromfile(out_bin, dtype=np.float32).reshape(h, w)

            # Python 侧的预处理结果
            py_in = preprocess(sample.rgb, variant.res)
            py_on_cpp_in = runner.infer(cpp_in)                                     # 层 2:同输入,隔离预处理
            py_out = runner.infer(py_in)                                            # 层 3:Python 全流程
            # 原图尺寸
            size = (NYU_HEIGHT, NYU_WIDTH)
            # 记录这张图的对拍结果
            rows.append({
                "name": sample.name,
                # 归一化域的差换回灰度级:Δ灰度 = Δ归一化 × std × 255(取最大 std,给出保守上界)
                "preprocess_max_gray_levels": float(np.abs(cpp_in - py_in).max() * max(IMAGE_STD) * 255),
                # 同一份输入下两边输出的最大差
                "infer_same_input_max_abs": float(np.abs(cpp_out - py_on_cpp_in).max()),
                # 各自走完整流程时输出的最大差
                "end_to_end_max_abs": float(np.abs(cpp_out - py_out).max()),
                # 两边的输出用同一套后处理和打分
                "abs_rel_cpp": evaluate_prediction(postprocess(cpp_out, size), sample.depth).abs_rel,
                "abs_rel_py": evaluate_prediction(postprocess(py_out, size), sample.depth).abs_rel,
            })
            # 打印这张图的结果
            log.info("%s %s", sample.name, {k: round(v, 6) for k, v in rows[-1].items() if k != "name"})

        # 基准固定用 drift 划分的第一张图
        png = tmpdir / "bench.png"
        cv2.imwrite(str(png), cv2.cvtColor(next(iter_split("drift", limit=1)).rgb, cv2.COLOR_RGB2BGR))
        # 计时前先释放 Python 侧的模型
        runner.close()
        # C++ 基准结果的临时路径
        js = tmpdir / "bench.json"
        # 运行 C++ 基准并取回它的输出
        out = subprocess.run([str(binary), *model_args, "--image", str(png), "--bench", str(bench_iters),
                              "--json", str(js)], check=True, capture_output=True, text=True).stdout
        # 打印 C++ 程序的输出
        log.info("C++ bench %s\n%s", variant.key, out)
        # 读取 C++ 写出的分段耗时
        bench = json.loads(js.read_text())

    # 汇总:各项差值取各图的最大值,AbsRel 差取平均
    summary = {
        "n": len(rows),
        "preprocess_max_gray_levels": max(r["preprocess_max_gray_levels"] for r in rows),
        "infer_same_input_max_abs": max(r["infer_same_input_max_abs"] for r in rows),
        "end_to_end_max_abs": max(r["end_to_end_max_abs"] for r in rows),
        "abs_rel_mean_diff": float(np.mean([r["abs_rel_cpp"] - r["abs_rel_py"] for r in rows])),
    }
    # 打印汇总
    log.info("summary: %s", summary)
    # 两个结果文件都只更新当前变体的条目,其余条目保留
    for name, payload in (("cpp_parity.json", {"summary": summary, "rows": rows}), ("cpp_bench.json", bench)):
        # 结果文件路径
        path = RESULTS_DIR / name
        # 已有结果则读入
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        # 按变体键写入
        report[variant.key] = payload
        # 写回
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
