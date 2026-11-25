"""批量构建 TensorRT 引擎。

用法(env: dl):
    python scripts/07_build_trt.py                                  # 默认变体集合
    python scripts/07_build_trt.py --variants trt:r518:fp16 trt:r518:int8mix
    python scripts/07_build_trt.py --determinism r364 --repeat 3    # 复现"弱类型 FP16 精度随计时漂移"

产物:
    artifacts/engines/da2s_<res>.<precision>.engine     引擎(绑定 GPU 型号 + TRT 版本,换机必重建)
    artifacts/engines/da2s_<res>.<precision>.layers.json  IEngineInspector 逐层信息
    artifacts/engines/timing.cache                      共享 timing cache(加速重建、稳定 tactic 选择)
    results/engines.json                                每个引擎的构建耗时、大小、kernel 精度分布、融合 MHA 数

注意:构建期间 builder 会实测每个 tactic 的耗时,**不要与其他 GPU 任务并行**,
否则 tactic 选择被干扰,引擎可能比独占构建时慢。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dadeploy.config import ENGINE_DIR, MAIN_RESOLUTIONS, RESOLUTIONS, RESULTS_DIR  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.trt_recipes import TRT_RECIPES  # noqa: E402
from dadeploy.trt_utils import build_engine, report_to_dict  # noqa: E402
from dadeploy.variants import Variant, parse_many  # noqa: E402

# 模块级 logger
log = get_logger("build")

# 共享的 timing cache 文件
TIMING_CACHE = ENGINE_DIR / "timing.cache"
# 主矩阵的配方
MAIN_RECIPES: tuple[str, ...] = ("fp32", "tf32", "fp16", "int8", "int8res", "int8mix", "fp8")
CURVE_RECIPES: tuple[str, ...] = ("fp16",)       # r266 只为分辨率曲线服务


def default_variants() -> list[str]:
    """主矩阵(两个主档位 × 全部配方)+ 分辨率曲线(r266 FP16)。"""
    # 两个主档位与全部主配方的组合
    keys = [f"trt:{r}:{p}" for r in MAIN_RESOLUTIONS for p in MAIN_RECIPES]
    # 追加 r266 档的曲线配方
    keys += [f"trt:r266:{p}" for p in CURVE_RECIPES]
    return keys


def merge_report(entries: dict[str, dict]) -> None:
    """把本次构建结果合并进 results/engines.json(按变体键覆盖)。"""
    # 构建记录文件
    path = RESULTS_DIR / "engines.json"
    # 已有记录则读入
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    # 按变体键覆盖
    report.update(entries)
    # 目录不存在时创建
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写回记录
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def build_variant(variant: Variant, force: bool) -> dict | None:
    """构建单个变体;ONNX 缺失时跳过并告警(而不是让整批失败)。"""
    # 该变体对应的 ONNX 路径
    onnx_path = variant.onnx_path()
    # 引擎输出路径
    engine_path = variant.engine_path()
    # ONNX 不存在时告警并跳过
    if not onnx_path.exists():
        log.warning("skip %s: %s missing (先运行 05_quantize_modelopt.py)", variant.key, onnx_path.name)
        return None
    # 引擎比 ONNX 新且未要求强制重建时跳过
    if engine_path.exists() and not force and engine_path.stat().st_mtime > onnx_path.stat().st_mtime:
        log.info("skip %s: engine up to date (use --force to rebuild)", variant.key)
        return None
    # 按配方构建引擎,使用共享的 timing cache
    report = build_engine(onnx_path, engine_path, TRT_RECIPES[variant.precision], variant.key, TIMING_CACHE)
    # 构建记录转成字典返回
    return report_to_dict(report)


def run_determinism(res_name: str, repeat: int) -> None:
    """同一 ONNX、同一组 flag 重复构建弱类型 FP16 引擎,记录每次实际选中的 kernel 精度。

    不使用 timing cache:cache 会让后续构建直接复用第一次的计时结果,掩盖问题本身。
    """
    # 弱类型 FP16 配方对应的变体
    variant = Variant.parse(f"trt:{res_name}:fp16w")
    # 各次构建的记录
    runs = {}
    # 重复构建 repeat 次
    for i in range(repeat):
        # 每次的引擎单独存放
        path = ENGINE_DIR / "determinism" / f"da2s_{res_name}.fp16w.{i}.engine"
        # 不使用 timing cache 构建
        report = build_engine(variant.onnx_path(), path, TRT_RECIPES["fp16w"], f"{variant.key}#{i}", None)
        # 记录这一次的结果
        runs[f"{variant.key}#{i}"] = report_to_dict(report)
    # 合并进构建记录文件
    merge_report(runs)


# 入口:批量构建引擎,或执行可复现性实验
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="批量构建 TensorRT 引擎")
    parser.add_argument("--variants", nargs="*", default=None, help="变体列表,缺省为主矩阵")
    parser.add_argument("--force", action="store_true", help="即使引擎比 ONNX 新也重建")
    parser.add_argument("--determinism", choices=list(RESOLUTIONS), default=None,
                        help="改为执行弱类型 FP16 可复现性实验")
    # 可复现性实验的重复次数
    parser.add_argument("--repeat", type=int, default=3)
    # 解析参数
    args = parser.parse_args()

    # 指定了 --determinism 时只做可复现性实验
    if args.determinism:
        run_determinism(args.determinism, args.repeat)
        return 0

    # 解析变体列表,没给时用默认集合
    variants = parse_many(args.variants or default_variants())
    # 本脚本只构建 trt 变体
    bad = [v.key for v in variants if v.backend != "trt"]
    if bad:
        raise ValueError(f"07_build_trt.py only builds trt variants, got {bad}")
    # 本次构建成功的记录
    entries = {}
    # 逐个变体构建
    for v in variants:
        rep = build_variant(v, args.force)
        # 构建了新引擎才记录
        if rep is not None:
            entries[v.key] = rep
            merge_report({v.key: rep})          # 每建完一个就落盘,中途失败不丢前面的结果
    # 打印构建的引擎数
    log.info("built %d engines", len(entries))
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
