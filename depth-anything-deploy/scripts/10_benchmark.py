"""交错轮次延迟基准。

用法:
    # env: dl(GPU,先确保没有其他 GPU 任务)
    python scripts/10_benchmark.py --variants trt:r518:tf32 trt:r518:fp16 trt:r518:int8mix trt:r518:fp8
    # env: ov(CPU)
    python scripts/10_benchmark.py --variants ov:r518:fp32 ov:r518:int8

协议(噪声对策见 dadeploy/timing.py):
    - 所有变体一次加载好,BENCH_ROUNDS 轮,每轮随机打乱顺序;每段先 BENCH_WARMUP 次预热(不计),
      再 BENCH_ITERS 次计时。输入是固定一张真实图片的预处理结果,排除图片差异;
    - 分段:preprocess(CPU)/ h2d / compute / d2h / postprocess(CPU)/ total(整段墙钟);
    - 报告 P50 / P90,以及 compute 各轮中位数的相对极差(>10% 视为不稳定,结果里标记)。

环境限制(如实写进 README):
    - GPU 是笔记本 RTX 4060,动态频率,数字只代表这台机器;
    - CPU 在 WSL2 里只分到 4 个 vCPU,无法从虚拟机内部绑定物理 P 核,
      Windows 调度可能把 vCPU 放到 E 核上,所以 CPU 数字的不确定度比 GPU 大。

产物:results/bench_<backend>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dadeploy.config import (  # noqa: E402
    BENCH_ITERS,
    BENCH_ROUNDS,
    BENCH_WARMUP,
    NYU_HEIGHT,
    NYU_WIDTH,
    RESULTS_DIR,
)
from dadeploy.data import iter_split  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.preprocess import postprocess, preprocess  # noqa: E402
from dadeploy.runners import create_runner  # noqa: E402
from dadeploy.timing import interleaved_schedule, round_spread, summarize  # noqa: E402
from dadeploy.variants import parse_many  # noqa: E402

# 模块级 logger
log = get_logger("bench")

# 各轮中位数的相对极差超过这个值就标记为不稳定
UNSTABLE_SPREAD: float = 0.10


# 入口:按交错轮次给一组变体计时,结果并入 results/bench_<backend>.json
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="交错轮次延迟基准")
    # 变体列表,必填
    parser.add_argument("--variants", nargs="+", required=True)
    # 轮数
    parser.add_argument("--rounds", type=int, default=BENCH_ROUNDS)
    # 每段的计时次数
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    # 每段的预热次数
    parser.add_argument("--warmup", type=int, default=BENCH_WARMUP)
    parser.add_argument("--out", default=None, help="输出文件名;缺省 bench_<首个变体的后端>.json")
    # 解析参数
    args = parser.parse_args()

    # 解析变体列表
    variants = parse_many(args.variants)
    # 涉及的后端集合
    backends = {v.backend for v in variants}
    # 一次只允许一个后端
    if len(backends) != 1:
        raise ValueError(f"一次只测一个后端,避免 GPU / CPU 任务互相干扰:{backends}")
    # 固定用 drift 划分的第一张图作输入
    sample = next(iter_split("drift", limit=1))
    # 一次加载好全部变体
    runners = {v.key: create_runner(v) for v in variants}
    # 变体字符串到变体对象的映射
    by_key = {v.key: v for v in variants}

    # 各变体各分段的耗时样本
    samples: dict = {k: defaultdict(list) for k in runners}
    # 各变体每一轮的推理耗时中位数
    per_round: dict = {k: [] for k in runners}
    # 按交错计划逐轮执行,轮次从 1 开始
    for r, order in enumerate(interleaved_schedule(list(runners), args.rounds), 1):
        # 按这一轮的顺序逐个变体测量
        for key in order:
            # 取出 Runner 和分辨率
            runner = runners[key]
            res = by_key[key].res
            # 预热,不计时
            for _ in range(args.warmup):
                runner.infer(preprocess(sample.rgb, res))
            # 这一段的推理耗时样本
            compute = []
            # 计时循环
            for _ in range(args.iters):
                # 依次执行预处理、推理、后处理,在各段之间取时间戳
                t0 = time.perf_counter()
                x = preprocess(sample.rgb, res)
                t1 = time.perf_counter()
                raw, seg = runner.infer_timed(x)
                t2 = time.perf_counter()
                postprocess(raw, (NYU_HEIGHT, NYU_WIDTH))
                t3 = time.perf_counter()
                # 记录预处理、后处理和整段的墙钟耗时(毫秒)
                samples[key]["preprocess"].append((t1 - t0) * 1e3)
                samples[key]["postprocess"].append((t3 - t2) * 1e3)
                samples[key]["total"].append((t3 - t0) * 1e3)
                # 记录后端给出的各分段耗时
                for name, ms in seg.items():
                    samples[key][name].append(ms)
                # 推理耗时另存一份,用于计算这一轮的中位数
                compute.append(seg["compute"])
            # 记录这一轮的推理耗时中位数
            per_round[key].append(summarize(compute)["p50"])
        # 打印这一轮各变体的中位数
        log.info("round %d/%d done: %s", r, args.rounds,
                 {k: round(per_round[k][-1], 2) for k in runners})

    # 各变体的汇总结果
    report = {}
    # 逐个变体汇总
    for key, segs in samples.items():
        # 每个分段的统计量
        entry = {name: summarize(values) for name, values in segs.items()}
        # 推理耗时各轮中位数的相对极差
        spread = round_spread(per_round[key])
        # 记录各轮中位数和极差
        entry["compute_round_p50"] = per_round[key]
        entry["compute_round_spread"] = spread
        # 极差超过阈值时标记为不稳定
        entry["unstable"] = spread > UNSTABLE_SPREAD
        # 由整段耗时的中位数换算帧率
        entry["fps_total_p50"] = 1e3 / entry["total"]["p50"]
        report[key] = entry
        # 打印该变体的摘要
        log.info("%s compute P50=%.2f P90=%.2f total P50=%.2f (%.1f FPS) spread=%.1f%%%s",
                 key, entry["compute"]["p50"], entry["compute"]["p90"], entry["total"]["p50"],
                 entry["fps_total_p50"], spread * 100, " UNSTABLE" if entry["unstable"] else "")
    # 释放全部 Runner
    for runner in runners.values():
        runner.close()

    # 输出文件,默认按后端命名
    out = RESULTS_DIR / (args.out or f"bench_{variants[0].backend}.json")
    # 已有结果则读入
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    # 按变体键覆盖
    previous.update(report)
    # 写回结果
    out.write_text(json.dumps(previous, indent=1), encoding="utf-8")
    # 打印结果文件位置
    log.info("wrote %s", out)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
