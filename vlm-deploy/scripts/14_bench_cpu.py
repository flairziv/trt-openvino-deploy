"""P4 CPU 计时:OpenVINO GenAI 各压缩配方的首字延迟分解与 decode 速度(CPU 4 线程)。

用法(env: ovgenai;Makefile 已封装):
    python scripts/14_bench_cpu.py --variants ov-fp16 ov-int8 ov-int4 ov-int4awq ov-int4-vis8w ov-int4-vis8s

分段(GenAI 的 VLMPipeline 自带分段计时,不用自己拆):
    视觉段  = prepare_embeddings:视觉塔 + 图像 embedding 拼进文字 embedding;
    TTFT    = 从 generate 开始到第一个 token(含视觉段与 prefill);prefill ≈ TTFT − 视觉段;
    decode  = 1000 / TPOT(每输出 token 的平均耗时),强制生成 DECODE_TOKENS 个 token(忽略 EOS)。

计时纪律:
    - 每个配方要单独加载(两个配方同时驻留会超过 WSL 的 8 GB),所以不能像 GPU 那样逐请求交错。
      退而求其次用 ABBA:配方按 A B C … 测一遍,再按 … C B A 测一遍,各取两遍的样本合并。
      CPU 降频与后台负载的漂移在两遍里方向相反,大致抵消;两遍中位数之差另报,作为漂移幅度的证据。
    - 每遍先跑 1 次预热请求(丢弃),再对 NUM_IMAGES 张图各测 1 次。
    - WSL2 不能绑核,宿主 Windows 的调度会把线程放到 E 核上 —— 数字只代表这台机器的这种配置。

产物:results/bench/cpu.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.backends import CPU_THREADS
from vlmdeploy.config import RESULTS_DIR, TOKEN_BUDGETS
from vlmdeploy.imaging import image_tokens, resize_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.tasks import iter_split
from vlmdeploy.timing import summarize
from vlmdeploy.variants import ov_ir_dir

# 模块级 logger
log = get_logger("bench_cpu")
# 计时用的图片数
NUM_IMAGES = 6
# 每个请求固定生成的 token 数
DECODE_TOKENS = 32
# 计时用的图像 token 档位
BUDGET = "t256"


def workload() -> list[tuple]:
    """取 dev 集前 NUM_IMAGES 张不同的图,预缩放到计时档位。"""
    # 测量项列表和已经用过的图片 id
    items, seen = [], set()
    # 按固定顺序读 dev 集
    for sample in iter_split("dev"):
        # 同一张图只用第一道题
        if sample.image_id in seen:
            continue
        # 记下这张图
        seen.add(sample.image_id)
        # 保存样本和缩放后的图片
        items.append((sample, resize_to_budget(sample.image(), TOKEN_BUDGETS[BUDGET])))
        # 取够图片数后结束
        if len(items) >= NUM_IMAGES:
            break
    return items


def measure(variant: str, items: list[tuple]) -> list[dict]:
    """加载一个配方,预热 1 次后对每张图各测 1 次,返回逐请求的分段耗时。"""
    # 重依赖只在这里导入
    import numpy as np
    import openvino as ov
    import openvino_genai as ov_genai

    # 加载计时起点
    t0 = time.time()
    # 创建 VLM 管线,在 CPU 上推理并限定线程数
    pipe = ov_genai.VLMPipeline(str(ov_ir_dir(variant)), "CPU", INFERENCE_NUM_THREADS=CPU_THREADS)
    # 加载耗时
    load_s = time.time() - t0
    # 生成参数
    config = ov_genai.GenerationConfig()
    # 上下限相同并忽略 EOS:每个请求都恰好生成 DECODE_TOKENS 个 token
    config.max_new_tokens = DECODE_TOKENS
    config.min_new_tokens = DECODE_TOKENS
    config.ignore_eos = True
    # 逐请求的计时结果
    out = []
    for i, (sample, img) in enumerate([items[0], *items]):   # 第 0 个是预热
        # 图片转成 [1, H, W, 3] 的 uint8 张量
        tensor = ov.Tensor(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None]))
        # 生成并取这次请求的性能指标
        pm = pipe.generate(sample.prompt, images=[tensor], generation_config=config).perf_metrics
        # 预热请求不记录
        if i == 0:
            continue
        # 记录首字延迟、视觉段耗时、每 token 耗时、输入 token 数、图像 token 数和加载耗时
        out.append({"ttft": pm.get_ttft().mean, "vision": pm.get_prepare_embeddings_duration().mean,
                    "tpot": pm.get_tpot().mean, "input_tokens": int(pm.get_num_input_tokens()),
                    "image_tokens": image_tokens(img.height, img.width), "load_s": load_s})
    del pipe  # 显式释放管线:下一个配方加载之前必须先把内存让出来
    return out


def main() -> int:
    """入口:按 ABBA 顺序测全部配方,汇总后写入 results/bench/cpu.json。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="CPU 计时(ABBA)")
    # 要测的变体,必填
    parser.add_argument("--variants", nargs="+", required=True)
    # 解析参数
    args = parser.parse_args()

    # 准备计时用的图片
    items = workload()
    passes = [args.variants, list(reversed(args.variants))]  # ABBA 的两遍:正序一遍、倒序一遍
    # samples[配方] = [第 1 遍的样本, 第 2 遍的样本]:两遍分开存,汇总时既要合并,也要比较两遍各自的中位数
    samples: dict[str, list[list[dict]]] = {v: [] for v in args.variants}
    # 两遍依次执行
    for p, order in enumerate(passes):
        # 按本遍的顺序逐个配方测量
        for v in order:
            # 计时起点
            t0 = time.time()
            # 加载、测量并保存本遍的样本
            samples[v].append(measure(v, items))
            # 打印本遍的首字延迟中位数和耗时
            log.info("pass %d %s: TTFT p50 %.0f ms (%.0fs)", p, v,
                     statistics.median(m["ttft"] for m in samples[v][-1]), time.time() - t0)

    # 结果:档位、图片数、生成 token 数、线程数、两遍的顺序、时间和各变体数据
    result = {"budget": BUDGET, "images": NUM_IMAGES, "decode_tokens": DECODE_TOKENS, "threads": CPU_THREADS,
              "order": passes, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": {}}
    # 逐个变体汇总
    for v in args.variants:
        allm = [m for run in samples[v] for m in run]  # 两遍的样本合并
        p50 = [statistics.median(m["ttft"] for m in run) for run in samples[v]]  # 每遍各自的 TTFT 中位数
        # 本变体的统计:首字延迟、视觉段、prefill、解码速度、两遍漂移、token 数和加载耗时
        result["variants"][v] = {
            "ttft_ms": summarize([m["ttft"] for m in allm]),
            "vision_ms": summarize([m["vision"] for m in allm]),
            "prefill_ms": summarize([m["ttft"] - m["vision"] for m in allm]),
            "decode_tps": summarize([1000 / m["tpot"] for m in allm]),
            # ABBA 漂移 = (第 2 遍中位数 − 第 1 遍中位数) / 两者的中位数(只有两个数,即均值);正值表示第 2 遍更慢
            "ttft_pass_drift": round((p50[1] - p50[0]) / statistics.median(p50), 3),
            "input_tokens_median": statistics.median(m["input_tokens"] for m in allm),
            "image_tokens_median": statistics.median(m["image_tokens"] for m in allm),
            "load_seconds": round(statistics.median(m["load_s"] for m in allm), 1),
        }
        # 取出本变体的统计用于打印
        r = result["variants"][v]
        # 打印首字延迟及其分解、解码速度和两遍漂移
        log.info("%-16s TTFT %6.0f ms = vision %6.0f + prefill %5.0f; decode %5.1f tok/s; ABBA drift %+.1f%%", v,
                 r["ttft_ms"]["p50"], r["vision_ms"]["p50"], r["prefill_ms"]["p50"], r["decode_tps"]["p50"],
                 100 * r["ttft_pass_drift"])
    # 创建结果目录
    (RESULTS_DIR / "bench").mkdir(parents=True, exist_ok=True)
    # 写出结果
    (RESULTS_DIR / "bench" / "cpu.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
