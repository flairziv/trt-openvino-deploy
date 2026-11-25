"""P2 / P3 GPU 计时:首字延迟(TTFT)分解 + decode 速度 + 显存,TensorRT-LLM 任一精度变体。

用法(env: trtllm,带 PYTHONPATH=vlmdeploy/patches/site;Makefile 已封装):
    python scripts/11_bench_gpu.py --variant trtllm-bf16
    python scripts/11_bench_gpu.py --variant trtllm-awq-img
    python scripts/11_bench_gpu.py --variant trtllm-bf16-vistrt --square   # P5:视觉塔引擎端到端(方图档位)

一个视觉问答请求在 GPU 栈上分四段:
    ① 预处理(CPU,主进程):HF 图像处理器切块、归一化,分词;
    ② 视觉编码(GPU):视觉塔 32 层,处理合并前的 4×N 个块;
    ③ prefill(GPU):语言模型一次处理"N 个图像 token + 文字 token";
    ④ decode(GPU):逐 token 生成,每步读一遍全部权重 —— 显存带宽受限。

怎么拆:
    - 每个测量单元是一对请求:**带图请求**,以及把其中的图像占位 token 全部换成普通文字 token、长度逐个相同的
      **纯文本请求**。两者的 prefill 计算量相同,差别只在有没有视觉编码,所以
      视觉段 ② = TTFT(带图) − TTFT(纯文本),③ = TTFT(纯文本)。差值在同一轮内配对相减再取中位数。
    - TTFT 取引擎侧时间戳(first_token_time − arrival_time,TensorRT-LLM 的 return_perf_metrics),
      不含 ①;① 用客户端墙钟减去引擎侧总时长估出(含进程间通信,是上界)。
    - decode:两种请求都强制生成 DECODE_TOKENS 个 token(ignore_eos),
      速度 = (DECODE_TOKENS − 1) / (last_token_time − first_token_time)。
    - 批大小恒为 1(端侧单路场景),请求逐个同步提交,不与其他请求重叠。

计时纪律(vlmdeploy/timing.py):1 轮预热丢弃;之后 ROUNDS 轮,每轮把全部 (图, 档位, 请求类型) 随机打乱各测一次;
报 P50 / P90 与各轮中位数的相对极差。GPU 是笔记本动态频率,数字只代表这台机器。

产物:results/bench/gpu_<variant>.json(带 --square 时文件名多一个 _square 后缀)
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
from vlmdeploy.config import MODEL_DIR, RESULTS_DIR, SQUARE_BUDGETS, TOKEN_BUDGETS
from vlmdeploy.imaging import resize_to_budget, square_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.smoke import chat_prompt
from vlmdeploy.sysinfo import gpu_memory_used_mb
from vlmdeploy.tasks import iter_split
from vlmdeploy.timing import interleaved_schedule, round_spread, summarize
from vlmdeploy.variants import VISION_ENGINE_DIR, trtllm_checkpoint

# 模块级 logger
log = get_logger("bench_gpu")
NUM_IMAGES = 12        # dev 集里取前 12 张不同的图;只计时不打分,所以用 dev 而不碰 test
# 计时轮数,不含预热
ROUNDS = 5
# 每个请求固定生成的 token 数
DECODE_TOKENS = 64
# 纯文本请求用来顶替图像占位 token 的普通 token:Qwen2 词表里的 " the"。
# 选一个高频词只是为了让输入"像文字";prefill 的计算量与 token 内容无关,只与长度有关。
FILLER_TOKEN = 279
# 带图请求里与图像有关的三种特殊 token(起止标记 + 占位符),纯文本请求里全部换成 FILLER_TOKEN
IMAGE_SPECIAL = {"<|vision_start|>", "<|vision_end|>", "<|image_pad|>"}


def build_workload(processor, budgets: dict[str, int], resize) -> list[dict]:  # noqa: ANN001
    """每张图 × 每个档位:带图请求、长度相同的纯文本请求。"""
    # 图像特殊 token 的 id
    special = {processor.tokenizer.convert_tokens_to_ids(t) for t in IMAGE_SPECIAL}
    # 测量项列表和已经用过的图片 id
    items, seen = [], set()
    # 按固定顺序读 dev 集
    for sample in iter_split("dev"):
        if sample.image_id in seen:   # 一张图可能有多道题,只用第一道
            continue
        # 记下这张图
        seen.add(sample.image_id)
        # 解码图片
        image = sample.image()
        # 套聊天模板
        prompt = chat_prompt(processor, sample.prompt)
        # 每个档位生成一个测量项
        for budget, tokens in budgets.items():
            # 缩放到档位尺寸
            img = resize(image, tokens)
            # ids 是带图请求的完整输入序列(图像占位符已由处理器按实际 token 数展开);
            # 把其中的图像特殊 token 逐个换成 FILLER_TOKEN,就得到一条长度完全相同的纯文本序列
            ids = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"][0].tolist()
            # 长度相同的纯文本序列
            text_ids = [FILLER_TOKEN if t in special else t for t in ids]
            # 记录图片序号、档位、提示词、图片、纯文本序列和输入长度
            items.append({"image": len(seen) - 1, "budget": budget, "prompt": prompt, "img": img,
                          "text_ids": text_ids, "input_len": len(ids)})
        # 取够图片数后结束
        if len(seen) >= NUM_IMAGES:
            break
    return items


def one_request(llm, request: dict, params) -> dict:  # noqa: ANN001
    """同步提交一个请求,返回引擎侧与客户端两套耗时(毫秒)。"""
    # 客户端计时起点
    t0 = time.perf_counter()
    # 同步提交单个请求
    out = llm.generate([request], sampling_params=params, use_tqdm=False)[0]
    # 客户端测到的总耗时(毫秒)
    wall = (time.perf_counter() - t0) * 1000
    # 第一个候选输出
    comp = out.outputs[0]
    # 引擎侧的时间戳
    tm = comp.request_perf_metrics.timing_metrics
    # 请求到达、首个 token、最后一个 token 的时刻(秒)
    arrival, first, last = (tm.arrival_time.total_seconds(), tm.first_token_time.total_seconds(),
                            tm.last_token_time.total_seconds())
    # 生成的 token 数
    n_out = len(comp.token_ids)
    # 返回首字延迟、引擎侧总耗时、客户端额外开销、输出 token 数和解码速度
    return {"ttft": (first - arrival) * 1000, "engine_total": (last - arrival) * 1000,
            "overhead": wall - (last - arrival) * 1000, "n_out": n_out,
            "decode_tps": (n_out - 1) / (last - first) if n_out > 1 and last > first else float("nan")}


def main() -> int:
    """入口:加载变体,按交错轮次计时,把分段结果写入 results/bench/。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="GPU TTFT 分解 + decode 计时")
    # 变体名
    parser.add_argument("--variant", default="trtllm-bf16")
    # 计时轮数
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    # 使用方图档位
    parser.add_argument("--square", action="store_true", help="方图档位 s256/s576/s1024(P5 引擎对比用)")
    # 只测指定的档位
    parser.add_argument("--budgets", default="", help="只测这些档位(逗号分隔);结果并入已有的同名 JSON")
    # 解析参数
    args = parser.parse_args()
    # 档位表:方图档位或普通档位
    budgets = SQUARE_BUDGETS if args.square else TOKEN_BUDGETS
    # 指定了档位时只保留这些
    if args.budgets:
        budgets = {b: budgets[b] for b in args.budgets.split(",")}
    # 对应的缩放函数
    resize = square_to_budget if args.square else resize_to_budget
    import os

    # 视觉塔引擎经环境变量传给 worker 进程里的 sitecustomize 补丁(见 vlmdeploy/patches/site),必须在拉起 LLM 前设置
    if args.variant.endswith("-vistrt"):
        os.environ["VLMDEPLOY_VISION_TRT"] = str(VISION_ENGINE_DIR)
        if args.square:
            # 方图档位的网格边长 = 2·sqrt(token 数)(2×2 合并前)。只让 worker 加载这几个引擎:
            # 三个引擎同时常驻在 8 GB 上放不下(见 sitecustomize 里 _VisionEngines 的注释)。
            # 各档位的网格边长
            sides = [2 * round(t ** 0.5) for t in budgets.values()]
            # 写入环境变量,格式为逗号分隔的 高x宽
            os.environ["VLMDEPLOY_VISION_TRT_GRIDS"] = ",".join(f"{s}x{s}" for s in sides)
    # 其他变体清掉 VLMDEPLOY_VISION_TRT
    else:
        os.environ.pop("VLMDEPLOY_VISION_TRT", None)

    # 重依赖在设置好环境变量之后才导入
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    from transformers import AutoProcessor

    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    # 生成全部测量项
    work = build_workload(processor, budgets, resize)
    # 打印图片数、档位数和各档位输入长度的中位数
    log.info("workload: %d images x %d budgets, input lengths %s", NUM_IMAGES, len(budgets),
             {b: statistics.median(w["input_len"] for w in work if w["budget"] == b) for b in budgets})

    # 加载前的整卡显存
    mem_before = gpu_memory_used_mb()
    # 加载计时起点
    t0 = time.time()
    # 变体对应的模型目录
    ckpt = trtllm_checkpoint(args.variant)
    # 必须关闭 KV cache 块复用,TensorRT-LLM 默认开启。同一请求每轮重复提交,
    # 第 2 轮起前缀直接命中缓存,prefill 被跳过。试跑实测不关时纯文本 TTFT 不随长度变化,
    # 285 / 590 / 928 token 都是 20 ms。端侧单路场景每个请求的图都不同,命中缓存不是真实情况。
    llm = LLM(model=str(ckpt), kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25, enable_block_reuse=False),
              max_batch_size=8, max_num_tokens=4096, max_seq_len=2048)
    # 加载耗时
    load_s = time.time() - t0
    # 加载后的整卡显存
    mem_loaded = gpu_memory_used_mb()
    # max_tokens = min_tokens 再加 ignore_eos:每个请求恰好生成 DECODE_TOKENS 个 token;
    # return_perf_metrics 让返回结果带上引擎侧时间戳(one_request 里读的 timing_metrics)
    params = SamplingParams(max_tokens=DECODE_TOKENS, min_tokens=DECODE_TOKENS, ignore_eos=True, temperature=0.0,
                            return_perf_metrics=True)

    # 测量项 = 每个 (图, 档位) × {带图 vqa, 纯文本 text};i 是 work 里的下标
    keys = [(i, kind) for i in range(len(work)) for kind in ("vqa", "text")]
    plan = interleaved_schedule(keys, args.rounds + 1, seed=0)   # 多排的 1 轮是预热;种子固定,顺序可复现
    # 每个测量项各轮的计时结果
    samples: dict[tuple[int, str], list[dict]] = {k: [] for k in keys}
    # 逐轮执行
    for r, order in enumerate(plan):
        # 本轮计时起点
        t_round = time.time()
        # 按本轮的顺序逐个请求
        for i, kind in order:
            # 取测量项
            w = work[i]
            # 带图请求传提示词和图片,纯文本请求直接传 token 序列
            req = ({"prompt": w["prompt"], "multi_modal_data": {"image": [w["img"]]}} if kind == "vqa"
                   else {"prompt_token_ids": w["text_ids"]})
            # 提交请求并计时
            m = one_request(llm, req, params)
            if r > 0:   # 第 0 轮是预热:首次遇到某个长度时的 kernel 选择、内存池扩张都落在这一轮
                samples[(i, kind)].append(m)
        # 打印本轮耗时
        log.info("round %d/%d done in %.0fs%s", r, args.rounds, time.time() - t_round, " (warm-up)" if r == 0 else "")
    mem_peak = gpu_memory_used_mb()   # 测完那一刻的整卡读数(不是峰值),结果里记为 end
    # 关闭模型,结束 worker 进程
    llm.shutdown()

    # 结果:变体、模型目录、轮数、图片数、生成 token 数、加载耗时、显存、时间和各档位数据
    result = {"variant": args.variant, "checkpoint": str(ckpt), "rounds": args.rounds, "images": NUM_IMAGES,
              "decode_tokens": DECODE_TOKENS, "load_seconds": round(load_s, 1),
              "gpu_mem_mb": {"before_load": mem_before, "after_load": mem_loaded, "end": mem_peak},
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "budgets": {}}
    # 结果目录
    out_dir = RESULTS_DIR / "bench"
    # 结果文件,方图档位带 _square 后缀
    out_file = out_dir / f"gpu_{args.variant}{'_square' if args.square else ''}.json"
    if args.budgets and out_file.exists():
        # 一档一个进程时,各档结果并入同一个文件;每档自带本进程的加载时间与显存读数
        result["budgets"] = json.loads(out_file.read_text(encoding="utf-8")).get("budgets", {})
    # 标记是否一档一个进程
    result["one_budget_per_process"] = bool(args.budgets)
    # 逐个档位汇总
    for budget in budgets:
        # 属于本档位的测量项下标
        idx = [i for i, w in enumerate(work) if w["budget"] == budget]
        # 本档位全部带图请求的样本
        vqa = [m for i in idx for m in samples[(i, "vqa")]]
        # 本档位全部纯文本请求的样本
        txt = [m for i in idx for m in samples[(i, "text")]]
        # 配对:同一张图、同一轮的带图 / 纯文本请求相减
        vision = [a["ttft"] - b["ttft"] for i in idx for a, b in zip(samples[(i, "vqa")], samples[(i, "text")],
                                                                       strict=True)]
        # 每一轮内各图带图 TTFT 的中位数,用来算跨轮极差(ttft_vqa_round_spread)
        per_round = [statistics.median(samples[(i, "vqa")][r]["ttft"] for i in idx) for r in range(args.rounds)]
        # 本档位的统计:输入长度、带图和纯文本的首字延迟、视觉段耗时及占比、客户端开销、解码速度
        result["budgets"][budget] = {
            "input_len_median": statistics.median(work[i]["input_len"] for i in idx),
            "ttft_vqa_ms": summarize([m["ttft"] for m in vqa]),
            "ttft_text_ms": summarize([m["ttft"] for m in txt]),
            "vision_ms": summarize(vision),
            "vision_share": round(statistics.median(vision) / statistics.median(m["ttft"] for m in vqa), 3),
            "client_overhead_vqa_ms": summarize([m["overhead"] for m in vqa]),
            "client_overhead_text_ms": summarize([m["overhead"] for m in txt]),
            "decode_tps_vqa": summarize([m["decode_tps"] for m in vqa]),
            "decode_tps_text": summarize([m["decode_tps"] for m in txt]),
            "ttft_vqa_round_spread": round(round_spread(per_round), 3),
            "load_seconds": result["load_seconds"], "gpu_mem_mb": result["gpu_mem_mb"], "time": result["time"],
        }
        # 取出本档位的统计用于打印
        b = result["budgets"][budget]
        # 打印首字延迟及其分解、客户端开销和解码速度
        log.info("%s: TTFT %.0f ms (vision %.0f = %.0f%%, prefill %.0f), client overhead %.0f ms, decode %.1f tok/s",
                 budget, b["ttft_vqa_ms"]["p50"], b["vision_ms"]["p50"], 100 * b["vision_share"],
                 b["ttft_text_ms"]["p50"], b["client_overhead_vqa_ms"]["p50"], b["decode_tps_text"]["p50"])
    # 创建结果目录
    out_dir.mkdir(parents=True, exist_ok=True)
    # 写出结果
    out_file.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
