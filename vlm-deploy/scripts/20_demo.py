"""P6 演示:一张图 + 一个问题,在任一部署变体上流式作答,现场给出首字延迟与生成速度。

用法(GPU 变体用 env trtllm 并带 PYTHONPATH=vlmdeploy/patches/site;OpenVINO 变体用 env ovgenai;Makefile 已封装):
    python scripts/20_demo.py --variant trtllm-fp8-img --budget t576          # GPU:FP8 语言模型,dev 集前 4 张图
    python scripts/20_demo.py --variant trtllm-bf16-vistrt --budget s576      # GPU:视觉塔引擎(只能用方图档位)
    python scripts/20_demo.py --variant ov-int4-vis8w --budget t256           # CPU:INT4 语言模型 + INT8 权重视觉塔
    python scripts/20_demo.py --variant ov-int4-vis8w --image a.jpg --question "What does the sign say?"

与评测 / 计时脚本的关系:
    - 加载走评测用的同一个入口 variants.make_answerer,演示的就是被评测的那套配置,不是另写的一条路径
      (计时脚本为了拿逐请求的性能指标各自构造后端,参数与 backends.py 里的相同);
    - 不同之处:这里流式输出,默认最多生成 48 个 token(评测只要短答案,上限 16),所以也能问开放式问题;
    - 这里的耗时是**单次观察值**:只丢弃开头 2 次预热请求,没有重复、没有交错轮次,用来直观感受"首字要等多久"。
      结论性的数字只看 11_bench_gpu.py / 14_bench_cpu.py 的结果。

两条栈的"首字延迟"口径不同,所以结果里分开记、不混着比:
    - TensorRT-LLM:客户端墙钟,从提交请求到收到第一段文字,含主进程里的图像预处理与进程间通信;
      引擎侧的时间戳(不含预处理)在流式模式下若可用,另记为 engine_ttft_ms。
    - OpenVINO GenAI:管线自带的分段计时(与 14_bench_cpu.py 同一来源),另给出视觉段。

默认的题目来自 dev 集(决策集),不碰测试集;用 --image 可以换成自己的图片。

产物:results/demo/<variant>_<budget>.json(逐题的问题、回答、耗时;README 的演示表由 90_report.py 从它生成)。
只有全部用默认设置(dev 集原题、默认题数与生成上限)的运行才落盘;改了任何一项的临时演示只打印,不覆盖这份记录。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import RESULTS_DIR, SQUARE_BUDGETS, TOKEN_BUDGETS
from vlmdeploy.imaging import image_tokens, resize_to_budget, square_side_tokens, square_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.smoke import chat_prompt
from vlmdeploy.sysinfo import gpu_memory_used_mb
from vlmdeploy.tasks import iter_split, score
from vlmdeploy.variants import is_gpu, make_answerer

# 模块级 logger
log = get_logger("demo")
# 演示结果目录
DEMO_DIR = RESULTS_DIR / "demo"
NUM_IMAGES = 4          # 默认演示几张图(dev 集里前几张不同的图)
WARMUP_REQUESTS = 2     # 开头丢弃几次请求(依次取前几张图;图不够时循环使用)
MAX_NEW_TOKENS = 48     # 够一两句话;评测用的 16 只够短答案
MIN_DECODE_TOKENS = 8   # 生成的 token 少于这个数时不报 decode 速度(两三个 token 的间隔全是抖动)


def write_piece(piece: str) -> None:
    """把一段新生成的文字立刻打到终端(不换行、不缓冲)。"""
    # 直接写标准输出,不加换行
    sys.stdout.write(piece)
    # 立即刷新
    sys.stdout.flush()


def ask_trtllm(answerer, question: str, img, max_new_tokens: int,  # noqa: ANN001
               echo: Callable[[str], None]) -> dict:
    """在 TensorRT-LLM 上流式回答一个问题。

    Args:
        answerer: variants.make_answerer 返回的 TrtllmAnswerer(用它已经加载好的 llm 与 processor)。
        question: 问题文字(不含聊天模板)。
        img: 已按档位预缩放的 PIL.Image。
        max_new_tokens: 最多生成多少个 token。
        echo: 每收到一段新文字就调用一次(预热时传空函数)。

    Returns:
        回答与耗时(毫秒);decode_tps 在生成过短时为 None。
    """
    # tensorrt_llm 只在这里导入
    from tensorrt_llm import SamplingParams

    # temperature=0 即贪心;return_perf_metrics 让结果带回引擎侧时间戳(下面的 engine_ttft_ms 靠它)
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, return_perf_metrics=True)
    # 提示词与评测、计时共用同一个 chat_prompt
    request = {"prompt": chat_prompt(answerer.processor, question), "multi_modal_data": {"image": [img]}}
    # 累计文字、输出 token 数、首段和末段的到达时刻、最近一次的输出对象
    text, n_out, first, last, comp = "", 0, None, None, None
    # 计时起点
    t0 = time.perf_counter()
    # 流式生成,每到一段处理一次
    for out in answerer.llm.generate_async(request, sampling_params=params, streaming=True):
        # 第一个候选输出
        comp = out.outputs[0]
        # 这一段的到达时刻
        last = time.perf_counter()
        first = first or last   # 只在第一段到达时赋值,之后保持不变
        echo(comp.text_diff)    # text_diff 是这一次新增的文字;comp.text 是累计的全文
        # 更新累计文字和输出 token 数
        text, n_out = comp.text, len(comp.token_ids)
    # 整个请求的墙钟耗时
    wall = time.perf_counter() - t0
    if first is None:   # 一个 token 都没生成(理论上不会发生):首字延迟记成整个请求的时长
        first = last = t0 + wall
    # 结果:回答、首字延迟、输出 token 数、总耗时,解码速度先置空
    result = {"answer": text.strip(), "ttft_ms": round((first - t0) * 1000, 1), "new_tokens": n_out,
              "wall_ms": round(wall * 1000, 1), "decode_tps": None}
    # 生成够长时才计算解码速度,首个 token 不计入
    if n_out >= MIN_DECODE_TOKENS and last > first:
        result["decode_tps"] = round((n_out - 1) / (last - first), 1)
    # 引擎侧时间戳:流式模式下不保证有,取不到就只留客户端口径
    # 取到的话,与 11_bench_gpu.py 报的 TTFT 是同一口径
    metrics = getattr(comp, "request_perf_metrics", None)
    # 取到时记录引擎侧的首字延迟
    if metrics is not None:
        tm = metrics.timing_metrics
        result["engine_ttft_ms"] = round((tm.first_token_time - tm.arrival_time).total_seconds() * 1000, 1)
    return result


def ask_openvino(answerer, question: str, img, max_new_tokens: int,  # noqa: ANN001
                 echo: Callable[[str], None]) -> dict:
    """在 OpenVINO GenAI 上流式回答一个问题;参数与返回值同 ask_trtllm,另含视觉段耗时。"""
    # 重依赖只在这里导入
    import numpy as np
    import openvino as ov
    import openvino_genai as ov_genai

    # 生成参数
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens   # 贪心是默认(do_sample=False)

    # 管线每生成一段文字就调一次这个回调;返回 RUNNING 表示继续生成
    def streamer(piece: str) -> ov_genai.StreamingStatus:
        echo(piece)
        return ov_genai.StreamingStatus.RUNNING

    tensor = ov.Tensor(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None]))   # [1,H,W,3] RGB
    # 计时起点
    t0 = time.perf_counter()
    # 流式生成
    out = answerer.pipe.generate(question, images=[tensor], generation_config=config, streamer=streamer)
    # 整个请求的墙钟耗时
    wall = time.perf_counter() - t0
    # 耗时取管线自带的指标(毫秒):视觉段 = prepare_embeddings 的时长,decode 速度由每 token 耗时 TPOT 换算
    pm = out.perf_metrics
    # 生成的 token 数
    n_out = int(pm.get_num_generated_tokens())
    # 返回回答、首字延迟、视觉段耗时、输出 token 数、总耗时和解码速度
    return {"answer": out.texts[0].strip(), "ttft_ms": round(pm.get_ttft().mean, 1),
            "vision_ms": round(pm.get_prepare_embeddings_duration().mean, 1), "new_tokens": n_out,
            "wall_ms": round(wall * 1000, 1),
            "decode_tps": round(1000 / pm.get_tpot().mean, 1) if n_out >= MIN_DECODE_TOKENS else None}


def load_items(args: argparse.Namespace) -> list[dict]:
    """准备演示用的 (图片, 问题):用户给了 --image 就用它,否则取 dev 集前 --num 张不同的图。

    Returns:
        每项含 name(图片来源)、question、img(原图),dev 题另带 sample(用来打分)。
    """
    # 指定了图片时只演示这一张
    if args.image:
        # Pillow 只在这里导入
        from PIL import Image

        # 没有给问题时用默认的描述指令
        question = args.question or "Describe this image in one sentence."
        # 读入图片并转成 RGB
        return [{"name": Path(args.image).name, "question": question, "img": Image.open(args.image).convert("RGB")}]
    # 演示项列表和已经用过的图片 id
    items, seen = [], set()
    # 按固定顺序读 dev 集
    for sample in iter_split("dev"):
        if sample.image_id in seen:   # 同一张图在 dev 集里可能有多道题,每张图只取遇到的第 1 题
            continue
        # 记下这张图
        seen.add(sample.image_id)
        # 不带 --question 时用评测的原题(含"用一个词或短语回答"的后缀),这样可以对照人工答案打分
        items.append({"name": f"dev:{sample.qid}", "question": args.question or sample.prompt, "img": sample.image(),
                      "sample": None if args.question else sample})
        # 取够图片数后结束
        if len(items) >= args.num:
            break
    return items


def main() -> int:
    """入口:加载变体,预热后逐题流式作答并打印耗时,结果写入 results/demo/。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="演示:流式视觉问答 + 首字延迟")
    # 变体名,必填
    parser.add_argument("--variant", required=True, help="trtllm-* 或 ov-*(见 vlmdeploy/variants.py)")
    # 图像 token 档位
    parser.add_argument("--budget", choices=(*TOKEN_BUDGETS, *SQUARE_BUDGETS), default="t576")
    # 自己的图片路径
    parser.add_argument("--image", default=None, help="自己的图片路径;缺省用 dev 集的图")
    # 问题文字
    parser.add_argument("--question", default=None, help="问题;缺省用 dev 题原题(给了 --image 时缺省为一句话描述)")
    # 演示的图片数
    parser.add_argument("--num", type=int, default=NUM_IMAGES, help="用 dev 集时演示几张图")
    # 生成 token 数上限
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    # 解析参数
    args = parser.parse_args()
    # 两条参数校验都用 parser.error:打印用法和报错后以退出码 2 结束
    if not args.variant.startswith(("trtllm-", "ov-")):
        parser.error("演示只支持部署变体 trtllm-* / ov-*(hf-bf16 是评测参考,不是部署形态)")
    # 是否方图档位
    square = args.budget in SQUARE_BUDGETS
    # 视觉塔引擎变体只能配方图档位
    if args.variant.endswith("-vistrt") and not square:
        parser.error("-vistrt 变体的视觉塔引擎是静态形状,只能配方图档位 s256 / s576 / s1024")

    # 档位对应的图像 token 数
    tokens = SQUARE_BUDGETS[args.budget] if square else TOKEN_BUDGETS[args.budget]
    # 对应的缩放函数
    resize = square_to_budget if square else resize_to_budget
    # 准备演示用的图片和问题
    items = load_items(args)
    # 图片缩放到档位尺寸
    for item in items:
        item["img"] = resize(item["img"], tokens)

    # 是否 GPU 变体
    on_gpu = is_gpu(args.variant)
    # 加载前后的整卡显存一并记下。演示是单次观察,容易受机器上其他负载影响:2026-11-15 首跑时 -vistrt 档的
    # 引擎侧 TTFT 是 427–558 ms(之后三次运行都是 158–178 ms,原因未查明)。那一次没有记显存,无从判断是不是显存被挤占;
    # 有了这两个读数,下次数字异常时至少能排除或确认这一条。
    mem_before = gpu_memory_used_mb() if on_gpu else None
    # 方图档位只有一个网格(边长 = 2 × 每边 token 数 个 patch),-vistrt 变体据此只加载对应的那一个引擎
    answerer = make_answerer(args.variant, vision_grid=2 * square_side_tokens(tokens) if square else None)
    # 按后端选择作答函数
    ask = ask_trtllm if on_gpu else ask_openvino
    # 加载后的整卡显存
    mem_loaded = gpu_memory_used_mb() if on_gpu else None
    # 打印加载耗时
    log.info("loaded %s in %.1fs; warming up", args.variant, answerer.load_seconds)
    # GPU 变体另打印加载前后的显存
    if on_gpu:
        log.info("GPU memory used: %s MB before load, %s MB after load (whole card)", mem_before, mem_loaded)
    # 逐题记录
    rows = []
    try:
        # 预热(丢弃):首次请求里有 kernel 选择、内存池扩张、处理器初始化等一次性开销,不代表稳态
        for i in range(WARMUP_REQUESTS):
            item = items[i % len(items)]   # 只有一张图(--image、--num 1)时同一张图问两次
            # 预热请求不输出文字
            ask(answerer, item["question"], item["img"], args.max_new_tokens, lambda piece: None)
        # 逐题演示
        for item in items:
            # 缩放后的图片
            img = item["img"]
            # 图像 token 数
            n_tokens = image_tokens(img.height, img.width)
            # 打印图片来源、尺寸和图像 token 数
            print(f"\n[{item['name']}] {img.width}x{img.height} -> {n_tokens} image tokens")
            # 打印问题的第一行
            print(f"Q: {item['question'].splitlines()[0]}")
            # 回答的前缀,后面接流式输出
            write_piece("A: ")
            # 流式作答
            r = ask(answerer, item["question"], img, args.max_new_tokens, write_piece)
            # 解码速度的显示文字,生成过短时不报
            speed = f"{r['decode_tps']:.1f} tok/s" if r["decode_tps"] else "n/a (too short)"
            # 视觉段耗时的显示文字,只有 OpenVINO 有
            vision = f" (vision {r['vision_ms']:.0f} ms)" if "vision_ms" in r else ""
            # 打印首字延迟、输出 token 数、解码速度和总耗时
            print(f"\n   first token {r['ttft_ms']:.0f} ms{vision} | {r['new_tokens']} tokens | decode {speed} | "
                  f"total {r['wall_ms']:.0f} ms")
            # 本题记录:图片、问题、图像 token 数和作答结果
            row = {"image": item["name"], "question": item["question"].splitlines()[0], "image_tokens": n_tokens, **r}
            # dev 集原题另记参考答案和得分
            if item.get("sample") is not None:
                # 人工答案按出现次数从多到少排(10 个标注者里最常见的在前)
                row["references"] = [a for a, _ in Counter(item["sample"].answers).most_common()]
                # 按评测口径打分
                row["vqa_acc"] = round(score(item["sample"], r["answer"]), 4)
                # 打印前 5 个参考答案和得分
                print(f"   reference: {', '.join(row['references'][:5])} | VQA acc {row['vqa_acc']:.2f}")
            # 保存本题记录
            rows.append(row)
    finally:
        # 成败都关闭后端
        answerer.close()

    # README 的演示表引用这份记录,所以只有全默认的运行才写它;改了图片、问题、题数或生成上限的临时演示只看终端输出
    if args.image or args.question or args.num != NUM_IMAGES or args.max_new_tokens != MAX_NEW_TOKENS:
        log.info("non-default image / question / --num / --max-new-tokens: results not saved")
        return 0
    # 演示记录:变体、档位、生成上限、加载耗时、时间、预热次数、显存、首字延迟口径和逐题记录
    result = {"variant": args.variant, "budget": args.budget, "max_new_tokens": args.max_new_tokens,
              "load_seconds": round(answerer.load_seconds, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "warmup_requests": WARMUP_REQUESTS,
              "gpu_mem_mb": {"before_load": mem_before, "after_load": mem_loaded} if on_gpu else None,
              "ttft_source": "client wall clock" if on_gpu else "openvino_genai perf_metrics",
              "rows": rows}
    # 创建演示结果目录
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    # 记录文件
    out = DEMO_DIR / f"{args.variant}_{args.budget}.json"
    # 写出记录
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    # 打印记录文件路径
    log.info("saved %s", out)
    return 0


if __name__ == "__main__":   # TensorRT-LLM 会用 spawn 另起 worker 进程,入口必须有这道保护
    sys.exit(main())
