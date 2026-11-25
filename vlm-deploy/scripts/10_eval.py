"""评测:任一变体 × 任一划分 × 任一 token 档位,逐题推理 + 打分,结果可断点续跑。

用法(GPU 变体用 env trtllm 并带 PYTHONPATH=vlmdeploy/patches/site;OpenVINO 变体用 env ovgenai;Makefile 已封装):
    python scripts/10_eval.py --variant hf-bf16      --split test --budget native    # P1:复现官方分数
    python scripts/10_eval.py --variant trtllm-bf16  --split test --budget t576      # P2
    python scripts/10_eval.py --variant trtllm-fp8-img --split dev --budget t576     # P3 决策(dev 集)
    python scripts/10_eval.py --variant ov-int4      --split test --budget t256 --limit 500   # P4

划分(定义见 vlmdeploy/tasks.py):test = TextVQA val 5000 题;rwqa = RealWorldQA 765 题;
dev = TextVQA train 分片的 3/4 图片(做决策用);calib 只给量化校准,不在这里评测。

档位:native = 不预缩放,交给 HF 处理器按默认像素范围处理(官方评测的设置,只用于 P1 复现);
t256 / t576 / t1024 = 预缩放到对应图像 token 上限(跨栈、跨精度对比一律用预缩放,保证输入完全一致)。

设计决定:
    - 逐题写 JSONL、每批 flush,重跑时跳过已完成的 qid。全量一次要 0.5–2 小时,WSL 只有 8 GB,
      笔记本还可能休眠 —— 任何中断都不应该从头再来。汇总只读 JSONL,与推理进程解耦。
    - 汇总带 95% 置信区间。TextVQA 5000 题的区间约 ±1.1 分:两个变体差 0.5 分不能叫"掉点",
      所以量化对比另用配对差(同一题两个变体得分相减)的区间,见 summarize 里的 paired_vs。
    - P1 的验收线:|复现 − 官方| ≤ 1.5(约 1.4 倍置信半宽),过线才说明取题、提示词、打分没有写错。

产物:
    results/eval/<variant>_<split>_<budget>.jsonl   逐题(qid、答案、得分、图像 token 数)
    results/eval/<variant>_<split>_<budget>.json    汇总
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import RESULTS_DIR, SQUARE_BUDGETS, TOKEN_BUDGETS
from vlmdeploy.imaging import image_tokens, resize_to_budget, smart_resize, square_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.tasks import iter_split, score
from vlmdeploy.variants import make_answerer

# 模块级 logger
log = get_logger("eval")
# 评测结果目录
EVAL_DIR = RESULTS_DIR / "eval"
# Qwen2-VL-2B-Instruct 官方报告分数(模型卡)。只有 hf-bf16 / native 与之比较。
OFFICIAL = {"test": 79.7, "rwqa": 62.9}
# 与官方分数相差多少分以内算通过
TOLERANCE = 1.5
# 一次提交给后端的题数:TensorRT-LLM 批量调度,太小吃不满 GPU,太大则预处理后的像素张量占内存
# (t1024 一张图的 pixel_values 约 19 MB)。其他后端逐题,批大小只影响 flush 频率。
CHUNK = {"trtllm": 32}


def tag(variant: str, split: str, budget: str, limit: int | None) -> str:
    """结果文件名的主干:<变体>_<划分>_<档位>,试跑时带 _n<题数> 后缀。"""
    return f"{variant}_{split}_{budget}" + (f"_n{limit}" if limit else "")   # 带后缀的试跑结果不会覆盖正式结果


def load_rows(path: Path) -> list[dict]:
    """读逐题 JSONL;文件不存在时返回空列表(首次运行)。"""
    # 文件不存在时返回空列表
    if not path.exists():
        return []
    # 逐行解析 JSON,跳过空行
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(variant: str, split: str, budget: str, limit: int | None, out: Path) -> dict:
    """推理并追加写入 out;返回本次运行的耗时信息。"""
    # 已经答过的题号
    done = {r["qid"] for r in load_rows(out)}
    # 只先收集待答 qid,不把题目(含图片字节)整批留在内存里:TextVQA val 的图片字节合计约 1 GB,
    # 与 TensorRT-LLM 同驻 8 GB 的 WSL 太挤。推理时再按分片顺序流式读取。
    todo_ids = [s.qid for s in iter_split(split, limit=limit) if s.qid not in done]
    # 打印已答和待答的题数
    log.info("%s: %d done, %d to go", out.name, len(done), len(todo_ids))
    if not todo_ids:   # 全部答完:不加载模型,直接返回去做汇总
        return {"seconds": 0.0, "answered": 0}
    # 待答题号的集合
    pending = set(todo_ids)
    # 只产出待答题目的生成器
    stream = (s for s in iter_split(split, limit=limit) if s.qid in pending)
    # 方图档位只有一个网格(边长 = 2·√token 个 patch),-vistrt 变体据此只加载对应的那一个视觉塔引擎
    grid = 2 * round(SQUARE_BUDGETS[budget] ** 0.5) if budget in SQUARE_BUDGETS else None
    # 模型加载只经过 make_answerer(重依赖在它里面才导入),本脚本因此在 trtllm、ovgenai 两个环境里都能跑
    answerer = make_answerer(variant, vision_grid=grid)
    # 打印加载耗时
    log.info("loaded %s in %.1fs", variant, answerer.load_seconds)
    # 变体名的第一段是栈名(trtllm-fp8-img -> trtllm);CHUNK 里没有的栈逐题推理,8 只决定多久落一次盘
    chunk = CHUNK.get(variant.split("-")[0], 8)
    # 计时起点和已答题数
    t0, n = time.time(), 0
    try:
        # 以追加方式打开结果文件
        with out.open("a", encoding="utf-8") as f:
            # 每次取一批题,取不到时结束
            while True:
                # 本批的样本、图片和图像 token 数
                batch = []
                for sample in itertools.islice(stream, chunk):   # stream 是生成器,每次接着上次的位置再取 chunk 题
                    # 解码图片
                    img = sample.image()
                    if budget in SQUARE_BUDGETS:   # P5:方图,保证所有图落在视觉塔引擎的同一个静态网格上
                        # 裁成方图并缩放到档位尺寸
                        img = square_to_budget(img, SQUARE_BUDGETS[budget])
                        # 图像 token 数
                        tokens = image_tokens(img.height, img.width)
                    # 普通档位:保持长宽比缩到 token 预算内
                    elif budget != "native":
                        img = resize_to_budget(img, TOKEN_BUDGETS[budget])
                        tokens = image_tokens(img.height, img.width)
                    else:   # 原生分辨率:按处理器默认像素范围的 smart_resize 算出实际 token 数
                        tokens = image_tokens(*smart_resize(img.height, img.width))
                    # 加入本批
                    batch.append((sample, img, tokens))
                # 没有待答的题时结束
                if not batch:
                    break
                # 整批作答
                answers = answerer.answer([(s, im) for s, im, _ in batch])
                # 逐题写出一行 JSON:题号、答案、得分和图像 token 数
                for (sample, _, tokens), ans in zip(batch, answers, strict=True):
                    f.write(json.dumps({"qid": sample.qid, "answer": ans, "score": score(sample, ans),
                                        "image_tokens": tokens}, ensure_ascii=False) + "\n")
                f.flush()   # 每批落盘一次:进程中途被杀,最多丢这一批
                # 累计已答题数
                n += len(batch)
                # n 刚跨过 200 的整数倍(批大小不一定整除 200,不能写 n % 200 == 0),或全部答完时,报一次进度
                if n % 200 < len(batch) or n == len(todo_ids):
                    log.info("%d/%d, %.2f s/question", n, len(todo_ids), (time.time() - t0) / n)
    finally:
        answerer.close()   # 成败都要关:释放显存、结束 worker 进程
    # 返回耗时、答题数、加载耗时和后端的附加信息
    return {"seconds": round(time.time() - t0, 1), "answered": n, "load_seconds": round(answerer.load_seconds, 1),
            **answerer.info}


def mean_ci(values: list[float]) -> tuple[float, float]:
    """均值与 95% 置信半宽(正态近似),单位:分(×100)。"""
    # 95% 置信半宽,样本不足两个时为 NaN
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    # 均值和半宽都换算成百分制
    return 100 * statistics.fmean(values), 100 * half


def summarize(variant: str, split: str, budget: str, limit: int | None, run_info: dict,
              paired_vs: str | None) -> dict:
    """从逐题 JSONL 算汇总(均值、95% CI、可选的配对差),写入同名 .json 并返回。"""
    # 结果文件名的主干
    name = tag(variant, split, budget, limit)
    # 重读整份 JSONL,不用本次推理留在内存里的结果:续跑时,汇总自然包含之前几次答完的题
    rows = load_rows(EVAL_DIR / f"{name}.jsonl")
    # 得分的均值和 95% 置信半宽
    mean, half = mean_ci([r["score"] for r in rows])
    # 汇总:变体、划分、档位、题数、分数、置信半宽、平均图像 token 数、运行信息和时间
    result = {"variant": variant, "split": split, "budget": budget, "num": len(rows), "score": round(mean, 2),
              "ci95": round(half, 2), "mean_image_tokens": round(statistics.fmean(r["image_tokens"] for r in rows), 1),
              "run": run_info, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    # 只有 HF 基线在原生分辨率下的全量结果才与官方分数比较
    if variant == "hf-bf16" and budget == "native" and split in OFFICIAL and not limit:
        # 官方分数
        result["official"] = OFFICIAL[split]
        # 与官方分数的差
        result["delta_vs_official"] = round(mean - OFFICIAL[split], 2)
        # 差值在容差内算通过
        result["pass"] = abs(result["delta_vs_official"]) <= TOLERANCE
        # 固定 1.5 分的线对 5000 题(CI 约 ±1.1)合适,对 765 题的 RealWorldQA(CI 约 ±3.5)比统计噪声还窄。
        # 不事后放宽验收线,而是另报"差是否落在本次复现的 95% 置信区间内",两个判据都写进 README。
        result["within_ci95"] = abs(result["delta_vs_official"]) <= result["ci95"]
    if paired_vs:   # 配对差:同一题上 本变体 − 参照变体,区间比两个独立区间窄得多
        # 参照变体优先找同样题数的结果,找不到就用它的全量结果(题目按固定顺序读取,前 N 题是全量的子集)
        ref_path = EVAL_DIR / f"{tag(paired_vs, split, budget, limit)}.jsonl"
        if not ref_path.exists():
            ref_path = EVAL_DIR / f"{tag(paired_vs, split, budget, None)}.jsonl"
        # 题号到参照变体结果的映射
        ref = {r["qid"]: r for r in load_rows(ref_path)}
        # 每对 = (得分差, 两个变体的答案是否逐字相同);只取两边都答过的题
        pairs = [(r["score"] - ref[r["qid"]]["score"], r["answer"] == ref[r["qid"]]["answer"])
                 for r in rows if r["qid"] in ref]
        # 有共同答过的题才计算配对差
        if pairs:
            # 得分差的均值和 95% 置信半宽
            d, dh = mean_ci([p[0] for p in pairs])
            # 记录参照变体、配对题数、差值、置信半宽和答案相同的比例
            result["paired"] = {"vs": paired_vs, "n": len(pairs), "delta": round(d, 2), "ci95": round(dh, 2),
                                "same_answer": round(sum(p[1] for p in pairs) / len(pairs), 4)}
    # 写出汇总文件
    (EVAL_DIR / f"{name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    """入口:推理(可断点续跑)后汇总,打印一行结果。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="评测(任一变体 × 划分 × token 档位)")
    # 变体名,必填
    parser.add_argument("--variant", required=True)
    # 数据划分,必填
    parser.add_argument("--split", choices=("test", "rwqa", "dev"), required=True)
    # 图像 token 档位
    parser.add_argument("--budget", choices=("native", *TOKEN_BUDGETS, *SQUARE_BUDGETS), default="t576")
    # 只跑前 N 题
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题(试跑;正式结果不加)")
    # 配对比较的参照变体
    parser.add_argument("--paired-vs", default=None, help="与哪个变体做逐题配对比较(同划分同档位)")
    # 解析参数
    args = parser.parse_args()

    # 创建评测结果目录
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # 逐题结果文件
    out = EVAL_DIR / f"{tag(args.variant, args.split, args.budget, args.limit)}.jsonl"
    # 推理,已答过的题跳过
    info = run(args.variant, args.split, args.budget, args.limit, out)
    # 汇总并写出结果
    result = summarize(args.variant, args.split, args.budget, args.limit, info, args.paired_vs)
    # 打印汇总结果,不含运行信息
    log.info("%s", json.dumps({k: v for k, v in result.items() if k != "run"}, ensure_ascii=False))
    return 0


# worker 启动时会重新导入本脚本:没有这道保护,导入时就会再跑一遍 main()、再起 worker
if __name__ == "__main__":   # TensorRT-LLM 会用 spawn 另起 worker 进程,入口必须有这道保护
    sys.exit(main())
