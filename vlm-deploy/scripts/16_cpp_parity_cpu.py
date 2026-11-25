r"""CPU 栈 C++ 程序与 Python 的一致性核对 + 配对计时(cpp/vlm_infer_ov vs OpenVINO GenAI 的 Python 接口)。

用法(env: ovgenai;需先 `make cpp-cpu`;CPU 独占;Makefile 已封装为 `make parity-cpu`):
    python scripts/16_cpp_parity_cpu.py --variant ov-int4-vis8w

要回答的问题只有一个:"C++ 程序和评测用的 Python 路径是不是同一回事?"是,C++ 的数字才能引用。
两边调的是同一个库(libopenvino_genai.so)、同一份 IR、同样的配置,所以判据很严:

    1. 回答:dev 集前 --num 道题,两边的回答应当**逐字相同**(只去首尾空白)。
       Python 侧走评测的入口 variants.make_answerer,用它已经加载好的管线和生成参数 —— 核对的就是被评测的那条路径。
    2. 输入 token 数:两边报告的输入序列长度(图像 token + 文字 token)应当逐题相同。
       它排除的是"图片或提示词到了管线里其实不一样"—— 比如 C++ 读图时通道顺序错了、提示词多了一个换行。
    3. 计时:同一批图,两边各测两遍,报首字延迟分解与 decode 速度。

怎么保证两边喂进去的东西相同:
    图片由本脚本预缩放到档位之内(与评测相同,见 imaging.resize_to_budget),存成 PNG 交给 C++ 程序。
    PNG 是无损格式,C++ 读回来的像素与 Python 内存里的逐个相同(本脚本会把 PNG 再读回来核对一遍)。
    提示词单独存成文本文件:它本身带换行("问题\n请用一个词或短语回答"),没法和路径写在同一行。

计时纪律(与 14_bench_cpu.py 同一套思路):
    - 两个实现不能同时驻留内存、也不能同时占用 4 个核,所以按 ABBA 交替:Python → C++ → C++ → Python,
      每遍各自加载、测完释放。CPU 降频与后台负载的漂移在两遍里方向相反,大致抵消;
    - 每遍先用"最后一张图"预热 1 次(丢弃),再对每张图各测 1 次。预热故意不用第 1 张图:
      管线会复用上一次请求的 KV cache,若预热与第 1 个正式请求相同,那个样本的 prefill 几乎被整段跳过
      (见 cpp/vlm_infer_ov.cpp 里 RunBatch 的说明);
    - 每个请求强制生成 DECODE_TOKENS 个 token(忽略结束符),decode 速度才在同样多的 token 上量。
    预期:两边是同一个库,推理耗时不该有差别。差几个百分点属于机器状态的漂移,不能读成"C++ 更快 / 更慢"。

产物:results/cpp_parity_cpu.json(README 的"C++ runtime"表由 90_report.py 从它生成)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.backends import CPU_THREADS, MAX_NEW_TOKENS
from vlmdeploy.config import REPO_ROOT, RESULTS_DIR, TOKEN_BUDGETS
from vlmdeploy.imaging import image_tokens, resize_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.tasks import iter_split, score
from vlmdeploy.timing import summarize
from vlmdeploy.variants import make_answerer, ov_ir_dir

# 模块级 logger
log = get_logger("cpp_parity")
# C++ 程序的路径
BINARY = REPO_ROOT / "cpp" / "build" / "vlm_infer_ov"
BUDGET = "t256"  # 图像 token 档位,与 CPU 栈的评测和计时相同
NUM_PARITY = 16  # 对拍的题数
NUM_IMAGES = 6  # 计时用的图片数,与 14_bench_cpu.py 相同
DECODE_TOKENS = 32  # 计时时每个请求固定生成的 token 数,与 14_bench_cpu.py 相同


def prepare(tmpdir: Path, num: int, distinct_images: bool) -> list[dict]:
    """取 dev 集的题,预缩放后把图片和提示词写到临时目录,供两边共用。

    Args:
        tmpdir: 临时目录。
        num: 要几道题。
        distinct_images: True 时每张图只取遇到的第 1 题(计时用:保证是 num 张不同的图)。

    Returns:
        每题一个字典:sample(题目)、img(预缩放后的 PIL 图)、png / txt(写到磁盘的图片与提示词路径)。
    """
    # numpy 和 Pillow 只在这里导入
    import numpy as np
    from PIL import Image

    # 题目列表和已经用过的图片 id
    items, seen = [], set()
    # 按固定顺序读 dev 集,不使用测试集
    for sample in iter_split("dev"):
        # 计时模式下同一张图只用第一道题
        if distinct_images and sample.image_id in seen:
            continue
        # 记下这张图
        seen.add(sample.image_id)
        # 图片预缩放到档位之内
        img = resize_to_budget(sample.image(), TOKEN_BUDGETS[BUDGET])
        # 文件名主干:计时用 img 前缀,对拍用 q 前缀,后接两位序号
        tag = f"{'img' if distinct_images else 'q'}{len(items):02d}"
        # 图片和提示词的落盘路径
        png, txt = tmpdir / f"{tag}.png", tmpdir / f"{tag}.txt"
        # 存成 PNG。不写入原图的色彩配置文件:它不影响像素值,但格式不规范时 libpng 会告警
        img.save(png, icc_profile=None)
        # 读回来核对,落盘的像素必须与内存里的完全相同
        with Image.open(png) as back:
            if not np.array_equal(np.asarray(back.convert("RGB")), np.asarray(img)):
                raise RuntimeError(f"{png} does not round-trip losslessly")
        # 提示词按原样写出,不转换换行符,换行是模型输入的一部分
        txt.write_text(sample.prompt, encoding="utf-8", newline="")
        # 保存样本、图片和两个文件的路径
        items.append({"sample": sample, "img": img, "png": png, "txt": txt})
        # 取够题数后结束
        if len(items) >= num:
            break
    return items


def run_cpp(variant: str, items: list[dict], tmpdir: Path, tag: str, extra: list[str]) -> list[dict]:
    """用 C++ 程序回答一批请求,返回它写出的逐请求记录(回答 + 分段计时)。"""
    # 请求清单和结果文件的路径
    listing, out = tmpdir / f"{tag}.tsv", tmpdir / f"{tag}.jsonl"
    # 写出请求清单,每行依次是图片路径、制表符、提示词文件路径
    listing.write_text("".join(f"{it['png']}\t{it['txt']}\n" for it in items), encoding="utf-8", newline="")
    # 调用 C++ 程序。线程数和生成上限显式传入,以 backends.py 的取值为准。
    # 返回非零退出码时抛异常。C++ 程序的进度信息直接显示在终端,不捕获。
    subprocess.run([str(BINARY), "--model", str(ov_ir_dir(variant)), "--requests", str(listing), "--out", str(out),
                    "--threads", str(CPU_THREADS), *extra], check=True)
    # 读回结果,每个请求一行 JSON
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 结果行数与请求数不一致时报错
    if len(rows) != len(items):
        raise RuntimeError(f"C++ answered {len(rows)} of {len(items)} requests")
    return rows


def as_tensor(img):  # noqa: ANN001, ANN201 —— PIL.Image → ov.Tensor
    """PIL 图 → [1, H, W, 3] uint8 张量,与 backends.OpenVinoAnswerer 里的写法相同。"""
    # numpy 和 openvino 只在这里导入
    import numpy as np
    import openvino as ov

    # 加上批维度并保证内存连续
    return ov.Tensor(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None]))


def run_python(variant: str, items: list[dict], bench: bool) -> list[dict]:
    """用 Python 接口回答同一批请求,返回与 run_cpp 相同字段的记录。

    Args:
        variant: 变体名。
        items: prepare 的返回值。
        bench: True 时按计时的设置跑(强制生成 DECODE_TOKENS 个 token,先预热 1 次)。
    """
    # openvino_genai 只在这里导入
    import openvino_genai as ov_genai

    # 与评测走同一个入口加载模型
    answerer = make_answerer(variant)
    config = answerer.config  # 评测用的生成参数
    # 计时模式:改用固定生成长度的参数,并先预热一次
    if bench:
        config = ov_genai.GenerationConfig()
        # 上下限相同并忽略结束符:每个请求都恰好生成 DECODE_TOKENS 个 token
        config.max_new_tokens = DECODE_TOKENS
        config.min_new_tokens = DECODE_TOKENS
        config.ignore_eos = True
        # 用最后一张图预热,结果丢弃,原因见文件头的计时纪律
        answerer.pipe.generate(items[-1]["sample"].prompt, images=[as_tensor(items[-1]["img"])],
                               generation_config=config)
    # 逐请求的记录
    rows = []
    # 逐题作答
    for it in items:
        tensor = as_tensor(it["img"])  # 转张量不计入耗时
        # 计时起点
        t0 = time.perf_counter()
        # 生成
        result = answerer.pipe.generate(it["sample"].prompt, images=[tensor], generation_config=config)
        # 墙钟耗时(毫秒)
        wall = (time.perf_counter() - t0) * 1000
        # 这次请求的性能指标
        pm = result.perf_metrics
        # 各项指标返回均值和标准差,一次 generate 只有一个样本,取均值
        rows.append({"answer": result.texts[0], "ttft_ms": pm.get_ttft().mean,
                     "vision_ms": pm.get_prepare_embeddings_duration().mean, "tpot_ms": pm.get_tpot().mean,
                     "input_tokens": int(pm.get_num_input_tokens()),
                     "new_tokens": int(pm.get_num_generated_tokens()), "wall_ms": wall,
                     "load_s": answerer.load_seconds})
    # 释放管线,下一步启动 C++ 进程前要让出内存
    answerer.close()
    del answerer
    return rows


def segments(rows: list[dict]) -> dict:
    """把一个实现两遍测量的样本汇总成各段的 P50 / P90。"""
    return {"ttft_ms": summarize([r["ttft_ms"] for r in rows]),
            "vision_ms": summarize([r["vision_ms"] for r in rows]),
            # prefill 不直接测量,逐请求用首字延迟减去视觉段后再汇总
            "prefill_ms": summarize([r["ttft_ms"] - r["vision_ms"] for r in rows]),
            # 每 token 耗时换算成每秒 token 数
            "decode_tps": summarize([1000 / r["tpot_ms"] for r in rows]),
            "wall_ms": summarize([r["wall_ms"] for r in rows]),
            "load_seconds": round(statistics.median(r["load_s"] for r in rows), 1)}


def main() -> int:
    """入口:逐题对拍回答与输入 token 数,再按 ABBA 做配对计时,结果写入 results/cpp_parity_cpu.json。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="CPU 栈 C++ 程序与 Python 的对拍 + 配对计时")
    # 变体名
    parser.add_argument("--variant", default="ov-int4-vis8w", help="ov-* 变体(见 vlmdeploy/variants.py)")
    # 对拍的题数
    parser.add_argument("--num", type=int, default=NUM_PARITY, help="对拍几道题")
    # 解析参数
    args = parser.parse_args()
    # 只接受 CPU 栈的变体
    if not args.variant.startswith("ov-"):
        parser.error("只有 CPU 栈(ov-*)有 C++ 程序")
    # C++ 程序还没有构建时报错
    if not BINARY.exists():
        raise FileNotFoundError(f"{BINARY} 不存在,请先运行 make cpp-cpu")

    # 临时目录存放图片、提示词、请求清单和 C++ 的输出,用完自动删除
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # 第 1 步:对拍回答和输入 token 数
        # 准备对拍用的题目
        items = prepare(tmpdir, args.num, distinct_images=False)
        # 打印题数和档位
        log.info("parity: %d dev questions at %s", len(items), BUDGET)
        # Python 作答
        py = run_python(args.variant, items, bench=False)
        # C++ 作答,生成上限与评测相同
        cpp = run_cpp(args.variant, items, tmpdir, "parity", ["--max-new-tokens", str(MAX_NEW_TOKENS)])
        # 逐题的对拍记录
        rows = []
        # 逐题比较
        for it, p, c in zip(items, py, cpp, strict=True):
            # 样本和预缩放后的图片
            sample, img = it["sample"], it["img"]
            # 两边的回答都只去掉首尾空白后比较,与评测的处理相同
            a_py, a_cpp = p["answer"].strip(), c["answer"].strip()
            # 记录题号、问题、图像 token 数、两边的回答、输入 token 数和得分
            rows.append({"qid": sample.qid, "question": sample.question,
                         "image_tokens": image_tokens(img.height, img.width),
                         "answer_python": a_py, "answer_cpp": a_cpp, "same_answer": a_py == a_cpp,
                         "input_tokens_python": p["input_tokens"], "input_tokens_cpp": c["input_tokens"],
                         # 两边的回答用同一个打分函数打分
                         "vqa_acc_python": round(score(sample, a_py), 4),
                         "vqa_acc_cpp": round(score(sample, a_cpp), 4)})
            # 回答不同时打印出来
            if not rows[-1]["same_answer"]:
                log.info("  %s differs: python %r vs cpp %r", sample.qid, a_py, a_cpp)
        # 对拍汇总:题数、回答相同的题数、输入 token 数相同的题数和两边的平均分
        parity = {"n": len(rows),
                  "same_answer": sum(r["same_answer"] for r in rows),
                  "same_input_tokens": sum(r["input_tokens_python"] == r["input_tokens_cpp"] for r in rows),
                  "mean_vqa_acc_python": round(statistics.fmean(r["vqa_acc_python"] for r in rows), 4),
                  "mean_vqa_acc_cpp": round(statistics.fmean(r["vqa_acc_cpp"] for r in rows), 4)}
        # 打印对拍结果
        log.info("parity: %d/%d answers identical, %d/%d input lengths identical", parity["same_answer"],
                 parity["n"], parity["same_input_tokens"], parity["n"])

        # 第 2 步:按 ABBA 顺序做配对计时
        # 准备计时用的图片
        images = prepare(tmpdir, NUM_IMAGES, distinct_images=True)
        # 四遍的执行顺序
        order = ["python", "cpp", "cpp", "python"]
        # 两个实现各自的全部样本
        samples: dict[str, list[dict]] = {"python": [], "cpp": []}
        per_pass: dict[str, list[float]] = {"python": [], "cpp": []}   # 每遍各自的首字延迟中位数
        # 逐遍执行
        for n, impl in enumerate(order):
            # 按实现选择作答方式,计时模式下固定生成长度并预热一次
            if impl == "python":
                got = run_python(args.variant, images, bench=True)
            else:
                got = run_cpp(args.variant, images, tmpdir, f"bench{n}",
                              ["--decode-tokens", str(DECODE_TOKENS), "--warmup", "1"])
            # 并入这个实现的样本
            samples[impl] += got
            # 记录本遍的首字延迟中位数
            per_pass[impl].append(statistics.median(r["ttft_ms"] for r in got))
            # 打印本遍结果
            log.info("pass %d %-6s TTFT p50 %.0f ms", n, impl, per_pass[impl][-1])
        # 两个实现各自的分段统计
        bench = {impl: segments(rows_) for impl, rows_ in samples.items()}
        # 计算两遍之间的漂移并打印
        for impl in bench:
            # 漂移等于两遍中位数之差除以两者的均值
            a, b = per_pass[impl]
            bench[impl]["ttft_pass_drift"] = round((b - a) / statistics.fmean((a, b)), 3)
            log.info("%-6s TTFT %5.0f ms = vision %5.0f + prefill %5.0f; decode %5.1f tok/s; pass drift %+.1f%%", impl,
                     bench[impl]["ttft_ms"]["p50"], bench[impl]["vision_ms"]["p50"], bench[impl]["prefill_ms"]["p50"],
                     bench[impl]["decode_tps"]["p50"], 100 * bench[impl]["ttft_pass_drift"])

    # 结果:变体、档位、线程数、时间、对拍汇总、计时统计和逐题记录
    result = {"variant": args.variant, "budget": BUDGET, "threads": CPU_THREADS,
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "parity": parity,
              "bench": {"order": order, "images": NUM_IMAGES, "decode_tokens": DECODE_TOKENS, **bench,
                        # C++ 与 Python 的首字延迟 P50 之比,应接近 1,偏离程度要对照漂移幅度判断
                        "ttft_ratio_cpp_over_python": round(bench["cpp"]["ttft_ms"]["p50"]
                                                            / bench["python"]["ttft_ms"]["p50"], 3)},
              "rows": rows}
    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 结果文件
    out = RESULTS_DIR / "cpp_parity_cpu.json"
    # 写出结果
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    # 打印结果文件路径
    log.info("saved %s", out)
    # 回答或输入长度不一致时返回非零退出码,供 Makefile 判断成败
    return 0 if parity["same_answer"] == parity["n"] == parity["same_input_tokens"] else 1


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
