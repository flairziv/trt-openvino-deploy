"""把 results/ 下的机器可读结果汇总成 README 表格(与项目 A 的 12_report.py 同一约定)。

用法(任一环境,只依赖标准库):
    python3 scripts/90_report.py              # 刷新全部小节
    python3 scripts/90_report.py --only P5    # 只刷新指定小节(results/ 不完整时用,其余小节原样保留)

小节名:SMOKE(P0)、P1–P5、DEMO(P6 演示记录)、CPP(CPU 栈 C++ 程序的对拍与计时)。

README 里 `<!-- RESULTS:xxx:BEGIN -->` 与 `<!-- RESULTS:xxx:END -->` 之间的内容由本脚本覆盖。
数字**只**从 results/ 来,不允许手工改 —— 避免文档与实测脱节。编号 90 是为了让它始终排在各阶段脚本之后。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# 仓库根目录
ROOT = Path(__file__).resolve().parent.parent
# 结果目录
RESULTS = ROOT / "results"
# 要回填的 README
README = ROOT / "README.md"


def load(name: str) -> dict:
    """读 results/ 下的一个 JSON;文件不存在时返回空字典(该项尚未生成)。"""
    # 结果文件路径
    path = RESULTS / name
    # 全脚本的容错都靠这个空字典:某项没跑过,调用方判空后跳过对应的行或小节,不会抛异常
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def peak_rss_gb(name: str) -> str:
    """从 /usr/bin/time -v 的输出里取峰值常驻内存(Maximum resident set size,单位 KB)。"""
    # 记录文件路径
    path = RESULTS / name
    # 文件不存在时显示占位符
    if not path.exists():
        return "—"
    # 匹配峰值常驻内存那一行
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text(encoding="utf-8"))
    # KB 换算成 GiB,匹配不到时显示占位符
    return f"{int(match.group(1)) / 2**20:.2f} GiB" if match else "—"


def table(headers: list[str], rows: list[list[str]]) -> str:
    """拼一张 Markdown 表格。"""
    # 表头行和分隔行
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    # 数据行
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    # 用换行连接
    return "\n".join(lines)


def smoke() -> str:
    """P0:三方冒烟对照表。"""
    # 表格的数据行
    rows = []
    # HF 参考的首题提示词
    hf_prompt = load("smoke_hf_t256.json").get("first_prompt")
    # 各后端的结果文件名、显示名和精度说明
    specs = (("hf", "HF transformers", "BF16,GPU"), ("trtllm", "TensorRT-LLM 1.3.0rc18", "BF16,GPU"),
             ("openvino", "OpenVINO GenAI 2026.4.1", "FP16 权重,FP32 计算,CPU 4 线程"))
    # 逐个后端生成一行
    for key, name, precision in specs:
        # 读冒烟结果
        r = load(f"smoke_{key}_t256.json")
        # 没跑过的后端跳过
        if not r:
            continue
        # CPU 后端没有显存可报,报 /usr/bin/time 记下的进程峰值内存;GPU 后端报整卡显存
        if key == "openvino":
            memory = f"进程峰值 {peak_rss_gb('smoke_ov_time.txt')}"
        else:
            memory = f"整卡 {r['gpu_mem_used_mb']} MB" if r.get("gpu_mem_used_mb") else "—"
        if key == "trtllm":   # GPU 两个后端出自同一个 chat_prompt,结果里存了首题提示词供核对
            prompt_check = "提示词文本相同" if hf_prompt and r.get("first_prompt") == hf_prompt else "提示词不同"
        else:
            # 其他后端显示输入 token 数的核对结果
            prompt_check = r.get("input_tokens_match_hf", "—")
        # 一行:后端、精度、平均分、与 HF 一致的题数、输入核对、加载耗时和内存
        rows.append([name, precision, f"{r['mean_vqa_acc']:.3f}", r.get("agree_with_hf_raw", "参考"),
                     prompt_check, f"{r['seconds'].get('load', 0):.1f}", memory])
    if not rows:
        return ""   # 一个后端都没跑过:这一节在 README 里显示"尚未生成"
    # 生成表格
    body = table(["后端", "精度 / 设备", "8 题平均 VQA acc", "答案与 HF 逐字相同", "输入与 HF 一致",
                  "加载 s", "峰值显存 / 内存"], rows)
    # OpenVINO 的冒烟结果
    ov = load("smoke_openvino_t256.json")
    # 表格下面的说明
    notes = [f"OpenVINO 导出(optimum-cli,FP16):进程峰值 {peak_rss_gb('ov_export_time.txt')}。"]
    # 有延迟数据时补一条说明
    if ov.get("ttft_ms_mean"):
        notes.append(f"OpenVINO 单次运行观察值(含首次调用开销,不是基准):TTFT 均值 {ov['ttft_ms_mean'] / 1000:.1f} s,"
                     f"其中视觉编码 + embedding 拼接 {ov['prepare_embeddings_ms_mean'] / 1000:.1f} s。")
    # 表格和说明之间空一行
    return body + "\n\n" + "\n".join(f"- {n}" for n in notes)


# ---------------------------------------------------------------------------
# P1–P5:各阶段表格。缺数据的行直接跳过,整节没数据时返回空串(README 里显示"尚未生成")。
# ---------------------------------------------------------------------------
def ev(variant: str, split: str, budget: str, limit: int | None = None) -> dict:
    """读一条评测汇总:results/eval/<变体>_<划分>_<档位>[_n<题数>].json。"""
    # 文件名规则与 10_eval.py 的 tag() 一致,那边改了这里要跟着改
    return load(f"eval/{variant}_{split}_{budget}" + (f"_n{limit}" if limit else "") + ".json")


def fmt_score(r: dict) -> str:
    """格式化成“分数 ± 95% 置信半宽”;没有数据时为“—”。"""
    # 有数据时格式化,否则显示占位符
    return f"{r['score']:.2f} ± {r['ci95']:.2f}" if r else "—"


def fmt_paired(r: dict) -> str:
    """格式化逐题配对差及其 95% 置信半宽;没有配对数据时为“—”。"""
    # 配对数据,没有时为 None
    p = r.get("paired") if r else None
    # 带正负号的差值和置信半宽
    return f"{p['delta']:+.2f} ± {p['ci95']:.2f}" if p else "—"


def fmt_same(r: dict) -> str:
    """格式化与参照变体答案逐字相同的比例。"""
    # 配对数据,没有时为 None
    p = r.get("paired") if r else None
    # 换算成百分比
    return f"{100 * p['same_answer']:.1f}%" if p else "—"


def p1() -> str:
    """P1:HF BF16 原生分辨率复现官方分数。"""
    # 表格的数据行
    rows = []
    # 两个数据集各一行
    for split, name in (("test", "TextVQA val"), ("rwqa", "RealWorldQA")):
        # HF 基线在原生分辨率下的评测结果
        r = ev("hf-bf16", split, "native")
        # 没有结果时跳过
        if not r:
            continue
        # 最后两格是两个判据:绝对差 ≤ 1.5 取 10_eval.py 写进结果的 pass,绝对差 ≤ CI 半宽在这里现算
        rows.append([name, str(r["num"]), f"{r['mean_image_tokens']:.0f}", fmt_score(r), f"{r['official']}",
                     f"{r['delta_vs_official']:+.2f}", "✅" if r["pass"] else "❌",
                     "✅" if abs(r["delta_vs_official"]) <= r["ci95"] else "❌"])
    # 没有数据时返回空串
    if not rows:
        return ""
    # 生成表格
    return table(
        ["数据集", "题数", "平均图像 token", "复现(95% CI)", "官方", "差", "绝对差 ≤ 1.5", "绝对差 ≤ CI 半宽"], rows)


def p2() -> str:
    """P2:GPU BF16 三档 token 的精度表与 TTFT 分解表。"""
    # 精度表的数据行
    acc_rows = []
    # 三个档位各一行
    for budget in ("t256", "t576", "t1024"):
        # TextVQA 和 RealWorldQA 的评测结果
        t, w = ev("trtllm-bf16", "test", budget), ev("trtllm-bf16", "rwqa", budget)
        # 任一有结果就输出这一行
        if t or w:
            acc_rows.append([budget, f"{t.get('mean_image_tokens', 0):.0f}" if t else "—", fmt_score(t), fmt_score(w)])
    # P1 的原生分辨率结果
    native_t, native_w = ev("hf-bf16", "test", "native"), ev("hf-bf16", "rwqa", "native")
    # 有结果时作为对照行加在最后
    if native_t:
        acc_rows.append(["原生(HF,P1)", f"{native_t['mean_image_tokens']:.0f}", fmt_score(native_t),
                         fmt_score(native_w)])
    # GPU 基线的计时结果
    b = load("bench/gpu_trtllm-bf16.json")
    # 本节的各个段落
    parts = []
    # 精度表
    if acc_rows:
        parts.append(table(["档位", "TextVQA 平均图像 token", "TextVQA val", "RealWorldQA"], acc_rows))
    # 计时表和说明
    if b:
        # 计时表的数据行
        rows = []
        # 每个档位一行
        for budget, m in b["budgets"].items():
            # 一行:档位、输入长度、首字延迟及其分解、视觉占比、客户端开销、解码速度和跨轮极差
            rows.append([budget, f"{m['input_len_median']:.0f}", f"{m['ttft_vqa_ms']['p50']:.0f}",
                         f"{m['vision_ms']['p50']:.0f}", f"{m['ttft_text_ms']['p50']:.0f}",
                         f"{100 * m['vision_share']:.0f}%", f"{m['client_overhead_vqa_ms']['p50']:.0f}",
                         f"{m['decode_tps_text']['p50']:.1f}", f"{100 * m['ttft_vqa_round_spread']:.1f}%"])
        # 生成计时表
        parts.append(table(["档位", "输入 token", "TTFT ms", "视觉段 ms", "prefill ms", "视觉占比",
                            "预处理+通信 ms", "decode tok/s", "跨轮极差"], rows))
        # 显存读数
        mem = b["gpu_mem_mb"]
        # 计时方法和显存的说明
        parts.append(f"- 计时:{b['images']} 张 dev 图 × {b['rounds']} 轮交错,P50;"
                     f"decode 强制生成 {b['decode_tokens']} 个 token。"
                     f"视觉段 = 同轮配对的(带图 TTFT − 同长度纯文本 TTFT)。\n"
                     f"- 显存(nvidia-smi 整卡口径,含 Windows 桌面占用):加载前 {mem['before_load']} MB → "
                     f"加载后 {mem['after_load']} MB → 测完 {mem['end']} MB。")
    # 各段之间空一行
    return "\n\n".join(parts)


def p3() -> str:
    """P3:GPU 量化各变体的精度(配对差)与计时。"""
    # 要列出的变体,基线在前
    variants = ("trtllm-bf16", "trtllm-fp8-txt", "trtllm-fp8-img", "trtllm-awq-txt", "trtllm-awq-img")
    # 表格的数据行
    rows = []
    # 每个变体一行
    for v in variants:
        # 三个划分的评测结果
        dev, test, rw = ev(v, "dev", "t576"), ev(v, "test", "t576"), ev(v, "rwqa", "t576")
        # dev 和 test 都没有结果时跳过
        if not (dev or test):
            continue
        # 量化记录
        q = load(f"quantize/{v.removeprefix('trtllm-')}.json")
        # t576 档的计时结果
        bench = load(f"bench/gpu_{v}.json").get("budgets", {}).get("t576", {})
        # 权重体积,基线没有量化记录时用固定值
        size = f"{q['weights_bytes'] / 1e9:.2f}" if q else "4.42"
        # 一行:变体、体积、各划分的分数和配对差、答案相同比例、首字延迟、prefill 和解码速度
        rows.append([v.removeprefix("trtllm-"), size, fmt_score(dev), fmt_paired(dev), fmt_score(test),
                     fmt_paired(test), fmt_same(test), fmt_paired(rw),
                     f"{bench['ttft_vqa_ms']['p50']:.0f}" if bench else "—",
                     f"{bench['ttft_text_ms']['p50']:.0f}" if bench else "—",
                     f"{bench['decode_tps_text']['p50']:.1f}" if bench else "—"])
    # 没有数据时返回空串
    if not rows:
        return ""
    # 生成表格并附上说明
    return table(["变体", "权重 GB", "dev", "dev 配对差", "TextVQA val", "val 配对差", "答案与 BF16 相同",
                  "RealWorldQA 配对差", "TTFT ms", "prefill ms", "decode tok/s"], rows) + (
        "\n\n- 全部在 t576 档;配对差 = 同一题上(量化 − BF16)的平均分差 ± 95% CI,单位:分。"
        "\n- 计时同 P2(12 张 dev 图 × 5 轮交错,P50);prefill = 同长度纯文本请求的 TTFT。")


def p4() -> str:
    """P4:CPU 各压缩配方的体积、精度与计时。"""
    # CPU 各变体的计时结果
    b = load("bench/cpu.json").get("variants", {})
    # 要列出的变体,基线在前
    names = ("ov-fp16", "ov-int8", "ov-int4", "ov-int4awq", "ov-int4-vis8w", "ov-int4-vis8s")
    # 精度评测的题数
    limit = None
    for v in names:   # 精度子集的题数从文件名里取(CPU 太慢,只评 test 的前 N 题)
        for f in sorted((RESULTS / "eval").glob(f"{v}_test_t256_n*.json")) if (RESULTS / "eval").exists() else []:
            # 文件名里 _n 后面的数字
            limit = int(f.stem.rsplit("_n", 1)[1])
    # 表格的数据行
    rows = []
    # 每个变体一行
    for v in names:
        # 精度结果
        r = ev(v, "test", "t256", limit) if limit else {}
        # 压缩记录
        c = load(f"compress/{v.removeprefix('ov-')}.json")
        # 计时结果
        m = b.get(v)
        # 精度和计时都没有时跳过
        if not (r or m):
            continue
        # 语言模型体积,基线没有压缩记录时用固定值
        lm_gb = f"{c['bytes']['openvino_language_model'] / 1e9:.2f}" if c else ("3.09" if v == "ov-fp16" else "—")
        # 视觉塔体积,处理方式相同
        vis_gb = f"{c['bytes']['openvino_vision_embeddings_merger_model'] / 1e9:.2f}" if c else (
            "1.33" if v == "ov-fp16" else "—")
        # 一行:配方、两个子模型的体积、分数、配对差、首字延迟及其分解、解码速度和两遍漂移
        rows.append([v.removeprefix("ov-"), lm_gb, vis_gb, fmt_score(r), fmt_paired(r),
                     f"{m['ttft_ms']['p50'] / 1000:.2f}" if m else "—",
                     f"{m['vision_ms']['p50'] / 1000:.2f}" if m else "—",
                     f"{m['prefill_ms']['p50'] / 1000:.2f}" if m else "—",
                     f"{m['decode_tps']['p50']:.1f}" if m else "—",
                     f"{100 * m['ttft_pass_drift']:+.1f}%" if m else "—"])
    # 没有数据时返回空串
    if not rows:
        return ""
    # 同样题数的 GPU 基线结果,用在说明里
    gpu = ev("trtllm-bf16", "test", "t256", limit) if limit else {}
    # 表格下面的说明,没有精度结果时为空
    note = (f"\n\n- 精度:TextVQA val 前 {limit} 题,t256;fp16 行的配对差相对 GPU BF16(同 {limit} 题 {fmt_score(gpu)}),"
            "其余行相对 fp16。\n- 计时:6 张 dev 图,配方按 ABBA 顺序各加载两次,P50;"
            "CPU 4 线程(WSL2 的 4 个 vCPU,不能绑核)。"
            "\"ABBA 漂移\"是两遍中位数之差,衡量这个数字的可信程度。") if limit else ""
    # 生成表格并附上说明
    return table(["配方", "语言模型 GB", "视觉塔 GB", f"TextVQA(n={limit})", "配对差", "TTFT s", "视觉段 s",
                  "prefill s", "decode tok/s", "ABBA 漂移"], rows) + note


def p5() -> str:
    """P5:视觉塔引擎的孤立计时、端到端 TTFT 与配对精度。"""
    # 孤立计时表的数据行
    rows = []
    # 三个方图档位各一行
    for budget in ("s256", "s576", "s1024"):
        # 视觉塔引擎的对拍和计时结果
        r = load(f"vision_trt_{budget}.json")
        # 没有结果时跳过
        if not r:
            continue
        # 两种实现的 GPU 时间
        g = r["gpu_ms"]
        # 对拍结果
        par = r["parity"]
        # 最后一格:键名叫 engine_mb,实际单位是 MiB(按 2**20 字节算的),表头也按 MiB 标
        rows.append([budget, str(r["patches"]), f"{g['hf_bf16_sdpa']['p50']:.1f}", f"{g['trt_fp16']['p50']:.1f}",
                     f"{r['speedup_p50']:.2f}×", f"{par['bf16_vs_fp32']['rel_err_max']:.4f}",
                     f"{par['trt_vs_fp32']['rel_err_max']:.4f}", f"{par['trt_vs_bf16']['cosine_min']:.5f}",
                     str(r["build"]["fused_attention_layers"]), f"{r['build']['engine_mb']:.0f}"])
    # 本节的各个段落
    parts = []
    # 孤立视觉塔的表格和说明
    if rows:
        parts.append("**孤立视觉塔**(GPU 时间,CUDA event,5 轮交错 × 每轮 10 次,P50):\n\n" + table(
            ["档位(方图)", "合并前块数", "HF BF16 ms", "TRT FP16 ms", "加速", "BF16 对 FP32 相对误差",
             "引擎对 FP32 相对误差", "引擎对 BF16 余弦(最小)", "融合注意力层", "引擎 MiB"], rows)
            + "\n\n- 相对误差 = 输出与 FP32 真值之差的范数 / 真值的范数,取 16 张 dev 图中的最大值;"
              "FP32 真值是同一份权重升到 FP32 的视觉塔。")
    # 基线(HF 视觉塔)与引擎两份方图计时结果,两份都有才逐档对比
    base, eng = load("bench/gpu_trtllm-bf16_square.json"), load("bench/gpu_trtllm-bf16-vistrt_square.json")
    if base and eng:
        # 端到端表的数据行
        e2e = []
        # 每个档位一行
        for budget in base["budgets"]:
            # 基线和引擎在这个档位的结果
            a_, b_ = base["budgets"][budget], eng["budgets"].get(budget)
            # 引擎没测这个档位时跳过
            if not b_:
                continue
            # 一行:档位、两边的首字延迟和加速比、两边的视觉段、视觉占比的变化
            e2e.append([budget, f"{a_['ttft_vqa_ms']['p50']:.0f}", f"{b_['ttft_vqa_ms']['p50']:.0f}",
                        f"{a_['ttft_vqa_ms']['p50'] / b_['ttft_vqa_ms']['p50']:.2f}×",
                        f"{a_['vision_ms']['p50']:.0f}", f"{b_['vision_ms']['p50']:.0f}",
                        f"{100 * a_['vision_share']:.0f}% → {100 * b_['vision_share']:.0f}%"])
        # 端到端表格和测量方式的说明
        parts.append("**接回 TensorRT-LLM 的端到端 TTFT**(方图,12 张 dev 图 × 5 轮,P50;"
                     "每个档位、每个配置各起一个进程先后测量 —— 8 GB 显存装不下三个引擎加 HF 模型):"
                     "\n\n" + table(["档位", "TTFT ms(HF 视觉塔)", "TTFT ms(引擎)", "TTFT 加速", "视觉段 ms(HF)",
                                       "视觉段 ms(引擎)", "视觉占比"], e2e))
    # 两种视觉塔在 s576 档的精度结果
    acc_b, acc_e = ev("trtllm-bf16", "test", "s576"), ev("trtllm-bf16-vistrt", "test", "s576")
    # 两边都有时写一段精度对比
    if acc_b and acc_e:
        parts.append(f"**精度**(TextVQA val 全量,方图 s576):HF 视觉塔 {fmt_score(acc_b)},引擎 {fmt_score(acc_e)};"
                     f"逐题配对差 {fmt_paired(acc_e)},答案逐字相同 {fmt_same(acc_e)}。")
    # 各段之间空一行
    return "\n\n".join(parts)


def fmt_median(rows: list[dict], key: str, unit: str) -> str:
    """一组演示记录里某个耗时字段(毫秒)的中位数;unit 为 "s" 时换算成秒。没有这个字段时为“—”。"""
    # 取出有这个字段的值
    values = [q[key] for q in rows if q.get(key) is not None]
    # 没有数据时显示占位符
    if not values:
        return "—"
    # 按单位格式化中位数
    return f"{statistics.median(values) / 1000:.2f} s" if unit == "s" else f"{statistics.median(values):.0f} ms"


# 演示表里档位的排列顺序
DEMO_BUDGET_ORDER = ("t256", "t576", "t1024", "s256", "s576", "s1024")


def demo_sort_key(run: dict) -> tuple:
    """演示表的行序:GPU 在前、CPU 在后;同一条栈内按档位;同一档位内 BF16 / FP16 基线排第一,其余按名字。"""
    # 变体名和档位
    variant, budget = run["variant"], run["budget"]
    # 是否基线变体
    is_baseline = variant in ("trtllm-bf16", "ov-fp16")
    # 档位的序号,没登记的排在最后
    budget_rank = DEMO_BUDGET_ORDER.index(budget) if budget in DEMO_BUDGET_ORDER else len(DEMO_BUDGET_ORDER)
    # 元组逐项比较,False 排在 True 前:所以"是不是 GPU""是不是基线"两项都取反,让 GPU 行、基线行排在前面
    return (not variant.startswith("trtllm-"), budget_rank, not is_baseline, variant)


def demo() -> str:
    """P6:演示记录(每次运行一行:同一批 dev 题的回答与单次观察到的首字延迟)。

    results/demo/ 里有什么文件就列什么行:换过 DEMO_GPU / DEMO_CPU 后,旧变体的记录要手工删掉。
    """
    # 读入全部演示记录
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((RESULTS / "demo").glob("*.json"))]
    # 按演示表的行序排序
    runs.sort(key=demo_sort_key)
    # 没有记录时返回空串
    if not runs:
        return ""
    # 表格的数据行
    rows = []
    # 每次运行一行
    for r in runs:
        unit = "ms" if r["variant"].startswith("trtllm-") else "s"   # GPU 是百毫秒级,CPU 是秒级
        # 有得分的题的得分
        acc = [q["vqa_acc"] for q in r["rows"] if "vqa_acc" in q]
        # engine_ttft_ms 只有 GPU 记录里有、vision_ms 只有 CPU 记录里有,缺的那格显示"—";运行时刻只留"月-日 时:分"
        rows.append([r["variant"], r["budget"], str(len(r["rows"])), fmt_median(r["rows"], "ttft_ms", unit),
                     fmt_median(r["rows"], "engine_ttft_ms", unit), fmt_median(r["rows"], "vision_ms", unit),
                     f"{statistics.fmean(acc):.2f}" if acc else "—",
                     " / ".join(q["answer"] or "(空)" for q in r["rows"]), r["time"][5:16]])
    # 生成表格
    body = table(["变体", "档位", "题数", "首字延迟(中位)", "引擎侧 TTFT(中位)", "视觉段(中位)",
                  "VQA acc 均值", "回答(按题号)", "运行时刻"], rows)
    # 各次运行问的是同一批题时才列题目;不是的话(有人改过脚本里的默认题数)只提示,不拿第一行的题冒充全部
    asked = {tuple(q["question"] for q in r["rows"]) for r in runs}
    if len(asked) == 1:
        # 第一次运行的逐题记录
        first = runs[0]["rows"]
        # 编号的题目列表,附最常见的两个人工答案
        questions = "\n".join(f"{i}. {q['question']}(最常见的人工答案:{'、'.join(q.get('references', [])[:2]) or '—'})"
                              for i, q in enumerate(first, 1))
        # 追加题目列表
        body += f"\n\n题目(dev 集前 {len(first)} 张不同的图,各取第 1 题):\n\n{questions}"
    else:
        # 追加提示
        body += "\n\n各次运行的题目不完全相同,逐题内容见 results/demo/ 下的记录。"
    # 预热次数从记录里读,不在这里写死;各次运行不一致时并列写出
    warmups = sorted({r.get("warmup_requests") for r in runs if r.get("warmup_requests") is not None})
    # 追加两条口径说明后返回
    return (body + "\n\n"
            f"- 单次观察值:丢弃开头 {' / '.join(map(str, warmups)) or '若干'} 次预热请求后每题只问一次,不是基准;"
            "结论性的数字见 P2–P5 的表。\n"
            "- 首字延迟的口径:GPU 是客户端墙钟(含图像预处理与进程间通信),另列引擎侧时间戳(与 P2 / P5 同口径);"
            "CPU 是 OpenVINO GenAI 管线自带的计时(与 P4 同一来源),另列其中的视觉段。")


def cpp() -> str:
    """CPU 栈 C++ 程序(cpp/vlm_infer_ov)与 Python 的对拍和配对计时,来自 16_cpp_parity_cpu.py。"""
    # C++ 对拍和计时的结果
    r = load("cpp_parity_cpu.json")
    # 没有结果时返回空串
    if not r:
        return ""
    # 对拍部分和计时部分
    parity, bench = r["parity"], r["bench"]
    # 表格的数据行
    rows = []
    # Python 和 C++ 各一行
    for impl, label in (("python", "Python(OpenVINO GenAI 的 Python 接口)"), ("cpp", "C++(`cpp/vlm_infer_ov`)")):
        # 这个实现的计时统计
        b = bench[impl]
        # 一行:实现、首字延迟、视觉段、prefill、解码速度、两遍漂移和加载耗时
        rows.append([label, f"{b['ttft_ms']['p50'] / 1000:.2f}", f"{b['vision_ms']['p50'] / 1000:.2f}",
                     f"{b['prefill_ms']['p50'] / 1000:.2f}", f"{b['decode_tps']['p50']:.1f}",
                     f"{100 * b['ttft_pass_drift']:+.1f}%", f"{b['load_seconds']:.1f}"])
    # 生成表格
    out = table(["实现", "TTFT s", "视觉段 s", "prefill s", "decode tok/s", "两遍漂移", "加载 s"], rows)
    # 表格下面的说明:运行配置、对拍结果和计时方法
    notes = [
        f"- 变体 `{r['variant']}`,档位 {r['budget']},{r['threads']} 线程;运行时刻 {r['time'][5:16]}。",
        f"- 对拍(dev 集前 {parity['n']} 题):回答逐字相同 {parity['same_answer']}/{parity['n']},"
        f"输入 token 数相同 {parity['same_input_tokens']}/{parity['n']};"
        f"VQA acc 均值 Python {parity['mean_vqa_acc_python']:.4f} / C++ {parity['mean_vqa_acc_cpp']:.4f}。",
        f"- 计时:{bench['images']} 张图,每个请求强制生成 {bench['decode_tokens']} 个 token,"
        f"顺序 {' → '.join(bench['order'])},每遍预热 1 次;表里是两遍合并后的 P50。"
        f"C++ / Python 的 TTFT 之比 {bench['ttft_ratio_cpp_over_python']:.3f}。",
    ]
    # 表格和说明之间空一行
    return out + "\n\n" + "\n".join(notes)


# 小节名 → 生成函数。新增一节:在这里登记,并在 README 里放一对同名的 BEGIN / END 标记
SECTIONS = {"SMOKE": smoke, "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "DEMO": demo, "CPP": cpp}


def main() -> int:
    """入口:回填 README 中 RESULTS 标记之间的内容;--only 指定只刷新哪些小节。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="用 results/ 回填 README 的结果表")
    # 只刷新指定的小节
    parser.add_argument("--only", nargs="+", choices=sorted(SECTIONS), default=None, metavar="SECTION",
                        help="只刷新这些小节(缺省全部)。results/ 不完整时用它,避免把没有数据的小节刷成“尚未生成”")
    # 解析参数
    args = parser.parse_args()

    # 读入 README 全文
    text = README.read_text(encoding="utf-8")
    # 逐个小节处理
    for name, build in SECTIONS.items():
        # 不在 --only 列表里的小节跳过
        if args.only and name not in args.only:
            continue
        # 标记之间允许为空(首次生成时 BEGIN / END 紧挨着),所以正文连同结尾换行都放进可空的 .*?
        pattern = re.compile(rf"(<!-- RESULTS:{name}:BEGIN -->\n).*?(<!-- RESULTS:{name}:END -->)", re.S)
        # README 里没有这对标记时告警并跳过
        if not pattern.search(text):
            print(f"[warn] README has no RESULTS:{name} markers", file=sys.stderr)
            continue
        # 生成小节内容,没有数据时写占位文字
        body = build() or "_(尚未生成)_"
        # group(1) / group(2) 是 BEGIN / END 两个标记,原样放回,只替换它们之间的内容
        text = pattern.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text)
    # newline="\n":在 Windows 侧跑也写 LF 行尾
    README.write_text(text, encoding="utf-8", newline="\n")
    # 打印已更新的文件
    print(f"updated {README}")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
