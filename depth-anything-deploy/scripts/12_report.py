"""把 results/ 下的机器可读结果汇总成 README 表格。

用法(任一环境,只依赖标准库 + numpy 无关):python scripts/12_report.py

README 中 `<!-- RESULTS:xxx:BEGIN -->` 与 `<!-- RESULTS:xxx:END -->` 之间的内容由本脚本覆盖,
数字**只**从 results/*.json 来,不允许手工改 —— 避免文档与实测脱节。

口径:
    - "Δ AbsRel" 是相对同分辨率 PyTorch FP32 的配对差,括号内 95% bootstrap CI;
    - CI 不含 0(且边界在四位小数下非零)时标 *,表示在 654 张上可分辨;
    - 延迟只来自 10_benchmark.py 的交错轮次,sensitivity 里的快速延迟不进表;
    - 轮间极差 > 10% 的延迟标 ~,读数时应视为不稳定。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# 仓库根目录
ROOT = Path(__file__).resolve().parent.parent
# 结果目录
RESULTS = ROOT / "results"
# 要回填的 README
README = ROOT / "README.md"

# 分辨率的显示顺序
RES_ORDER = ("r518", "r364", "r266")
# 配方的显示顺序
PRECISION_ORDER = ("fp32", "tf32", "fp16", "fp8", "int8", "int8res", "int8mix",
                   "int8sqoff", "int8sq050", "int8sq075", "int8ac")
DELTA_DISPLAY_EPS = 5e-5   # AbsRel 表保留四位小数,CI 边界低于半个末位即显示为 0.0000
# 分辨率档位在表格里的显示名
RES_LABEL = {"r518": "518×686 (1813 tok)", "r364": "364×476 (884 tok)", "r266": "266×350 (475 tok)"}


# 读取 results 下的一个 JSON,文件不存在时返回空字典
def load(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# 读取全部测试集评测结果,键是变体字符串
def load_evals() -> dict[str, dict]:
    # 变体到评测结果的映射
    out = {}
    # 按文件名顺序遍历测试集结果文件
    for path in sorted((RESULTS / "eval").glob("*__test.json")):
        # 读取一个结果
        data = json.loads(path.read_text(encoding="utf-8"))
        # 去掉逐图明细,只留汇总
        data.pop("per_image", None)
        # 按变体字符串登记
        out[data["variant"]] = data
    return out


# 拼一张 Markdown 表格
def table(headers: list[str], rows: list[list[str]]) -> str:
    # 表头行和分隔行
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    # 数据行
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    # 用换行连接
    return "\n".join(lines)


# 格式化相对 FP32 的 AbsRel 配对差及其置信区间
def fmt_delta(entry: dict) -> str:
    # 取配对差,基准行和没有基准的行没有这一项
    d = entry.get("delta_abs_rel")
    # 没有配对差时:PyTorch 行显示基准,其余显示短横
    if not d:
        return "基准" if entry["variant"].startswith("torch") else "—"
    # 置信区间不含 0 时标星号
    # 四位小数下仍为 0 的差值(如 TRT FP32 相对 PyTorch FP32 的 1e-6 级舍入)只是数值噪声,不标显著。
    star = "*" if d["ci95_low"] >= DELTA_DISPLAY_EPS or d["ci95_high"] <= -DELTA_DISPLAY_EPS else ""
    # 均值差后面跟置信区间
    return f"{d['mean']:+.4f}{star} [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]"


# 格式化某一分段的 P50 耗时,不稳定的结果加标记
def fmt_ms(bench: dict, seg: str) -> str:
    # 没有基准数据或没有这一分段时显示短横
    if not bench or seg not in bench:
        return "—"
    # 保留两位小数
    return ("~" if bench.get("unstable") else "") + f"{bench[seg]['p50']:.2f}"


# 排序键:先按分辨率,再按配方,最后按后端
def sort_key(key: str) -> tuple:
    # 拆出后端、分辨率、配方
    backend, res, prec = key.split(":")
    # 没登记的配方排到最后
    return (RES_ORDER.index(res), PRECISION_ORDER.index(prec) if prec in PRECISION_ORDER else 99, backend)


# 生成一个后端的结果矩阵表
def matrix(backend: str) -> str:
    # 全部测试集评测结果
    evals = load_evals()
    # 该后端的延迟基准
    bench = load(f"bench_{backend}.json")
    # 引擎构建记录
    engines = load("engines.json")
    # OpenVINO 配方记录
    ov = load("openvino.json")
    # 该后端的变体,按显示顺序排序
    keys = sorted((k for k in evals if k.startswith(f"{backend}:")), key=sort_key)
    # GPU 表里加上 PyTorch 基准行
    if backend == "trt":
        keys = sorted(keys + [k for k in evals if k.startswith("torch:")], key=sort_key)
    # 表格的数据行
    rows = []
    # 逐个变体生成一行
    for key in keys:
        # 评测结果
        e = evals[key]
        # 延迟基准,可能没有
        b = bench.get(key, {})
        # 拆出分辨率和配方
        _, res, prec = key.split(":")
        # 产物大小,默认显示短横
        size = "—"
        # TensorRT 变体取引擎大小
        if key in engines:
            size = f"{engines[key]['engine_mb']:.0f}"
        # OpenVINO 变体取 IR 大小
        elif key.startswith("ov:") and f"{res}.{prec}" in ov:   # PyTorch 基准行没有部署产物,不能借用同名 IR 的大小
            size = f"{ov[f'{res}.{prec}']['ir_mb']:.0f}"
        # PyTorch 行和没有偏差数据的行不显示 SQNR
        sqnr = "—" if e.get("drift_sqnr_db") in (None, float("inf")) or key.startswith("torch") else f"{e['drift_sqnr_db']:.1f}"
        # 有延迟基准时显示帧率
        fps = f"{b['fps_total_p50']:.1f}" if b else "—"
        # 一行:分辨率、配方、精度、配对差、偏差、延迟、帧率、产物大小
        rows.append([RES_LABEL[res], ("PyTorch " if key.startswith("torch") else "") + prec, f"{e['abs_rel']:.4f}",
                     fmt_delta(e), f"{e['delta1']:.4f}", sqnr, fmt_ms(b, "compute"), fmt_ms(b, "total"), fps, size])
    # 加上表头后返回
    return table(["分辨率", "配方", "AbsRel", "Δ AbsRel vs FP32 [95% CI]", "δ1", "SQNR dB", "推理 P50 ms",
                  "端到端 P50 ms", "FPS", "产物 MB"], rows)


# 生成各引擎的 kernel 精度分布表
def kernels() -> str:
    # 引擎构建记录
    engines = load("engines.json")
    # 数据行
    rows = []
    # 跳过可复现性实验的条目,其余按显示顺序排序
    for key in sorted((k for k in engines if "#" not in k), key=sort_key):
        # 取该引擎的记录
        e = engines[key]
        # 精度分布,按层数从多到少
        prec = ", ".join(f"{k} {v}" for k, v in sorted(e["precision"].items(), key=lambda x: -x[1]))
        # 一行:引擎、精度分布、融合注意力数、kernel 数、构建耗时
        rows.append([key, prec, str(e["fused_mha"]), str(e["num_layers"]), f"{e['build_seconds']:.0f}"])
    return table(["引擎", "计算层实际 kernel 精度", "融合 MHA", "kernel 数", "构建 s"], rows)


# 生成两个主档位的敏感度实验表
def sensitivity() -> str:
    # 各档位的表格片段
    parts = []
    # 逐个主档位
    for res in ("r518", "r364"):
        # 读取该档位的实验报告
        d = load(f"sensitivity_{res}.json")
        # 没有报告时跳过
        if not d:
            continue
        # 前两行是基线和上限
        rows = [["全部 INT8", f"{d['full_int8']['abs_rel']:.4f}", f"{d['full_int8']['sqnr_db']:.1f}"],
                ["全部 FP16(拆掉全部 Q/DQ)", f"{d['all_fp16']['abs_rel']:.4f}", f"{d['all_fp16']['sqnr_db']:.1f}"]]
        # 每个子层类型的留一法结果
        rows += [[f"仅 {r['type']} 回退 FP16", f"{r['abs_rel']:.4f}", f"{r['sqnr_db']:.1f}"] for r in d["type_loo"]]
        # 最后一行是最终选择
        rows.append([f"最终 int8mix({len(d['selected_groups'])} 组 FP16)", f"{d['final']['abs_rel']:.4f}",
                     f"{d['final']['sqnr_db']:.1f}"])
        # 表前写明 FP32 的 val 精度和接受线
        parts.append(f"**{RES_LABEL[res]}**,FP32 val AbsRel = {d['fp32_val_abs_rel']:.4f},"
                     f"接受线 ≤ {d['target_abs_rel']:.4f}:\n\n" + table(["配置(val 100 张,真实引擎)", "AbsRel", "SQNR dB"], rows))
    # 各档位之间空一行
    return "\n\n".join(parts)


# 生成激活统计表
def activation() -> str:
    # 读取 r518 档的激活统计
    d = load("activation_stats_r518.json")
    # 没有统计时返回空
    if not d:
        return ""
    # 张量种类在表格里的显示名
    label = {"qkv_in": "q/k/v 输入(norm1 输出)", "o_proj_in": "o_proj 输入", "fc1_in": "fc1 输入(norm2 输出)",
             "fc2_in": "fc2 输入(GELU 输出)", "resid_attn": "残差流(注意力相加后)", "resid_out": "残差流(block 输出)"}
    # 每类张量一行:离群比的中位数和最大值、两种 SQNR
    rows = [[label[k], f"{v['channel_outlier_median']:.1f}", f"{v['channel_outlier_max']:.1f}",
             f"{v['sqnr_max_db_mean']:.1f}", f"{v['sqnr_clip_db_mean']:.1f}"] for k, v in d["by_kind"].items()]
    # 加上表头后返回
    return table(["量化点", "通道离群比 中位", "通道离群比 最大", "单张量 INT8 SQNR(max 校准)", "SQNR(99.99% 截断)"], rows)


# 生成各引擎的耗时占比表
def profile() -> str:
    # 读取剖析结果
    d = load("profile_trt.json")
    # 类别的显示顺序
    cats = ("attention_mha", "attention_gemm", "linear_gemm", "kgen", "conv", "reformat", "other")
    # 每个引擎一行,各类别显示占比
    rows = [[k] + [f"{v['by_category_share'].get(c, 0):.1%}" for c in cats] for k, v in d.items()]
    # 没有数据时返回空
    return table(["引擎"] + list(cats), rows) if rows else ""


# 生成 fc2 异常的对照表
def anomaly() -> str:
    # 读取探针结果
    d = load("fc2_anomaly.json")
    # 每个配置一行:ORT 仿真与真实引擎的 SQNR
    rows = [[k, f"{v['ort_cuda_sqnr_db']:.1f}", f"{v['trt_sqnr_db']:.1f}"] for k, v in d.items()]
    # 没有数据时返回空
    return table(["拆掉的 Q/DQ", "ORT CUDA SQNR dB", "TensorRT SQNR dB"], rows) if rows else ""


# 生成 C++ 运行时的对拍与延迟表
def cpp() -> str:
    # GPU 栈(da_infer)和 CPU 栈(da_infer_ov)的结果在同一对文件里,按变体名作键,所以合成一张表。
    bench, parity = load("cpp_bench.json"), load("cpp_parity.json")
    # 数据行
    rows = []
    # 每个变体一行
    for key, b in bench.items():
        # 该变体的对拍汇总
        s = parity[key]["summary"]
        # 分段耗时一律走 fmt_ms:OpenVINO 版在 CPU 上推理,没有 H2D / D2H 两段,缺的格子显示成"—"
        rows.append([key, fmt_ms(b, "preprocess"), fmt_ms(b, "h2d"), fmt_ms(b, "compute"), fmt_ms(b, "d2h"),
                     fmt_ms(b, "postprocess"), fmt_ms(b, "total"),
                     f"{1000 / b['total']['p50']:.1f}", f"{s['infer_same_input_max_abs']:.1e}",
                     f"{s['preprocess_max_gray_levels']:.2f}", f"{s['abs_rel_mean_diff']:+.1e}"])
    # 没有数据时返回空
    return table(["变体", "预处理", "H2D", "推理", "D2H", "后处理", "总计 ms", "FPS", "同输入输出最大差",
                  "预处理最大差(灰度级)", "AbsRel 差"], rows) if rows else ""


# README 小节名到生成函数的映射
SECTIONS = {"GPU": lambda: matrix("trt"), "CPU": lambda: matrix("ov"), "KERNELS": kernels,
            "SENSITIVITY": sensitivity, "ACTIVATION": activation, "PROFILE": profile,
            "ANOMALY": anomaly, "CPP": cpp}


# 入口:逐节生成表格,替换 README 里对应标记之间的内容
def main() -> int:
    # 读入 README
    text = README.read_text(encoding="utf-8")
    # 逐节处理
    for name, build in SECTIONS.items():
        # 标记之间允许为空(首次生成时 BEGIN / END 紧挨着),所以正文连同结尾换行都放进可空的 .*?
        pattern = re.compile(rf"(<!-- RESULTS:{name}:BEGIN -->\n).*?(<!-- RESULTS:{name}:END -->)", re.S)
        # README 里没有这一节的标记时告警并跳过
        if not pattern.search(text):
            print(f"[warn] README has no RESULTS:{name} markers", file=sys.stderr)
            continue
        # 生成内容,为空时写占位文字
        body = build() or "_(尚未生成)_"
        # 替换两个标记之间的内容,标记本身保留
        text = pattern.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text)
    # 写回 README,行尾固定为 LF
    README.write_text(text, encoding="utf-8", newline="\n")
    # 打印结果
    print(f"updated {README}")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
