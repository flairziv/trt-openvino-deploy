"""P0 CPU 冒烟:optimum-intel 导出 FP16 基础 IR → OpenVINO GenAI VLMPipeline 在 CPU 上答同一批题。

用法(env: ovgenai):
    python scripts/03_smoke_cpu.py --export-only          # 只导出(Makefile 用 /usr/bin/time 包住它记录峰值内存)
    python scripts/03_smoke_cpu.py --num 8 --budget t256  # 用已导出的 IR 答题,并与 HF 参考逐题比对

为什么导出 FP16、而不是直接导 INT4:
    P4 要比较多种压缩配方(INT8 / INT4 / INT4+AWQ、视觉塔 INT8 只压权重 / 静态量化)。从同一份 FP16 基础 IR
    出发各自压缩,配方之间的差异就只来自压缩本身;每个配方都从 PyTorch 重新导出,既慢,
    又可能把导出过程的差异混进来。FP16 只是存储格式:本机 CPU 没有 FP16 / BF16 计算指令,推理时按 FP32 算。

导出的内存风险:WSL 只有 8 GB。optimum 要把 PyTorch 模型整个加载进来再转换,峰值可能超过 8 GB(还有 4 GB swap)。
所以单独做一步并记录峰值;超了就需要临时调大 WSL 内存(要用户同意,因为要重启 WSL)。

产物:
    $VLM_ARTIFACTS/openvino/fp16/      语言模型 / 视觉塔 / 文本 embedding 各自的 IR + tokenizer
    results/smoke_openvino_<budget>.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.backends import CPU_THREADS, MAX_NEW_TOKENS
from vlmdeploy.config import MODEL_DIR, OV_DIR
from vlmdeploy.logging_ import get_logger
from vlmdeploy.smoke import load_smoke_set, print_table, summarize

# 模块级 logger
log = get_logger("smoke_cpu")
# FP16 基础 IR 的目录
FP16_DIR = OV_DIR / "fp16"


def export_fp16() -> float:
    """调用 optimum-cli 导出 FP16 IR(已存在则跳过)。返回耗时秒数。"""
    # IR 已存在时跳过导出
    if (FP16_DIR / "openvino_language_model.xml").exists():
        log.info("FP16 IR already exists: %s", FP16_DIR)
        return 0.0
    # 导出命令:用 optimum-cli 导出 OpenVINO IR,权重为 FP16
    cmd = ["optimum-cli", "export", "openvino", "--model", str(MODEL_DIR), "--task", "image-text-to-text",
           "--weight-format", "fp16", str(FP16_DIR)]
    # 打印导出命令
    log.info("export: %s", " ".join(cmd))
    # 计时起点
    t0 = time.time()
    # 执行导出,失败时抛异常
    subprocess.run(cmd, check=True)
    # 打印耗时和输出目录
    log.info("exported in %.0fs -> %s", time.time() - t0, FP16_DIR)
    return time.time() - t0


def hf_input_lengths(items: list) -> list[int]:
    """HF 处理器为每道题生成的输入 token 数(含展开后的图像 token),作为提示词一致性的标尺。

    OpenVINO GenAI 自己套聊天模板、自己在文字前插图像占位符,我们看不到它最终的提示词。
    但只要它报告的输入 token 数与 HF 逐题相同,就排除了"多一个换行"一类的差异
    (GPU 冒烟里 TensorRT-LLM 的默认输入加载器正是多了一个换行,导致答案与 HF 不同)。
    """
    # transformers 只在这里导入
    from transformers import AutoProcessor

    from vlmdeploy.smoke import chat_prompt

    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    # 各题的输入 token 数
    lengths = []
    # 逐题计算
    for sample, img in items:
        # 套聊天模板后分词并预处理图片,只做预处理不推理
        enc = processor(text=[chat_prompt(processor, sample.prompt)], images=[img], return_tensors="pt")
        # 记录输入序列的长度
        lengths.append(int(enc["input_ids"].shape[1]))
    return lengths


def answer(num: int, budget: str) -> None:
    """OpenVINO GenAI 在 CPU 上答冒烟题,核对输入 token 数并与 HF 参考答案比对。"""
    # numpy 和 openvino 只在这里导入
    import numpy as np
    import openvino as ov
    import openvino_genai as ov_genai

    # 取冒烟题并预缩放图片
    items = load_smoke_set(num, budget)
    # HF 处理器给出的各题输入 token 数
    expected_len = hf_input_lengths(items)
    # 加载计时起点
    t0 = time.time()
    # 创建 VLM 管线,在 CPU 上推理并限定线程数
    pipe = ov_genai.VLMPipeline(str(FP16_DIR), "CPU", INFERENCE_NUM_THREADS=CPU_THREADS)
    # 加载耗时
    load_s = time.time() - t0
    # 生成参数
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS   # 与 GPU 冒烟、评测相同;greedy 是默认(do_sample=False)

    # 各题的答案、首字延迟、视觉段耗时和输入 token 数
    answers, ttft, prep, n_input = [], [], [], []
    # 推理计时起点
    t0 = time.time()
    # 逐题作答
    for sample, img in items:
        # 只传问题文字:GenAI 自己套聊天模板,并按 Qwen2-VL 的规则把图像占位符放在文字前
        tensor = ov.Tensor(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None]))  # [1, H, W, 3] RGB uint8
        # 生成答案
        result = pipe.generate(sample.prompt, images=[tensor], generation_config=config)
        # 记录去掉首尾空白的答案
        answers.append(result.texts[0].strip())
        # 取这次请求的性能指标
        pm = result.perf_metrics
        # 首字延迟(毫秒)
        ttft.append(pm.get_ttft().mean)
        # 管线报告的输入 token 数
        n_input.append(int(pm.get_num_input_tokens()))
        if hasattr(pm, "get_prepare_embeddings_duration"):   # VLM 专有指标:视觉编码 + embedding 拼接
            prep.append(pm.get_prepare_embeddings_duration().mean)
    # 推理耗时
    infer_s = time.time() - t0

    # 输入 token 数与 HF 一致的题数
    same_len = sum(a == b for a, b in zip(n_input, expected_len, strict=True))
    # 附加信息:设备、线程数、IR 目录、两边的输入 token 数、延迟均值和版本号
    extra = {"device": "CPU", "threads": CPU_THREADS, "ir": str(FP16_DIR),
             "input_tokens_genai": n_input, "input_tokens_hf": expected_len,
             "input_tokens_match_hf": f"{same_len}/{len(items)}",
             "ttft_ms_mean": round(sum(ttft) / len(ttft), 1),
             "prepare_embeddings_ms_mean": round(sum(prep) / len(prep), 1) if prep else None,
             "versions": {"openvino": ov.__version__, "openvino_genai": ov_genai.__version__}}
    # 汇总结果并与 HF 参考逐题比对
    result = summarize("openvino", budget, items, answers, {"load": round(load_s, 1), "infer": round(infer_s, 1)},
                       extra)
    # 打印逐题结果
    print_table(result, log.info)
    # 打印输入 token 数的核对结果
    log.info("input tokens match HF: %s (genai %s / hf %s)", extra["input_tokens_match_hf"], n_input, expected_len)
    # 打印加载耗时、推理耗时和延迟均值
    log.info("load %.1fs, %d questions in %.1fs, TTFT mean %.0f ms, vision+embed mean %s ms", load_s, num, infer_s,
             extra["ttft_ms_mean"], extra["prepare_embeddings_ms_mean"])


def main() -> int:
    """入口:--export-only 只导出 FP16 基础 IR,否则跑冒烟答题。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="P0 CPU 冒烟:导出 + OpenVINO GenAI 答题")
    # 只导出不答题
    parser.add_argument("--export-only", action="store_true")
    # 题数
    parser.add_argument("--num", type=int, default=8)
    # 图像 token 档位
    parser.add_argument("--budget", default="t256")
    # 解析参数
    args = parser.parse_args()
    # 不带 --export-only 时也先调一次:IR 已存在就直接跳过,不存在则现导出
    export_fp16()
    # 没有指定 --export-only 时继续答题
    if not args.export_only:
        answer(args.num, args.budget)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
