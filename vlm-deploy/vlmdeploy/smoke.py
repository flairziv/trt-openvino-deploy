"""P0 冒烟测试的公共部分:取同一批题、预缩放、与 HF 参考答案逐题比对、写结果。

冒烟测试回答的问题只有一个:"这条推理栈装好了、输出是对的吗?"
判据不是绝对准确率(8 道题统计意义很弱),而是与 HF transformers BF16 参考逐题一致:
    - TensorRT-LLM 与 HF 同为 BF16 GPU,只有 kernel 实现不同,应几乎逐字相同;
    - OpenVINO 在 CPU 上以 FP32 计算(本机没有 BF16 指令),数值路径不同,允许少量答案不同。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from vlmdeploy.config import RESULTS_DIR, TOKEN_BUDGETS
from vlmdeploy.imaging import image_tokens, resize_to_budget
from vlmdeploy.tasks import VqaSample, iter_samples, normalize_answer, score


def load_smoke_set(num: int, budget: str) -> list[tuple[VqaSample, object]]:
    """取 TextVQA 验证集前 num 题,图片按 token 预算预缩放。

    Returns:
        [(样本, 预缩放后的 PIL.Image)],顺序固定。
    """
    # 档位名对应的图像 token 上限
    tokens = TOKEN_BUDGETS[budget]
    # 取前 num 题,图片预缩放到预算内
    return [(s, resize_to_budget(s.image(), tokens)) for s in iter_samples("textvqa", "validation", limit=num)]


# 虽然放在冒烟模块里,这是全项目共用的提示词函数:GPU 两个后端、量化校准、计时、演示都调它。
def chat_prompt(processor, question: str) -> str:  # noqa: ANN001 —— transformers 的 AutoProcessor
    r"""用模型自带的聊天模板生成提示词:单张图在前、文字在后,末尾带 assistant 起始标记。

    GPU 两个后端都用这一个函数生成提示词,保证送进模型的 token 逐个相同。
    不用 TensorRT-LLM 的 default_multimodal_input_loader:它在图片占位符和文字之间插了一个换行
    (qwen2_vl 的占位符分隔符是 "\n"),而 Qwen2-VL 官方模板里两者紧挨着 —— 多一个 token,
    冒烟测试里就出现了与 HF 不同的答案。
    """
    # 一条 user 消息:先是图片占位,后是问题文字
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    # 套聊天模板,返回字符串并在末尾加上 assistant 起始标记
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def summarize(backend: str, budget: str, items: list[tuple[VqaSample, object]], answers: list[str],
              seconds: dict[str, float], extra: dict | None = None) -> dict:
    """汇总一个后端的冒烟结果;若已有 HF 参考结果,逐题比对。

    Args:
        backend: "hf" / "trtllm" / "openvino"。
        budget: token 档位名。
        items: load_smoke_set 的返回值。
        answers: 与 items 一一对应的模型输出。
        seconds: 各阶段耗时(加载、推理等),只作记录,不作性能结论。
        extra: 其他要落盘的信息(显存、版本等)。

    Returns:
        结果字典(同时写入 results/smoke_<backend>_<budget>.json)。
    """
    # 逐题的结果行
    rows = []
    # 题目与答案一一对应,长度不一致时报错
    for (sample, img), ans in zip(items, answers, strict=True):
        # 记录题号、问题、答案、图像 token 数和得分
        rows.append({"qid": sample.qid, "question": sample.question, "answer": ans,
                     "image_tokens": image_tokens(img.height, img.width), "vqa_acc": round(score(sample, ans), 4)})
    # 汇总:后端、档位、题数、时间、平均分、耗时、附加信息和逐题结果
    result = {"backend": backend, "budget": budget, "num": len(rows), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "mean_vqa_acc": round(sum(r["vqa_acc"] for r in rows) / len(rows), 4), "seconds": seconds,
              **(extra or {}), "rows": rows}
    # HF 参考结果的文件
    ref_path = RESULTS_DIR / f"smoke_hf_{budget}.json"
    # 当前后端不是 hf 且参考结果存在时逐题比对
    if backend != "hf" and ref_path.exists():
        # 题号到 HF 答案的映射
        ref = {r["qid"]: r["answer"] for r in json.loads(ref_path.read_text(encoding="utf-8"))["rows"]}
        # 两种口径:raw 是逐字相同(只去首尾空白),norm 是 EvalAI 归一化后相同(忽略大小写、标点)
        same_raw = [r["answer"].strip() == ref.get(r["qid"], "").strip() for r in rows]
        same_norm = [normalize_answer(r["answer"]) == normalize_answer(ref.get(r["qid"], "")) for r in rows]
        # 把 HF 的答案和是否逐字相同写回每一行
        for r, raw in zip(rows, same_raw, strict=True):
            r["hf_answer"], r["same_as_hf"] = ref.get(r["qid"]), raw
        # 记录两种口径下一致的题数
        result["agree_with_hf_raw"] = f"{sum(same_raw)}/{len(rows)}"
        result["agree_with_hf_normalized"] = f"{sum(same_norm)}/{len(rows)}"
    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 结果文件路径
    out = RESULTS_DIR / f"smoke_{backend}_{budget}.json"
    # 写出结果
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def print_table(result: dict, log: Callable[[str], None] = print) -> None:
    """把逐题结果打成一张易读的表。"""
    # 第一行打印汇总,做过比对时附上与 HF 一致的题数
    log(f"[{result['backend']}] budget={result['budget']} mean VQA acc={result['mean_vqa_acc']:.3f} "
        + (f"agree with HF: raw {result['agree_with_hf_raw']}, normalized {result['agree_with_hf_normalized']}"
           if "agree_with_hf_raw" in result else ""))
    # 逐题打印一行
    for r in result["rows"]:
        # 行尾标记:与 HF 一致、不一致时附上 HF 的答案、没有比对时为空
        mark = "" if "same_as_hf" not in r else ("  ==HF" if r["same_as_hf"] else f"  != HF '{r['hf_answer']}'")
        # 打印题号、图像 token 数、得分、问题和答案
        log(f"  {r['qid']:>6} tok={r['image_tokens']:4d} acc={r['vqa_acc']:.2f}  {r['question'][:48]:48s} -> "
            f"{r['answer'][:40]!r}{mark}")
