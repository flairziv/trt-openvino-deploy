"""P0 GPU 冒烟:HF transformers BF16 参考 / TensorRT-LLM BF16 答同一批题,逐题比对。

用法(env: trtllm;必须带 PYTHONPATH=vlmdeploy/patches/site,Makefile 已设置):
    python scripts/02_smoke_gpu.py --backend hf     --num 8 --budget t256   # 先跑,产出参考答案
    python scripts/02_smoke_gpu.py --backend trtllm --num 8 --budget t256   # 再跑,与参考逐题比对

两者的输入完全相同:同样的题、同样预缩放到 token 预算内的图(处理器里不再发生缩放)、同样的聊天模板
(图在文字前)、同样的贪心解码与 16 个新 token 上限。所以两者差异只来自推理实现 —— TensorRT-LLM 与 HF
同为 BF16,预期几乎逐字相同。

本脚本的耗时只作记录(含加载、首次调用的预热与 CUDA graph 捕获),不作性能结论;计时在 P2 用交错轮次做。

产物:results/smoke_<backend>_<budget>.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.backends import MAX_NEW_TOKENS
from vlmdeploy.config import MODEL_DIR
from vlmdeploy.logging_ import get_logger
from vlmdeploy.smoke import chat_prompt, load_smoke_set, print_table, summarize
from vlmdeploy.sysinfo import gpu_memory_used_mb

# 模块级 logger
log = get_logger("smoke_gpu")


def run_hf(items: list) -> tuple[list[str], dict, dict]:
    """HF transformers BF16 逐题作答;返回 (答案列表, 各阶段耗时, 附加信息)。"""
    # torch 和 transformers 只在这里导入
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    # 加载计时起点
    t0 = time.time()
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    # BF16 是 Qwen2-VL 的官方 dtype;SDPA 而非 flash-attn 2(本环境没有),eager 会显式构造 N×N 注意力矩阵
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to("cuda").eval()
    # 加载耗时
    load_s = time.time() - t0

    # 各题的答案和提示词
    answers, prompts = [], []
    # 推理计时起点
    t0 = time.time()
    # 逐题作答
    for sample, img in items:
        # 套聊天模板
        text = chat_prompt(processor, sample.prompt)
        # 分词并预处理图片,张量移到 GPU
        inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
        # 关闭梯度记录,生成答案
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)   # do_sample=False 即贪心
        # generate 返回的是"输入 + 新生成"的完整序列,切掉输入那一段,只解码新生成的 token
        answers.append(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                       .strip())
        # 记下这道题的提示词
        prompts.append(text)
    # 推理耗时
    infer_s = time.time() - t0
    # gpu_mem_used_mb 是整卡读数(nvidia-smi,含其他进程);torch_max_allocated_mb 是本进程 PyTorch 分配器的峰值。
    # 第一题的提示词原文也落盘,供两个后端的结果互相核对
    extra = {"gpu_mem_used_mb": gpu_memory_used_mb(),
             "torch_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
             "first_prompt": prompts[0]}
    # 返回答案、加载和推理耗时、附加信息
    return answers, {"load": round(load_s, 1), "infer": round(infer_s, 1)}, extra


def run_trtllm(items: list) -> tuple[list[str], dict, dict]:
    """TensorRT-LLM BF16 作答;返回值同 run_hf。"""
    # tensorrt_llm 只在这里导入
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    from transformers import AutoProcessor

    # 加载计时起点
    t0 = time.time()
    # 显存:TRT-LLM 默认把剩余显存的 90% 划给 KV cache 池。本模型 KV 每 token 只有 28 KB,
    # 8 道题用不了几 MB,只给 25%,给视觉塔的激活和 CUDA graph 留出余量(整卡 8 GB,Windows 桌面还占约 1 GB)。
    llm = LLM(model=str(MODEL_DIR), kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25),
              max_batch_size=8, max_num_tokens=4096, max_seq_len=2048)
    # 加载耗时
    load_s = time.time() - t0

    # 直接给"提示词 + 图片",由 TRT-LLM 的 Qwen2-VL 输入处理器(内部同样调用 HF 处理器)做分词与图像预处理;
    # 提示词与 HF 参考出自同一个 chat_prompt,输入 token 逐个相同。
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    # 每题一个请求:提示词加一张图
    inputs = [{"prompt": chat_prompt(processor, s.prompt), "multi_modal_data": {"image": [img]}} for s, img in items]
    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)   # temperature=0 即贪心
    # 推理计时起点
    t0 = time.time()
    outputs = llm.generate(inputs, sampling_params=params)   # 全部题一次提交(HF 那边是逐题)
    # 推理耗时
    infer_s = time.time() - t0
    # 取每个请求的第一个候选答案并去掉首尾空白
    answers = [o.outputs[0].text.strip() for o in outputs]
    # 附加信息:整卡显存和第一题的提示词
    extra = {"gpu_mem_used_mb": gpu_memory_used_mb(), "first_prompt": inputs[0]["prompt"]}
    llm.shutdown()   # 结束 worker 进程、释放显存
    # 返回答案、加载和推理耗时、附加信息
    return answers, {"load": round(load_s, 1), "infer_batched": round(infer_s, 1)}, extra


def main() -> int:
    """入口:跑 --backend 指定的后端,汇总并与 HF 参考逐题比对。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="P0 GPU 冒烟")
    # 后端,必填
    parser.add_argument("--backend", choices=("hf", "trtllm"), required=True)
    # 题数
    parser.add_argument("--num", type=int, default=8)
    # 图像 token 档位
    parser.add_argument("--budget", default="t256")
    # 解析参数
    args = parser.parse_args()

    # 取冒烟题并预缩放图片
    items = load_smoke_set(args.num, args.budget)
    # 按后端选择作答函数并执行
    answers, seconds, extra = (run_hf if args.backend == "hf" else run_trtllm)(items)
    # 汇总结果,存在 HF 参考时逐题比对
    result = summarize(args.backend, args.budget, items, answers, seconds, extra)
    # 打印逐题结果
    print_table(result, log.info)
    # 打印耗时和显存
    log.info("seconds %s, GPU memory used %s MB", seconds, extra.get("gpu_mem_used_mb"))
    return 0


# spawn 出来的 worker 会重新导入本脚本:没有这道保护,导入时就会再跑一遍 main()、再起 worker
if __name__ == "__main__":   # TRT-LLM 单卡会用 spawn 另起 worker 进程,入口必须有这道保护
    sys.exit(main())
