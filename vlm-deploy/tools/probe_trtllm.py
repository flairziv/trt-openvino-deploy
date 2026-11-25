"""TensorRT-LLM 上 Qwen2-VL 的分段排障探针:语言模型 / 视觉塔 / 拼接,哪一段出了问题?

用法(env: trtllm,需带 PYTHONPATH=vlmdeploy/patches/site):
    python tools/probe_trtllm.py --mode text      # 纯文字题:只走语言模型
    python tools/probe_trtllm.py --mode image     # 带图题:视觉塔 + 拼接 + 语言模型
    python tools/probe_trtllm.py --mode logprobs  # 带图题,打印首个生成 token 的 logprob(NaN 会在这里现形)

背景:P0 冒烟里 TensorRT-LLM 对 8 道题全部输出 "!!!!!!!!"。"!" 是 Qwen 词表的 0 号 token,
logits 全为 NaN 时 argmax 就落在 0 号 —— 这是"计算出了 NaN",不是"模型答错"。
排障的第一步是把三段拆开:纯文字题正常 → 语言模型没问题,问题在视觉段或拼接;纯文字也坏 → 先查语言模型。

为了便于在异常栈里看到完整信息,探针设置 TLLM_WORKER_USE_SINGLE_PROCESS=1,让模型跑在当前进程里。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 必须赶在导入 tensorrt_llm 之前设置才生效,所以放在文件顶部而不是 main() 里
os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import MODEL_DIR
from vlmdeploy.smoke import load_smoke_set

# 两道答案确定的纯文字题(首都、加法):答对答错一眼可见
TEXT_PROMPTS = ["What is the capital of France? Answer with one word.", "1 + 2 = ? Answer with a number."]


def main() -> int:
    """命令行入口:按 --mode 对 TensorRT-LLM 的 Qwen2-VL 做分段排障。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="TensorRT-LLM Qwen2-VL 分段排障")
    # 探测模式,必填
    parser.add_argument("--mode", choices=("text", "image", "logprobs"), required=True)
    # 带图题的题数
    parser.add_argument("--num", type=int, default=2)
    # 二分开关:每次只动一个,看 NaN 跟着哪个开关走
    parser.add_argument("--attn", default="TRTLLM", choices=("TRTLLM", "FLASHINFER", "VANILLA"),
                        help="语言模型注意力后端(默认 TRTLLM 自研 kernel)")
    # 是否关闭 CUDA graph
    parser.add_argument("--no-cuda-graph", action="store_true", help="关闭 decode 阶段的 CUDA graph")
    # 是否关闭 overlap scheduler
    parser.add_argument("--no-overlap", action="store_true", help="关闭 overlap scheduler")
    # 模型精度
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    # 解析参数
    args = parser.parse_args()

    # tensorrt_llm 只在这里导入
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.inputs import default_multimodal_input_loader
    from tensorrt_llm.llmapi import KvCacheConfig

    extra = {"cuda_graph_config": None} if args.no_cuda_graph else {}   # 不传 = 默认开启(default_factory)
    # 按开关加载模型
    llm = LLM(model=str(MODEL_DIR), kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25),
              max_batch_size=4, max_num_tokens=4096, max_seq_len=2048, attn_backend=args.attn, dtype=args.dtype,
              disable_overlap_scheduler=args.no_overlap, **extra)
    # 采样参数:最多生成 8 个 token,贪心,logprobs 模式下返回首选 token 的 logprob
    params = SamplingParams(max_tokens=8, temperature=0.0, logprobs=1 if args.mode == "logprobs" else None)
    # 纯文字题:套聊天模板
    if args.mode == "text":
        messages = [[{"role": "user", "content": p}] for p in TEXT_PROMPTS]
        inputs = [llm.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    else:
        # 带图题:取冒烟题,用 TensorRT-LLM 自带的加载器生成输入
        items = load_smoke_set(args.num, "t256")
        inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer, model_dir=str(MODEL_DIR),
                                                 model_type="qwen2_vl", modality="image",
                                                 prompts=[s.prompt for s, _ in items],
                                                 media=[[img] for _, img in items], image_data_format="pil")
    # 逐题生成并打印
    for out in llm.generate(inputs, sampling_params=params):
        # 第一个候选输出
        comp = out.outputs[0]
        # token id 比文字更直接:"全是 0 号 token"一眼可见
        print(f"[probe] tokens={list(comp.token_ids)} text={comp.text!r}", flush=True)
        # logprobs 模式另打印首个 token 的 logprob
        if args.mode == "logprobs":
            print(f"[probe] logprobs[0]={comp.logprobs[0] if comp.logprobs else None}", flush=True)
    # 关闭模型
    llm.shutdown()
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
