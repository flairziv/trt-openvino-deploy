"""三个推理后端的统一答题接口:HF transformers(参考)/ TensorRT-LLM(GPU 栈)/ OpenVINO GenAI(CPU 栈)。

评测脚本只认 `Answerer.answer(batch) -> list[str]`,所以"换后端 / 换精度"不会改变取题、提示词、打分中的任何一步,
各后端分数之差只来自推理实现本身。这是两栈对照能成立的前提。

三个后端的提示词来源:
    - HF 与 TensorRT-LLM:同一个 `chat_prompt`(模型自带聊天模板,图在文字前),输入 token 逐个相同(P0 已核对);
    - OpenVINO GenAI:只接受问题文字,自己套模板;P0 用"输入 token 数与 HF 逐题相同"核对过模板一致。
图片一律由调用方预缩放好(见 imaging.py);后端处理器里的 smart_resize 因此是恒等变换,不再插值。

重依赖(torch / tensorrt_llm / openvino_genai)只在各自构造函数里导入,本模块在任一环境都能 import。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vlmdeploy.config import MODEL_DIR
from vlmdeploy.logging_ import get_logger
from vlmdeploy.smoke import chat_prompt

# 模块级 logger
log = get_logger("backends")

# 下面两个是协议常量,不只本模块用:冒烟脚本导入前者,冒烟、压缩、CPU 计时脚本导入后者。

# 两套评测数据的参考答案都是短语或单个选项字母;与 lmms-eval 的设置一致。
MAX_NEW_TOKENS = 16
# OpenVINO 线程数 = WSL 的 4 个 vCPU(按"端侧 4 核预算"呈现,与项目 A 相同)。
CPU_THREADS = 4


@dataclass
class Answerer:
    """后端基类。子类实现 `_answer` 与可选的 `close`。

    Attributes:
        name: 结果文件里的变体名,如 "hf-bf16" / "trtllm-fp8-img" / "ov-int4"。
        load_seconds: 加载耗时(只作记录,不进入任何性能结论)。
    """

    name: str
    load_seconds: float = 0.0
    info: dict = field(default_factory=dict)  # 各后端自己填的附加信息(模型位置、线程数等),评测时随结果落盘

    def answer(self, batch: Sequence[tuple]) -> list[str]:
        """回答一批 (VqaSample, PIL.Image);返回与输入一一对应的答案(已 strip)。"""
        # strip 放在基类里做,三个后端的行为保证一致
        return [a.strip() for a in self._answer(batch)]

    # 由子类实现:返回与输入一一对应的原始答案
    def _answer(self, batch: Sequence[tuple]) -> list[str]:
        raise NotImplementedError

    # 默认什么都不做:三个后端里只有 TensorRT-LLM 需要关 worker 进程
    def close(self) -> None:
        """释放显存 / 子进程。"""


class HfAnswerer(Answerer):
    """HF transformers BF16,逐题 batch=1。

    不做批量:HF 的 Qwen2-VL 批量生成需要左填充,还要把不同尺寸的图拼进一个 batch,
    会引入"填充是否影响答案"这个新变量。参考后端要的是干净,不是吞吐。
    """

    # 构造函数:加载处理器和模型到 GPU
    def __init__(self, name: str = "hf-bf16") -> None:
        super().__init__(name)
        # torch 和 transformers 只在这里导入
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        # 加载计时起点
        t0 = time.time()
        # 保存 torch 模块供其他方法使用
        self._torch = torch
        # 加载处理器:分词器、图像预处理和聊天模板
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        # SDPA 而非 flash-attn 2(环境里没有);eager 会显式构造 N×N 注意力矩阵,原生分辨率下显存吃不消
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        # 记录加载耗时
        self.load_seconds = time.time() - t0
        # 已回答的题数,用于定期归还显存缓存
        self._count = 0

    # 逐题作答
    def _answer(self, batch: Sequence[tuple]) -> list[str]:
        # 各题的答案
        out = []
        # 逐题处理
        for sample, img in batch:
            # 套聊天模板后分词并预处理图片,张量移到 GPU
            inputs = self.processor(text=[chat_prompt(self.processor, sample.prompt)], images=[img],
                                    return_tensors="pt").to("cuda")
            # 关闭梯度记录,贪心解码生成答案
            with self._torch.inference_mode():
                ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            # generate 返回的是"输入 + 新生成"的完整序列,切掉输入部分再解码
            out.append(self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                                                   skip_special_tokens=True)[0])
            # 删除张量引用,显存才能回到缓存池
            del inputs, ids
            # 原生分辨率下每张图的序列长度都不同(几百到两千多 token),PyTorch 缓存分配器为每种尺寸留一块,
            # 缓存只涨不跌。P1 实测:跑到约 700 题时整卡显存 7.8 GB 占满,WDDM 开始把显存换到共享内存,
            # 每题从 0.7 s 劣化到 4–7 s(GPU 利用率仍显示 100%,很迷惑)。
            # 定期归还缓存即可;每题都还则每题都要重新 cudaMalloc,实测 0.7 → 1.3 s/题,所以每 50 题一次。
            self._count += 1
            if self._count % 50 == 0:
                self._torch.cuda.empty_cache()
        return out


class TrtllmAnswerer(Answerer):
    """TensorRT-LLM(PyTorch 后端),批量提交。

    Args:
        name: 变体名。
        checkpoint: 模型目录 —— 原始 HF 目录即 BF16;ModelOpt 导出的量化 checkpoint 目录即对应精度
            (TensorRT-LLM 读其中的 hf_quant_config.json 决定 kernel,视觉塔一律不量化,见 modeling_qwen2vl.py)。
        max_seq_len: 序列上限。t1024 档最长约 1024 图像 token + 150 文字 token + 16 生成 token。
        vision_engines: 视觉塔 TensorRT 引擎目录(P5);None 表示用 TensorRT-LLM 自带的 HF 视觉塔。
            通过环境变量传给 worker 进程里的 sitecustomize 补丁,必须在构造 LLM(拉起 worker)之前设置。
    """

    # 构造函数:设置视觉塔引擎的环境变量,启动 TensorRT-LLM 并加载处理器
    def __init__(self, name: str = "trtllm-bf16", checkpoint: Path = MODEL_DIR, max_seq_len: int = 2048,
                 vision_engines: Path | None = None) -> None:
        super().__init__(name)
        import os

        # 需要视觉塔引擎时通过环境变量把引擎目录传给 worker 进程
        if vision_engines is not None:
            os.environ["VLMDEPLOY_VISION_TRT"] = str(vision_engines)
        else:
            # 不用引擎时清掉,免得沿用上一次留在环境里的值
            os.environ.pop("VLMDEPLOY_VISION_TRT", None)
        # tensorrt_llm 在环境变量设置之后才导入
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig
        from transformers import AutoProcessor

        # 加载计时起点
        t0 = time.time()
        # 显存:TensorRT-LLM 默认把剩余显存的 90% 划给 KV cache。本模型每 token 的 KV 只有 28 KB,
        # 一批 32 题 × 1.2k token ≈ 1 GB 都不到,只给 25%,给视觉塔激活与 CUDA graph 留出余量(整卡 8 GB,桌面还占约 1 GB)。
        # 关闭 KV 块复用:与计时脚本保持同一配置(评测里同一张图常有 2 道题,复用会让第二题跳过部分 prefill;
        # 结果数值上应相同,但"评测与计时用的是同一种引擎配置"更容易解释)。
        self.llm = LLM(model=str(checkpoint), kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25,
                                                                            enable_block_reuse=False),
                       max_batch_size=8, max_num_tokens=4096, max_seq_len=max_seq_len)
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)   # 只用它的聊天模板,与 HF 参考同源
        self.params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)   # temperature=0 即贪心
        # 记录加载耗时
        self.load_seconds = time.time() - t0
        # 附加信息:模型目录和视觉塔引擎目录
        self.info = {"checkpoint": str(checkpoint), "vision_engines": str(vision_engines) if vision_engines else None}

    # 整批提交作答
    def _answer(self, batch: Sequence[tuple]) -> list[str]:
        # 每题一个请求:提示词加一张图
        inputs = [{"prompt": chat_prompt(self.processor, s.prompt), "multi_modal_data": {"image": [img]}}
                  for s, img in batch]
        # 整批一次提交,批处理由 TensorRT-LLM 自己调度
        return [o.outputs[0].text for o in self.llm.generate(inputs, sampling_params=self.params, use_tqdm=False)]

    def close(self) -> None:
        """关闭 TensorRT-LLM:结束 worker 进程,释放显存。"""
        self.llm.shutdown()


class OpenVinoAnswerer(Answerer):
    """OpenVINO GenAI VLMPipeline,CPU,逐题(GenAI 的 VLM 管线一次处理一个请求)。

    Args:
        name: 变体名。
        ir_dir: 导出 / 压缩后的 IR 目录(语言模型、视觉塔、文本 embedding 各一个 IR + tokenizer)。
    """

    # 构造函数:加载 IR 并编译到 CPU
    def __init__(self, name: str, ir_dir: Path) -> None:
        super().__init__(name)
        # openvino 只在这里导入
        import openvino as ov
        import openvino_genai as ov_genai

        # 加载计时起点
        t0 = time.time()
        # 保存 openvino 模块供其他方法使用
        self._ov = ov
        # 创建 VLM 管线,在 CPU 上推理并限定线程数
        self.pipe = ov_genai.VLMPipeline(str(ir_dir), "CPU", INFERENCE_NUM_THREADS=CPU_THREADS)
        # 生成参数,只设置最大生成长度
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = MAX_NEW_TOKENS   # 贪心是默认(do_sample=False)
        # 记录加载耗时
        self.load_seconds = time.time() - t0
        # 附加信息:IR 目录和线程数
        self.info = {"ir": str(ir_dir), "threads": CPU_THREADS}

    # 逐题作答
    def _answer(self, batch: Sequence[tuple]) -> list[str]:
        # numpy 只在这里导入
        import numpy as np

        # 各题的答案
        out = []
        # 逐题处理
        for sample, img in batch:
            # 图片转成 uint8 张量,最前面增加 batch 维
            tensor = self._ov.Tensor(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None]))  # [1,H,W,3] RGB
            # 只传题目文字和图片,聊天模板由管线添加;取第一个候选答案
            out.append(self.pipe.generate(sample.prompt, images=[tensor], generation_config=self.config).texts[0])
        return out
