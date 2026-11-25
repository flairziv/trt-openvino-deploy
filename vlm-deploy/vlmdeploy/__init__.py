"""vlm-deploy:Qwen2-VL-2B 在笔记本上的两栈部署(TensorRT-LLM GPU / OpenVINO GenAI CPU)。

本包必须保持"导入即轻量":顶层不 import torch / tensorrt_llm / openvino,
两个 conda 环境(trtllm / ovgenai)以及只有标准库的下载步骤都要能 import 它。
重依赖只在用到的子模块里导入。
"""

# 每个阶段(P0–P6)一个次版本
__version__ = "0.7.0"   # 与 pyproject.toml 的 version 保持一致
