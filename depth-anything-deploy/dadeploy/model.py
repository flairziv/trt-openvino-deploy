"""PyTorch 模型加载与导出包装。

HF 的 `DepthAnythingForDepthEstimation.forward` 返回 `DepthEstimatorOutput` 数据类,
导出 ONNX / 当参考模型用时都只需要 `predicted_depth` 一个张量,因此包一层只返回该张量的 Module,
同时把输出名、形状契约固定下来:

    输入  pixel_values    [1, 3, H, W] float32(已归一化)
    输出  predicted_depth [1, H, W]    float32(相对视差,越大越近)
"""

from __future__ import annotations

import torch
from torch import nn

from .config import HF_MODEL_DIR


class DepthOnly(nn.Module):
    """只返回 predicted_depth 的薄包装,隐藏 HF 的输出数据类。"""

    # 构造函数:保存被包装的 HF 模型
    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.model = hf_model

    # 前向:调用 HF 模型,只返回 predicted_depth 张量
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(pixel_values=pixel_values).predicted_depth


def load_model(device: str = "cuda") -> DepthOnly:
    """加载本地 HF 权重并切到 eval。

    Args:
        device: "cuda" 或 "cpu"。

    Returns:
        DepthOnly 模型(FP32)。

    Raises:
        FileNotFoundError: 权重目录不存在(先运行 scripts/00_download_model.py)。
    """
    # 权重文件不存在时提示先下载
    if not (HF_MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError(f"{HF_MODEL_DIR} 缺少权重,请先运行 scripts/00_download_model.py")
    from transformers import AutoModelForDepthEstimation  # 惰性导入:ov 环境不装 transformers

    # 从本地目录加载 HF 模型,切到推理模式
    hf = AutoModelForDepthEstimation.from_pretrained(str(HF_MODEL_DIR)).eval()
    # 套上包装,切到推理模式并移到指定设备
    return DepthOnly(hf).eval().to(device)


def disable_tf32() -> None:
    """关闭 TF32,保证 PyTorch 参考是严格 FP32。

    Ampere+ 上 cuDNN 卷积默认允许 TF32(10-bit 尾数),不关掉的话"FP32 参考"本身就带误差,
    会把 TensorRT TF32 / FP16 的偏差测小。
    """
    # 关闭矩阵乘法的 TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    # 关闭 cuDNN 卷积的 TF32
    torch.backends.cudnn.allow_tf32 = False
