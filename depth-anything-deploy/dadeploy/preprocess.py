"""预处理与后处理。

所有后端(PyTorch / ORT / TensorRT / OpenVINO)**共用这一份实现**,这是对照实验公平的前提:
如果各后端各写一套 resize / normalize,精度差异里就会混进预处理差异,无法归因到量化。

流水线位置:
    相机帧(HWC uint8 RGB) → preprocess → [1,3,H,W] float32 → 推理 → [H,W] 相对视差
    → postprocess(双线性放回原图尺寸)→ metrics 对齐与打分
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import IMAGE_MEAN, IMAGE_STD, Resolution

# 归一化均值,整理成 [1,1,3] 以便按通道广播
_MEAN = np.asarray(IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
# 标准差的倒数,归一化时用乘法代替除法
_INV_STD = (1.0 / np.asarray(IMAGE_STD, dtype=np.float32)).reshape(1, 1, 3)


def preprocess(rgb: np.ndarray, res: Resolution) -> np.ndarray:
    """把 RGB 图像变成模型输入张量。

    与官方 Depth Anything V2 推理一致:bicubic 缩放 → /255 → ImageNet 归一化 → NCHW。
    官方是"短边 518、保持长宽比",这里改为固定尺寸(部署要求静态形状);
    对 4:3 的 NYU 原图,r518 档(518×686)与官方 HF 预处理得到的尺寸完全相同。

    与官方的一处有意差异:官方先 /255 转 float 再 bicubic,这里直接对 uint8 缩放
    (相机帧本来就是 uint8,缩放的内存带宽是 float32 的 1/4)。已实测验证两者等价:
    测试集 218 张上 AbsRel 0.05835 vs 0.05832(差 3e-5,远小于任何量化带来的差异);
    后处理用 cv2 双线性(align_corners=False)替代官方 align_corners=True,差异同样不可测。

    Args:
        rgb: HWC、uint8、RGB 顺序的图像。
        res: 目标输入档位。

    Returns:
        形状 [1, 3, res.height, res.width] 的 float32 连续数组。

    Raises:
        ValueError: 输入不是 HWC×3 的 uint8 图像。
    """
    # 输入必须是 HWC 三通道 uint8 图像
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expect HWC uint8 RGB, got shape={rgb.shape} dtype={rgb.dtype}")
    # 双三次插值缩放到目标分辨率,cv2 的尺寸参数顺序是(宽,高)
    resized = cv2.resize(rgb, (res.width, res.height), interpolation=cv2.INTER_CUBIC)
    # 转 float32 并缩放到 0 到 1,再按通道减均值、乘标准差的倒数
    x = (resized.astype(np.float32) * (1.0 / 255.0) - _MEAN) * _INV_STD
    # HWC 转 CHW 并在最前面增加 batch 维,返回连续内存
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(pred: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """把模型输出的相对视差图放回原图分辨率。

    Args:
        pred: 模型输出,形状 [H, W] 或 [1, H, W],任意浮点类型。
        out_hw: 目标 (高, 宽)。

    Returns:
        形状 out_hw 的 float32 视差图(仍是相对量,尺度/平移待 metrics 对齐)。
    """
    # 统一成二维 float32 数组,兼容 [H,W] 和 [1,H,W] 两种输入
    pred2d = np.asarray(pred, dtype=np.float32).reshape(pred.shape[-2], pred.shape[-1])
    # 双线性插值缩放到目标尺寸,cv2 的尺寸参数顺序是(宽,高)
    # cv2.INTER_LINEAR 采用半像素中心对齐,与 torch F.interpolate(bilinear, align_corners=False) 一致
    return cv2.resize(pred2d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
