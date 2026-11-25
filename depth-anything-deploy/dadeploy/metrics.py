"""评测指标。

两类指标,回答两个不同的问题:

1. **任务精度**(对真值):模型部署后深度估计还准不准。
   Depth Anything V2 输出的是**相对视差**(affine-invariant inverse depth),没有绝对尺度,
   因此按 MiDaS / Depth Anything 的标准协议:在**视差域**对每张图做最小二乘尺度-平移对齐,
   再转回深度,在 Eigen 裁剪区域内计算 AbsRel / δ1 / RMSE。

2. **量化偏差**(对 FP32 参考输出):同一模型换精度后,输出变了多少。
   不需要真值,直接比较模型原始输出,用于逐层敏感度分析与回退决策。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DELTA1_THRESHOLD, EIGEN_CROP, MAX_DEPTH_M, MIN_DEPTH_M


# ---------------------------------------------------------------------------
# 任务精度
# ---------------------------------------------------------------------------
def eigen_valid_mask(gt_depth: np.ndarray) -> np.ndarray:
    """生成 Eigen 协议的有效像素掩码:深度在 (MIN, MAX) 内且落在标准裁剪框中。

    Args:
        gt_depth: 真值深度图(米),形状 [480, 640]。

    Returns:
        同形状的 bool 掩码。
    """
    # 真值深度落在有效范围内的像素
    mask = (gt_depth > MIN_DEPTH_M) & (gt_depth < MAX_DEPTH_M)
    # Eigen 裁剪框掩码,先全部置为 False
    crop = np.zeros_like(mask)
    # 取出裁剪框的上下左右边界
    top, bottom, left, right = EIGEN_CROP
    # 框内置为 True
    crop[top:bottom, left:right] = True
    # 两个条件同时满足才算有效像素
    return mask & crop


def align_disparity_to_depth(pred_disp: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """视差域最小二乘尺度-平移对齐,返回对齐后的深度图。

    求解 min_{s,t} || s·pred + t − 1/gt ||²(只在 mask 内),闭式解即 2×2 正规方程。
    对齐后的视差下限截断到 1/MAX_DEPTH,避免出现负深度 / 无穷大深度。

    Args:
        pred_disp: 模型输出的相对视差,已放回原图尺寸。
        gt_depth: 真值深度(米)。
        mask: 有效像素掩码。

    Returns:
        与 gt 同尺度的预测深度图(米)。

    Raises:
        ValueError: 有效像素不足以求解(< 2)。
    """
    # 取有效像素上的预测视差,转 float64 做最小二乘
    p = pred_disp[mask].astype(np.float64)
    # 真值深度取倒数,得到真值视差
    g = 1.0 / gt_depth[mask].astype(np.float64)
    # 有效像素少于 2 个时无法求解
    if p.size < 2:
        raise ValueError("not enough valid pixels for scale-shift alignment")
    # 设计矩阵:第一列是预测视差,第二列全为 1
    a = np.stack([p, np.ones_like(p)], axis=1)
    # 最小二乘求出尺度和平移
    (scale, shift), *_ = np.linalg.lstsq(a, g, rcond=None)
    # 把尺度和平移作用到整张预测图
    aligned = scale * pred_disp.astype(np.float64) + shift
    # 视差下限截断到最大深度的倒数,避免负深度和无穷大
    aligned = np.maximum(aligned, 1.0 / MAX_DEPTH_M)
    # 视差取倒数还原为深度,转回 float32
    return (1.0 / aligned).astype(np.float32)


@dataclass
class DepthMetrics:
    """单张图的深度指标。AbsRel / RMSE 越小越好,δ1 越大越好。"""

    # 绝对相对误差
    abs_rel: float
    # 预测与真值的比值小于阈值的像素占比
    delta1: float
    # 均方根误差(米)
    rmse: float


def depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> DepthMetrics:
    """计算单张图的 AbsRel / δ1 / RMSE(仅 mask 内像素)。

    Args:
        pred_depth: 已对齐的预测深度(米)。
        gt_depth: 真值深度(米)。
        mask: 有效像素掩码。

    Returns:
        DepthMetrics。
    """
    # 取有效像素上的预测深度
    p = pred_depth[mask].astype(np.float64)
    # 取有效像素上的真值深度
    g = gt_depth[mask].astype(np.float64)
    # 逐像素取预测与真值两个方向比值中较大的一个
    ratio = np.maximum(p / g, g / p)
    return DepthMetrics(
        # 绝对误差除以真值后取平均
        abs_rel=float(np.mean(np.abs(p - g) / g)),
        # 比值小于阈值的像素占比
        delta1=float(np.mean(ratio < DELTA1_THRESHOLD)),
        # 误差平方取平均后开方
        rmse=float(np.sqrt(np.mean((p - g) ** 2))),
    )


def evaluate_prediction(pred_disp_full: np.ndarray, gt_depth: np.ndarray) -> DepthMetrics:
    """一步完成"掩码 → 对齐 → 打分"。

    Args:
        pred_disp_full: 放回原图尺寸的相对视差。
        gt_depth: 真值深度(米)。

    Returns:
        DepthMetrics。
    """
    # 生成有效像素掩码
    mask = eigen_valid_mask(gt_depth)
    # 在视差域对齐后转成深度
    pred_depth = align_disparity_to_depth(pred_disp_full, gt_depth, mask)
    # 在掩码内计算三项指标
    return depth_metrics(pred_depth, gt_depth, mask)


# ---------------------------------------------------------------------------
# 量化偏差(对 FP32 参考输出)
# ---------------------------------------------------------------------------
@dataclass
class DriftMetrics:
    """量化后输出相对 FP32 参考的偏差。

    Attributes:
        rel_l1: mean|q − r| / mean|r|,越小越好。
        sqnr_db: 信号量化噪声比 10·log10(Σr² / Σ(q−r)²),越大越好(每 +6 dB ≈ 误差减半)。
    """

    rel_l1: float
    sqnr_db: float


def output_drift(pred: np.ndarray, ref: np.ndarray) -> DriftMetrics:
    """比较同分辨率下的原始模型输出。

    Args:
        pred: 被测变体的原始输出。
        ref: PyTorch FP32 参考输出(同形状)。

    Returns:
        DriftMetrics。

    Raises:
        ValueError: 形状不一致。
    """
    # 两个输出的形状必须一致
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    # 被测输出展平成一维 float64
    q = pred.astype(np.float64).ravel()
    # 参考输出展平成一维 float64
    r = ref.astype(np.float64).ravel()
    # 逐元素误差
    err = q - r
    # 误差能量
    noise = float(np.sum(err**2))
    # 误差为零时记为无穷大,否则按信号能量与误差能量之比换算成分贝
    sqnr = float("inf") if noise == 0.0 else 10.0 * np.log10(float(np.sum(r**2)) / noise)
    # 返回相对 L1 误差与 SQNR
    return DriftMetrics(rel_l1=float(np.mean(np.abs(err)) / np.mean(np.abs(r))), sqnr_db=sqnr)


# ---------------------------------------------------------------------------
# 配对比较
# ---------------------------------------------------------------------------
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """同一组图片上两个变体逐图指标之差 (b − a) 的均值与 95% bootstrap 置信区间。

    为什么用配对:两个变体在**同一批图**上评测,图与图之间的难度差异远大于量化带来的差异。
    直接比两个均值,难度方差会把真实差异淹没;逐图相减后再重采样,难度项被抵消,
    区间才能回答"INT8 比 FP16 差,是真的差还是噪声"。

    Args:
        a: 基准变体的逐图指标。
        b: 被测变体的逐图指标(与 a 同序同长)。
        n_boot: 重采样次数。
        seed: 随机种子。

    Returns:
        (均值差, 2.5% 分位, 97.5% 分位)。

    Raises:
        ValueError: 长度不一致或为空。
    """
    # 两组逐图指标都转成 float64 数组
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # 两组必须等长且非空
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"paired arrays must match and be non-empty: {a.shape} vs {b.shape}")
    # 逐图差值:被测减基准
    diff = b - a
    # 固定种子的随机数发生器
    rng = np.random.default_rng(seed)
    # 生成 n_boot 组有放回的重采样下标,每组长度等于样本数
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    # 每组重采样的均值
    boots = diff[idx].mean(axis=1)
    # 返回原始均值差,以及重采样均值的 2.5% 和 97.5% 分位
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
