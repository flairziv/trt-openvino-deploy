"""计时工具:分段统计与交错轮次编排。

笔记本平台计时的两个噪声源与对策:
    1. **降频漂移**:GPU/CPU 温度上升后频率下降,先测的变体天然占便宜。
       对策:交错轮次(interleaved rounds)—— 每轮把所有变体随机打乱各测一段,
       最终每个变体的样本均匀分布在整个时间窗口里,漂移被平摊。
    2. **长尾抖动**:调度 / 中断引起的偶发慢帧。
       对策:报中位数(P50)与 P90,不报均值;同时报每轮中位数的极差,作为稳定性证据。
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np


def summarize(samples_ms: Sequence[float]) -> dict[str, float]:
    """把一组耗时样本(毫秒)汇总成统计量。

    Args:
        samples_ms: 耗时样本。

    Returns:
        {"n", "p50", "p90", "mean", "std", "min"}。

    Raises:
        ValueError: 样本为空。
    """
    # 没有样本时报错
    if len(samples_ms) == 0:
        raise ValueError("no samples")
    # 转成 float64 数组
    arr = np.asarray(samples_ms, dtype=np.float64)
    # 返回样本数、中位数、P90、均值、标准差、最小值
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
    }


def interleaved_schedule(keys: Sequence[str], rounds: int, seed: int = 0) -> list[list[str]]:
    """生成交错轮次计划:每轮包含全部变体一次,顺序随机。

    Args:
        keys: 变体字符串。
        rounds: 轮数。
        seed: 随机种子(固定以便复现计划本身)。

    Returns:
        rounds 个列表,每个是 keys 的一个排列。
    """
    # 独立的随机数发生器,种子固定则计划可复现
    rng = random.Random(seed)
    # 各轮的执行顺序
    plan = []
    # 每轮生成一个随机排列
    for _ in range(rounds):
        # 复制一份,不改动入参
        order = list(keys)
        # 原地打乱
        rng.shuffle(order)
        # 记入计划
        plan.append(order)
    return plan


def round_spread(per_round_p50: Sequence[float]) -> float:
    """各轮中位数的相对极差 (max−min)/median,衡量跨轮稳定性。"""
    # 转成 float64 数组
    arr = np.asarray(per_round_p50, dtype=np.float64)
    # 极差除以中位数,得到相对波动
    return float((arr.max() - arr.min()) / np.median(arr))
