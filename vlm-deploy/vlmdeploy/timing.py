"""计时协议:分段统计、交错轮次编排、跨轮稳定性。

与项目 A 的 dadeploy/timing.py 同一套方法,改为纯标准库实现(两个 conda 环境都要用,不依赖 numpy)。

笔记本平台计时的两个噪声源与对策:
    1. **频率漂移**:GPU / CPU 升温后降频,先测的变体天然占便宜。
       对策:交错轮次 —— 每轮把全部测量项随机打乱各测一次,每个变体的样本均匀分布在整个时间窗口里,漂移被平摊。
    2. **长尾抖动**:调度、中断、Windows 宿主的后台任务引起的偶发慢样本。
       对策:报中位数(P50)与 P90,不报均值;另报各轮中位数的相对极差,作为"这个数字稳不稳"的证据。

差值类指标(如"视觉段 = 带图请求 TTFT − 同长度纯文本请求 TTFT")必须在**同一轮内配对相减**再取中位数,
而不是两个中位数相减:同一轮的两个样本经历的频率状态最接近,配对能消掉大部分漂移。
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位数(与 numpy.percentile 默认的 "linear" 方法一致)。

    Args:
        values: 样本。
        q: 0–100。

    Raises:
        ValueError: 样本为空。
    """
    # 没有样本时报错
    if not values:
        raise ValueError("no samples")
    # 排序
    data = sorted(values)
    # pos 是目标分位在排好序的样本里的小数下标;落在两个样本之间时按距离线性插值
    pos = (len(data) - 1) * q / 100
    # 目标位置两侧的整数下标
    lo, hi = math.floor(pos), math.ceil(pos)
    # 按距离做线性插值
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def summarize(samples: Sequence[float], digits: int = 2) -> dict[str, float]:
    """把一组样本汇总成 {"n", "p50", "p90", "mean", "min"}(单位与输入相同)。"""
    # 依次是样本数、中位数、P90、均值和最小值,保留 digits 位小数
    return {"n": len(samples), "p50": round(percentile(samples, 50), digits),
            "p90": round(percentile(samples, 90), digits), "mean": round(statistics.fmean(samples), digits),
            "min": round(min(samples), digits)}


def interleaved_schedule(items: Sequence, rounds: int, seed: int = 0) -> list[list]:
    """交错轮次计划:每轮包含全部测量项各一次,顺序随机(种子固定,计划本身可复现)。"""
    # 独立的随机数发生器,种子固定则计划可复现
    rng = random.Random(seed)
    # 各轮的执行顺序
    plan = []
    # 每轮生成一个随机排列
    for _ in range(rounds):
        # 复制一份,不改动入参
        order = list(items)
        # 原地打乱
        rng.shuffle(order)
        # 记入计划
        plan.append(order)
    return plan


def round_spread(per_round_p50: Sequence[float]) -> float:
    """各轮中位数的相对极差 (max − min) / median。小于 5% 视为稳定。"""
    # 各轮中位数的中位数
    med = statistics.median(per_round_p50)
    # 极差除以中位数,中位数为 0 时返回 NaN
    return (max(per_round_p50) - min(per_round_p50)) / med if med else float("nan")


# 目前只有单元测试调用它:README 里的加速比是两个 P50 直接相除,没有带区间。
def ratio_ci(a: Sequence[float], b: Sequence[float], seed: int = 0, n_boot: int = 2000) -> tuple[float, float, float]:
    """两组**配对**样本的中位数比 median(a)/median(b) 及其 bootstrap 95% 置信区间。

    用于报"X 比 Y 快几倍":只给点估计不给区间,在笔记本上很容易把噪声当成提速。

    Returns:
        (点估计, 下界, 上界)。
    """
    # 两组样本必须等长且非空
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and of equal length")
    # 固定种子的随机数发生器
    rng = random.Random(seed)
    # 样本下标
    idx = range(len(a))
    # 各次重采样得到的比值
    boots = []
    # 重复 n_boot 次
    for _ in range(n_boot):
        # 有放回地重抽下标;a、b 用同一批下标,配对关系才不被打散
        pick = [rng.choice(idx) for _ in idx]
        # 重采样样本上的中位数之比
        boots.append(statistics.median(a[i] for i in pick) / statistics.median(b[i] for i in pick))
    # 返回原始样本的中位数之比,以及重采样比值的 2.5% 和 97.5% 分位
    return statistics.median(a) / statistics.median(b), percentile(boots, 2.5), percentile(boots, 97.5)
