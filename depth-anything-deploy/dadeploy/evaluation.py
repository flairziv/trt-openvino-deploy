"""评测主循环:一个 Runner 跑一个划分,得到任务精度 + 量化偏差。

所有变体的评测都走这一个函数,保证预处理、对齐、打分、偏差计算完全一致。

流水线(每张图):
    rgb(480×640) → preprocess(res) → runner.infer → 原始输出 [H,W]
        ├─ 若该图在 drift 参考集里:与 FP32 参考比 → rel_l1 / SQNR(原始输出、同分辨率)
        └─ postprocess 放回 480×640 → 视差域尺度-平移对齐 → AbsRel / δ1 / RMSE(Eigen 裁剪)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import NYU_HEIGHT, NYU_WIDTH
from .data import iter_samples
from .logging_ import get_logger
from .metrics import evaluate_prediction, output_drift
from .preprocess import postprocess, preprocess

# 模块级 logger
log = get_logger("eval")


@dataclass
class EvalResult:
    """一个变体在一个划分上的评测结果(写入 results/eval/<variant>__<split>.json)。

    Attributes:
        variant: 变体字符串。
        split: 划分名。
        n: 图片数。
        abs_rel / delta1 / rmse: 逐图指标的均值。
        drift_rel_l1 / drift_sqnr_db: 相对 FP32 参考的偏差均值(仅在提供参考时有值)。
        per_image: 逐图指标(用于配对比较的 bootstrap 置信区间)。
        seconds: 评测墙钟时间。
    """

    variant: str
    split: str
    n: int
    abs_rel: float
    delta1: float
    rmse: float
    drift_rel_l1: float | None = None
    drift_sqnr_db: float | None = None
    # 参与偏差统计的图片数
    drift_n: int = 0
    per_image: dict[str, list[float]] = field(default_factory=dict)
    seconds: float = 0.0

    # 转成普通字典,便于写入 JSON
    def to_dict(self) -> dict:  # noqa: D102
        return asdict(self)

    def headline(self) -> str:
        """一行摘要,用于日志。"""
        # 有偏差数据时在摘要末尾追加偏差部分
        drift = (f" drift rel_l1={self.drift_rel_l1:.2e} sqnr={self.drift_sqnr_db:.1f}dB(n={self.drift_n})"
                 if self.drift_rel_l1 is not None else "")
        # 拼出变体、划分、图片数与三项精度指标
        return (f"{self.variant} [{self.split} n={self.n}] AbsRel={self.abs_rel:.4f} "
                f"δ1={self.delta1:.4f} RMSE={self.rmse:.4f}{drift}")


def evaluate(runner, split: str, names: list[str],  # noqa: ANN001 —— runner: runners.Runner
             reference: tuple[list[str], np.ndarray] | None = None,
             keep: Callable[[str, np.ndarray], None] | None = None,
             log_every: int = 200) -> EvalResult:
    """评测一个 Runner。

    Args:
        runner: 任意后端的 Runner。
        split: 划分名(只写进结果)。
        names: 要评测的样本名(顺序即评测顺序)。
        reference: (参考样本名, 参考输出数组);名字在其中的样本会额外计算偏差。
        keep: 回调 keep(name, raw_output),用于参考脚本收集原始输出。
        log_every: 每多少张打一次进度。

    Returns:
        EvalResult。
    """
    # 参考样本名到参考数组下标的映射,没有参考时为空
    ref_index = {n: i for i, n in enumerate(reference[0])} if reference else {}
    # 逐图指标的容器
    per = {"abs_rel": [], "delta1": [], "rmse": []}
    # 逐图的相对 L1 偏差
    drift_l1: list[float] = []
    # 逐图的 SQNR
    drift_sqnr: list[float] = []
    # 评测计时起点
    t0 = time.time()
    # 按顺序逐张评测,序号从 1 开始
    for i, sample in enumerate(iter_samples(names), 1):
        # 预处理后推理,得到模型原始输出
        raw = runner.infer(preprocess(sample.rgb, runner.res))
        # 提供了回调时把原始输出交给调用方
        if keep is not None:
            keep(sample.name, raw)
        # 该样本有 FP32 参考时计算量化偏差
        if sample.name in ref_index:
            # 取出对应的参考输出并比较
            d = output_drift(raw, np.asarray(reference[1][ref_index[sample.name]]))
            # 记录这张图的两项偏差
            drift_l1.append(d.rel_l1)
            drift_sqnr.append(d.sqnr_db)
        # 放回原图尺寸后对齐并打分
        m = evaluate_prediction(postprocess(raw, (NYU_HEIGHT, NYU_WIDTH)), sample.depth)
        # 记录这张图的三项指标
        per["abs_rel"].append(m.abs_rel)
        per["delta1"].append(m.delta1)
        per["rmse"].append(m.rmse)
        # 按间隔打印进度和当前的平均 AbsRel
        if log_every and i % log_every == 0:
            log.info("%s %s %d/%d AbsRel(running)=%.4f", runner.key, split, i, len(names), np.mean(per["abs_rel"]))

    # 求平均前去掉无穷大的 SQNR
    finite_sqnr = [s for s in drift_sqnr if np.isfinite(s)]
    # 汇总成评测结果,各项指标取逐图均值
    return EvalResult(
        variant=runner.key,
        split=split,
        n=len(per["abs_rel"]),
        abs_rel=float(np.mean(per["abs_rel"])),
        delta1=float(np.mean(per["delta1"])),
        rmse=float(np.mean(per["rmse"])),
        # 没有偏差数据时为 None
        drift_rel_l1=float(np.mean(drift_l1)) if drift_l1 else None,
        # 全部为无穷大时记无穷大,否则取有限值的平均
        drift_sqnr_db=(float(np.mean(finite_sqnr)) if finite_sqnr else float("inf")) if drift_sqnr else None,
        drift_n=len(drift_l1),
        per_image=per,
        # 评测耗时(秒)
        seconds=round(time.time() - t0, 1),
    )
