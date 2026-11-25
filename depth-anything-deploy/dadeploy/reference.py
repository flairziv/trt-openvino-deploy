"""PyTorch FP32 参考输出的落盘与读取。

为什么要落盘:量化偏差(drift)= 被测变体原始输出 vs FP32 参考输出。OpenVINO 变体跑在 `ov`
环境(没有 torch),无法现场算参考,所以由 scripts/03_reference.py 在 `dl` 环境里一次算好存下来,
两个环境共用同一份基准。

存储格式:artifacts/reference/<res>_<split>.npy(float32,[N, H, W],np.load 时 mmap 读取)
         + 同名 .json 记录样本名顺序,读取时逐一核对,防止清单变化后错位比较。
         参考必须保持 FP32 —— 存成 FP16 会引入 ~5e-4 的相对误差,和要测的 FP16 引擎偏差同一量级。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import REF_DIR, Resolution


def reference_paths(res: Resolution, split: str) -> tuple[Path, Path]:
    """返回 (数组路径, 名单路径)。"""
    # 两个文件共用的路径前缀:档位名加划分名
    stem = REF_DIR / f"{res.name}_{split}"
    # 分别加上 .npy 和 .json 后缀
    return stem.with_suffix(".npy"), stem.with_suffix(".json")


def save_reference(res: Resolution, split: str, names: list[str], outputs: np.ndarray) -> None:
    """保存参考输出。

    Args:
        res: 分辨率档位。
        split: 划分名("drift" / "val")。
        names: 样本名,顺序与 outputs 第 0 维一致。
        outputs: [N, H, W] float32。

    Raises:
        ValueError: 形状或 dtype 不符。
    """
    # 数组必须是 float32,形状为 [样本数,高,宽]
    if outputs.dtype != np.float32 or outputs.shape != (len(names), res.height, res.width):
        raise ValueError(f"bad reference array {outputs.shape} {outputs.dtype}")
    # 取数组与名单的保存路径
    arr_path, names_path = reference_paths(res, split)
    # 目录不存在时创建
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存参考输出数组
    np.save(arr_path, outputs)
    # 保存样本名顺序
    names_path.write_text(json.dumps(names), encoding="utf-8")


def load_reference(res: Resolution, split: str, expect_names: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """读取参考输出(mmap,不占内存)。

    Args:
        res: 分辨率档位。
        split: 划分名。
        expect_names: 若给出,则逐一核对样本名顺序。

    Returns:
        (样本名列表, [N, H, W] 只读数组)。

    Raises:
        FileNotFoundError: 尚未生成参考。
        ValueError: 样本名顺序与期望不一致。
    """
    # 取数组与名单的路径
    arr_path, names_path = reference_paths(res, split)
    # 参考文件不存在时提示先生成
    if not arr_path.exists():
        raise FileNotFoundError(f"{arr_path} 不存在,请先运行 scripts/03_reference.py")
    # 读取落盘时记录的样本名顺序
    names = json.loads(names_path.read_text(encoding="utf-8"))
    # 给了期望名单时核对顺序,不一致则报错
    if expect_names is not None and names != list(expect_names):
        raise ValueError(f"reference {arr_path.name} sample order mismatch; 重新运行 03_reference.py")
    # 以内存映射方式只读打开数组,不整体读入内存
    return names, np.load(arr_path, mmap_mode="r")
