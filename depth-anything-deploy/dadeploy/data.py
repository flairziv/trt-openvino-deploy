"""NYUv2 数据读取。

数据来源:HF 镜像 `sayakpaul/nyu_depth_v2`(fast-depth 预处理版):
    - val/official/*.h5   官方 labeled 集的 654 张 test 划分,文件名 = labeled 集 1-based 索引,
                          由 scripts/01_prepare_data.py 用官方 splits.mat 的 testNdxs 逐个核对;
    - train/<scene>/*.h5  训练场景帧:从 12 个训练 tar 的中段各截 40 MB 字节窗口恢复,
                          切成 calib(量化校准)/ val(敏感度分析与精度控制)两个互斥子集;
                          与 test 场景不相交,避免"拿测试图校准 / 调参"的数据泄漏。
每个 h5 含 `rgb` (3,480,640) uint8 与 `depth` (480,640) float32(米)。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import DRIFT_STRIDE, MANIFEST_PATH, NYU_DIR


@dataclass
class Sample:
    """一张评测 / 校准样本。

    Attributes:
        name: 样本名(相对 data/nyu 的路径),用作结果与参考输出的键。
        rgb: HWC uint8 RGB 图。
        depth: 真值深度(米),[480, 640]。
    """

    name: str
    rgb: np.ndarray
    depth: np.ndarray


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取单个 fast-depth h5。

    Args:
        path: h5 文件路径。

    Returns:
        (rgb HWC uint8, depth HW float32)。
    """
    # 以只读方式打开 h5 文件
    with h5py.File(path, "r") as f:
        # 图像由 CHW 转为 HWC,并整理成连续内存
        rgb = np.ascontiguousarray(np.asarray(f["rgb"]).transpose(1, 2, 0))
        # 深度图转为 float32
        depth = np.asarray(f["depth"], dtype=np.float32)
    return rgb, depth


def load_manifest() -> dict:
    """读取 01_prepare_data.py 生成的数据清单。

    Raises:
        FileNotFoundError: 尚未执行数据准备脚本。
    """
    # 清单不存在时提示先运行数据准备脚本
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} 不存在,请先运行 scripts/01_prepare_data.py")
    # 读取并解析清单
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def split_names(split: str) -> list[str]:
    """返回某个划分的样本名列表(清单顺序)。

    Args:
        split: "test" / "calib" / "val",或 "drift"(测试集按 DRIFT_STRIDE 均匀抽样的子集)。

    Raises:
        KeyError: 未知划分名。
    """
    # drift 划分:测试集每隔 DRIFT_STRIDE 张取一张
    if split == "drift":
        return load_manifest()["test"][::DRIFT_STRIDE]
    # 其余划分直接取清单里对应的列表
    return list(load_manifest()[split])


def iter_samples(names: Iterable[str]) -> Iterator[Sample]:
    """按给定名字顺序逐张读取样本(流式,内存占用恒定)。"""
    # 按给定顺序逐张处理
    for name in names:
        # 读取该样本的图像与深度
        rgb, depth = load_h5(NYU_DIR / name)
        # 逐张产出,不在内存里累积
        yield Sample(name=name, rgb=rgb, depth=depth)


def iter_split(split: str, limit: int | None = None) -> Iterator[Sample]:
    """按清单顺序迭代某个划分的样本。

    Args:
        split: 见 :func:`split_names`。
        limit: 只取前 limit 张(调试与快速冒烟用);None 表示全部。

    Yields:
        Sample。
    """
    # 取该划分的全部样本名
    names = split_names(split)
    # 指定了 limit 时只保留前 limit 个
    if limit is not None:
        names = names[:limit]
    # 交给 iter_samples 逐张读取
    yield from iter_samples(names)
