"""运行环境读数:目前只有整卡显存。只依赖标准库,任何环境都能 import。"""

from __future__ import annotations

import subprocess


def gpu_memory_used_mb() -> int | None:
    """整卡已用显存(MB)。

    nvidia-smi 口径:包含 Windows 桌面占用(约 0.5–1 GB)和其他进程,不是本进程的占用。
    用它看的是"8 GB 还剩多少",以及上一个 GPU 进程的显存是否已经归还。

    Returns:
        已用显存(MB);没有 nvidia-smi 或读数失败时为 None(CPU 环境、驱动异常)。
    """
    # 调用 nvidia-smi 查询已用显存,失败时返回 None
    try:
        # 只输出数值,不带表头和单位
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        # 每块显卡一行,只取第一行(第一块显卡)
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        # 读数只是辅助信息:没有 nvidia-smi、命令失败或超时、输出不是数字,都不该让主流程崩
        return None
