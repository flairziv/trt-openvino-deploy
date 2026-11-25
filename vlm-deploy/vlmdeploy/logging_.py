"""统一日志。

所有脚本通过 :func:`get_logger` 拿 logger,格式固定为 `时间 级别 [模块] 消息`,
便于把多个脚本的输出拼进同一个 run log 后按时间排查。
(模块名带下划线是为了避免遮蔽标准库 `logging`。)
"""

from __future__ import annotations

import logging
import sys

# 日志格式:时间、级别首字母、模块名、消息
_FORMAT = "%(asctime)s %(levelname).1s [%(name)s] %(message)s"
# 根 logger 是否已经初始化
_configured = False


def get_logger(name: str) -> logging.Logger:
    """返回已配置好格式的 logger。

    Args:
        name: logger 名,建议传脚本或模块的短名。

    Returns:
        logging.Logger 实例(首次调用时初始化根 handler,之后复用)。
    """
    # 声明要修改模块级标志
    global _configured
    # 首次调用时初始化根 logger
    if not _configured:
        # 输出到 stdout(basicConfig 默认是 stderr):便于和子进程的输出按时间拼在一起
        logging.basicConfig(level=logging.INFO, format=_FORMAT, datefmt="%H:%M:%S", stream=sys.stdout)
        # 置位,避免重复初始化
        _configured = True
    # 返回指定名字的 logger
    return logging.getLogger(name)
