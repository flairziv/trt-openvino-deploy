"""命令行入口:多连接分块断点下载(实现见 vlmdeploy/fetch.py,只依赖标准库)。

用法:
    python tools/parallel_fetch.py <url> <out> [--size N] [--sha256 HEX] [--workers 8] [--chunk-mb 16]

envs/setup_trtllm.sh 用它预取 2.7 GB 的 TensorRT-LLM wheel:pip 单连接下载断流后只能从头再来,
先落盘并校验 sha256,再本地安装。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.fetch import FetchError, fetch


def main() -> int:
    """命令行入口:多连接分块下载一个 URL。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="多连接分块断点下载")
    # 下载地址
    parser.add_argument("url")
    # 保存路径
    parser.add_argument("out")
    # 期望的字节数
    parser.add_argument("--size", type=int, default=None, help="期望字节数;缺省时用 HEAD 获取")
    # 期望的 sha256
    parser.add_argument("--sha256", default=None, help="期望的 sha256(十六进制)")
    # 并发连接数
    parser.add_argument("--workers", type=int, default=8)
    # 分块大小,单位 MB
    parser.add_argument("--chunk-mb", type=int, default=16)
    # 解析参数
    args = parser.parse_args()
    # 下载并校验,进度信息直接打印
    try:
        fetch(args.url, args.out, size=args.size, sha256=args.sha256, workers=args.workers, chunk_mb=args.chunk_mb,
              log=lambda msg: print(msg, flush=True))
    # 失败时把原因写到标准错误
    except FetchError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        # 非零退出码是给调用方看的:envs/setup_trtllm.sh 开了 set -e,下载或校验失败就此停下,不会接着去装
        return 1
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
