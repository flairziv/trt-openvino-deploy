"""多连接分块断点下载器。

背景:
    HF 镜像对单连接限速(实测 ~90 KB/s)且会中途断流(curl: (18))。
    把文件切成固定大小的块、多线程并行 Range 请求,每块独立重试,
    最后按块序拼接,并按目标字节数校验完整性。

用法:
    python tools/parallel_fetch.py <url> <out_path> --size <bytes> [--offset <bytes>] [--workers 8] [--chunk-mb 16]

    --offset 让它只取文件中间的一段字节窗口(例如从大 tar 中段抽样校准图),此时 --size 是窗口长度。

设计说明:
    - 每块落盘为 <out>.parts/<idx>.part,已完整的块下次直接跳过(可断点续跑)。
    - 只依赖标准库,任何 conda 环境都能跑。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import sys
import time
import urllib.request

# 请求头里的客户端标识
_USER_AGENT = "Mozilla/5.0 (depth-anything-deploy fetcher)"


def _fetch_range(url: str, start: int, end: int, part_path: str, retries: int = 50) -> int:
    """下载 [start, end] 闭区间字节到 part_path,支持块内续传。

    Args:
        url: 资源地址(需支持 HTTP Range)。
        start: 起始字节(含)。
        end: 结束字节(含)。
        part_path: 该块的落盘路径。
        retries: 最大重试次数。

    Returns:
        该块的字节数。

    Raises:
        RuntimeError: 重试耗尽仍未下完。
    """
    # 该块应有的字节数
    expected = end - start + 1
    # 失败后重试,最多 retries 次
    for attempt in range(retries):
        # 块文件已有的字节数,用于续传
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        # 已经下完则直接返回
        if have >= expected:
            return expected
        # 从已有字节之后继续请求
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": _USER_AGENT}
        )
        # 下载并追加写入块文件,网络异常时退避后重试
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, "ab") as fout:
                shutil.copyfileobj(resp, fout, length=1 << 20)
        except Exception:  # noqa: BLE001 —— 网络异常一律退避重试
            # 等待时间逐次加长,最长 15 秒
            time.sleep(min(2 + attempt, 15))
    # 重试耗尽
    raise RuntimeError(f"chunk {part_path} failed after {retries} retries")


# 入口:多连接分块下载一个文件或其中的一段
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="多连接分块断点下载")
    # 下载地址
    parser.add_argument("url")
    # 保存路径
    parser.add_argument("out")
    parser.add_argument("--size", type=int, required=True, help="目标字节数(用于切块与校验)")
    parser.add_argument("--offset", type=int, default=0, help="起始字节偏移(取文件中段窗口时使用)")
    # 并发连接数
    parser.add_argument("--workers", type=int, default=8)
    # 分块大小(MB)
    parser.add_argument("--chunk-mb", type=int, default=16)
    # 解析参数
    args = parser.parse_args()

    # 目标文件已存在且大小相符时跳过
    if os.path.exists(args.out) and os.path.getsize(args.out) == args.size:
        print(f"[skip] {args.out} already complete")
        return 0

    # 分块大小换算成字节
    chunk = args.chunk_mb << 20
    # 分块的存放目录
    parts_dir = args.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    # 各块的序号和起止字节,最后一块可能不足一个分块
    ranges = [(i, s, min(s + chunk, args.size) - 1) for i, s in enumerate(range(0, args.size, chunk))]

    # 计时起点
    t0 = time.time()
    # 线程池并发下载各块
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        # 提交全部分块任务,记录任务与块序号的对应关系
        futs = {
            pool.submit(
                _fetch_range, args.url, args.offset + s, args.offset + e, os.path.join(parts_dir, f"{i:05d}.part")
            ): i
            for i, s, e in ranges
        }
        # 按完成的先后处理
        for n_done, fut in enumerate(cf.as_completed(futs), 1):
            # 取结果,块下载失败时在这里抛出异常
            fut.result()
            # 每完成 8 块或全部完成时打印进度
            if n_done % 8 == 0 or n_done == len(ranges):
                mb = n_done * chunk / (1 << 20)
                print(f"[{n_done}/{len(ranges)}] ~{mb:.0f} MB, {mb / (time.time() - t0):.2f} MB/s", flush=True)

    # 按块序拼接,再做长度校验
    # 临时文件路径
    tmp = args.out + ".tmp"
    # 按块序号依次拼接到临时文件
    with open(tmp, "wb") as fout:
        for i, _, _ in ranges:
            with open(os.path.join(parts_dir, f"{i:05d}.part"), "rb") as fin:
                shutil.copyfileobj(fin, fout, length=1 << 22)
    # 总长度不符时报错退出
    if os.path.getsize(tmp) != args.size:
        print(f"[error] size mismatch: {os.path.getsize(tmp)} != {args.size}", file=sys.stderr)
        return 1
    # 校验通过后改名为正式文件
    os.replace(tmp, args.out)
    # 删除分块目录
    shutil.rmtree(parts_dir)
    # 打印完成信息
    print(f"[done] {args.out} {args.size} bytes in {time.time() - t0:.0f}s")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
