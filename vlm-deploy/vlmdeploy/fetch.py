"""多连接分块断点下载(只依赖标准库,任何环境都能 import)。

从项目 A 的 tools/parallel_fetch.py 演进而来。背景:HF 镜像、pypi.nvidia.com 单连接只有 3~4 MB/s
(2026-11-09 实测),大文件中途断流时 pip / curl 只能从头再来。这里把文件切成固定大小的块,
多线程并行发 Range 请求,每块独立重试、可断点续跑,最后按块序拼接并校验。

相对 A 的版本,加了三处"下完的文件一定是对的"的保护:
    1. 每个请求都要求 206 Partial Content。服务器若忽略 Range 回 200 整文件,块会被整文件内容撑大、
       拼出来是错的 —— 这种情况直接判失败,而不是把错误内容写进去;
    2. 块长度超过预期(上次异常写入)就删块重下,不再"多了也当完成";
    3. 支持整文件 sha256 校验(HF 的 LFS 文件、pypi.nvidia.com 的 wheel 都公开 sha256)。
另外 size 可省略,自动取 HEAD 的 Content-Length。代理沿用环境变量 https_proxy(WSL 里是 Clash)。
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import shutil
import time
import urllib.request
from collections.abc import Callable

# 请求头里的客户端标识
_USER_AGENT = "Mozilla/5.0 (vlm-deploy fetcher)"
# 单次请求的超时(秒)
_TIMEOUT_S = 60


class FetchError(RuntimeError):
    """下载无法完成或校验失败(不可通过重试解决的错误)。"""


def head_size(url: str) -> int:
    """用 HEAD 请求取文件字节数(urllib 会跟随重定向)。

    Raises:
        FetchError: 响应没有 Content-Length。
    """
    # 构造 HEAD 请求
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    # 发送请求并读取 Content-Length
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        length = resp.headers.get("Content-Length")
    # 响应没有给出长度时报错
    if not length:
        raise FetchError(f"no Content-Length for {url}")
    return int(length)


def sha256_of(path: str) -> str:
    """流式计算文件 sha256(4 MB 一块,内存占用恒定)。"""
    # 创建 sha256 计算对象
    digest = hashlib.sha256()
    # 以二进制方式打开文件
    with open(path, "rb") as fin:
        # 每次读 4 MB,读到文件末尾为止
        for block in iter(lambda: fin.read(1 << 22), b""):
            digest.update(block)
    # 返回十六进制摘要
    return digest.hexdigest()


def _fetch_range(url: str, start: int, end: int, part_path: str, retries: int = 50) -> int:
    """把 [start, end] 闭区间字节下载到 part_path,块内可续传。

    Returns:
        该块字节数。

    Raises:
        FetchError: 服务器不支持 Range,或重试耗尽。
    """
    # 该块应有的字节数
    expected = end - start + 1
    # 失败后重试,最多 retries 次
    for attempt in range(retries):
        # 每次尝试都先看块文件已有多少字节,下面的 Range 从它之后接着请求 —— 这就是块内续传
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        # 已经下完则直接返回
        if have == expected:
            return expected
        if have > expected:  # 异常写入的残块:整块作废重下,不能截断了事(无法知道哪段是坏的)
            os.remove(part_path)
            have = 0
        # 从已有字节之后继续请求
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": _USER_AGENT}
        )
        # 下载并追加写入块文件
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                # 状态码不是 206 说明服务器没有按 Range 返回
                if resp.status != 206:
                    raise FetchError(f"server ignored Range request (HTTP {resp.status}): {url}")
                # 以追加方式写入块文件
                with open(part_path, "ab") as fout:
                    shutil.copyfileobj(resp, fout, length=1 << 20)
        except FetchError:
            raise  # 服务器不支持 Range,重试没用,原样抛出
        except Exception:  # 故意捕获全部异常 —— 断流、超时、5xx:一律退避后续传
            time.sleep(min(2 + attempt, 15))  # 等待逐次加长,最多 15 秒
    # 重试耗尽
    raise FetchError(f"chunk {part_path} failed after {retries} retries")


def fetch(url: str, out: str, size: int | None = None, sha256: str | None = None,
          workers: int = 8, chunk_mb: int = 16, log: Callable[[str], None] = print) -> str:
    """多连接下载 url 到 out,完成后校验长度与(可选的)sha256。已完整且校验通过的文件直接跳过。

    Args:
        url: 资源地址(需支持 HTTP Range)。
        out: 目标路径。下载过程中的分块放在 <out>.parts/,完成后删除。
        size: 期望字节数;None 时用 HEAD 获取。
        sha256: 期望的十六进制 sha256;None 时只校验长度。
        workers: 并发连接数。
        chunk_mb: 分块大小(MB)。
        log: 进度输出函数。

    Returns:
        out(便于链式调用)。

    Raises:
        FetchError: 长度或 sha256 不符,或某块无法下载。
    """
    # 没有给出长度时用 HEAD 请求获取
    size = size if size is not None else head_size(url)
    # 目标文件已存在且长度相符时,校验通过就跳过
    if os.path.exists(out) and os.path.getsize(out) == size:
        if sha256 is None or sha256_of(out) == sha256:
            log(f"[skip] {out} already complete")
            return out
        os.remove(out)  # 长度对但内容错:不能留着冒充成品

    # 创建目标文件所在的目录
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # 分块大小换算成字节
    chunk = chunk_mb << 20
    # 分块的存放目录
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    # (块序号, 起, 止),闭区间;最后一块可能不足一个 chunk
    ranges = [(i, s, min(s + chunk, size) - 1) for i, s in enumerate(range(0, size, chunk))]

    # 计时起点
    t0 = time.time()
    # 线程池并发下载各块
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        # 提交全部分块任务
        futs = [pool.submit(_fetch_range, url, s, e, os.path.join(parts_dir, f"{i:05d}.part")) for i, s, e in ranges]
        # 已完成的字节数
        done_bytes = 0
        # 按完成的先后处理
        for n_done, fut in enumerate(cf.as_completed(futs), 1):
            # 取该块的字节数,块下载失败时在这里抛出异常
            done_bytes += fut.result()
            # 每完成 8 块或全部完成时打印进度
            if n_done % 8 == 0 or n_done == len(ranges):
                elapsed = max(time.time() - t0, 1e-6)
                log(f"[{n_done}/{len(ranges)}] {done_bytes / 2**20:.0f}/{size / 2**20:.0f} MB, "
                    f"{done_bytes / 2**20 / elapsed:.2f} MB/s  {os.path.basename(out)}")

    # 临时文件路径
    tmp = out + ".tmp"
    # 按块序号依次拼接到临时文件
    with open(tmp, "wb") as fout:
        for i, _, _ in ranges:
            with open(os.path.join(parts_dir, f"{i:05d}.part"), "rb") as fin:
                shutil.copyfileobj(fin, fout, length=1 << 22)
    # 总长度不符时报错
    if os.path.getsize(tmp) != size:
        raise FetchError(f"size mismatch for {out}: {os.path.getsize(tmp)} != {size}")
    # 需要时校验 sha256
    if sha256 is not None:
        got = sha256_of(tmp)
        if got != sha256:
            # 分块里有坏数据:连同临时文件一起删掉,不能留着下次续传
            os.remove(tmp)
            shutil.rmtree(parts_dir, ignore_errors=True)
            raise FetchError(f"sha256 mismatch for {out}: {got} != {sha256}")
    # 校验通过才原子地改名成正式文件:中途失败不会留下"看起来完整"的半成品
    os.replace(tmp, out)
    # 删除分块目录
    shutil.rmtree(parts_dir, ignore_errors=True)
    # 打印完成信息
    log(f"[done] {out} {size} bytes in {time.time() - t0:.0f}s" + (" (sha256 ok)" if sha256 else ""))
    return out
