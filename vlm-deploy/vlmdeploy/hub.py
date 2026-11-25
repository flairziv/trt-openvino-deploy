"""HF Hub(镜像)文件清单与下载编排,只依赖标准库。

为什么不用 huggingface_hub.snapshot_download:
    - 下载步骤要在任何环境(包括还没建好的 conda 环境)里都能跑;
    - snapshot_download 单连接,镜像限速 3~4 MB/s,断流后整个文件重来。
    这里用 Hub 的 tree API 拿文件清单和 LFS 文件的 sha256,小文件直接下,大文件交给
    vlmdeploy.fetch 多连接下载,并逐个校验 sha256 —— 下完的文件保证和仓库里的一字不差。
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass

from vlmdeploy.fetch import fetch

# 请求头里的客户端标识
_USER_AGENT = "Mozilla/5.0 (vlm-deploy hub)"
_SMALL_FILE_BYTES = 32 << 20   # 小于 32 MB 的文件单连接直接下,分块没有意义


@dataclass(frozen=True)
class RemoteFile:
    """仓库里的一个文件。

    Attributes:
        path: 仓库内相对路径。
        size: 字节数。
        sha256: LFS 文件的 sha256;普通 git 文件为 None(tree API 只给 git blob 的 sha1,不能拿来校验内容)。
    """

    path: str
    size: int
    sha256: str | None


# 发送 GET 请求并把响应解析成 JSON
def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def list_files(endpoint: str, repo: str, kind: str = "model", revision: str = "main") -> list[RemoteFile]:
    """递归列出仓库文件。

    Args:
        endpoint: Hub 地址,如 https://hf-mirror.com。
        repo: 仓库名,如 Qwen/Qwen2-VL-2B-Instruct。
        kind: "model" 或 "dataset"。
        revision: 分支或 commit。

    Returns:
        文件列表(不含目录)。
    """
    # 模型和数据集的 API 路径前缀不同
    prefix = "models" if kind == "model" else "datasets"
    # 递归列出文件的 tree 接口地址
    url = f"{endpoint}/api/{prefix}/{repo}/tree/{urllib.parse.quote(revision)}?recursive=true"
    # 请求文件清单
    entries = _get_json(url)
    # 文件列表
    files = []
    # 逐个条目处理
    for entry in entries:  # type: ignore[union-attr]
        # 跳过目录
        if entry.get("type") != "file":
            continue
        # 大文件(LFS)的真实大小和 sha256 在 lfs 字段里(oid 就是 sha256);普通文件没有这个字段
        lfs = entry.get("lfs") or {}
        # 大小优先取 lfs 字段,其次取条目自身的 size;sha256 只有 LFS 文件才有
        files.append(RemoteFile(path=entry["path"], size=int(lfs.get("size") or entry.get("size") or 0),
                                sha256=lfs.get("oid")))
    return files


def resolve_url(endpoint: str, repo: str, path: str, kind: str = "model", revision: str = "main") -> str:
    """文件的下载地址(Hub 的 resolve 接口,会重定向到 CDN)。"""
    # 数据集的下载地址多一段 datasets
    base = endpoint if kind == "model" else f"{endpoint}/datasets"
    # 拼出 resolve 接口的下载地址,分支名和路径做 URL 转义
    return f"{base}/{repo}/resolve/{urllib.parse.quote(revision)}/{urllib.parse.quote(path)}"


def download_files(endpoint: str, repo: str, files: list[RemoteFile], out_dir: str, kind: str = "model",
                   log: Callable[[str], None] = print) -> list[dict]:
    """下载并校验一组文件,返回下载记录。

    Args:
        endpoint: Hub 地址。
        repo: 仓库名。
        files: 要下载的文件(来自 list_files)。
        out_dir: 本地目录,保留仓库内的相对路径结构。
        kind: "model" 或 "dataset"。
        log: 进度输出函数。

    Returns:
        每个文件一条记录:路径、大小、sha256、是否经过 sha256 校验、耗时。

    Raises:
        vlmdeploy.fetch.FetchError: 长度或 sha256 校验失败。
    """
    # 各文件的下载记录
    records = []
    # 逐个文件下载
    for remote in files:
        # 本地路径保留仓库内的相对路径
        out = os.path.join(out_dir, remote.path)
        # 下载地址
        url = resolve_url(endpoint, repo, remote.path, kind)
        # 计时起点
        t0 = time.time()
        # 大文件多连接分块下载
        if remote.size >= _SMALL_FILE_BYTES:
            fetch(url, out, size=remote.size, sha256=remote.sha256, log=log)
        else:
            # 单连接;64 MB 的块比小文件上限(32 MB)大,整个文件就是一块
            fetch(url, out, size=remote.size, sha256=remote.sha256, workers=1, chunk_mb=64, log=log)
        # 记录文件信息、是否校验过 sha256 和耗时
        records.append({**asdict(remote), "sha256_verified": remote.sha256 is not None,
                        "seconds": round(time.time() - t0, 1)})
    return records
