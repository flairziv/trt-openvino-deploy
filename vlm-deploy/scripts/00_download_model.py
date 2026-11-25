"""下载 Qwen2-VL-2B-Instruct 到 $VLM_ARTIFACTS/hf/(默认 D 盘),逐文件校验长度与 sha256。

用法(任一环境,只依赖标准库):
    HF_ENDPOINT=https://hf-mirror.com python3 scripts/00_download_model.py

产物:
    <MODEL_DIR>/*                       模型权重、tokenizer、处理器配置(4.43 GB)
    results/download_model.json         文件清单、大小、sha256、是否校验、耗时

模型权重本身不进仓库;许可 Apache-2.0(模型卡 license 字段)。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 直接运行脚本时 sys.path[0] 是 scripts/ 而不是仓库根目录,先把根目录加进去,下面才能 import vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import HF_ENDPOINT, MODEL_DIR, MODEL_REPO, RESULTS_DIR
from vlmdeploy.hub import download_files, list_files
from vlmdeploy.logging_ import get_logger

# 模块级 logger
log = get_logger("download_model")


def main() -> int:
    """列出模型仓库的文件清单,逐个下载并校验,记录写进 results/download_model.json。"""
    # 计时起点
    t0 = time.time()
    # 列出模型仓库的全部文件
    files = list_files(HF_ENDPOINT, MODEL_REPO, kind="model")
    # 全部文件的总字节数
    total = sum(f.size for f in files)
    # 打印仓库名、文件数、总大小和本地目录
    log.info("%s: %d files, %.2f GB -> %s", MODEL_REPO, len(files), total / 1e9, MODEL_DIR)
    # 逐个下载并校验
    records = download_files(HF_ENDPOINT, MODEL_REPO, files, str(MODEL_DIR), kind="model", log=log.info)

    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 下载记录:仓库、镜像地址、本地目录、总字节数、耗时和逐文件记录
    summary = {"repo": MODEL_REPO, "endpoint": HF_ENDPOINT, "local_dir": str(MODEL_DIR), "total_bytes": total,
               "seconds": round(time.time() - t0, 1), "files": records}
    # 写出下载记录
    (RESULTS_DIR / "download_model.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")
    # 打印耗时和经过 sha256 校验的文件数
    log.info("done in %.0fs, %d/%d files sha256-verified", summary["seconds"],
             sum(r["sha256_verified"] for r in records), len(records))
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
