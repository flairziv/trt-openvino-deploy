"""下载 Depth Anything V2-Small(HF 格式)权重。

用法:
    HF_ENDPOINT=https://hf-mirror.com python scripts/00_download_model.py

说明:
    - 使用 HF 版 `depth-anything/Depth-Anything-V2-Small-hf`(transformers 原生实现),
      避免依赖 GitHub 上的官方仓库代码(本机 WSL 访问 GitHub 不稳定)。
    - Small 版权重为 Apache-2.0 许可;Base / Large 为 CC-BY-NC-4.0,不可商用,本仓库只用 Small。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dadeploy.config import HF_MODEL_DIR, HF_REPO_ID  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402

# 模块级 logger
log = get_logger("download")


# 入口:下载模型权重到本地目录
def main() -> int:
    # huggingface_hub 只在这里导入
    from huggingface_hub import snapshot_download

    # 打印当前使用的 Hub 地址
    log.info("HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT", "<default huggingface.co>"))
    # 下载整个模型仓库到本地目录
    path = snapshot_download(HF_REPO_ID, local_dir=str(HF_MODEL_DIR))
    # 统计下载目录下各文件的总大小(MiB)
    size_mb = sum(f.stat().st_size for f in Path(path).glob("*")) / 2**20
    # 打印模型位置与大小
    log.info("model ready at %s (%.1f MB)", path, size_mb)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
