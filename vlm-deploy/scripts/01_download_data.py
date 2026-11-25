"""下载评测数据分片到 data/raw/,逐文件校验长度与 sha256。

用法(任一环境,只依赖标准库):
    HF_ENDPOINT=https://hf-mirror.com python3 scripts/01_download_data.py [--only textvqa realworldqa]

下载什么、为什么(清单定义在 vlmdeploy/config.py 的 DATASETS):
    TextVQA     val 3 个分片(5000 题,测试集)+ train 第 1 个分片(抽校准集与 dev 集),约 1.2 GB
    RealWorldQA 官方 xai-org 仓库 2 个分片(765 题,车载 / 真实场景空间理解),约 0.68 GB

产物:
    data/raw/<dataset>/data/*.parquet
    results/download_data.json     每个文件的大小、sha256、许可、耗时

数据只用于评测,不随仓库分发(data/ 在 .gitignore 里)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import DATASETS, HF_ENDPOINT, RAW_DIR, RESULTS_DIR
from vlmdeploy.hub import download_files, list_files
from vlmdeploy.logging_ import get_logger

# 模块级 logger
log = get_logger("download_data")


def main() -> int:
    """按 config.DATASETS 的清单下载评测数据分片,记录写进 results/download_data.json。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="下载评测数据分片")
    # 只下载指定的数据集,默认全部
    parser.add_argument("--only", nargs="*", choices=sorted(DATASETS), default=sorted(DATASETS))
    # 解析参数
    args = parser.parse_args()

    # 各数据集的下载记录
    summary = {}
    # 逐个数据集下载
    for name in args.only:
        # 数据集的下载清单
        spec = DATASETS[name]
        # 清单里要下载的文件
        wanted = set(spec.files)
        # 从仓库文件列表里筛出清单中的文件
        files = [f for f in list_files(HF_ENDPOINT, spec.repo, kind="dataset") if f.path in wanted]
        # 清单里有而仓库里没有的文件
        missing = wanted - {f.path for f in files}
        if missing:  # 上游改了分片命名时立刻失败,而不是悄悄少下几个分片、测试集变小
            raise SystemExit(f"{spec.repo}: files not found on hub: {sorted(missing)}")
        # 本地保存目录
        out_dir = RAW_DIR / spec.local_dir
        # 打印仓库名、文件数、总大小和本地目录
        log.info("%s: %d files, %.2f GB -> %s", spec.repo, len(files), sum(f.size for f in files) / 1e9, out_dir)
        # 计时起点
        t0 = time.time()
        # 逐个下载并校验
        records = download_files(HF_ENDPOINT, spec.repo, files, str(out_dir), kind="dataset", log=log.info)
        # 记录仓库、许可、本地目录、耗时和逐文件记录
        summary[name] = {"repo": spec.repo, "license": spec.license, "local_dir": str(out_dir),
                         "seconds": round(time.time() - t0, 1), "files": records}

    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 下载记录文件
    path = RESULTS_DIR / "download_data.json"
    # 并入已有记录而不是覆盖:--only 只下一个数据集时,另一个数据集的记录还在
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    # 用本次的记录更新
    merged.update(summary)
    # 写回
    path.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
    # 打印各数据集的耗时
    log.info("done: %s", ", ".join(f"{k} {v['seconds']:.0f}s" for k, v in summary.items()))
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
