"""在 654 张测试集上评测变体:任务精度 + 量化偏差 + 与基准的配对置信区间。

用法:
    # env: dl
    python scripts/09_evaluate.py --variants trt:r518:fp16 trt:r518:int8 trt:r518:int8mix
    # env: ov
    python scripts/09_evaluate.py --variants ov:r518:fp32 ov:r518:int8

口径:
    - 全部变体走 dadeploy.evaluation.evaluate,预处理 / 对齐 / 掩码完全一致;
    - 同分辨率 PyTorch FP32 是基准。报告 ΔAbsRel 及其 95% 配对 bootstrap 置信区间:
      区间不含 0 才能说"确实变差 / 变好",否则只能说"在 654 张上分辨不出";
    - 量化偏差只在 drift 子集(109 张)上算,相对落盘的 FP32 参考输出。

产物:results/eval/<变体>__test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dadeploy.config import RESULTS_DIR  # noqa: E402
from dadeploy.data import split_names  # noqa: E402
from dadeploy.evaluation import evaluate  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.metrics import paired_bootstrap  # noqa: E402
from dadeploy.reference import load_reference  # noqa: E402
from dadeploy.runners import create_runner  # noqa: E402
from dadeploy.variants import parse_many  # noqa: E402

# 模块级 logger
log = get_logger("evaluate")


# 入口:逐个变体在测试集上评测,并与同分辨率的 FP32 基准做配对比较
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="测试集评测(精度 + 偏差 + 配对 CI)")
    # 变体列表,必填
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None, help="只评前 N 张(冒烟用,结果不落盘)")
    # 解析参数
    args = parser.parse_args()

    # 评测结果目录,不存在时创建
    out_dir = RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 测试集的全部样本名
    names = split_names("test")
    # 指定了 limit 时只取前 N 张
    if args.limit:
        names = names[: args.limit]

    # 逐个变体评测
    for variant in parse_many(args.variants):
        # 读取该分辨率的 drift 参考输出,并核对样本顺序
        reference = load_reference(variant.res, "drift", split_names("drift"))
        # 创建 Runner,用完自动释放
        with create_runner(variant) as runner:
            # 评测测试集,同时计算量化偏差
            result = evaluate(runner, "test", names, reference=reference)
        # 打印结果摘要
        log.info(result.headline())
        # 冒烟模式不落盘
        if args.limit:
            continue
        # 结果转成字典
        payload = result.to_dict()
        # 同分辨率 FP32 基准的结果文件
        base_path = out_dir / f"torch_{variant.res.name}_fp32__test.json"
        # 基准存在且当前不是基准本身时做配对比较
        if base_path.exists() and variant.backend != "torch":
            # 基准的逐图 AbsRel
            base = json.loads(base_path.read_text())["per_image"]["abs_rel"]
            # 配对 bootstrap:均值差和 95% 置信区间
            mean, low, high = paired_bootstrap(np.asarray(base), np.asarray(result.per_image["abs_rel"]))
            # 写入结果
            payload["delta_abs_rel"] = {"mean": mean, "ci95_low": low, "ci95_high": high}
            # 打印配对差
            log.info("%s ΔAbsRel vs FP32 = %+.5f [%+.5f, %+.5f]", variant.key, mean, low, high)
        # 写出该变体的结果文件
        (out_dir / f"{variant.slug}__test.json").write_text(json.dumps(payload))
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
