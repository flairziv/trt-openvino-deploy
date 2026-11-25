"""生成 PyTorch FP32 参考:精度基准 + 量化偏差的参考输出。

用法(env: dl):
    python scripts/03_reference.py                # 三个档位全部
    python scripts/03_reference.py --res r518

每个档位做三件事:
    1. 在 654 张测试集上评测 PyTorch FP32(TF32 关闭)→ results/eval/torch_<res>_fp32__test.json
       —— 这是所有部署变体的精度基准;
    2. 保存 drift 子集(测试集每 6 张取 1 张,109 张)的原始输出 → 报告各变体的量化偏差;
    3. 保存 val 集(100 张,训练场景)的原始输出 → 敏感度分析 / 回退决策只用它,不碰测试集。

协议健全性检查:r518 档就是官方 HF 预处理对 NYU 4:3 输入的结果(短边 518、长边取 14 的倍数 686),
其 AbsRel 应与 Depth Anything V2 论文的 ViT-S 零样本 NYUv2 结果处于同一量级,否则说明对齐 / 掩码有误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dadeploy.config import RESOLUTIONS, RESULTS_DIR  # noqa: E402
from dadeploy.data import split_names  # noqa: E402
from dadeploy.evaluation import evaluate  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.reference import save_reference  # noqa: E402
from dadeploy.runners import TorchRunner  # noqa: E402

# 模块级 logger
log = get_logger("reference")


def run_one(res_name: str) -> None:
    """对一个档位生成基准与参考输出。"""
    # 取该档位的分辨率
    res = RESOLUTIONS[res_name]
    # PyTorch 基准的变体字符串
    key = f"torch:{res_name}:fp32"
    # 创建 PyTorch Runner
    runner = TorchRunner(key, res)

    # 1 + 2:测试集评测,同时收集 drift 子集的原始输出
    # drift 子集的样本名
    drift_names = split_names("drift")
    # 转成集合,便于判断是否属于 drift 子集
    wanted = set(drift_names)
    # 收集到的原始输出
    collected: dict[str, np.ndarray] = {}

    # 回调:只保存属于 drift 子集的原始输出
    def keep(name: str, raw: np.ndarray) -> None:
        if name in wanted:
            # 拷贝一份再保存
            collected[name] = raw.copy()

    # 评测全部测试集,同时通过回调收集输出
    result = evaluate(runner, "test", split_names("test"), keep=keep)
    # 打印结果摘要
    log.info(result.headline())
    # 评测结果目录,不存在时创建
    out_dir = RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 写出测试集上的评测结果
    (out_dir / f"torch_{res_name}_fp32__test.json").write_text(json.dumps(result.to_dict()))
    # 按 drift 样本名的顺序堆叠后保存参考输出
    save_reference(res, "drift", drift_names, np.stack([collected[n] for n in drift_names]))

    # 3:val 集原始输出(同时记录 val 上的 FP32 指标,作为 accuracy-control 的起点)
    # val 划分的样本名
    val_names = split_names("val")
    # 收集到的 val 原始输出
    val_out: dict[str, np.ndarray] = {}
    # 评测 val 集,回调把每张图的原始输出存进字典
    val_result = evaluate(runner, "val", val_names, keep=lambda n, r: val_out.__setitem__(n, r.copy()))
    # 打印结果摘要
    log.info(val_result.headline())
    # 写出 val 集上的评测结果
    (out_dir / f"torch_{res_name}_fp32__val.json").write_text(json.dumps(val_result.to_dict()))
    # 保存 val 集的参考输出
    save_reference(res, "val", val_names, np.stack([val_out[n] for n in val_names]))
    # 打印保存的张数
    log.info("%s: saved references drift=%d val=%d", res_name, len(drift_names), len(val_names))


# 入口:逐个档位生成基准与参考输出
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="PyTorch FP32 精度基准 + 参考输出")
    # 要处理的档位,默认全部
    parser.add_argument("--res", nargs="*", default=list(RESOLUTIONS), choices=list(RESOLUTIONS))
    # 解析参数
    args = parser.parse_args()
    # 逐个档位处理
    for name in args.res:
        run_one(name)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
