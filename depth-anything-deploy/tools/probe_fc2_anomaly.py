"""复现 TensorRT 10.16 混合精度异常:INT8 fc1 + FP16 fc2 时输出严重失真。

用法(env: dl,GPU 独占):python tools/probe_fc2_anomaly.py

现象:在 INT8 QDQ 图上只拆掉 fc2 的激活 Q/DQ,TRT 选择 fc1 `i8f32`(INT8 进、FP32 直出)→ GELU → FP16 fc2,
SQNR 从 ~21 dB 跌到 ~3.5 dB;ORT CUDA EP 执行同一张图没有这个现象,其它拆法也正常。
结论的边界:这是从外部黑盒观察到的,**根因未证实**;只能说"疑似 Myelin 代码生成缺陷"。
不使用 timing cache,排除缓存污染。产物:results/fc2_anomaly.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx

from dadeploy.config import ARTIFACTS_DIR, ONNX_DIR, RESOLUTIONS, RESULTS_DIR
from dadeploy.data import iter_split
from dadeploy.groups import bypass_pairs, find_qdq_pairs, pairs_by_group
from dadeploy.metrics import output_drift
from dadeploy.preprocess import preprocess
from dadeploy.reference import load_reference
from dadeploy.runners import OrtRunner, TrtRunner
from dadeploy.trt_recipes import TRT_RECIPES
from dadeploy.trt_utils import build_engine

# 实验用的分辨率档位
RES = "r364"
# 参与比较的 val 图片数
NUM = 16


# 入口:对几种拆 Q/DQ 的方式分别比较 ORT 仿真与真实引擎的 SQNR
def main() -> int:
    # 取分辨率
    res = RESOLUTIONS[RES]
    # 读取 val 的 FP32 参考输出
    _, ref = load_reference(res, "val")
    # 预处理 val 集前 NUM 张图
    xs = [preprocess(s.rgb, res) for s in iter_split("val", limit=NUM)]
    # 加载全 INT8 图
    model = onnx.load(str(ONNX_DIR / f"da2s_{RES}.int8.onnx"))
    # 量化点分组
    groups = pairs_by_group(find_qdq_pairs(model))

    # 取分组名以指定后缀结尾的全部量化点
    def of(suffix: str) -> list:
        return [p for g, ps in groups.items() if g.endswith(suffix) for p in ps]

    # fc2 的全部量化点
    fc2 = of(".fc2")
    # 四种拆法:fc2 全拆、只拆激活、只拆权重、fc1 和 fc2 都拆
    cases = {
        "fc2_activation_and_weight": fc2,
        "fc2_activation_only": [p for p in fc2 if not p.is_weight],
        "fc2_weight_only": [p for p in fc2 if p.is_weight],
        "fc1_and_fc2": of(".fc1") + fc2,
    }
    # 工作目录
    work = ARTIFACTS_DIR / "fc2_probe"
    work.mkdir(parents=True, exist_ok=True)

    # 在这批图上计算相对 FP32 参考的平均 SQNR
    def sqnr(runner) -> float:  # noqa: ANN001
        return float(np.mean([output_drift(runner.infer(x), np.asarray(ref[i])).sqnr_db for i, x in enumerate(xs)]))

    # 各拆法的结果
    report = {}
    # 逐个拆法实验
    for name, pairs in cases.items():
        # 拆掉对应的量化点
        graph = bypass_pairs(model, pairs)
        # 候选图路径
        path = work / "candidate.onnx"
        # 保存候选图
        onnx.save(graph, str(path))
        # ORT 在 CUDA 上仿真的 SQNR
        ort_db = sqnr(OrtRunner("ort", res, graph.SerializeToString()))
        # 不使用 timing cache,强类型构建引擎
        build_engine(path, work / "candidate.engine", TRT_RECIPES["int8"], f"probe:{name}", None)
        # 加载引擎
        runner = TrtRunner("trt", res, work / "candidate.engine")
        # 记录两边的 SQNR
        report[name] = {"ort_cuda_sqnr_db": ort_db, "trt_sqnr_db": sqnr(runner)}
        # 释放引擎
        runner.close()
        # 打印这一拆法的结果
        print(name, report[name], flush=True)
    # 写出结果
    (RESULTS_DIR / "fc2_anomaly.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    # 删除工作目录
    shutil.rmtree(work, ignore_errors=True)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
