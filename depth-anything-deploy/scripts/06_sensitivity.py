"""INT8 敏感度分析:在**真实 TensorRT 引擎**上做子层类型消融 + 贪心回退 + 逐块回收。

用法(env: dl,需先有 da2s_<res>.int8.onnx、val 参考输出、04 的激活统计):
    python scripts/06_sensitivity.py --res r518

为什么不用 ORT 假量化做排序(本项目踩过的坑,见 README "坑 4"):
    - ORT CPU EP 对 FP16 QDQ 图的仿真完全失真:同一模型 SQNR 2.7 dB vs CUDA EP 17.0 dB;
    - 即便是 CUDA EP,与 TensorRT 真实 INT8 引擎相比 SQNR 均值差约 4 dB、逐图相关性≈0
      (TRT 把 Q/DQ 融进 kernel,取整、累加精度与 ORT 逐算子执行不同)。
    仿真既然排不准,决策就只能在真实引擎上做;为控制成本,粒度从 64 个分组提到 9 个子层类型。

流程(每个"实验" = 在同一张 INT8 图上拆掉一批 Q/DQ → 强类型构建引擎 → val 100 张评测):
    1. 基线:全 INT8;天花板:拆掉全部 Q/DQ(等价 FP16)。
    2. 类型级留一法:每次只把一个子层类型(如全部 12 个 block 的 residual)拆成 FP16,记录 AbsRel 改善。
    3. 贪心累加:按改善从大到小依次回退类型,直到 AbsRel ≤ FP32 + ACCEPT_ABSREL_DELTA。
    4. 逐块回收:在已回退的类型里,按 04 激活统计"从最好量化到最难量化"的顺序,
       尝试把单个 block 重新量化回 INT8;仍达标就保留 INT8 —— 用尽量少的 FP16 换达标。
    每一步都同时记录引擎的 INT8 层占比与 GPU 纯推理延迟,结果表直接就是"精度-速度"决策表。

与校准的关系:所有实验共用 05 产出的同一份 INT8 校准(只删 Q/DQ,不重新校准),
差异只来自"哪些位置量化",归因干净;最终选中的图原样保存为 da2s_<res>.int8mix.onnx。

数据纪律:决策只看 val 集(训练场景)。测试集只在 09_evaluate.py 的最终报告里出现。

产物:
    results/sensitivity_<res>.json
    artifacts/onnx/da2s_<res>.int8mix.onnx   逐块回收后的最终选择(回收步的差异接近 val 噪声,可能过拟合)
    artifacts/onnx/da2s_<res>.int8res.onnx   规则化对照:全部 residual 分组保持 FP16

已知异常(tools/probe_fc2_anomaly.py 可复现):只拆 fc2 激活的 Q/DQ 时,TRT 引擎 SQNR 暴跌到 ~3.5 dB,
而 ORT 仿真同一张图正常。这正是必须在真实引擎上验证每个候选配置的原因。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import onnx  # noqa: E402

from dadeploy.config import (  # noqa: E402
    ACCEPT_ABSREL_DELTA,
    ARTIFACTS_DIR,
    ENGINE_DIR,
    NYU_HEIGHT,
    NYU_WIDTH,
    ONNX_DIR,
    RESOLUTIONS,
    RESULTS_DIR,
)
from dadeploy.data import iter_split  # noqa: E402
from dadeploy.groups import QdqPair, bypass_pairs, find_qdq_pairs, pairs_by_group  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.metrics import evaluate_prediction, output_drift  # noqa: E402
from dadeploy.preprocess import postprocess, preprocess  # noqa: E402
from dadeploy.reference import load_reference  # noqa: E402
from dadeploy.runners import TrtRunner  # noqa: E402
from dadeploy.trt_recipes import TRT_RECIPES  # noqa: E402
from dadeploy.trt_utils import build_engine  # noqa: E402

# 模块级 logger
log = get_logger("sensitivity")

# 共享的 timing cache 文件
TIMING_CACHE = ENGINE_DIR / "timing.cache"
# 快速延迟测量的迭代次数
LATENCY_ITERS = 50
# 子层类型 → 激活统计里对应的张量种类(用于逐块回收时的排序;None 表示无统计,按网络顺序)
STATS_KIND = {"residual": "resid_out", "fc2": "fc2_in", "fc1": "fc1_in", "qkv": "qkv_in", "o_proj": "o_proj_in"}


def sublayer_of(group: str) -> str:
    """b03.fc2 → fc2;neck.fusion → neck.fusion。"""
    return group.split(".", 1)[1] if group.startswith("b") else group


@dataclass
class Experiment:
    """一次消融实验的结果。"""

    # 实验标签
    tag: str
    # 保持 FP16 的分组
    fp16_groups: list[str]
    # val 集的平均 AbsRel
    abs_rel: float
    # 相对 FP32 参考的平均 SQNR(dB)
    sqnr_db: float
    # 相对 FP32 参考的平均相对 L1 偏差
    rel_l1: float
    # GPU 纯推理耗时的中位数(毫秒)
    gpu_ms: float
    # 引擎里 INT8 计算层的个数
    int8_layers: int
    # 引擎里计算层的总数
    compute_layers: int
    # 引擎构建耗时(秒)
    build_seconds: float
    # 附加信息
    notes: dict = field(default_factory=dict)


class EngineBench:
    """把"拆 Q/DQ → 构建 → 评测"封装成一次函数调用,并缓存同一配置的结果。"""

    # 构造函数:加载 INT8 图、量化点分组和 val 数据
    def __init__(self, res_name: str) -> None:
        # 档位名与分辨率
        self.res_name = res_name
        self.res = RESOLUTIONS[res_name]
        # 加载全 INT8 的 QDQ 图
        self.model = onnx.load(str(ONNX_DIR / f"da2s_{res_name}.int8.onnx"))
        # 找出全部量化点并按分组聚合
        self.groups = pairs_by_group(find_qdq_pairs(self.model))
        # 读入 val 集全部样本
        samples = list(iter_split("val"))
        # 读取 val 的 FP32 参考输出,并核对样本顺序
        _, ref = load_reference(self.res, "val", [s.name for s in samples])
        # 预处理后的输入,提前算好供各次实验复用
        self.inputs = [preprocess(s.rgb, self.res) for s in samples]
        # 真值深度
        self.depths = [s.depth for s in samples]
        # 参考输出转成数组
        self.ref = np.asarray(ref)
        # 候选图和候选引擎的工作目录
        self.work = ARTIFACTS_DIR / "sensitivity" / res_name
        self.work.mkdir(parents=True, exist_ok=True)
        # 实验结果缓存,键是保持 FP16 的分组集合
        self.cache: dict[frozenset[str], Experiment] = {}

    # 生成候选图:拆掉指定分组的全部量化点
    def graph_for(self, fp16_groups: set[str]) -> onnx.ModelProto:
        # 收集这些分组的量化点
        pairs: list[QdqPair] = [p for g in fp16_groups for p in self.groups[g]]
        return bypass_pairs(self.model, pairs)

    def run(self, tag: str, fp16_groups: set[str]) -> Experiment:
        """评估"这些分组保持 FP16、其余 INT8"的配置。"""
        # 以分组集合作缓存键
        key = frozenset(fp16_groups)
        # 同一配置做过实验时直接返回
        if key in self.cache:
            return self.cache[key]
        # 用 "int8" 配方(强类型)构建:精度完全由这张图的 Q/DQ 决定,builder 不会再按计时改精度。
        # 共享 timing cache:只有 Q/DQ 变化的层才重新试 tactic,单次实验从 ~90 s 降到 ~10 s。
        # 候选图路径
        onnx_path = self.work / "candidate.onnx"
        # 候选引擎路径
        engine_path = self.work / "candidate.engine"
        # 保存候选图
        onnx.save(self.graph_for(fp16_groups), str(onnx_path))
        # 强类型构建候选引擎
        report = build_engine(onnx_path, engine_path, TRT_RECIPES["int8"], f"sens:{self.res_name}:{tag}",
                              TIMING_CACHE)
        # 加载候选引擎
        runner = TrtRunner(tag, self.res, engine_path)
        # 逐图的 SQNR、相对 L1 偏差和 AbsRel
        sqnr, l1, absrel = [], [], []
        # 在 val 集上逐张评测
        for x, depth, ref in zip(self.inputs, self.depths, self.ref):
            # 推理得到原始输出
            raw = runner.infer(x)
            # 与 FP32 参考比较
            d = output_drift(raw, ref)
            # 记录两项偏差
            sqnr.append(d.sqnr_db)
            l1.append(d.rel_l1)
            # 放回原图尺寸后对齐打分,记录 AbsRel
            absrel.append(evaluate_prediction(postprocess(raw, (NYU_HEIGHT, NYU_WIDTH)), depth).abs_rel)
        # 这里的延迟只作决策参考:实测同类配置间抖动可达 2.8~4.6 ms,结论一律以 10_benchmark.py 为准
        for _ in range(10):                       # 预热后取 GPU 纯推理时间中位数(快速参考,非正式计时)
            runner.infer_timed(self.inputs[0])
        # 重复测量纯推理耗时
        lat = [runner.infer_timed(self.inputs[0])[1]["compute"] for _ in range(LATENCY_ITERS)]
        # 释放引擎占用的显存
        runner.close()
        del runner
        # 汇总成实验结果
        exp = Experiment(
            tag=tag,
            fp16_groups=sorted(fp16_groups),
            abs_rel=float(np.mean(absrel)),
            sqnr_db=float(np.mean(sqnr)),
            rel_l1=float(np.mean(l1)),
            gpu_ms=float(np.median(lat)),
            int8_layers=report.precision.get("int8", 0),
            compute_layers=sum(report.precision.values()),
            build_seconds=report.build_seconds,
        )
        # 打印这次实验的摘要
        log.info("[%s] fp16_groups=%d AbsRel=%.4f SQNR=%.1fdB GPU=%.2fms INT8 layers=%d/%d",
                 tag, len(fp16_groups), exp.abs_rel, exp.sqnr_db, exp.gpu_ms, exp.int8_layers, exp.compute_layers)
        # 写入缓存
        self.cache[key] = exp
        return exp

    # 返回属于某个子层类型的全部分组
    def groups_of_type(self, sublayer: str) -> set[str]:
        return {g for g in self.groups if sublayer_of(g) == sublayer}


def friendliness_order(res_name: str, groups: list[str]) -> list[str]:
    """按 04 激活统计把分组从"最好量化"排到"最难量化"(per-tensor INT8 SQNR 从高到低)。

    统计缺失时退回网络顺序;这只影响回收尝试的顺序,不影响正确性(每一步都实测验证)。
    """
    # 该档位的激活统计文件
    path = RESULTS_DIR / f"activation_stats_{res_name}.json"
    if not path.exists():
        path = RESULTS_DIR / "activation_stats_r518.json"       # 激活分布对分辨率不敏感,可借用主档统计
    # 没有任何统计时保持原顺序
    if not path.exists():
        return groups
    # 读取逐点统计
    rows = json.loads(path.read_text())["rows"]
    # (block,张量种类)到截断后 SQNR 的映射
    score = {(r["block"], r["kind"]): r["sqnr_clip_db"] for r in rows}

    # 排序键:分组对应的 SQNR,没有统计的排在最后
    def key(group: str) -> float:
        # 子层类型对应的张量种类
        kind = STATS_KIND.get(sublayer_of(group))
        # 没有统计或不属于 block 的分组给最小值
        if kind is None or not group.startswith("b"):
            return -1e9
        # 取该 block 该种类的 SQNR,缺失时给最小值
        return score.get((int(group[1:3]), kind), -1e9)

    # 按 SQNR 从高到低排序
    return sorted(groups, key=key, reverse=True)


def export_graphs(model: onnx.ModelProto, groups: dict, res_name: str, selected: set[str]) -> Path:
    """保存最终选择(int8mix)与规则化对照(int8res),返回 int8mix 路径。"""
    # int8mix 图的路径
    mix_path = ONNX_DIR / f"da2s_{res_name}.int8mix.onnx"
    # 拆掉选中分组的量化点后保存为 int8mix
    onnx.save(bypass_pairs(model, [p for g in selected for p in groups[g]]), str(mix_path))
    # 全部残差分组的量化点
    residual = [p for g, ps in groups.items() if g.endswith(".residual") for p in ps]
    # 拆掉残差流的量化点后保存为 int8res
    onnx.save(bypass_pairs(model, residual), str(ONNX_DIR / f"da2s_{res_name}.int8res.onnx"))
    return mix_path


# 入口:执行消融、贪心回退和逐块回收,或只按已有报告导出图
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="真实 TRT 引擎上的 INT8 子层消融与回退选择")
    # 分辨率档位,必填
    parser.add_argument("--res", required=True, choices=list(RESOLUTIONS))
    parser.add_argument("--refine-steps", type=int, default=12, help="逐块回收最多尝试的分组数")
    parser.add_argument("--export-only", action="store_true", help="不跑实验,按已有报告重新导出 int8mix / int8res 图")
    # 解析参数
    args = parser.parse_args()

    # 只导出模式:按已有报告里的选择重新生成两张图
    if args.export_only:
        # 加载全 INT8 图
        model = onnx.load(str(ONNX_DIR / f"da2s_{args.res}.int8.onnx"))
        # 量化点分组
        groups = pairs_by_group(find_qdq_pairs(model))
        # 读取已有报告
        report = json.loads((RESULTS_DIR / f"sensitivity_{args.res}.json").read_text(encoding="utf-8"))
        # 按报告里选中的分组导出
        export_graphs(model, groups, args.res, set(report["selected_groups"]))
        return 0

    # 准备实验环境
    bench = EngineBench(args.res)
    # FP32 在 val 集上的 AbsRel
    fp32_val = json.loads((RESULTS_DIR / "eval" / f"torch_{args.res}_fp32__val.json").read_text())["abs_rel"]
    # 达标线:FP32 加上允许的上升量
    target = fp32_val + ACCEPT_ABSREL_DELTA
    # 全部子层类型
    types = sorted({sublayer_of(g) for g in bench.groups})
    # 计时起点
    t0 = time.time()
    # 打印实验规模和达标线
    log.info("res=%s groups=%d types=%s FP32 val AbsRel=%.4f target<=%.4f",
             args.res, len(bench.groups), types, fp32_val, target)

    # 基线:全部量化点保留
    full = bench.run("full_int8", set())
    # 上限:全部量化点拆掉
    ceiling = bench.run("all_fp16", set(bench.groups))

    # ---- 1. 类型级留一法
    # 留一法的结果行
    loo = []
    # 每次只把一个子层类型保持 FP16
    for t in types:
        # 做一次实验
        e = bench.run(f"fp16:{t}", bench.groups_of_type(t))
        # 记录该类型的指标和相对基线的改善
        loo.append({"type": t, "groups": len(bench.groups_of_type(t)), "abs_rel": e.abs_rel, "sqnr_db": e.sqnr_db,
                    "gpu_ms": e.gpu_ms, "gain_abs_rel": full.abs_rel - e.abs_rel, "gain_db": e.sqnr_db - full.sqnr_db})
    # 按 AbsRel 改善从大到小排序
    loo.sort(key=lambda r: r["gain_abs_rel"], reverse=True)
    # 打印排序结果
    log.info("type LOO ranking: %s", [(r["type"], round(r["gain_abs_rel"], 4)) for r in loo])

    # ---- 2. 贪心累加类型
    # 排序依据是 AbsRel 改善而不是 SQNR:SQNR 衡量输出像不像 FP32,接受标准却是任务精度;
    # 两者并不等价(尺度-平移对齐会吸收大量偏差),所以 int8mix 达标不代表输出逼近 FP32。
    # 已回退的类型
    chosen_types: list[str] = []
    # 保持 FP16 的分组
    fp16: set[str] = set()
    # 贪心过程的记录,第一行是基线
    greedy = [{"types": [], **asdict(full)}]
    # 当前配置的实验结果
    current = full
    # 按改善从大到小依次回退
    for r in loo:
        # 已达标则停止
        if current.abs_rel <= target:
            break
        # 回退这一类型
        chosen_types.append(r["type"])
        # 把该类型的分组并入 FP16 集合
        fp16 |= bench.groups_of_type(r["type"])
        # 评估回退后的配置
        current = bench.run("greedy:" + "+".join(chosen_types), fp16)
        # 记录这一步
        greedy.append({"types": list(chosen_types), **asdict(current)})
    # 是否达标
    accepted = current.abs_rel <= target
    # 全部回退仍未达标时告警
    if not accepted:
        log.warning("all types reverted but target not met; using all-FP16 configuration")

    # ---- 3. 逐块回收:在已回退的分组里,尝试把"最好量化"的 block 重新量化
    # 注意:单步差异(~0.0005)接近 val 100 张的噪声,回收结果可能过拟合 val,
    # 所以同时导出规则化的 int8res 对照,由测试集判断回收是否真有价值。
    # 逐块回收的记录
    refine = []
    # 达标后才尝试回收
    if accepted:
        # 候选分组按从最好量化到最难量化排序
        candidates = friendliness_order(args.res, sorted(fp16))
        # 最多尝试 refine_steps 个分组
        for g in candidates[: args.refine_steps]:
            # 试着把这个分组重新量化
            trial = fp16 - {g}
            # 评估回收后的配置
            e = bench.run(f"refine:-{g}", trial)
            # 仍然达标则保留回收
            kept = e.abs_rel <= target
            # 记录这一步
            refine.append({"group": g, "requantized": kept, "abs_rel": e.abs_rel, "gpu_ms": e.gpu_ms})
            # 保留回收时更新当前配置
            if kept:
                fp16, current = trial, e
        # 打印回收结果
        log.info("refine: re-quantized %d of %d tried", sum(r["requantized"] for r in refine), len(refine))

    # ---- 产出:最终图 + 报告
    # 导出最终选择和规则化对照
    final_path = export_graphs(bench.model, bench.groups, args.res, fp16)
    # 实验报告
    report = {
        "res": args.res,
        "method": "real-TensorRT ablation on val (100 imgs); shared INT8 calibration; Q/DQ bypass",
        "fp32_val_abs_rel": fp32_val,
        "target_abs_rel": target,
        "full_int8": asdict(full),
        "all_fp16": asdict(ceiling),
        "type_loo": loo,
        "greedy": greedy,
        "refine": refine,
        "accepted": accepted,
        "selected_types": chosen_types,
        "selected_groups": sorted(fp16),
        "final": asdict(current),
        "experiments": [asdict(e) for e in bench.cache.values()],
        "seconds": round(time.time() - t0, 1),
    }
    # 报告文件路径
    out = RESULTS_DIR / f"sensitivity_{args.res}.json"
    # 写出报告
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(bench.work, ignore_errors=True)               # 候选引擎/图只是中间件,不留盘
    # 打印最终配置与两个参照的对比
    log.info("final: %d groups in FP16 -> AbsRel=%.4f GPU=%.2fms (full INT8 %.4f / %.2fms, all FP16 %.4f / %.2fms)",
             len(fp16), current.abs_rel, current.gpu_ms, full.abs_rel, full.gpu_ms, ceiling.abs_rel, ceiling.gpu_ms)
    # 打印产物位置
    log.info("wrote %s and %s", out, final_path.name)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
