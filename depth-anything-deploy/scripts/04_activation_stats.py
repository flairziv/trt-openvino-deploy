"""激活统计:在每个 INT8 量化点上量一量"这个张量有多难量化"。

用法(env: dl):
    python scripts/04_activation_stats.py --res r518 --num 16

敏感度分析(06)回答"哪一组量化误差最伤输出",本脚本回答"为什么"——
在 PyTorch FP32 模型上挂 hook,对 modelopt 实际插 Q/DQ 的每类张量统计:

    channel_outlier   各通道 absmax 的 最大值 / 中位数。per-tensor 量化只有一个 scale,
                      若少数通道比其余大几十倍,要么为它们牺牲所有通道的分辨率,要么截断它们。
                      这正是 SmoothQuant 要"把难度从激活挪到权重"的对象。
    tail_ratio        |x| 的 最大值 / 99.99 分位。描述"少数 token 的极端值"(与通道离群是两回事)。
    sqnr_max_db       用 absmax 做 scale 的对称 per-tensor INT8,单张量量化后的 SQNR。
    sqnr_clip_db      用 99.99 分位截断后的 SQNR(近似熵校准会选的截断点)。
                      两者都低 → 这个张量本身就不适合 per-tensor INT8,与校准方法无关。

被统计的张量(与 modelopt 的量化点一一对应,见 dadeploy/groups.py):
    qkv_in   = norm1 输出      o_proj_in = 注意力输出      fc1_in = norm2 输出
    fc2_in   = GELU 输出       resid_attn = 注意力残差相加后的残差流
    resid_out = block 输出(MLP 残差相加后的残差流)

产物:results/activation_stats_<res>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dadeploy.config import RESOLUTIONS, RESULTS_DIR  # noqa: E402
from dadeploy.data import iter_split  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.model import disable_tf32, load_model  # noqa: E402
from dadeploy.preprocess import preprocess  # noqa: E402

# 模块级 logger
log = get_logger("actstats")

# 统计的六类张量,顺序即报告顺序
TENSOR_KINDS: tuple[str, ...] = ("qkv_in", "o_proj_in", "fc1_in", "fc2_in", "resid_attn", "resid_out")


def int8_sqnr(x: torch.Tensor, clip: float) -> float:
    """对称 per-tensor INT8 量化 x(截断到 ±clip)后的 SQNR(dB)。"""
    # 量化步长:截断值对应 127
    scale = clip / 127.0
    # 量化:除以步长取整,截断到 int8 范围,再乘回步长
    xq = torch.clamp(torch.round(x / scale), -128, 127) * scale
    # 量化误差的能量
    noise = torch.sum((x - xq) ** 2).item()
    # 误差为零时记无穷大,否则按信号能量与误差能量之比换算成分贝
    return float("inf") if noise == 0 else 10.0 * np.log10(torch.sum(x**2).item() / noise)


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    """单个激活张量([1, tokens, channels])的统计量。"""
    x = x.detach().float().reshape(-1, x.shape[-1])            # [tokens, channels]
    # 逐元素绝对值
    absx = x.abs()
    # 每个通道的最大绝对值
    ch_max = absx.amax(dim=0)
    # 整个张量的最大绝对值
    absmax = float(ch_max.max())
    # 展平成一维
    flat = absx.flatten()
    # 求绝对值的 99.99 分位
    # torch.quantile 对超大张量有元素数上限,用 kthvalue 精确求高分位
    k = max(1, int(round(0.9999 * flat.numel())))
    p9999 = float(flat.kthvalue(k).values)
    # 返回各项统计量
    return {
        "absmax": absmax,
        "p9999": p9999,
        # 通道离群度:最大通道幅值除以各通道幅值的中位数
        "channel_outlier": absmax / float(ch_max.median().clamp_min(1e-12)),
        # 长尾比:最大值除以 99.99 分位
        "tail_ratio": absmax / max(p9999, 1e-12),
        # 以最大值为截断点的 SQNR
        "sqnr_max_db": int8_sqnr(x, absmax),
        # 以 99.99 分位为截断点的 SQNR
        "sqnr_clip_db": int8_sqnr(x, p9999),
        # 幅值最大的 3 个通道的序号
        "top_channels": [int(c) for c in torch.topk(ch_max, 3).indices.tolist()],
    }


def attach_hooks(model: torch.nn.Module, sink: dict) -> list:
    """在每个 block 的 6 类量化点挂 hook,统计结果追加到 sink[(block, kind)]。"""
    # 已注册的 hook 句柄
    handles = []
    # 遍历主干的每个 block
    for i, blk in enumerate(model.model.backbone.encoder.layer):
        # 生成记录函数:把张量统计追加到对应的(block,类型)列表
        def rec(kind: str, block: int = i):
            return lambda t: sink[(block, kind)].append(tensor_stats(t))

        # norm1 的输出,即 q/k/v 投影的输入
        handles.append(blk.norm1.register_forward_hook(lambda m, a, o, r=rec("qkv_in"): r(o)))
        # 注意力输出投影的输入
        handles.append(blk.attention.o_proj.register_forward_pre_hook(lambda m, a, r=rec("o_proj_in"): r(a[0])))
        # norm2 的输入就是"注意力残差相加之后"的残差流
        handles.append(blk.norm2.register_forward_pre_hook(lambda m, a, r=rec("resid_attn"): r(a[0])))
        # norm2 的输出,即 fc1 的输入
        handles.append(blk.norm2.register_forward_hook(lambda m, a, o, r=rec("fc1_in"): r(o)))
        # fc2 的输入,即 GELU 的输出
        handles.append(blk.mlp.fc2.register_forward_pre_hook(lambda m, a, r=rec("fc2_in"): r(a[0])))
        # block 的输出,输出是元组时取第一个元素
        handles.append(blk.register_forward_hook(
            lambda m, a, o, r=rec("resid_out"): r(o[0] if isinstance(o, tuple) else o)))
    return handles


# 入口:在校准图片上统计各量化点的激活分布
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="逐量化点激活统计")
    # 分辨率档位
    parser.add_argument("--res", default="r518", choices=list(RESOLUTIONS))
    parser.add_argument("--num", type=int, default=16, help="calib 前多少张")
    # 解析参数
    args = parser.parse_args()

    # 关闭 TF32
    disable_tf32()
    # 在 GPU 上加载 FP32 模型
    model = load_model("cuda")
    # 取该档位的分辨率
    res = RESOLUTIONS[args.res]
    # 统计结果的容器,键是(block,类型)
    sink: dict = defaultdict(list)
    # 挂上全部 hook
    handles = attach_hooks(model, sink)
    # 关闭梯度记录
    with torch.inference_mode():
        # 逐张校准图片前向,hook 在前向过程中收集统计
        for sample in iter_split("calib", limit=args.num):
            model(torch.from_numpy(preprocess(sample.rgb, res)).cuda())
    # 统计完成后移除全部 hook
    for h in handles:
        h.remove()

    # 逐个量化点的汇总行
    rows = []
    # 按 block 和类型的顺序遍历
    for (block, kind), stats in sorted(sink.items()):
        # 各统计量对图片取平均,通道序号除外
        agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0] if k != "top_channels"}
        agg["absmax"] = float(max(s["absmax"] for s in stats))          # 幅值取跨图最大,其余取均值
        # 每张图 top-3 通道的众数,看离群是否固定在同几个通道(固定 → 可被 SmoothQuant 一类方法吸收)
        tops = [c for s in stats for c in s["top_channels"]]
        # 出现次数最多的 3 个通道
        agg["stable_top_channels"] = [int(c) for c, _ in sorted(
            ((c, tops.count(c)) for c in set(tops)), key=lambda x: -x[1])[:3]]
        # 记录这一行
        rows.append({"block": block, "kind": kind, **agg})

    # 按张量类型的汇总
    by_kind = {}
    # 逐类汇总
    for kind in TENSOR_KINDS:
        # 取出这一类在各 block 的行
        sel = [r for r in rows if r["kind"] == kind]
        # 跨 block 统计:离群度的中位数和最大值、两种 SQNR 的均值、最大幅值
        by_kind[kind] = {
            "channel_outlier_median": float(np.median([r["channel_outlier"] for r in sel])),
            "channel_outlier_max": float(max(r["channel_outlier"] for r in sel)),
            "sqnr_max_db_mean": float(np.mean([r["sqnr_max_db"] for r in sel])),
            "sqnr_clip_db_mean": float(np.mean([r["sqnr_clip_db"] for r in sel])),
            "absmax_max": float(max(r["absmax"] for r in sel)),
        }
        # 打印这一类的摘要
        log.info("%-10s outlier median=%6.1f max=%7.1f | INT8 SQNR max-calib=%5.1fdB clip=%5.1fdB | absmax=%.1f",
                 kind, by_kind[kind]["channel_outlier_median"], by_kind[kind]["channel_outlier_max"],
                 by_kind[kind]["sqnr_max_db_mean"], by_kind[kind]["sqnr_clip_db_mean"], by_kind[kind]["absmax_max"])

    # 结果文件路径
    out = RESULTS_DIR / f"activation_stats_{args.res}.json"
    # 写出汇总与逐点明细
    out.write_text(json.dumps({"res": args.res, "num": args.num, "by_kind": by_kind, "rows": rows}, indent=1))
    # 打印结果文件位置
    log.info("wrote %s", out)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
