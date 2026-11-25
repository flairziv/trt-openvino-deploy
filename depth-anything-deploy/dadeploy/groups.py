"""把 ONNX 节点 / Q-DQ 量化点按"语义层"分组,供敏感度分析与混合精度回退使用。

分组粒度的取舍:按单个节点(~250 个量化点)做留一法太慢且噪声大;按整个 block(12 个)又看不出
"是注意力的问题、MLP 的问题,还是残差流的问题"。这里取"block × 子层类型":

    patch_embed        补丁嵌入卷积
    bNN.qkv            第 NN 个 block 的 q/k/v 三个投影(共享同一个 LayerNorm 输出作为输入)
    bNN.o_proj         注意力输出投影
    bNN.fc1 / bNN.fc2  MLP 两层(fc2 的输入是 GELU 输出,分布形态与其他层明显不同)
    bNN.residual       残差流上的量化点:两个残差 Add 的输入、两个 LayerNorm 的输入。
                       modelopt 默认会在这里插 Q/DQ(为了让 TRT 把"GEMM + 残差相加"融合成 INT8 epilogue),
                       而 DINOv2 的残差流恰恰是少数通道幅值极大的地方 —— 这是本项目的核心嫌疑对象。
    neck.reassemble    DPT 颈部:主干四个阶段输出的 LayerNorm、token 重排回特征图的投影 / 转置卷积
    neck.fusion        DPT 颈部:逐级融合上采样的残差卷积
    head               深度头卷积

注意力内部的两个批量矩阵乘(Q·Kᵀ、softmax·V)在 INT8 模式下 modelopt 不插 Q/DQ(实测 0/24),
所以不单独成组。

节点名来自 TorchScript 导出器(scripts/02_export_onnx.py),modelopt 插入 Q/DQ 时不改原节点名,
因此同一套规则对 FP32 图和 QDQ 图都适用。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

import onnx

# 主干第 N 个 block 的节点名前缀,括号捕获 block 序号
_BLOCK = r"^/model/backbone/encoder/layer\.(\d+)/"
# 节点名规则表:每项是正则和分组名模板,按顺序匹配
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^/model/backbone/embeddings/"), "patch_embed"),
    (re.compile(_BLOCK + r"attention/[qkv]_proj/MatMul$"), "b{:02d}.qkv"),
    (re.compile(_BLOCK + r"attention/o_proj/MatMul$"), "b{:02d}.o_proj"),
    (re.compile(_BLOCK + r"mlp/fc1/MatMul$"), "b{:02d}.fc1"),
    (re.compile(_BLOCK + r"mlp/fc2/MatMul$"), "b{:02d}.fc2"),
    (re.compile(_BLOCK + r"(Add|Add_1|norm1/LayerNormalization|norm2/LayerNormalization)$"), "b{:02d}.residual"),
    (re.compile(r"^/model/backbone/layernorm(_\d+)?/LayerNormalization$"), "neck.reassemble"),
    (re.compile(r"^/model/neck/(reassemble_stage|convs)"), "neck.reassemble"),
    (re.compile(r"^/model/neck/fusion_stage/"), "neck.fusion"),
    (re.compile(r"^/model/head/"), "head"),
)

# 线性层与卷积类的算子类型
COMPUTE_OPS: frozenset[str] = frozenset({"MatMul", "Conv", "ConvTranspose", "Gemm"})


def group_of(node_name: str) -> str | None:
    """返回节点所属的分组名;不属于任何分组(例如 softmax、reshape)时返回 None。"""
    # 按规则表的顺序逐条尝试
    for pattern, template in _RULES:
        # 从节点名开头匹配
        m = pattern.match(node_name)
        # 命中后返回分组名,模板带占位符时填入 block 序号
        if m:
            return template.format(int(m.group(1))) if "{" in template else template
    # 没有规则命中
    return None


def group_sort_key(group: str) -> tuple[int, str]:
    """让分组按网络从前到后的顺序排列:patch_embed → b00..b11 → neck → head。"""
    # patch_embed 排在最前
    if group == "patch_embed":
        return (0, group)
    # 各 block 居中,组内按名字排序
    if group.startswith("b"):
        return (1, group)
    # 颈部和深度头排在最后
    return (2, group)


@dataclass(frozen=True)
class QdqPair:
    """图中的一个"量化点":QuantizeLinear → DequantizeLinear。

    Attributes:
        q: QuantizeLinear 节点名。
        dq: DequantizeLinear 节点名。
        source: 被量化的原始张量名(Q 的输入)。
        target: DQ 输出张量名(下游消费者读的是它)。
        is_weight: 被量化的是否为常量权重(initializer 或 Constant 产出)。
        group: 所属分组(由下游消费者决定)。
    """

    q: str
    dq: str
    source: str
    target: str
    is_weight: bool
    group: str


def find_qdq_pairs(model: onnx.ModelProto) -> list[QdqPair]:
    """枚举 QDQ 图中的全部量化点并分组。

    分组依据是 **DQ 输出的消费者**:同一个张量被量化,是"为了谁"量化的,就归谁。
    例如 LayerNorm1 的输出被量化是为了喂 q/k/v → 归 bNN.qkv;
    残差 Add 的输出被量化是为了喂下一个 LayerNorm / Add → 归 bNN.residual。

    Args:
        model: modelopt 产出的 QDQ ONNX。

    Returns:
        量化点列表。

    Raises:
        ValueError: 有量化点无法归入任何分组(命名规则需要更新)。
    """
    # 张量名到产生它的节点的映射
    producers = {o: n for n in model.graph.node for o in n.output}
    # 张量名到消费它的节点列表的映射
    consumers: dict[str, list[onnx.NodeProto]] = {}
    # 遍历每个节点的每个输入,登记消费关系
    for node in model.graph.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    # 全部 initializer 的名字
    initializers = {i.name for i in model.graph.initializer}

    # 已归组的量化点
    pairs: list[QdqPair] = []
    # 无法归入单一分组的量化点
    orphans: list[str] = []
    # 以每个 DequantizeLinear 为起点查找量化点
    for dq in model.graph.node:
        # 跳过其他类型的节点
        if dq.op_type != "DequantizeLinear":
            continue
        # 找到产生 DQ 输入的节点
        q = producers.get(dq.input[0])
        # 上游不是 QuantizeLinear 的不算量化点
        if q is None or q.op_type != "QuantizeLinear":
            continue                      # dq_only 形式的权重(INT8 initializer → DQ),不在 Q-DQ 对之列
        # 被量化的原始张量
        src = q.input[0]
        # 产生原始张量的节点,initializer 没有上游节点
        src_producer = producers.get(src)
        # 原始张量是 initializer 或 Constant 的输出时视为权重
        is_weight = src in initializers or (src_producer is not None and src_producer.op_type == "Constant")
        # 由 DQ 输出的各个消费者推出所属分组
        groups = {g for c in consumers.get(dq.output[0], []) if (g := group_of(c.name)) is not None}
        # 其中属于残差流的分组
        residual = {g for g in groups if g.endswith(".residual")}
        if len(groups) > 1 and len(residual) == 1:
            # 阶段输出处(block 3/6/9)残差流同时喂下一个 block 与颈部 LayerNorm:它首先是残差流,归残差组
            groups = residual
        # 仍不能归入唯一分组的记入孤儿列表
        if len(groups) != 1:
            orphans.append(f"{q.name} -> {sorted(groups) or 'no group'}")
            continue
        # 登记一个量化点
        pairs.append(QdqPair(q.name, dq.name, src, dq.output[0], is_weight, groups.pop()))
    # 存在孤儿时报错,提示更新命名规则
    if orphans:
        raise ValueError(f"{len(orphans)} QDQ pairs not attributable to a single group, e.g. {orphans[:3]}")
    return pairs


def pairs_by_group(pairs: list[QdqPair]) -> "OrderedDict[str, list[QdqPair]]":
    """按网络顺序把量化点聚合到分组。"""
    # 保持插入顺序的分组字典
    out: OrderedDict[str, list[QdqPair]] = OrderedDict()
    # 按网络前后顺序遍历量化点
    for p in sorted(pairs, key=lambda p: group_sort_key(p.group)):
        # 追加到所属分组
        out.setdefault(p.group, []).append(p)
    return out


def owner_nodes(model: onnx.ModelProto, groups: set[str]) -> list[str]:
    """返回属于这些分组的全部节点名 —— 传给 modelopt `nodes_to_exclude`,让它们的输入不再插 Q/DQ。"""
    # 筛出分组名在给定集合里的节点
    return [n.name for n in model.graph.node if group_of(n.name) in groups]


def exclude_patterns(node_names: list[str]) -> list[str]:
    """把节点名转成 modelopt `nodes_to_exclude` 需要的精确匹配正则。"""
    # 转义节点名并加上首尾锚定
    return [f"^{re.escape(n)}$" for n in node_names]


def bypass_pairs(model: onnx.ModelProto, pairs: list[QdqPair]) -> onnx.ModelProto:
    """"拆掉"一批量化点:下游消费者改读 Q 的原始输入,Q/DQ 节点删除。

    得到的图在这些位置等价于"不量化、保持高精度(FP16)",其余量化点不变 ——
    这就是在真实部署图上做"去掉某组量化,误差能减少多少"的留一法实验。

    Args:
        model: QDQ 模型(不会被修改)。
        pairs: 要拆掉的量化点。

    Returns:
        新的 ModelProto。
    """
    # 只改连线不重新校准:其余量化点的 scale 保持不变,实验之间的差异才能归因到"这组量化与否"。
    # dq_only=False 的前提很关键:权重以浮点 + Q→DQ 形式存放,拆掉后读到的就是原始权重;
    # 若权重已存成 INT8 initializer,这里拆不回精度(find_qdq_pairs 也不会把它算作量化点)。
    # 复制一份模型,不改动入参
    new = onnx.ModelProto()
    new.CopyFrom(model)
    rewire = {p.target: p.source for p in pairs}     # DQ 输出名 → Q 输入名;一个 DQ 的所有消费者一起改
    # 要删除的 Q 和 DQ 节点名
    drop = {p.q for p in pairs} | {p.dq for p in pairs}
    # 保留其余节点
    kept = [n for n in new.graph.node if n.name not in drop]
    # 把指向已删除 DQ 输出的输入改接到 Q 的原始输入
    for node in kept:
        for idx, name in enumerate(node.input):
            if name in rewire:
                node.input[idx] = rewire[name]
    # 用保留下来的节点替换图里的节点列表
    del new.graph.node[:]
    new.graph.node.extend(kept)
    return new
