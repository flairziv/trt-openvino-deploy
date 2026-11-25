"""P4 CPU 栈压缩:NNCF 对 OpenVINO IR 做权重压缩 / 静态量化,产出 OpenVINO GenAI 可直接加载的目录。

用法(env: ovgenai;Makefile 已封装,`make p4-compress` 依次跑完全部配方):
    python scripts/13_compress_cpu.py --collect              # 第 1 步:收集校准输入,落盘(只需一次)
    python scripts/13_compress_cpu.py --recipe int4awq       # 第 2 步:按配方压缩

所有配方都从同一份 FP16 基础 IR($VLM_ARTIFACTS/openvino/fp16,P0 导出)出发,只改动配方涉及的子模型,
其余子模型硬链接过去 —— 配方之间的差异只来自压缩本身(见 03_smoke_cpu.py 文档)。

Qwen2-VL 的 IR 由四个子模型组成,体积与压缩处理:
    language_model          3.1 GB   语言模型 28 层;权重压缩的主战场
    vision_embeddings_merger 1.3 GB  视觉塔 32 层 transformer + 2×2 合并 MLP;视觉段算力的绝大部分
    text_embeddings         0.47 GB  词表 embedding 查表(gather,不是矩阵乘);保持 FP16
    vision_embeddings       3 MB     patch embed(3D 卷积);太小,保持 FP16

配方(问题 4:CPU 上的杠杆在哪):
    int8        语言模型 INT8 非对称、逐通道,无需数据
    int4        语言模型 INT4 非对称、group 128,无需数据
    int4awq     同 int4,另用校准数据做 AWQ(按激活幅度给权重通道缩放后再量化)
    int4-vis8w  int4 + 视觉塔 INT8 仅权重(计算仍是 FP32,省带宽与内存;是否提速取决于 CPU 插件的动态量化)
    int4-vis8s  int4 + 视觉塔 INT8 静态量化(权重与激活都 INT8,走 VNNI 整数矩阵乘;需校准数据)

校准数据(calib 划分,与 test / dev 互不重叠):
    在 FP16 IR 上用 optimum-intel 跑 NUM_CALIB 道带图题(t256),在两个位置挂钩子截获真实输入:
    视觉塔 merger 的输入(patch 特征 + 旋转位置编码)、语言模型 prefill 的输入(拼好图像 embedding 的 inputs_embeds、
    3 维 mrope 位置等)。截获的是推理时真正送进子模型的张量,不需要自己重写 Qwen2-VL 的预处理与位置计算。
    收集时**不编译语言模型**:模型以 compile=False 加载(各子模型首次调用时才编译),语言模型的 forward 被换成
    "调用 prepare_inputs 拿到真实输入 → 记录 → 抛异常中止本次 generate"。语言模型按 FP32 编译要 6 GB 以上,
    而校准只需要它的输入,不需要它的输出;视觉塔与文本 embedding 照常编译运行。
    为什么分两步:收集与压缩分在两个进程,内存峰值不叠加(WSL 只有 8 GB),中间只经过磁盘上约 300 MB 的 npz。

产物:
    $VLM_ARTIFACTS/openvino/calib/*.npz            校准输入
    $VLM_ARTIFACTS/openvino/<recipe>/              配方目录(GenAI 直接加载)
    results/compress/<recipe>.json                 配方记录(参数、各子模型体积、耗时、峰值内存)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.backends import CPU_THREADS
from vlmdeploy.config import OV_DIR, RESULTS_DIR, TOKEN_BUDGETS
from vlmdeploy.imaging import resize_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.tasks import iter_split

# 模块级 logger
log = get_logger("compress_cpu")
# FP16 基础 IR 的目录
FP16_DIR = OV_DIR / "fp16"
# 校准输入的目录
CALIB_DIR = OV_DIR / "calib"
CALIB_BUDGET = "t256"     # CPU 栈的主档位(4 核 CPU 上 t576 的视觉段已要十几秒,端侧不会用)
# 校准题数
NUM_CALIB = 64
# 要压缩的两个子模型的文件名主干(各有一对 .xml / .bin)
LM = "openvino_language_model"
MERGER = "openvino_vision_embeddings_merger_model"

# 配方表:lm = 语言模型怎么压,vision = 视觉塔怎么压(没有 vision 键就不动视觉塔)
RECIPES: dict[str, dict] = {
    "int8": {"lm": "int8"},
    "int4": {"lm": "int4"},
    "int4awq": {"lm": "int4awq"},
    "int4-vis8w": {"lm": "int4", "vision": "int8w"},
    "int4-vis8s": {"lm": "int4", "vision": "int8s"},
}


# 内存是这个脚本最紧张的资源:收集、统计、压缩每一步都把峰值记进日志或配方记录
def peak_rss_gb() -> float:
    """本进程的峰值常驻内存(GiB)。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20   # Linux 上 ru_maxrss 单位是 KB


# ---------------------------------------------------------------------------
# 第 1 步:收集校准输入
# ---------------------------------------------------------------------------
class _PrefillCaptured(Exception):
    """语言模型的 prefill 输入已记录,中止本次 generate(不需要语言模型真的算)。"""


def collect() -> None:
    """在 FP16 IR 上跑 calib 题,截获视觉塔与语言模型 prefill 的真实输入并落盘。"""
    # 重依赖只在这里导入
    import numpy as np
    from optimum.intel import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    # 加载处理器
    processor = AutoProcessor.from_pretrained(FP16_DIR)
    # 加载 FP16 IR,不立即编译,限定 CPU 线程数
    model = OVModelForVisualCausalLM.from_pretrained(FP16_DIR, device="CPU", compile=False,
                                                     ov_config={"INFERENCE_NUM_THREADS": str(CPU_THREADS)})
    # 截获的视觉塔输入和语言模型输入
    merger_inputs, lm_inputs = [], []

    # 视觉塔子模型
    merger = model.vision_embeddings_merger
    # 留住原来的 forward:钩子只负责记录输入,视觉塔本身照常算
    merger_forward = merger.forward

    # 视觉塔的钩子:记录输入后调用原来的 forward
    def merger_hook(pixel_values, **kwargs):  # noqa: ANN001, ANN003, ANN202
        # attention_mask 不存:单张图时它是 [1, N, N] 的全零(块对角只有一块),压缩时按 N 重建,省下 N² 的体积
        merger_inputs.append({"hidden_states": np.asarray(pixel_values, dtype=np.float32),
                              "rotary_pos_emb": np.asarray(kwargs["rotary_pos_emb"], dtype=np.float32)})
        # 照常计算视觉塔
        return merger_forward(pixel_values, **kwargs)

    # 换上钩子
    merger.forward = merger_hook

    # 语言模型子模型
    lm = model.language_model

    # 语言模型的替身:记录 prefill 输入后中止本次生成
    def lm_forward(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        # prepare_inputs 在请求未编译时不碰 infer request(见 optimum 的 OVModelWithEmbedForCausalLM.prepare_inputs),
        # 只用到文本 embedding 子模型 —— 拿到的就是编译后真正会送进语言模型的
        # inputs_embeds / 3 维 mrope 位置 / beam_idx。
        inputs = lm.prepare_inputs(*args, **kwargs)
        # 转成 numpy 数组后保存
        lm_inputs.append({k: np.asarray(v) for k, v in inputs.items()})
        # 抛异常中止本次 generate
        raise _PrefillCaptured

    # 换上替身
    lm.forward = lm_forward

    # 计时起点
    t0 = time.time()
    # 逐题读取 calib 划分
    for sample in iter_split("calib", limit=NUM_CALIB):
        # 图片缩放到校准档位
        img = resize_to_budget(sample.image(), TOKEN_BUDGETS[CALIB_BUDGET])
        # 一条 user 消息:图片在前,文字在后
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample.prompt}]}]
        # 套聊天模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # generate 走到语言模型时会被 lm_forward 抛出的 _PrefillCaptured 打断 —— 输入已经截到,接着跑下一题
        try:
            model.generate(**processor(text=[text], images=[img], return_tensors="pt"), max_new_tokens=1)
        except _PrefillCaptured:
            pass
    # 创建校准输入目录
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    # 两类输入各存一个 npz,键名是 "样本序号/字段名",load_calib 按它还原成逐样本的字典
    for name, items in (("merger", merger_inputs), ("lm", lm_inputs)):
        np.savez(CALIB_DIR / f"{name}.npz", **{f"{i}/{k}": v for i, d in enumerate(items) for k, v in d.items()})
    # 打印截获的样本数、耗时和峰值内存
    log.info("collected %d merger / %d lm inputs in %.0fs, peak RSS %.2f GiB", len(merger_inputs), len(lm_inputs),
             time.time() - t0, peak_rss_gb())


def load_calib(name: str) -> list[dict]:
    """读回 collect 落盘的校准输入,还原成逐样本的字典列表。"""
    # numpy 只在这里导入
    import numpy as np

    # 读入 npz
    data = np.load(CALIB_DIR / f"{name}.npz")
    # 样本序号到字段字典的映射
    items: dict[int, dict] = {}
    # 逐个键还原
    for key in data.files:
        # 拆出样本序号和字段名
        i, field = key.split("/", 1)
        # 放回对应的样本
        items.setdefault(int(i), {})[field] = data[key]
    # 按样本序号排序后返回
    return [items[i] for i in sorted(items)]


# ---------------------------------------------------------------------------
# 第 2 步:按配方压缩
# ---------------------------------------------------------------------------
def compress_lm(model, how: str):  # noqa: ANN001, ANN201
    """按配方压缩语言模型(INT8 / INT4 / INT4 + AWQ),返回压缩后的 ov.Model。"""
    # nncf 只在这里导入
    import nncf

    # INT8 非对称,不需要数据
    if how == "int8":
        return nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    # INT4 非对称,不需要数据
    if how == "int4":
        # ratio=1.0:全部可压的层都进 INT4(默认 ratio 会按敏感度留一部分在 INT8,混进来就分不清是谁的功劳)。
        # all_layers 保持默认:嵌入与输出层留在 INT8 —— 语言模型的输出层与输入 embedding 共享权重,对误差最敏感。
        return nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=1.0)
    # INT4 加 AWQ,需要校准数据
    if how == "int4awq":
        # 读入语言模型的校准输入
        calib = load_calib("lm")

        # 把一条校准样本转成模型输入
        def transform(item: dict) -> dict:
            # 有状态模型(KV cache 存在模型内部状态里):每个样本都是一次独立的 prefill,beam_idx 恒为 [0]
            # 截获的字段里只保留模型真正有的输入名
            return {k: v for k, v in item.items() if k in {i.get_any_name() for i in model.inputs}}

        # 用全部校准样本做 AWQ 后压缩
        return nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=1.0,
                                     dataset=nncf.Dataset(calib, transform), awq=True, subset_size=len(calib))
    # 未知的压缩方式
    raise ValueError(how)


def compress_vision(model, how: str):  # noqa: ANN001, ANN201
    """按配方压缩视觉塔(INT8 仅权重 / INT8 静态量化),返回压缩后的 ov.Model。"""
    # nncf 和 numpy 只在这里导入
    import nncf
    import numpy as np

    if how == "int8w":
        # 仅权重:和语言模型的 INT8 一样不需要校准数据
        return nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    # INT8 静态量化,需要校准数据
    if how == "int8s":
        # 读入视觉塔的校准输入
        calib = load_calib("merger")

        # 把一条校准样本转成模型输入,补上注意力掩码
        def transform(item: dict) -> dict:
            n = item["hidden_states"].shape[0]  # 合并前的块数 N
            # 单张图:注意力掩码是 [1, N, N] 全零(加性掩码,0 = 不屏蔽)
            return {"hidden_states": item["hidden_states"], "rotary_pos_emb": item["rotary_pos_emb"],
                    "attention_mask": np.zeros((1, n, n), dtype=np.float32)}

        # 分批收集统计后量化
        return quantize_in_batches(model, nncf.Dataset(calib, transform), len(calib))
    # 未知的压缩方式
    raise ValueError(how)


STAT_NODES_PER_PASS = 40   # 每次统计收集最多挂多少个目标节点,见 quantize_in_batches


def quantize_in_batches(model, dataset, subset_size: int):  # noqa: ANN001, ANN201
    """与 nncf.quantize(model_type=TRANSFORMER, preset=MIXED) 同一条流水线,只把统计收集拆成多批。

    为什么不直接调 nncf.quantize:它每一步把该步需要的统计点一次性全挂到模型上,再编译这个"带额外输出的模型"。
    视觉塔第 2 步(MinMax + FastBiasCorrection)有 584 个统计点 / 324 个目标节点,编译出来的统计模型在 8 GB 的
    WSL 里放不下 —— 实测只喂 1 条样本,进程 RSS 也在 15 秒内涨到 7.4 GiB 并开始换页,两次把整个 WSL 虚拟机
    带进 OOM(内核日志:Out of memory: Killed process ... anon-rss:7632864kB)。所以与校准样本数无关,减样本没用。
    按目标节点分批后,每批的峰值实测为 5.30 / 6.31 / 7.16 GiB(每批 40 / 80 / 160 个节点),取 40。

    分批不改变结果:每个统计点的收集器只依赖它自己那条边上的张量,各批跑的是同一个模型、同一批样本;
    同一个目标节点上的统计点留在同一批(NNCF 会把同一条边上的收集器合并共享)。代价是多编译几次模型、
    每条样本多推理几遍。
    """
    # nncf 只在这里导入
    import nncf

    # 以下用到的是 NNCF 的内部函数(不在公开 API 里):把 nncf.quantize 的流水线拆开,自己编排统计收集
    from nncf.common.factory import build_graph
    from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
    from nncf.openvino.quantization.quantize_model import (
        compress_quantize_weights_transformation,
        dump_parameters,
        remove_friendly_name_duplicates,
    )
    from nncf.quantization.algorithms.pipeline import collect_statistics
    from nncf.quantization.algorithms.post_training.pipeline import create_ptq_pipeline

    # TRANSFORMER 模型类型:NNCF 会跳过对 INT8 不友好的位置(LayerNorm 前后、softmax 输入等),
    # 并对 MatMul 激活做 SmoothQuant —— 与项目 A 在 ViT 上验证有效的配置一致。
    params = {"preset": nncf.QuantizationPreset.MIXED, "subset_size": subset_size,
              "model_type": nncf.ModelType.TRANSFORMER}
    # 创建训练后量化流水线
    pipeline = create_ptq_pipeline(**params)
    # 去掉重复的节点名
    model = remove_friendly_name_duplicates(model)
    # 逐步执行流水线
    for step, algorithms in enumerate(pipeline.pipeline_steps):
        # 构建当前模型的计算图
        graph = build_graph(model)
        # 这一步需要的全部统计点
        points = pipeline.get_statistic_points_for_step(step, model, graph)
        # points.data 以目标节点名为键:按节点切批,同一个节点上的统计点自然留在同一批
        names = list(points.data)
        # 按每批的节点数上限切分
        batches = [names[i:i + STAT_NODES_PER_PASS] for i in range(0, len(names), STAT_NODES_PER_PASS)]
        # 打印步骤、算法名、目标节点数和批数
        log.info("step %d (%s): %d target nodes in %d passes", step, " + ".join(type(a).__name__ for a in algorithms),
                 len(names), len(batches))
        # 逐批收集统计
        for batch in batches:
            # 本批的统计点容器
            part = StatisticPointsContainer()
            # 放入本批节点的统计点
            for name in batch:
                part.data[name] = points.data[name]
            collect_statistics(part, model, graph, dataset)   # 收集器是共享对象:统计值直接落在 points 里
        # 打印峰值内存
        log.info("step %d statistics done, peak RSS %.2f GiB", step, peak_rss_gb())
        # 各批统计收齐后,用完整的 points(不是某一批的 part)执行这一步的算法
        model = pipeline.run_step(step, points, model, graph)
    # 以下与 nncf.quantize 的收尾相同:把权重上的 FakeQuantize 折叠成 INT8 常量,并在模型元信息里记下参数
    compress_quantize_weights_transformation(model)
    dump_parameters(model, {"preset": params["preset"], "subset_size": subset_size, "model_type": params["model_type"],
                            "statistics_nodes_per_pass": STAT_NODES_PER_PASS})
    return model


def link_or_copy(src: Path, dst: Path) -> None:
    """未改动的文件硬链接过去(D 盘 NTFS 支持),失败再复制。"""
    # 先尝试硬链接
    try:
        os.link(src, dst)
    # 硬链接失败时改为复制
    except OSError:
        shutil.copy2(src, dst)


def build(recipe: str) -> None:
    """生成一个配方目录:压缩涉及的子模型,其余文件硬链接,最后写配方记录。"""
    # openvino 只在这里导入
    import openvino as ov

    # 配方内容
    spec = RECIPES[recipe]
    # 配方的输出目录
    out = OV_DIR / recipe
    # 临时目录,写完后改名
    tmp = OV_DIR / f".{recipe}.tmp"
    # 清掉上次留下的临时目录
    if tmp.exists():
        shutil.rmtree(tmp)
    # 创建临时目录
    tmp.mkdir(parents=True)
    # OpenVINO 入口对象
    core = ov.Core()
    # 被替换的子模型及其压缩耗时
    replaced: dict[str, float] = {}
    # 总计时起点
    t_total = time.time()
    # 语言模型:组合配方(int4-vis*)复用 int4 配方已经压好的 IR,不重复压缩
    if spec["lm"] != recipe and (OV_DIR / spec["lm"] / f"{LM}.xml").exists():
        # 链接已压好的 xml 和 bin
        for ext in (".xml", ".bin"):
            link_or_copy(OV_DIR / spec["lm"] / f"{LM}{ext}", tmp / f"{LM}{ext}")
        # 复用时耗时记为 0
        replaced[LM] = 0.0
    else:
        # 计时起点
        t0 = time.time()
        # 读入 FP16 语言模型并压缩
        model = compress_lm(core.read_model(FP16_DIR / f"{LM}.xml"), spec["lm"])
        # 保存 IR,未压缩的浮点权重存为 FP16
        ov.save_model(model, tmp / f"{LM}.xml", compress_to_fp16=True)
        # 释放模型占用的内存
        del model
        # 记录压缩耗时
        replaced[LM] = round(time.time() - t0, 1)
    # 配方涉及视觉塔时压缩视觉塔
    if "vision" in spec:
        # 计时起点
        t0 = time.time()
        # 读入 FP16 视觉塔并压缩
        model = compress_vision(core.read_model(FP16_DIR / f"{MERGER}.xml"), spec["vision"])
        # 保存 IR
        ov.save_model(model, tmp / f"{MERGER}.xml", compress_to_fp16=True)
        # 释放模型占用的内存
        del model
        # 记录压缩耗时
        replaced[MERGER] = round(time.time() - t0, 1)
    # 其余文件(另外两个子模型、分词器、配置等)原样硬链接;上面已经写出的子模型不覆盖
    for f in FP16_DIR.iterdir():
        if f.is_file() and not (tmp / f.name).exists():
            link_or_copy(f, tmp / f.name)
    # 已有同名配方目录时先删除
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)   # 整个目录写完才改名:中途失败不会留下"看起来完整"的半成品目录

    # 子模型 bin 文件的字节数
    def size(stem: str) -> int:
        return (out / f"{stem}.bin").stat().st_size

    # 配方记录:配方、内容、目录、各子模型耗时、总耗时、峰值内存、各子模型体积和时间
    record = {"recipe": recipe, "spec": spec, "dir": str(out), "seconds": replaced,
              "total_seconds": round(time.time() - t_total, 1), "peak_rss_gib": round(peak_rss_gb(), 2),
              "bytes": {stem: size(stem) for stem in (LM, MERGER, "openvino_text_embeddings_model",
                                                       "openvino_vision_embeddings_model")},
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    # 创建记录目录
    (RESULTS_DIR / "compress").mkdir(parents=True, exist_ok=True)
    # 写出配方记录
    (RESULTS_DIR / "compress" / f"{recipe}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    # 打印输出目录、两个子模型的体积和峰值内存
    log.info("%s -> %s, LM %.2f GB, merger %.2f GB, peak RSS %.2f GiB", recipe, out, record["bytes"][LM] / 1e9,
             record["bytes"][MERGER] / 1e9, record["peak_rss_gib"])


def main() -> int:
    """入口:--collect 收集校准输入,--recipe 按配方压缩。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="P4 NNCF 压缩")
    # 两个选项互斥且必须给一个
    group = parser.add_mutually_exclusive_group(required=True)
    # 收集校准输入
    group.add_argument("--collect", action="store_true")
    # 按配方压缩
    group.add_argument("--recipe", choices=tuple(RECIPES))
    # 解析参数
    args = parser.parse_args()
    # 按选项执行对应的步骤
    if args.collect:
        collect()
    else:
        build(args.recipe)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
