"""P5 视觉塔 TensorRT 引擎:导出 ONNX → 强类型 FP16 引擎 → 与 HF BF16 视觉塔对拍 → 计时。

触发条件(README 问题 1):P2 证实视觉塔占 TTFT 超过 40%。

用法(env: trtllm;Makefile 已封装):
    python scripts/15_vision_trt.py --budget s576      # 方图档位 s256 / s576 / s1024,见 imaging.square_to_budget

本脚本只做"孤立的视觉塔":对拍 + GPU 时间。端到端(引擎接回 TensorRT-LLM,TTFT 实测)由
sitecustomize 的第 4 个补丁 + `11_bench_gpu.py --variant trtllm-bf16-vistrt --square` 完成,
精度由 `10_eval.py --variant trtllm-bf16-vistrt --budget s576` 与 trtllm-bf16 逐题配对。

做法沿用项目 A 在 Depth Anything(纯 ViT)上验证过的四条:
    1. **静态形状**:每个 token 档位单独建一个引擎。视觉塔输入是"合并前的块",N = 4 × 图像 token 数,
       形状随图片长宽比变化 —— 本脚本按方图(s576 → 24×24 token → 48×48 块 = 2304 块)建引擎,
       文件名 vision_g48x48.engine 按网格命名,运行期按请求的 image_grid_thw 查表;非方图只需按 (h, w) 再建一个。
    2. **常量折叠位置编码**:Qwen2-VL 用 2D 旋转位置编码(按块的行列号计算 cos / sin),对固定的 (h, w) 是常量。
       导出时把 grid_thw 固定成 Python 常量,cos / sin 与块对角注意力掩码(单张图时就是全局注意力)在 trace 时
       直接算成常量,引擎里不再有这段整型运算。
    3. **强类型 FP16**:ONNX 里的 dtype 就是 FP16(模型先 .half() 再导出),构建时用 STRONGLY_TYPED,
       不用"弱类型 + FP16 flag"—— 项目 A 实测后者的层精度选择会随计时漂移,两次构建结果不同。
       视觉塔里 LayerNorm 与 softmax 在 FP16 下是否溢出,由对拍数字说话。
    4. **确认融合 MHA**:构建后用 inspector 统计引擎里的层名,看注意力是否被融合成一个 kernel
       (项目 A 发现 INT8 下融合 MHA 反而更慢,这里只做 FP16)。

对拍:同一批 dev 图片,以 FP32 视觉塔(同一份权重升到 FP32)为真值,分别算 HF BF16 视觉塔(TensorRT-LLM 当前
实际在用的实现)和 FP16 引擎的余弦相似度与相对误差 —— 引擎只要不比 BF16 现状差,就可以无损替换;
另报引擎与 BF16 之间的直接差异。计时:引擎 enqueue 的 GPU 时间(CUDA event,交错轮次)vs HF 视觉塔前向。

产物:
    $VLM_ARTIFACTS/vision_trt/vision_g{H}x{W}.{onnx,engine}
    results/vision_trt_<budget>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import MODEL_DIR, RESULTS_DIR, SQUARE_BUDGETS
from vlmdeploy.imaging import square_side_tokens, square_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.tasks import iter_split
from vlmdeploy.timing import interleaved_schedule, summarize
from vlmdeploy.variants import VISION_ENGINE_DIR

# 模块级 logger
log = get_logger("vision_trt")
# ONNX 和引擎的输出目录
OUT_DIR = VISION_ENGINE_DIR
NUM_PARITY = 16       # 对拍用的 dev 图片数
ROUNDS = 5            # 计时的交错轮数(另有 1 轮预热,丢弃)
REPEATS = 10          # 每轮每个实现连续测 10 次取中位数
WORKSPACE = 2 << 30   # 2 GiB:8 GB 卡,不设上限时 TensorRT 会尝试吃满显存


class FixedGridVision:
    """把 HF 视觉塔包成"输入只有 pixel_values"的模块,grid_thw 在构造时固定(→ 导出时常量折叠)。"""

    @staticmethod
    def build(visual, side_tokens: int):  # noqa: ANN001, ANN205
        """返回只接受 pixel_values 的包装模块;side_tokens 是方图每边的图像 token 数。"""
        # torch 只在这里导入
        import torch

        side = side_tokens * 2   # 合并前每边的块数
        # grid_thw = (帧数, 高, 宽),以合并前的块为单位;单张图是 1 帧
        grid = torch.tensor([[1, side, side]], dtype=torch.long, device="cuda")

        # 包装模块:只接受 pixel_values,网格固定为上面的 grid
        class Wrapper(torch.nn.Module):
            # 构造函数:保存视觉塔
            def __init__(self) -> None:
                super().__init__()
                self.visual = visual

            # 前向:按固定网格调用视觉塔
            def forward(self, pixel_values):  # noqa: ANN001, ANN202
                out = self.visual(pixel_values, grid_thw=grid)
                # pooler_output 是 2×2 合并之后的图像 embedding;拿到的已经是张量时原样返回
                return getattr(out, "pooler_output", out)   # transformers 5.x 返回 BaseModelOutputWithPooling

        # 返回推理模式的包装模块
        return Wrapper().eval()


def square_inputs(processor, budget: str) -> list:  # noqa: ANN001
    """取 dev 集图片,中心裁成方图、缩放到该档位的方形尺寸,经 HF 处理器得到 pixel_values。"""
    # 各图的 pixel_values 和已经用过的图片 id
    out, seen = [], set()
    # 按固定顺序读 dev 集
    for sample in iter_split("dev"):
        # 同一张图只用一次
        if sample.image_id in seen:
            continue
        # 记下这张图
        seen.add(sample.image_id)
        # 裁成方图并缩放到档位尺寸
        img = square_to_budget(sample.image(), SQUARE_BUDGETS[budget])
        # 只做图像预处理
        enc = processor.image_processor(images=[img], return_tensors="pt")
        # 保存 pixel_values
        out.append(enc["pixel_values"])
        # 取够对拍用的图片数后结束
        if len(out) >= NUM_PARITY:
            break
    return out


def export_onnx(wrapper, example, path: Path) -> None:  # noqa: ANN001
    """把固定网格的视觉塔导出成 ONNX(导出期间绕开 transformers 5.x 的 enable_gqa 分支)。"""
    # 导出期间要临时修改的 transformers 模块
    from transformers.integrations import sdpa_attention

    # 创建输出目录
    path.parent.mkdir(parents=True, exist_ok=True)
    # transformers 5.x 只要模块带 num_key_value_groups(视觉塔恒为 1)且没有 mask,就给 SDPA 传
    # enable_gqa=True;TorchScript 导出器对这个参数直接断言失败("not implemented if enable_gqa is True")。
    # 导出期间关掉这条分支:改走 repeat_kv(key, 1),它在 n_rep == 1 时原样返回,数值上是同一个计算。
    use_gqa = sdpa_attention.use_gqa_in_sdpa
    # 换成恒返回 False 的函数,原函数留在 use_gqa 里
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
    try:
        # 执行导出
        _export(wrapper, example, path)
    finally:
        # 无论导出成败都恢复原函数:对库的模块级修改一定要还原
        sdpa_attention.use_gqa_in_sdpa = use_gqa


# 实际的导出调用
def _export(wrapper, example, path: Path) -> None:  # noqa: ANN001
    # torch 只在这里导入
    import torch

    # 关闭梯度记录
    with torch.inference_mode():
        # dynamo=False 即 TorchScript 导出器;输入 / 输出名 pixel_values / embeds 就是 TrtRunner 绑定地址时用的张量名
        torch.onnx.export(wrapper, (example,), str(path), input_names=["pixel_values"], output_names=["embeds"],
                          opset_version=18, dynamo=False, do_constant_folding=True)


def build_engine(onnx_path: Path, engine_path: Path) -> dict:
    """用强类型网络构建引擎并落盘;返回构建耗时、体积与融合注意力层数。"""
    # tensorrt 只在这里导入
    import tensorrt as trt

    # TensorRT 日志,只输出警告及以上
    logger = trt.Logger(trt.Logger.WARNING)
    # 构建器
    builder = trt.Builder(logger)
    # 强类型网络,精度由 ONNX 里的张量类型决定
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    # ONNX 解析器
    parser = trt.OnnxParser(network, logger)
    # 解析 ONNX,失败时汇总错误信息后报错
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError("ONNX parse failed: " + errors)
    # 构建配置
    config = builder.create_builder_config()
    # 限制工作区内存
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    # DETAILED:引擎里保留每层的 tactic / kernel 名,下面靠它确认注意力有没有被融合
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # 构建计时起点
    t0 = time.time()
    # 构建并序列化引擎
    blob = builder.build_serialized_network(network, config)
    # 构建失败时报错
    if blob is None:
        raise RuntimeError("engine build failed")
    # 引擎落盘
    engine_path.write_bytes(bytes(blob))
    # 反序列化引擎,用于读取层信息
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    # 用 inspector 导出各层信息
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    # 层列表
    layers = info.get("Layers", [])
    # 融合注意力在层信息里的痕迹:Myelin 编译的融合区域(ForeignNode)里出现 mha / fmha 一类的 kernel 名。
    # 层名本身常常只是"{ForeignNode[...]}",所以在整条层信息(含 tactic / kernel 名)里找。
    dumps = [json.dumps(layer).lower() for layer in layers]
    # 含 mha 字样的层
    mha = [d for d in dumps if "mha" in d]
    # 返回构建耗时、引擎体积、层数、Myelin 区域数、融合注意力层数和示例
    return {"build_seconds": round(time.time() - t0, 1), "engine_mb": round(blob.nbytes / 2**20, 1),
            "num_layers": len(layers), "myelin_regions": sum("foreignnode" in d for d in dumps),
            "fused_attention_layers": len(mha), "attention_layer_examples": [d[:160] for d in mha[:2]]}


class TrtRunner:
    """最小 TensorRT 执行器:单输入单输出,预分配输出。

    在 PyTorch 的当前流上执行:计时用的 CUDA event 也记在当前流上,两者同一条流,event 之间恰好框住引擎的 kernel。
    (若引擎跑在另一条流上再同步,event 量到的是"主机等待"的时间,混进了 Python 开销。)
    """

    # 构造函数:加载引擎,创建执行上下文,预分配输出
    def __init__(self, engine_path: Path) -> None:
        # tensorrt 和 torch 只在这里导入
        import tensorrt as trt
        import torch

        # 保存 torch 模块,供 __call__ 使用
        self.torch = torch
        # 反序列化引擎
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.read_bytes())
        # 执行上下文
        self.ctx = self.engine.create_execution_context()
        # 输出张量的形状
        out_shape = tuple(self.engine.get_tensor_shape("embeds"))
        # 在 GPU 上预分配 FP16 输出
        self.out = torch.empty(out_shape, dtype=torch.float16, device="cuda")

    def __call__(self, x):  # noqa: ANN001, ANN204
        """执行一次推理,返回预分配的输出张量(下次调用会被覆盖)。"""
        # 把输入 / 输出张量的显存地址直接交给引擎,不经过主机内存
        self.ctx.set_tensor_address("pixel_values", x.data_ptr())
        self.ctx.set_tensor_address("embeds", self.out.data_ptr())
        # 在 PyTorch 的当前流上执行,失败时报错
        if not self.ctx.execute_async_v3(self.torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("engine execution failed")
        return self.out


def gpu_ms(fn, x, repeats: int) -> float:  # noqa: ANN001
    """连续 repeats 次的 GPU 时间中位数(CUDA event,包含 kernel 而不含 Python 侧的输入准备)。"""
    # torch 只在这里导入
    import torch

    # 各次的 GPU 时间
    times = []
    # 连续测 repeats 次
    for _ in range(repeats):
        # 起止两个 CUDA 事件
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 前后各同步一次:先清空之前排队的 kernel,事后等 end 事件完成才能读 elapsed_time(毫秒)
        torch.cuda.synchronize()
        # 记录起点
        start.record()
        # 执行被测函数
        fn(x)
        # 记录终点
        end.record()
        torch.cuda.synchronize()
        # 两个事件之间的 GPU 时间
        times.append(start.elapsed_time(end))
    # 返回中位数
    return statistics.median(times)


def main() -> int:
    """入口:导出 → 构建 → 对拍 → 孤立计时,结果写入 results/vision_trt_<档位>.json。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="P5 视觉塔 TensorRT 引擎")
    # 方图档位
    parser.add_argument("--budget", choices=tuple(SQUARE_BUDGETS), default="s576")
    # 解析参数
    args = parser.parse_args()

    # 重依赖在解析参数之后才导入
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    # 方图每边的图像 token 数
    side_tokens = square_side_tokens(SQUARE_BUDGETS[args.budget])
    grid = f"g{2 * side_tokens}x{2 * side_tokens}"   # 合并前的块网格,运行期按它查引擎
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    # 加载完整的 BF16 模型
    full = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                           attn_implementation="sdpa")
    # 取出视觉塔放到 GPU,切到推理模式
    visual_bf16 = full.model.visual.to("cuda").eval()
    del full  # 只留下视觉塔;语言模型从没上过 GPU,随即释放
    # 对拍用的 pixel_values
    inputs = square_inputs(processor, args.budget)
    import copy

    # FP32 真值:同一份权重升到 FP32 再算一遍;算完立刻释放(FP32 视觉塔 2.7 GB,与后面的 FP16 副本不能同时驻留)
    visual_fp32 = copy.deepcopy(visual_bf16).float()
    # 包成固定网格的模块
    fp32_wrapper = FixedGridVision.build(visual_fp32, side_tokens)
    # 计算全部图片的 FP32 输出,结果放回 CPU
    with torch.inference_mode():
        truth = [fp32_wrapper(px.to("cuda", torch.float32)).float().cpu() for px in inputs]
    # 释放 FP32 副本
    del visual_fp32, fp32_wrapper
    # 归还缓存的显存
    torch.cuda.empty_cache()

    # ONNX 文件路径
    onnx_path = OUT_DIR / f"vision_{grid}.onnx"
    # 引擎文件路径
    engine_path = OUT_DIR / f"vision_{grid}.engine"
    # 导出用 FP16 副本(强类型:ONNX 里的 dtype 即引擎里的 dtype)
    visual_fp16 = copy.deepcopy(visual_bf16).half()
    # BF16 参考实现的包装模块
    ref_wrapper = FixedGridVision.build(visual_bf16, side_tokens)
    # 用于导出的 FP16 包装模块
    fp16_wrapper = FixedGridVision.build(visual_fp16, side_tokens)
    # 导出计时起点
    t0 = time.time()
    # 以第一张图为示例输入导出 ONNX
    export_onnx(fp16_wrapper, inputs[0].to("cuda", torch.float16), onnx_path)
    # 导出耗时
    export_s = time.time() - t0
    # 释放 FP16 副本
    del visual_fp16, fp16_wrapper
    # 归还缓存的显存
    torch.cuda.empty_cache()
    # 构建引擎
    build = build_engine(onnx_path, engine_path)
    # 打印导出耗时和构建信息
    log.info("exported in %.0fs, built %s", export_s, build)

    # 加载引擎
    runner = TrtRunner(engine_path)

    # 返回 (余弦相似度, 相对误差);相对误差 = 差的范数 / 基准的范数,b 是基准
    def compare(a, b) -> tuple[float, float]:  # noqa: ANN001
        return (torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                ((a - b).norm() / b.norm()).item())

    # 三组对比的逐图结果
    stats: dict[str, list[tuple[float, float]]] = {"bf16_vs_fp32": [], "trt_vs_fp32": [], "trt_vs_bf16": []}
    # 逐图对拍
    with torch.inference_mode():
        for px, t32 in zip(inputs, truth, strict=True):
            # HF BF16 视觉塔的输出
            bf16 = ref_wrapper(px.to("cuda", torch.bfloat16)).float().cpu()
            # FP16 引擎的输出
            got = runner(px.to("cuda", torch.float16).contiguous()).float().cpu()
            # 分别与 FP32 真值比较,再互相比较
            stats["bf16_vs_fp32"].append(compare(bf16, t32))
            stats["trt_vs_fp32"].append(compare(got, t32))
            stats["trt_vs_bf16"].append(compare(got, bf16))
    # 每组对比的余弦相似度最小值和均值、相对误差最大值和均值
    parity = {k: {"cosine_min": round(min(c for c, _ in v), 6),
                  "cosine_mean": round(statistics.mean(c for c, _ in v), 6),
                  "rel_err_max": round(max(r for _, r in v), 5),
                  "rel_err_mean": round(statistics.mean(r for _, r in v), 5)}
              for k, v in stats.items()}
    # 打印两种实现相对 FP32 的最大相对误差
    log.info("parity vs FP32: HF BF16 rel err %.4f, TRT FP16 rel err %.4f (max)",
             parity["bf16_vs_fp32"]["rel_err_max"], parity["trt_vs_fp32"]["rel_err_max"])

    # 计时用的输入:第一张图,两种精度各一份
    x_bf16 = inputs[0].to("cuda", torch.bfloat16)
    x_fp16 = inputs[0].to("cuda", torch.float16).contiguous()
    # 计时对象:名字 -> (可调用对象, 输入);两个实现各用自己的 dtype 喂同一张图
    fns = {"hf_bf16": (lambda x: ref_wrapper(x), x_bf16), "trt_fp16": (runner, x_fp16)}
    # 各实现各轮的 GPU 时间
    samples = {k: [] for k in fns}
    # 按交错轮次计时
    with torch.inference_mode():
        for r, order in enumerate(interleaved_schedule(list(fns), ROUNDS + 1, seed=0)):
            for k in order:
                # 取出被测函数和输入
                fn, x = fns[k]
                # 连续测 REPEATS 次取中位数
                ms = gpu_ms(fn, x, REPEATS)
                if r > 0:  # 第 0 轮是预热,不计
                    samples[k].append(ms)
    # 两种实现的统计
    hf, trt_ms = summarize(samples["hf_bf16"]), summarize(samples["trt_fp16"])
    # 结果:档位、网格、引擎路径、块数、token 数、导出和构建信息、对拍、计时、加速比和时间
    result = {"budget": args.budget, "grid": grid, "engine": str(engine_path), "patches": (side_tokens * 2) ** 2,
              "image_tokens": side_tokens**2,
              "export_seconds": round(export_s, 1), "build": build,
              "parity": {"n": len(inputs), **parity},
              "gpu_ms": {"hf_bf16_sdpa": hf, "trt_fp16": trt_ms},
              "speedup_p50": round(hf["p50"] / trt_ms["p50"], 2), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    # 写出结果
    (RESULTS_DIR / f"vision_trt_{args.budget}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
    # 打印两种实现的 P50 耗时和加速比
    log.info("vision tower %s: HF BF16 %.1f ms vs TRT FP16 %.1f ms (%.2fx)", args.budget, hf["p50"], trt_ms["p50"],
             result["speedup_p50"])
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
