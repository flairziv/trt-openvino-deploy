"""OpenVINO(CPU)路线:FP32 IR、NNCF INT8、SmoothQuant 扫描、Accuracy Control。

用法(env: ov):
    python scripts/08_quantize_openvino.py --res r364 --recipes fp32 int8 int8sqoff int8sq050 int8sq075 int8ac

各配方(与 dadeploy.variants.OV_RECIPES 一一对应):
    fp32       ov.convert_model + save_model(compress_to_fp16=False)。
               坑:save_model 默认把权重压成 FP16,基准会带入权重舍入误差,必须显式关掉。
    int8       nncf.quantize(model_type=TRANSFORMER, preset=MIXED),NNCF 默认 SmoothQuant α=0.95(MatMul)。
    int8sqoff  同上但 α=-1(关闭 SmoothQuant),作为对照。
    int8sq050 / int8sq075  α 扫描:α 越大,越多地把激活的通道离群"搬"到权重上。
    int8ac     nncf.quantize_with_accuracy_control:以 val 集 (1 − AbsRel) 为指标、max_drop = ACCEPT_ABSREL_DELTA,
               自动按影响排序逐层回退,直到达标 —— 与 GPU 侧 int8mix 使用同一个接受标准。

SmoothQuant 为什么在这里值得测:04 激活统计显示残差流的离群集中在固定通道(324 / 278 / 68 …)。
per-tensor 激活量化只有一个 scale,而 SmoothQuant 对每个通道施加 s:激活 ÷ s、权重 × s,
数学上等价,却把离群从"只能共用一个 scale 的激活"挪到"可以 per-channel 量化的权重"。

CPU 推理精度说明:本机 14900HX 有 AVX-VNNI、无 AVX512-FP16 / AMX,OpenVINO 的 FP32 / INT8 都是原生;
不提供 FP16 配方,因为 CPU 上 FP16 不会更快(权重仍要解压到 f32 计算)。

产物:
    artifacts/openvino/da2s_<res>.<recipe>.xml/.bin
    results/openvino.json   每个配方的耗时、IR 大小、FakeQuantize 数、val 集指标
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import nncf  # noqa: E402
import openvino as ov  # noqa: E402
from nncf.parameters import ModelType  # noqa: E402
from nncf.quantization.advanced_parameters import (  # noqa: E402
    AdvancedAccuracyRestorerParameters,
    AdvancedQuantizationParameters,
    AdvancedSmoothQuantParameters,
)

from dadeploy.config import (  # noqa: E402
    ACCEPT_ABSREL_DELTA,
    CALIB_NUM,
    NYU_HEIGHT,
    NYU_WIDTH,
    ONNX_DIR,
    OV_DIR,
    RESOLUTIONS,
    RESULTS_DIR,
)
from dadeploy.data import iter_split  # noqa: E402
from dadeploy.logging_ import get_logger  # noqa: E402
from dadeploy.metrics import evaluate_prediction  # noqa: E402
from dadeploy.preprocess import postprocess, preprocess  # noqa: E402
from dadeploy.variants import OV_RECIPES  # noqa: E402

# 模块级 logger
log = get_logger("openvino")

# 配方 → SmoothQuant MatMul α。None 表示使用 NNCF 默认值(当前版本 0.95)
SQ_ALPHA: dict[str, float | None] = {"int8": None, "int8sqoff": -1.0, "int8sq050": 0.5, "int8sq075": 0.75}
AC_MAX_ITERATIONS: int = 8          # accuracy control 最多回退轮数(CPU 上每轮都要跑一遍 val)


def advanced_params(alpha: float | None) -> AdvancedQuantizationParameters:
    """构造 NNCF 高级参数。卷积的 SmoothQuant 保持关闭(α=-1,NNCF 默认):只有 MatMul 需要。"""
    # 没有指定 α 时使用 NNCF 的默认参数
    if alpha is None:
        return AdvancedQuantizationParameters()
    # 只改 MatMul 的 α,卷积保持关闭
    return AdvancedQuantizationParameters(smooth_quant_alphas=AdvancedSmoothQuantParameters(convolution=-1, matmul=alpha))


def calib_dataset(res_name: str) -> nncf.Dataset:
    """calib 划分 → NNCF Dataset(transform_fn 返回模型输入)。"""
    # 取该档位的分辨率
    res = RESOLUTIONS[res_name]
    # 读入校准集前 CALIB_NUM 张
    samples = list(iter_split("calib", limit=CALIB_NUM))
    # 包装成 NNCF 数据集,取样时做预处理
    return nncf.Dataset(samples, lambda s: preprocess(s.rgb, res))


def val_samples(res_name: str) -> list:
    """val 划分(带真值),预处理放在 validation_fn 外,减少重复 IO。"""
    # 取该档位的分辨率
    res = RESOLUTIONS[res_name]
    # 逐张预处理,与真值深度配对
    return [(preprocess(s.rgb, res), s.depth) for s in iter_split("val")]


def make_validation_fn(res_name: str):
    """构造 accuracy control 的验证函数。

    NNCF 约定指标"越大越好",因此返回 1 − AbsRel:绝对 drop 0.001 ⇔ AbsRel 上升 0.001,
    与 GPU 侧的接受标准完全相同。同时返回逐样本值,NNCF 可据此排序候选回退层,比比较模型输出更贴近任务。
    """

    # 验证函数:返回平均指标和逐样本指标
    def validate(compiled_model: ov.CompiledModel, dataset) -> tuple[float, list[float]]:
        # 模型的第 0 个输出
        output = compiled_model.output(0)
        # 逐样本的指标
        per_sample = []
        # 逐张推理并打分
        for x, depth in dataset:
            # 推理并取出输出
            raw = compiled_model(x)[output]
            # 放回原图尺寸
            pred = postprocess(np.asarray(raw), (NYU_HEIGHT, NYU_WIDTH))
            # 指标取 1 减 AbsRel,越大越好
            per_sample.append(1.0 - evaluate_prediction(pred, depth).abs_rel)
        # 返回均值和逐样本值
        return float(np.mean(per_sample)), per_sample

    return validate


def val_abs_rel(xml_path: Path, res_name: str, samples: list) -> float:
    """用 CPU 插件评测 IR 在 val 集上的 AbsRel。"""
    # OpenVINO 的入口对象
    core = ov.Core()
    # 编译到 CPU,按 FP32 计算
    compiled = core.compile_model(str(xml_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    # 复用验证函数得到平均指标
    metric, _ = make_validation_fn(res_name)(compiled, samples)
    # 指标换回 AbsRel
    return 1.0 - metric


# 统计模型里 FakeQuantize 节点的个数
def count_fake_quantize(model: ov.Model) -> int:
    return sum(op.get_type_name() == "FakeQuantize" for op in model.get_ops())


def build_recipe(recipe: str, res_name: str, fp32_model: ov.Model, samples: list) -> ov.Model:
    """按配方产出 ov.Model。"""
    # fp32 配方直接返回原模型
    if recipe == "fp32":
        return fp32_model
    # 各量化配方共用的参数
    # MIXED:权重对称、激活非对称。GELU / ReLU 输出分布偏正,非对称能少浪费一半量化区间。
    # TRANSFORMER:让 NNCF 按 Transformer 模板放量化点(不量化 LayerNorm 等)并启用 SmoothQuant。
    # subset_size 必须显式给:默认 300 大于 calib 张数,且会掩盖实际用了多少校准数据。
    common = {"preset": nncf.QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER, "subset_size": CALIB_NUM}
    # int8ac 配方:带精度控制的量化
    if recipe == "int8ac":
        return nncf.quantize_with_accuracy_control(
            fp32_model,
            # 校准数据集
            calib_dataset(res_name),
            nncf.Dataset(samples, lambda item: item[0]),   # transform 取模型输入;validation_fn 拿到的仍是 (x, depth)
            # 验证函数
            make_validation_fn(res_name),
            # 允许的指标下降上限,按绝对值计
            max_drop=ACCEPT_ABSREL_DELTA,
            drop_type=nncf.DropType.ABSOLUTE,
            # 限制最多回退的轮数
            advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(max_num_iterations=AC_MAX_ITERATIONS),
            **common,
        )
    # 其余配方:普通量化,按配方设置 SmoothQuant 的 α
    return nncf.quantize(fp32_model, calib_dataset(res_name), advanced_parameters=advanced_params(SQ_ALPHA[recipe]), **common)


# 入口:按配方生成 OpenVINO IR,并记录到 results/openvino.json
def main() -> int:
    # 命令行参数
    parser = argparse.ArgumentParser(description="OpenVINO FP32 / NNCF INT8 / SmoothQuant / Accuracy Control")
    # 分辨率档位,必填
    parser.add_argument("--res", required=True, choices=list(RESOLUTIONS))
    # 要生成的配方,默认全部
    parser.add_argument("--recipes", nargs="*", default=list(OV_RECIPES), choices=list(OV_RECIPES))
    parser.add_argument("--force", action="store_true", help="产物已存在时也重新生成")
    # 解析参数
    args = parser.parse_args()

    # 创建输出目录
    OV_DIR.mkdir(parents=True, exist_ok=True)
    # 基础 FP32 ONNX
    src = ONNX_DIR / f"da2s_{args.res}.onnx"
    # 转成 OpenVINO 模型
    fp32_model = ov.convert_model(str(src))
    # 预处理好的 val 样本
    samples = val_samples(args.res)
    # 记录文件
    report_path = RESULTS_DIR / "openvino.json"

    # 逐个配方处理
    for recipe in args.recipes:
        # 该配方的 IR 路径
        xml = OV_DIR / f"da2s_{args.res}.{recipe}.xml"
        # 已存在且未要求强制重建时跳过
        if xml.exists() and not args.force:
            log.info("skip %s.%s: already exists", args.res, recipe)
            continue
        # 计时起点
        t0 = time.time()
        # nncf 可能原地改模型,每个配方都从 FP32 副本开始
        model = build_recipe(recipe, args.res, fp32_model.clone(), samples)
        # compress_to_fp16 默认 True 会把未量化的浮点权重舍入成 FP16,FP32 基准就不再是 FP32
        ov.save_model(model, str(xml), compress_to_fp16=False)
        # 该配方的记录:耗时、IR 大小、FakeQuantize 数、val 集 AbsRel
        info = {
            "seconds": round(time.time() - t0, 1),
            "ir_mb": round((xml.stat().st_size + xml.with_suffix(".bin").stat().st_size) / 2**20, 2),
            "fake_quantize": count_fake_quantize(model),
            "val_abs_rel": val_abs_rel(xml, args.res, samples),
        }
        # 打印记录
        log.info("%s.%s: %s", args.res, recipe, info)
        # 已有记录则读入,只更新本次的条目
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        # 以档位名加配方名作键
        report[f"{args.res}.{recipe}"] = info
        # 写回记录
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
