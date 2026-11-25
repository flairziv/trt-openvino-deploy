"""变体注册表:把 "后端:分辨率:精度" 字符串解析成产物路径与说明。

一个**变体(variant)**就是结果表里的一行,例如:

    torch:r518:fp32     PyTorch FP32(严格,关闭 TF32)—— 精度基准
    trt:r518:fp16       TensorRT 强类型 FP16 引擎
    trt:r364:int8mix    TensorRT INT8 + 敏感层回退 FP16
    ov:r518:int8        OpenVINO NNCF INT8(CPU)

所有脚本(构建 / 评测 / 计时 / 报告)都只认这个字符串,产物路径由本模块统一推导,
避免"脚本 A 写 xxx_int8.engine、脚本 B 读 xxx.int8.engine"这类命名漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ENGINE_DIR, ONNX_DIR, OV_DIR, RESOLUTIONS, Resolution

# 支持的后端名
BACKENDS: tuple[str, ...] = ("torch", "ort", "trt", "ov")

# OpenVINO 精度配方(产物由 scripts/08_quantize_openvino.py 生成)
OV_RECIPES: dict[str, str] = {
    "fp32": "OpenVINO FP32(save_model 时关闭默认的 FP16 权重压缩)",
    "int8": "NNCF INT8,model_type=TRANSFORMER(含 SmoothQuant 默认 α)",
    "int8sqoff": "NNCF INT8,关闭 SmoothQuant",
    "int8sq050": "NNCF INT8,SmoothQuant α=0.50",
    "int8sq075": "NNCF INT8,SmoothQuant α=0.75",
    "int8ac": "NNCF accuracy-control:以 val 集 AbsRel 为约束自动回退敏感层",
}


@dataclass(frozen=True)
class Variant:
    """一个被测变体。

    Attributes:
        backend: "torch" / "ort" / "trt" / "ov"。
        res: 输入分辨率档位。
        precision: 精度配方名(各后端的合法取值见 TRT_RECIPES / OV_RECIPES)。
    """

    backend: str
    res: Resolution
    precision: str

    @property
    def key(self) -> str:
        """规范化的变体字符串,同时用作结果文件名(冒号替换为下划线)。"""
        return f"{self.backend}:{self.res.name}:{self.precision}"

    @property
    def slug(self) -> str:
        """可作文件名的形式。"""
        return self.key.replace(":", "_")

    @classmethod
    def parse(cls, text: str) -> "Variant":
        """解析变体字符串。

        Args:
            text: 形如 "trt:r518:fp16"。

        Returns:
            Variant。

        Raises:
            ValueError: 格式错误、后端或分辨率未知。
        """
        # 去掉首尾空白后按冒号拆分
        parts = text.strip().split(":")
        # 必须恰好三段
        if len(parts) != 3:
            raise ValueError(f"variant must be backend:res:precision, got {text!r}")
        # 依次是后端、分辨率档位名、精度配方名
        backend, res_name, precision = parts
        # 校验后端名
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}, expect one of {BACKENDS}")
        # 校验分辨率档位名
        if res_name not in RESOLUTIONS:
            raise ValueError(f"unknown resolution {res_name!r}, expect one of {tuple(RESOLUTIONS)}")
        # ov 后端的配方名必须在 OV_RECIPES 中
        if backend == "ov" and precision not in OV_RECIPES:
            raise ValueError(f"unknown OpenVINO recipe {precision!r}, expect one of {tuple(OV_RECIPES)}")
        # torch 后端只有 fp32 一种
        if backend == "torch" and precision != "fp32":
            raise ValueError("torch backend only provides the fp32 reference")
        # 把档位名换成 Resolution 对象后构造实例
        return cls(backend=backend, res=RESOLUTIONS[res_name], precision=precision)

    # ------------------------------------------------------------------ 产物路径
    def onnx_path(self) -> Path:
        """该变体对应的 ONNX。基础 FP32 图为 da2s_<res>.onnx,派生图为 da2s_<res>.<tag>.onnx。"""
        # trt 后端读哪张 ONNX 由配方表决定
        if self.backend == "trt":
            from .trt_recipes import TRT_RECIPES  # 不依赖 tensorrt 的纯配置模块

            tag = TRT_RECIPES[self.precision].onnx_tag
        else:
            # 其余后端:fp32 读基础图,否则标签就是配方名
            tag = "" if self.precision == "fp32" else self.precision
        # 标签为空时是基础图,否则文件名里带上标签
        return ONNX_DIR / (f"da2s_{self.res.name}.onnx" if not tag else f"da2s_{self.res.name}.{tag}.onnx")

    def engine_path(self) -> Path:
        """TensorRT 引擎路径(只对 trt 后端有意义)。"""
        return ENGINE_DIR / f"da2s_{self.res.name}.{self.precision}.engine"

    def ov_path(self) -> Path:
        """OpenVINO IR(.xml,同名 .bin 在旁边)路径(只对 ov 后端有意义)。"""
        return OV_DIR / f"da2s_{self.res.name}.{self.precision}.xml"

    def artifact_path(self) -> Path | None:
        """该变体运行时直接加载的产物;torch 后端加载 HF 权重,返回 None。"""
        # 按后端取对应的路径方法并调用,没有对应产物的后端返回 None
        return {
            "trt": self.engine_path,
            "ov": self.ov_path,
            "ort": self.onnx_path,
        }.get(self.backend, lambda: None)()

    def describe(self) -> str:
        """一句话说明,用于结果表。"""
        # PyTorch 参考后端的固定说明
        if self.backend == "torch":
            return "PyTorch FP32(TF32 关闭),精度与量化偏差的基准"
        # ONNX Runtime 后端的固定说明
        if self.backend == "ort":
            return "ONNX Runtime(用于核对导出 / QDQ 假量化仿真)"
        # OpenVINO 后端取配方表里的说明
        if self.backend == "ov":
            return OV_RECIPES[self.precision]
        # TensorRT 后端取配方表里的说明
        from .trt_recipes import TRT_RECIPES

        return TRT_RECIPES[self.precision].description


def parse_many(texts: list[str]) -> list[Variant]:
    """批量解析,保持顺序并去重。"""
    # 用字典按变体字符串去重,并保持首次出现的顺序
    seen: dict[str, Variant] = {}
    # 逐个解析
    for text in texts:
        v = Variant.parse(text)
        # 已经出现过的变体不覆盖
        seen.setdefault(v.key, v)
    # 返回去重后的变体列表
    return list(seen.values())
