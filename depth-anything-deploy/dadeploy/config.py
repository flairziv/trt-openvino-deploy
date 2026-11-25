"""全局配置(单一事实来源)。

所有脚本共用的路径、分辨率档位与评测协议参数都定义在这里,脚本内**不允许**再写魔数。
修改本文件中的任一协议参数(裁剪框、深度范围、对齐方式)都会使历史 results/ 失效,
需要整套重跑,因此改动前请在 README 的"评测协议"一节同步说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 仓库根目录:本文件所在目录的上一级
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
# 数据根目录
DATA_DIR: Path = REPO_ROOT / "data"
# 下载的原始数据包与官方划分文件
RAW_DIR: Path = DATA_DIR / "raw"
# 整理后的 NYUv2 数据
NYU_DIR: Path = DATA_DIR / "nyu"
NYU_TEST_DIR: Path = NYU_DIR / "test"          # 654 张官方 test 划分(h5)
NYU_CALIB_DIR: Path = NYU_DIR / "calib"        # 训练场景抽样:calib(量化校准)+ val(敏感度分析 / 精度控制)
# 数据清单,记录 test / calib / val 各划分的样本名
MANIFEST_PATH: Path = NYU_DIR / "manifest.json"

# 模型产物根目录,内容都可以由脚本重新生成
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
# 本地的 HF 原始权重
HF_MODEL_DIR: Path = ARTIFACTS_DIR / "hf" / "Depth-Anything-V2-Small-hf"
# 导出与量化后的 ONNX
ONNX_DIR: Path = ARTIFACTS_DIR / "onnx"
# TensorRT 引擎
ENGINE_DIR: Path = ARTIFACTS_DIR / "engines"
# OpenVINO IR
OV_DIR: Path = ARTIFACTS_DIR / "openvino"
# 校准缓存目录,目前没有代码引用
CALIB_CACHE_DIR: Path = ARTIFACTS_DIR / "calib"
REF_DIR: Path = ARTIFACTS_DIR / "reference"    # PyTorch FP32 参考输出(量化偏差的基准)

# 机器可读的结果文件,README 表格由它生成
RESULTS_DIR: Path = REPO_ROOT / "results"

# Hub 上的模型仓库名
HF_REPO_ID: str = "depth-anything/Depth-Anything-V2-Small-hf"

# ---------------------------------------------------------------------------
# 模型输入档位
# ---------------------------------------------------------------------------
PATCH_SIZE: int = 14                            # DINOv2 ViT-S/14:每 14×14 像素一个 token


# 不可变数据类:实例创建后字段不能修改
@dataclass(frozen=True)
class Resolution:
    """一个固定输入尺寸档位。

    Attributes:
        name: 档位名,出现在产物文件名与结果表中。
        height: 输入高(必须是 PATCH_SIZE 的整数倍)。
        width: 输入宽(必须是 PATCH_SIZE 的整数倍)。
    """

    name: str
    height: int
    width: int

    # 字段赋值完成后自动调用,用于校验
    def __post_init__(self) -> None:
        # 高或宽不是 PATCH_SIZE 的整数倍时报错
        if self.height % PATCH_SIZE or self.width % PATCH_SIZE:
            raise ValueError(f"{self.name}: {self.height}x{self.width} 不是 {PATCH_SIZE} 的整数倍")

    @property
    def tokens(self) -> int:
        """ViT 主干的 patch token 数(不含 CLS)。注意力计算量 ∝ tokens²。"""
        # 行方向的 token 数乘以列方向的 token 数
        return (self.height // PATCH_SIZE) * (self.width // PATCH_SIZE)


# NYU 原图 480×640(4:3)。官方推理把短边缩到 518 并保持长宽比,对 4:3 输入即 518×686(686=49×14)。
# 部署要求静态形状,因此按"相机分辨率固定"的现实场景把三档都定死成 ≈4:3。
# 档位名到分辨率的映射
RESOLUTIONS: dict[str, Resolution] = {
    r.name: r
    for r in (
        Resolution("r518", 518, 686),   # 官方档:37×49 = 1813 tokens
        Resolution("r364", 364, 476),   # 降档:  26×34 =  884 tokens(≈0.49×)
        Resolution("r266", 266, 350),   # 低档:  19×25 =  475 tokens(≈0.26×,只用于分辨率曲线)
    )
}
MAIN_RESOLUTIONS: tuple[str, ...] = ("r518", "r364")   # 精度矩阵主档

# 预处理常量(与 HF preprocessor_config.json 一致:ImageNet 均值方差,bicubic 缩放)
IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

ONNX_OPSET: int = 17                            # ≥17 才有原生 LayerNormalization,TRT FP16 下数值更稳
# 模型输入张量名
INPUT_NAME: str = "pixel_values"
# 模型输出张量名
OUTPUT_NAME: str = "predicted_depth"

# ---------------------------------------------------------------------------
# 评测协议(NYUv2 Eigen test,654 张)
# ---------------------------------------------------------------------------
# NYU 原图的高
NYU_HEIGHT: int = 480
# NYU 原图的宽
NYU_WIDTH: int = 640
EIGEN_CROP: tuple[int, int, int, int] = (45, 471, 41, 601)   # (top, bottom, left, right),Eigen 标准裁剪
# 有效真值深度的下限(米)
MIN_DEPTH_M: float = 1e-3
# 有效真值深度的上限(米)
MAX_DEPTH_M: float = 10.0
# δ1 指标的比值阈值
DELTA1_THRESHOLD: float = 1.25

# 训练场景抽样(与 654 张测试图场景不相交)。两个子集互斥、各司其职:
#   calib —— 只用来统计激活范围(量化校准);
#   val   —— 所有"做决策"的地方都只看它:逐层敏感度排序、回退层选择、NNCF accuracy-control。
#            测试集只用于最终报告,任何配置选择都不碰测试集,避免"在测试集上调参"。
# 校准集张数
CALIB_NUM: int = 200
# 数据准备脚本抽样时用的随机种子
CALIB_SEED: int = 1
# 验证集张数
VAL_NUM: int = 100

# 统一的"可接受"标准(两条推理栈共用):相对同分辨率 FP32,val 集 AbsRel 上升不超过 0.001。
# GPU 侧 int8mix 的回退层选择、CPU 侧 NNCF accuracy-control 的 max_drop 都用它,两栈结论才可比。
ACCEPT_ABSREL_DELTA: float = 0.001

# 量化偏差(drift)报告子集:测试集每隔 DRIFT_STRIDE 张取 1 张(按场景均匀覆盖,109 张)。
# 需要预先落盘 PyTorch FP32 参考输出(各分辨率约 40~155 MB),故不取全量 654 张。
DRIFT_STRIDE: int = 6

# ---------------------------------------------------------------------------
# 计时协议
# ---------------------------------------------------------------------------
BENCH_ROUNDS: int = 5                           # 交错轮数:每轮随机打乱变体顺序,抵消笔记本降频漂移
# 每段计时前的预热次数
BENCH_WARMUP: int = 20
# 每段计时的迭代次数
BENCH_ITERS: int = 100
CPU_THREADS: int = 4                            # WSL 配额 4 vCPU;恰好等于 RK3588 大核数,按"端侧 4 核预算"呈现
