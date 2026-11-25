"""全局配置(单一事实来源)。

与项目 A 的 dadeploy/config.py 同一约定:路径、模型常量、数据清单、评测与计时协议都定义在这里,
脚本里不写魔数。改动任一"协议"参数都会让历史 results/ 失效,需要整套重跑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 仓库根目录 = 本文件往上两级;resolve() 取绝对路径,不依赖从哪个目录启动脚本
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
# 数据根目录
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"          # 下载的原始分片(parquet),只读
RESULTS_DIR: Path = REPO_ROOT / "results"  # 机器可读结果,README 表格的唯一来源
# 日志目录
LOGS_DIR: Path = REPO_ROOT / "logs"

# 大文件(模型、量化产物、OpenVINO IR、安装包)放 D 盘普通目录,不放仓库、也不放 WSL 家目录:
#   - D / E 盘都只剩约 36 GB,而 E 盘还承载着 WSL 虚拟盘(E:\WSL\Ubuntu);
#   - 写进 WSL ext4 的数据会让虚拟盘永久变大(vhdx 不会自动缩小),NTFS 目录删掉才是真的释放。
# 代价是 WSL 经 9p 读 /mnt/d 比读 ext4 慢,只影响模型加载时间(不计入任何计时)。
ARTIFACTS_DIR: Path = Path(os.environ.get("VLM_ARTIFACTS", "/mnt/d/vlm-artifacts"))
# HF 原始模型
HF_DIR: Path = ARTIFACTS_DIR / "hf"
# 安装用的 wheel
WHEEL_DIR: Path = ARTIFACTS_DIR / "wheels"
# OpenVINO IR
OV_DIR: Path = ARTIFACTS_DIR / "openvino"
TRTLLM_CKPT_DIR: Path = ARTIFACTS_DIR / "trtllm"   # ModelOpt 量化导出的 checkpoint

# Hub 地址,默认走镜像;和上面的大文件目录一样,可用环境变量覆盖
HF_ENDPOINT: str = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
# 全项目只有这一个模型
MODEL_REPO: str = "Qwen/Qwen2-VL-2B-Instruct"     # Apache-2.0(模型卡 license 字段)
# 模型的本地目录
MODEL_DIR: Path = HF_DIR / "Qwen2-VL-2B-Instruct"

# 结构常量,逐项核对过 config.json / preprocessor_config.json(2026-11-09):
#   视觉塔 32 层 / 1280 维 / 16 头,约 0.67B;语言部分 28 层 / 1536 维 / 12 头 / 2 个 KV 头(GQA),约 1.31B;
#   词表 151936,输入 embedding 与 lm_head 共享权重(tie_word_embeddings=true),约 0.23B。
# 共享权重这一条要留意:TensorRT-LLM 加载时把它弄丢不会报错,只会让输出全是 "!",
# patches/site/sitecustomize.py 的第 3 处修补修的就是它。
# 视觉塔的切块边长(像素)
PATCH_SIZE: int = 14
# 相邻块的合并边长:2 乘 2 个块合并成 1 个图像 token
MERGE_SIZE: int = 2
TOKEN_PIXELS: int = PATCH_SIZE * MERGE_SIZE       # 28:一个图像 token 覆盖 28×28 像素(14 像素切块,再 2×2 合并)
TEMPORAL_PATCH_SIZE: int = 2                       # 单张图会被复制成 2 帧送进 3D patch embed

# 归一化常量是 CLIP 的均值方差,不是项目 A 用的 ImageNet 值,不能照抄
IMAGE_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# 处理器默认的像素范围:下限 4 个 token,上限 16384 个 token。上限极高 —— 不设预算时,
# 一张 1080p 图就会产生上千个 token,TTFT 与精度都随之变化,所以评测必须显式控制 token 数。
# 面积下限(像素数)
DEFAULT_MIN_PIXELS: int = 3136
# 面积上限(像素数)
DEFAULT_MAX_PIXELS: int = 12845056

# 图像 token 预算档位(max_pixels = 预算 × 28²)。视觉塔内部处理的 patch 数是 token 数的 4 倍。
TOKEN_BUDGETS: dict[str, int] = {"t256": 256, "t576": 576, "t1024": 1024}
# 方图档位(中心裁剪 + 缩放到固定网格):只用于 P5 视觉塔静态形状引擎的端到端对比,见 imaging.square_to_budget。
SQUARE_BUDGETS: dict[str, int] = {"s256": 256, "s576": 576, "s1024": 1024}


# ---------------------------------------------------------------------------
# 评测数据(只下载评测真正用到的分片)
# ---------------------------------------------------------------------------
# 不可变数据类:实例创建后字段不能修改
@dataclass(frozen=True)
class HubDataset:
    """HF Hub 上的一个数据集及要下载的文件。

    Attributes:
        repo: 数据集仓库名(不含 "datasets/" 前缀)。
        files: 仓库内的相对路径。
        local_dir: 落盘到 RAW_DIR 下的子目录名。
        license: 许可(写进下载记录,便于 README 引用)。
    """

    repo: str
    files: tuple[str, ...]
    local_dir: str
    license: str


# 数据集短名到下载清单的映射
DATASETS: dict[str, HubDataset] = {
    # TextVQA:OCR 类问答,对图像 token 数最敏感。val 5000 题做测试;train 只取第 1 个分片(~300 MB),
    # 从中抽校准集与决策用的 dev 集 —— 三者互不重叠,沿用项目 A 的 calib / val / test 纪律。
    "textvqa": HubDataset(
        repo="lmms-lab/textvqa",
        files=(
            "data/validation-00000-of-00003.parquet",
            "data/validation-00001-of-00003.parquet",
            "data/validation-00002-of-00003.parquet",
            "data/train-00000-of-00020.parquet",
        ),
        local_dir="textvqa",
        license="CC BY 4.0 (TextVQA); images from OpenImages (CC BY 2.0)",
    ),
    # RealWorldQA:车载与真实场景的空间理解,765 题,用官方 xai-org 仓库。
    "realworldqa": HubDataset(
        repo="xai-org/RealworldQA",
        files=("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"),
        local_dir="realworldqa",
        license="CC BY-ND 4.0",
    ),
}
