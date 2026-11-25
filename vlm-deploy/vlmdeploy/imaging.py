"""图像 token 预算:按 Qwen2-VL 的缩放规则,把图片预先缩到"28 的整数倍且不超过预算"。

为什么要自己预缩放,而不是把 max_pixels 交给各推理栈的处理器:
    两条推理栈各有一份预处理实现 —— TensorRT-LLM 用 HF 的 Qwen2VLImageProcessor(Python + PIL),
    OpenVINO GenAI 用自己的 C++ 实现。只要处理器里真的发生了缩放,两份实现的插值细节就可能不同,
    "两栈精度差"里就混进了"预处理差"。
    这里先用与官方一致的 smart_resize 算出目标尺寸,自己缩好。之后两栈处理器再算 smart_resize,
    结果就是原尺寸(28 的倍数、且在像素范围内)—— 缩放变成恒等,预处理天然一致。
    这和项目 A 里 C++ 与 Python 对拍"同一份输入"是同一个思路。

token 与像素的关系:
    视觉塔按 14×14 切块,再把相邻 2×2 个块合并成 1 个 token,所以 1 个图像 token = 28×28 像素;
    448×448 → 16×16 = 256 个 token,但视觉塔内部处理的是合并前的 1024 个块。
"""

from __future__ import annotations

import math

from vlmdeploy.config import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, TOKEN_PIXELS

MAX_ASPECT_RATIO = 200   # 与官方实现一致:长宽比超过 200 直接拒绝


def smart_resize(height: int, width: int, factor: int = TOKEN_PIXELS, min_pixels: int = DEFAULT_MIN_PIXELS,
                 max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """Qwen2-VL 的目标尺寸规则(逐行对应 transformers 的 image_processing_qwen2_vl.smart_resize)。

    规则:先把高、宽各自四舍五入到 factor 的倍数;若面积超过 max_pixels,按面积等比缩小后向下取整;
    若面积低于 min_pixels,等比放大后向上取整。长宽比基本保持。

    注意 Python 的 round 是"四舍六入五成双",这里必须与官方实现用同一个 round,否则恰好落在 .5 的尺寸
    会差 28 像素(一行或一列 token)。

    Args:
        height: 原图高。
        width: 原图宽。
        factor: 尺寸必须是它的整数倍,默认 28。
        min_pixels: 面积下限。
        max_pixels: 面积上限(= token 预算 × 28²)。

    Returns:
        (目标高, 目标宽),都是 factor 的整数倍。

    Raises:
        ValueError: 长宽比超过 200。
    """
    # 长宽比超过上限时拒绝
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio must be <= {MAX_ASPECT_RATIO}, got {max(height, width) / min(height, width)}")
    # 高和宽各自就近取整到 factor 的倍数
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    # 面积超过上限:按面积比等比缩小,向下取整
    if h_bar * w_bar > max_pixels:
        # beta 是边长的缩放系数(面积比开方);向下取整保证不超过 max_pixels,max(factor, …) 防止某一边被缩成 0
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        # 面积低于下限:等比放大,向上取整
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    # 顺序是 (高, 宽);PIL 的 size 和 resize 用的是 (宽, 高),调用方别弄反
    return h_bar, w_bar


def image_tokens(height: int, width: int) -> int:
    """28 倍数尺寸的图像送进模型后产生的图像 token 数。

    Raises:
        ValueError: 尺寸不是 28 的整数倍(说明没有经过 smart_resize)。
    """
    # 宁可报错,也不让下面的整除悄悄算出一个错的 token 数
    if height % TOKEN_PIXELS or width % TOKEN_PIXELS:
        raise ValueError(f"{height}x{width} is not a multiple of {TOKEN_PIXELS}")
    # 行数乘以列数
    return (height // TOKEN_PIXELS) * (width // TOKEN_PIXELS)


def budget_size(height: int, width: int, budget_tokens: int) -> tuple[int, int]:
    """在 token 预算内、保持长宽比的目标尺寸。

    Args:
        height: 原图高。
        width: 原图宽。
        budget_tokens: 图像 token 上限。

    Returns:
        (目标高, 目标宽)。实际 token 数 ≤ 预算(非方图通常略小于预算)。
    """
    # token 预算换算成面积上限后计算目标尺寸
    return smart_resize(height, width, max_pixels=budget_tokens * TOKEN_PIXELS * TOKEN_PIXELS)


def resize_to_budget(image, budget_tokens: int):  # noqa: ANN001, ANN201 —— PIL.Image,避免顶层导入 Pillow
    """把 RGB 图片缩到 token 预算内的目标尺寸(bicubic,与 HF 处理器使用的 PIL 插值一致)。

    Args:
        image: PIL.Image(任意模式,会转成 RGB)。
        budget_tokens: 图像 token 上限。

    Returns:
        缩放后的 PIL.Image;尺寸已是目标尺寸时原样返回(不重复插值)。
    """
    # Pillow 只在这里导入
    from PIL import Image

    if image.mode != "RGB":  # convert 总会复制一份,已是 RGB 时跳过,保持"尺寸已对就原样返回"
        image = image.convert("RGB")
    # 按预算计算目标尺寸
    target_h, target_w = budget_size(image.height, image.width, budget_tokens)
    # 尺寸已符合时原样返回
    if (image.height, image.width) == (target_h, target_w):
        return image
    # resize 要的是 (宽, 高),与上面的 (高, 宽) 相反
    return image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)


def square_side_tokens(budget_tokens: int) -> int:
    """方图档位每边的 token 数(256 → 16,576 → 24,1024 → 32)。

    Raises:
        ValueError: 预算不是完全平方数。
    """
    # 整数开平方
    side = math.isqrt(budget_tokens)
    # 预算不是完全平方数时报错
    if side * side != budget_tokens:
        raise ValueError(f"square budget must be a perfect square, got {budget_tokens}")
    return side


def square_to_budget(image, budget_tokens: int):  # noqa: ANN001, ANN201 —— PIL.Image
    """中心裁成方图,再缩放到 (边长 token × 28)²。

    用途:视觉塔 TensorRT 引擎是静态形状(每个 (h, w) 网格一个引擎)。方图档位让同一档的所有图片落在同一个网格上,
    端到端对比(HF 视觉塔 vs 引擎)时两边看到的图完全相同。代价是裁掉了长边两侧的内容,所以方图档位的分数
    只和方图档位比,不和 t* 档位比。
    """
    # Pillow 只在这里导入
    from PIL import Image

    # 统一转成 RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    # 短边作边长,居中裁剪;crop 的框是 (左, 上, 右, 下)
    s = min(image.size)
    # 裁剪框的左上角,使裁剪区域居中
    left, top = (image.width - s) // 2, (image.height - s) // 2
    # 裁出正方形
    image = image.crop((left, top, left + s, top + s))
    # 目标边长(像素)
    px = square_side_tokens(budget_tokens) * TOKEN_PIXELS
    # 双三次插值缩放到目标边长
    return image.resize((px, px), resample=Image.Resampling.BICUBIC)
