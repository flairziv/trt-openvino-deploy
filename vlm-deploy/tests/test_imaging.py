"""imaging 模块的单元测试:预缩放的核心性质是"处理器再算一次 smart_resize 时尺寸不变"。

运行:python -m unittest discover -s tests -v(任一环境;用到 Pillow 的用例在没有 Pillow 时跳过)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import DEFAULT_MAX_PIXELS, TOKEN_BUDGETS, TOKEN_PIXELS
from vlmdeploy.imaging import budget_size, image_tokens, smart_resize

# 检查是否装了 Pillow,没装时跳过相关用例
try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:  # pragma: no cover —— 只有标准库的环境
    HAVE_PIL = False


# smart_resize、budget_size 和 image_tokens 的用例
class SmartResizeTest(unittest.TestCase):
    def test_square_budgets_hit_exactly(self) -> None:
        # 边长正好 448 / 672 / 896 的方图,在对应预算下不缩放,正好 16² / 24² / 32² 个 token(计时用的固定输入)
        for budget, side in ((256, 448), (576, 672), (1024, 896)):
            self.assertEqual(budget_size(side, side, budget), (side, side))
            self.assertEqual(image_tokens(side, side), budget)

    def test_non_square_undershoots_budget(self) -> None:
        # 超预算时官方规则是"等比缩小后向下取整",非方图的实际 token 数通常小于预算:
        # 4:3 的 768×1024 在 256 档只得到 13×18 = 234 个 token。评测记录里要存每张图的实际 token 数,
        # 不能假设等于预算。
        self.assertEqual(budget_size(768, 1024, 256), (364, 504))
        self.assertEqual(image_tokens(364, 504), 234)

    def test_never_exceeds_budget_and_keeps_multiple_of_28(self) -> None:
        # 各种尺寸和档位下,结果都是 28 的倍数且不超过预算
        for h, w in ((480, 640), (768, 1024), (1080, 1920), (333, 999), (1200, 400), (37, 4000)):
            for budget in TOKEN_BUDGETS.values():
                # 目标尺寸
                th, tw = budget_size(h, w, budget)
                # 高和宽都是 28 的倍数
                self.assertEqual(th % TOKEN_PIXELS, 0)
                self.assertEqual(tw % TOKEN_PIXELS, 0)
                # token 数不超过预算
                self.assertLessEqual(image_tokens(th, tw), budget)

    def test_idempotent_under_processor_defaults(self) -> None:
        # 关键性质:处理器拿到预缩放后的图,用默认 min/max 再算一次,必须原样返回 —— 否则两栈各自再插值一次
        for h, w in ((480, 640), (768, 1024), (1080, 1920), (500, 375)):
            for budget in TOKEN_BUDGETS.values():
                # 预缩放后的尺寸
                th, tw = budget_size(h, w, budget)
                # 按处理器默认上限再算一次,尺寸不变
                self.assertEqual(smart_resize(th, tw, max_pixels=DEFAULT_MAX_PIXELS), (th, tw))
                # 按本档位上限再算一次,尺寸也不变
                self.assertEqual(smart_resize(th, tw, max_pixels=budget * TOKEN_PIXELS**2), (th, tw))

    def test_bankers_rounding_matches_reference(self) -> None:
        # 官方实现用内置 round(五成双):1.5 → 2,2.5 → 2。预缩放必须用同一个 round,
        # 否则恰好落在 .5 的边长会和处理器差 28 像素(多或少一行 token)。
        self.assertEqual(round(42 / 28) * 28, 56)
        self.assertEqual(round(70 / 28) * 28, 56)
        self.assertEqual(smart_resize(70, 280), (56, 280))

    def test_rejects_extreme_aspect_ratio(self) -> None:
        # 长宽比超限时应抛 ValueError
        with self.assertRaises(ValueError):
            smart_resize(10, 3000)   # 长宽比 300,超过上限 200

    def test_image_tokens_requires_multiple(self) -> None:
        # 尺寸不是 28 的倍数时应抛 ValueError
        with self.assertRaises(ValueError):
            image_tokens(100, 100)   # 100 不是 28 的倍数


# resize_to_budget 的用例,需要 Pillow
@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class ResizeToBudgetTest(unittest.TestCase):
    def test_output_size_and_identity(self) -> None:
        from PIL import Image

        from vlmdeploy.imaging import resize_to_budget

        # 构造一张 1024x768 的纯色图
        img = Image.new("RGB", (1024, 768), (10, 20, 30))
        # 缩放到 256 token 档
        out = resize_to_budget(img, 256)
        # 输出尺寸与 budget_size 的结果一致
        self.assertEqual((out.height, out.width), budget_size(768, 1024, 256))
        self.assertIs(resize_to_budget(out, 256), out)   # 已是目标尺寸:不再插值


class MatchesTransformersTest(unittest.TestCase):
    """与 transformers 官方实现逐点对比(只在装了 transformers 的环境里跑,即 trtllm / ovgenai)。"""

    def test_grid_matches_reference(self) -> None:
        # 取官方实现作参照,没装 transformers 时跳过
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as ref
        except ImportError:
            self.skipTest("transformers not installed")
        # 按固定步长遍历高和宽
        for h in range(28, 2000, 37):
            for w in range(28, 2000, 53):
                # 长宽比超限的组合跳过
                if max(h, w) / min(h, w) > 200:
                    continue
                # 每个档位都比较
                for budget in TOKEN_BUDGETS.values():
                    # 档位对应的面积上限
                    max_px = budget * TOKEN_PIXELS**2
                    # 结果必须与官方实现相同
                    self.assertEqual(smart_resize(h, w, max_pixels=max_px), ref(h, w, factor=28, max_pixels=max_px),
                                     msg=f"{h}x{w} budget {budget}")


# 直接运行本文件时执行全部用例
if __name__ == "__main__":
    unittest.main()
