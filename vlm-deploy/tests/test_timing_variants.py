"""timing / variants / sysinfo 模块与文档行号校验工具的单元测试(纯标准库,两个环境都能跑)。

运行:python -m unittest discover -s tests -v
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import unittest
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import MODEL_DIR, OV_DIR, TRTLLM_CKPT_DIR
from vlmdeploy.sysinfo import gpu_memory_used_mb
from vlmdeploy.timing import interleaved_schedule, percentile, ratio_ci, round_spread, summarize
from vlmdeploy.variants import is_gpu, ov_ir_dir, trtllm_checkpoint


# timing 模块的用例
class TimingTest(unittest.TestCase):
    def test_percentile_matches_numpy_linear(self) -> None:
        # 乱序的 5 个样本
        data = [5.0, 1.0, 3.0, 2.0, 4.0]
        # 中位数
        self.assertEqual(percentile(data, 50), 3.0)
        self.assertAlmostEqual(percentile(data, 90), 4.6)   # numpy.percentile([1..5], 90) == 4.6
        # 只有一个样本时返回它本身
        self.assertEqual(percentile([7.0], 90), 7.0)

    def test_summarize_fields(self) -> None:
        # 汇总 4 个样本
        s = summarize([1.0, 2.0, 3.0, 4.0])
        # 核对样本数、中位数和最小值
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["p50"], 2.5)
        self.assertEqual(s["min"], 1.0)

    def test_schedule_is_reproducible_permutation(self) -> None:
        # 生成 4 轮计划
        plan = interleaved_schedule(["a", "b", "c"], rounds=4, seed=1)
        # 种子相同时计划相同
        self.assertEqual(plan, interleaved_schedule(["a", "b", "c"], rounds=4, seed=1))
        # 每轮都是全部测量项的一个排列
        for order in plan:
            self.assertEqual(sorted(order), ["a", "b", "c"])

    def test_round_spread(self) -> None:
        self.assertAlmostEqual(round_spread([9.0, 10.0, 11.0]), 0.2)   # (最大 − 最小) / 中位数 = (11 − 9) / 10

    def test_ratio_ci_brackets_point_estimate(self) -> None:
        # 两组等长的配对样本
        a = [2.0 + 0.01 * i for i in range(30)]
        b = [1.0 + 0.005 * i for i in range(30)]
        # 点估计和置信区间
        point, lo, hi = ratio_ci(a, b)
        # 点估计等于两个中位数之比
        self.assertAlmostEqual(point, statistics.median(a) / statistics.median(b))
        # 区间包含点估计
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)
        with self.assertRaises(ValueError):
            ratio_ci([1.0], [1.0, 2.0])   # 两组样本是配对的,长度不等直接报错


# variants 模块的用例
class VariantsTest(unittest.TestCase):
    def test_paths(self) -> None:
        # 基线用原始模型目录
        self.assertEqual(trtllm_checkpoint("trtllm-bf16"), MODEL_DIR)
        # 量化变体用对应的 checkpoint 子目录
        self.assertEqual(trtllm_checkpoint("trtllm-awq-img"), TRTLLM_CKPT_DIR / "awq-img")
        # OpenVINO 变体用对应的 IR 子目录
        self.assertEqual(ov_ir_dir("ov-int4-vis8s"), OV_DIR / "int4-vis8s")
        self.assertEqual(trtllm_checkpoint("trtllm-bf16-vistrt"), MODEL_DIR)   # -vistrt 只换视觉塔,语言模型目录不变
        with self.assertRaises(ValueError):
            trtllm_checkpoint("trtllm-int2-img")   # 不认识的名字:没有 int2 这种量化方法

    def test_device(self) -> None:
        # hf 和 trtllm 变体在 GPU 上,ov 变体不在
        self.assertTrue(is_gpu("hf-bf16"))
        self.assertTrue(is_gpu("trtllm-fp8-txt"))
        self.assertFalse(is_gpu("ov-int4"))


# sysinfo 模块的用例
class SysinfoTest(unittest.TestCase):
    def test_gpu_memory_is_int_or_none(self) -> None:
        # 没有 nvidia-smi 的环境(CPU 机器、CI)返回 None,而不是抛异常
        used = gpu_memory_used_mb()
        # 返回 None 或非负整数
        self.assertTrue(used is None or (isinstance(used, int) and used >= 0))


class DocsCheckTest(unittest.TestCase):
    """tools/check_docs.py:锚点在行号范围内才算对,行号过期要能报出来。"""

    def test_detects_stale_line_numbers(self) -> None:
        # 把 tools 目录加入模块搜索路径后导入 check_docs
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        import check_docs

        # 读入 config.py 的全部行
        source = (check_docs.ROOT / "vlmdeploy" / "config.py").read_text(encoding="utf-8").splitlines()
        # PATCH_SIZE 所在的行号
        line = next(i for i, text in enumerate(source, 1) if text.startswith("PATCH_SIZE"))
        # 构造一段导读:一行正确、一行行号过期、一行超出文件长度
        doc = "\n".join(["### `vlmdeploy/config.py`", "| 行 | 代码 | 说明 |", "|---|---|---|",
                         f"| {line} | `PATCH_SIZE: int` | 对 |",
                         f"| {line + 1}–{line + 2} | `PATCH_SIZE: int` | 行号过期 |",
                         "| 99999 | `x` | 超出文件长度 |"])
        # 写到临时文件后校验
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text(doc, encoding="utf-8")
            errors, covered = check_docs.check(path)
        # 三行表格里第 1 行是对的;行号过期、超出文件长度各报 1 个错
        self.assertEqual(len(errors), 2)
        # 超出文件长度的那一行不计入覆盖
        self.assertEqual(covered["vlmdeploy/config.py"], {line, line + 1, line + 2})


# 直接运行本文件时执行全部用例
if __name__ == "__main__":
    unittest.main()
