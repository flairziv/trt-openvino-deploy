"""核心纯函数的单元测试(不需要 GPU / 模型 / 数据集,两个 conda 环境都能跑)。

运行:python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import unittest
from pathlib import Path

import numpy as np

# 仓库根目录
ROOT = Path(__file__).resolve().parent.parent
# 加入模块搜索路径,保证能导入 dadeploy
sys.path.insert(0, str(ROOT))

from dadeploy.config import RESOLUTIONS, Resolution  # noqa: E402
from dadeploy.metrics import (  # noqa: E402
    align_disparity_to_depth,
    depth_metrics,
    eigen_valid_mask,
    output_drift,
    paired_bootstrap,
)
from dadeploy.preprocess import postprocess, preprocess  # noqa: E402
from dadeploy.timing import interleaved_schedule, round_spread, summarize  # noqa: E402
from dadeploy.variants import Variant, parse_many  # noqa: E402


# 按文件路径加载 scripts 下的脚本,文件名以数字开头不能直接 import
def _load_script(name: str):
    # 由文件路径创建模块规格
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    # 创建模块对象
    module = importlib.util.module_from_spec(spec)
    # 执行模块代码
    spec.loader.exec_module(module)
    return module


# 指标函数的测试
class MetricsTest(unittest.TestCase):
    # 每个用例之前准备随机真值深度和有效像素掩码
    def setUp(self) -> None:
        # 固定种子的随机数发生器
        rng = np.random.default_rng(0)
        # 0.5 到 9 米之间的随机真值深度
        self.gt = rng.uniform(0.5, 9.0, size=(480, 640)).astype(np.float32)
        # Eigen 有效像素掩码
        self.mask = eigen_valid_mask(self.gt)

    # 已知尺度和平移的预测,对齐后应还原真值
    def test_alignment_recovers_known_scale_and_shift(self) -> None:
        # 预测视差 = (1/gt − t)/s,对齐后应精确还原真值
        pred = (1.0 / self.gt - 0.05) / 3.0
        # 做尺度平移对齐
        aligned = align_disparity_to_depth(pred, self.gt, self.mask)
        # 计算指标
        m = depth_metrics(aligned, self.gt, self.mask)
        # 误差应接近 0,δ1 应接近 1
        self.assertLess(m.abs_rel, 1e-3)
        self.assertGreater(m.delta1, 0.999)

    # 掩码应排除裁剪框以外和深度无效的像素
    def test_eigen_mask_excludes_border_and_invalid_depth(self) -> None:
        # 复制一份真值
        gt = self.gt.copy()
        # 一个像素置为 0,低于有效下限
        gt[100, 100] = 0.0
        # 一个像素置为 20 米,超出有效上限
        gt[101, 101] = 20.0
        # 重新生成掩码
        mask = eigen_valid_mask(gt)
        # 左上角在裁剪框外
        self.assertFalse(mask[0, 0])
        # 深度为 0 的像素无效
        self.assertFalse(mask[100, 100])
        # 深度超限的像素无效
        self.assertFalse(mask[101, 101])
        # 框内的正常像素有效
        self.assertTrue(mask[200, 200])

    # 输出与参考完全相同时偏差为 0,SQNR 为无穷大
    def test_drift_identical_is_infinite_sqnr(self) -> None:
        d = output_drift(self.gt, self.gt)
        self.assertEqual(d.rel_l1, 0.0)
        self.assertEqual(d.sqnr_db, float("inf"))

    # 已知噪声水平下 SQNR 应为 20 dB
    def test_drift_known_noise_level(self) -> None:
        # 全 1 的参考输出
        ref = np.ones((10, 10), dtype=np.float32)
        d = output_drift(ref * 1.1, ref)            # 噪声功率 = 信号功率 × 0.01 → 20 dB
        self.assertAlmostEqual(d.sqnr_db, 20.0, places=3)

    # 两组指标相差一个常数时,均值差和置信区间都应等于该常数
    def test_paired_bootstrap_detects_constant_shift(self) -> None:
        # 500 个随机样本
        a = np.random.default_rng(1).normal(size=500)
        # 第二组整体加 0.5
        mean, low, high = paired_bootstrap(a, a + 0.5)
        # 均值差和区间上下界都应为 0.5
        self.assertAlmostEqual(mean, 0.5, places=6)
        self.assertAlmostEqual(low, 0.5, places=6)
        self.assertAlmostEqual(high, 0.5, places=6)


# 预处理与后处理的测试
class PreprocessTest(unittest.TestCase):
    # 输出的形状、类型和归一化数值
    def test_shape_dtype_and_normalization(self) -> None:
        # 取最小的档位
        res = RESOLUTIONS["r266"]
        # 全白图片经过预处理
        x = preprocess(np.full((480, 640, 3), 255, dtype=np.uint8), res)
        # 形状应为 [1,3,H,W]
        self.assertEqual(x.shape, (1, 3, res.height, res.width))
        # 类型应为 float32
        self.assertEqual(x.dtype, np.float32)
        # 像素值 255 归一化后应为 1 减均值再除以标准差
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), (1.0 - 0.485) / 0.229, places=4)

    # 非 uint8 输入应报错
    def test_rejects_non_uint8(self) -> None:
        with self.assertRaises(ValueError):
            preprocess(np.zeros((10, 10, 3), dtype=np.float32), RESOLUTIONS["r266"])

    # 后处理应把输出缩放到指定尺寸
    def test_postprocess_restores_size(self) -> None:
        self.assertEqual(postprocess(np.ones((1, 14, 28), dtype=np.float32), (48, 64)).shape, (48, 64))

    # 高宽不是 patch 尺寸整数倍的档位应报错
    def test_resolution_must_be_patch_multiple(self) -> None:
        with self.assertRaises(ValueError):
            Resolution("bad", 500, 686)


# 变体解析与产物路径的测试
class VariantTest(unittest.TestCase):
    # 解析后再取字符串应与原串一致
    def test_round_trip(self) -> None:
        self.assertEqual(Variant.parse("trt:r518:fp16").key, "trt:r518:fp16")
        self.assertEqual(Variant.parse("ov:r364:int8ac").slug, "ov_r364_int8ac")

    # 各种非法的变体字符串都应报错
    def test_invalid(self) -> None:
        # 依次是段数不足、未知后端、未知分辨率、ov 没有的配方、torch 不支持的精度
        for bad in ("trt:r518", "foo:r518:fp16", "trt:r999:fp16", "ov:r518:fp16", "torch:r518:int8"):
            with self.assertRaises(ValueError, msg=bad):
                Variant.parse(bad)

    # 批量解析应去重并保持首次出现的顺序
    def test_parse_many_dedups_in_order(self) -> None:
        keys = [v.key for v in parse_many(["ov:r364:fp32", "ov:r518:fp32", "ov:r364:fp32"])]
        self.assertEqual(keys, ["ov:r364:fp32", "ov:r518:fp32"])

    # 各配方对应的 ONNX 和引擎文件名
    def test_artifact_paths(self) -> None:
        self.assertEqual(Variant.parse("trt:r518:int8mix").onnx_path().name, "da2s_r518.int8mix.onnx")
        self.assertEqual(Variant.parse("trt:r518:fp32").onnx_path().name, "da2s_r518.onnx")
        self.assertEqual(Variant.parse("trt:r518:tf32").engine_path().name, "da2s_r518.tf32.engine")


# 计时工具的测试
class TimingTest(unittest.TestCase):
    # 汇总统计的样本数和中位数
    def test_summarize(self) -> None:
        s = summarize([1.0, 2.0, 3.0])
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["p50"], 2.0)

    # 每轮都是全部变体的一个排列,同一种子结果相同
    def test_schedule_is_permutation_each_round_and_reproducible(self) -> None:
        # 生成 4 轮计划
        plan = interleaved_schedule(["a", "b", "c"], 4, seed=7)
        # 应有 4 轮
        self.assertEqual(len(plan), 4)
        # 每轮排序后都等于原列表
        self.assertTrue(all(sorted(r) == ["a", "b", "c"] for r in plan))
        # 同样的参数再生成一次,结果应相同
        self.assertEqual(plan, interleaved_schedule(["a", "b", "c"], 4, seed=7))

    # 相对极差等于最大值减最小值再除以中位数
    def test_round_spread(self) -> None:
        self.assertAlmostEqual(round_spread([9.0, 10.0, 11.0]), 0.2)


# tar 字节窗口扫描的测试
class TarWindowTest(unittest.TestCase):
    # 只恢复完整落在窗口内的成员
    def test_recovers_only_complete_members(self) -> None:
        # 加载数据准备脚本
        prepare = _load_script("01_prepare_data")
        # 在内存里构造一个含 4 个成员的 tar
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
            # 每个成员 2000 字节,内容是成员序号
            for i in range(4):
                info = tarfile.TarInfo(f"train/scene/{i:05d}.h5")
                payload = bytes([i]) * 2000
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
        # 取出整个 tar 的字节
        blob = buf.getvalue()
        # 每个成员占 512(头)+ 2048(数据);从第 2 个成员的数据中间切入,到第 4 个成员的数据中间切出
        window = blob[512 + 2048 + 512 + 1024: 3 * (512 + 2048) + 512 + 1024]
        # 扫描窗口
        members = prepare.scan_tar_window(window)
        # 只有第 3 个成员完整落在窗口内
        self.assertEqual([m[0] for m in members], ["train/scene/00002.h5"])
        # 内容应与写入时一致
        self.assertEqual(members[0][1], bytes([2]) * 2000)


# 直接运行本文件时执行全部用例
if __name__ == "__main__":
    unittest.main()
