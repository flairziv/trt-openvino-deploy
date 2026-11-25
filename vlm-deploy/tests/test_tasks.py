"""tasks 模块的单元测试:打分函数的已知用例 + 数据读取的顺序稳定性。

运行:python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import DATASETS, RAW_DIR
from vlmdeploy.tasks import EvalAIAnswerProcessor, realworldqa_correct, vqa_accuracy

# 一道题的 10 个人工答案(dakota × 5、dakota digital × 3、另外两个各 1 次),VqaAccuracyTest 的用例共用
TEN = ("dakota", "dakota", "dakota", "dakota digital", "dakota digital", "dakota digital", "dakota", "dakota",
       "nous les gosses", "clos culombu")


# vqa_accuracy 和答案归一化的用例
class VqaAccuracyTest(unittest.TestCase):
    def test_majority_answer_scores_one(self) -> None:
        # "dakota" 有 5 个标注:留掉哪一个都还剩至少 4 个匹配,每次都是满分
        self.assertAlmostEqual(vqa_accuracy("dakota", TEN), 1.0)

    def test_minority_answer_scores_partial(self) -> None:
        # "dakota digital" 有 3 个标注:留掉其中 1 个时只剩 2 个匹配(2/3 分),留掉其余 7 个时仍是 3 个(1 分)
        self.assertAlmostEqual(vqa_accuracy("dakota digital", TEN), (3 * (2 / 3) + 7 * 1.0) / 10)

    def test_single_annotator_answer(self) -> None:
        # 只有 1 个标注者给的答案:留掉它时 0 分,其余 9 次各 1/3 分
        self.assertAlmostEqual(vqa_accuracy("clos culombu", TEN), 9 * (1 / 3) / 10)

    def test_normalization(self) -> None:
        # 归一化器
        proc = EvalAIAnswerProcessor()
        self.assertEqual(proc("The Dakota."), "dakota")   # 去冠词、去句点、转小写
        self.assertEqual(proc("two"), "2")                # 数字词转数字
        self.assertEqual(proc("dont"), "don't")           # 补回缩写里的撇号
        self.assertEqual(proc("1,000"), "1000")           # 去千分位逗号
        self.assertAlmostEqual(vqa_accuracy("Dakota.", TEN), 1.0)   # 打分前先做同样的归一化

    def test_wrong_answer_scores_zero(self) -> None:
        # 人工答案里没有的答案得 0 分
        self.assertEqual(vqa_accuracy("canon", TEN), 0.0)


# realworldqa_correct 的用例
class RealWorldQaTest(unittest.TestCase):
    def test_option_letter(self) -> None:
        # 选择题:取预测里第一个独立的选项字母比较
        self.assertEqual(realworldqa_correct("C", "C"), 1.0)
        self.assertEqual(realworldqa_correct("C.", "C"), 1.0)
        self.assertEqual(realworldqa_correct("The answer is B", "C"), 0.0)

    def test_words_and_numbers(self) -> None:
        # 非选择题:忽略大小写和标点,数字词转成数字后比较
        self.assertEqual(realworldqa_correct("Yes.", "Yes"), 1.0)
        self.assertEqual(realworldqa_correct("three", "3"), 1.0)
        self.assertEqual(realworldqa_correct("No", "Yes"), 0.0)


# calib / dev 划分的用例
class SplitTest(unittest.TestCase):
    def test_subset_is_deterministic_and_roughly_quarter(self) -> None:
        from vlmdeploy.tasks import subset_of

        # 构造 4000 个图片 id
        ids = [f"img{i:05d}" for i in range(4000)]
        # 第一次划分的结果
        first = [subset_of(i) for i in ids]
        # 再算一次结果相同
        self.assertEqual(first, [subset_of(i) for i in ids])
        # calib 所占的比例
        share = first.count("calib") / len(ids)
        # 比例在 0.2 到 0.3 之间
        self.assertGreater(share, 0.2)
        self.assertLess(share, 0.3)


# 数据集是否已下载
def _downloaded(dataset: str) -> bool:
    # 两个数据集可以分开下载(01_download_data.py --only),所以各看各的目录
    return (RAW_DIR / DATASETS[dataset].local_dir).exists()


# 数据读取的用例,需要已下载的数据集
@unittest.skipUnless(_downloaded("textvqa"), "TextVQA not downloaded")
class IterSamplesTest(unittest.TestCase):
    def test_order_is_stable_and_complete(self) -> None:
        # 没装 pyarrow 时跳过
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        from vlmdeploy.tasks import iter_samples

        # 读前 5 题的题号
        first = [s.qid for s in iter_samples("textvqa", "validation", limit=5)]
        # 再读一次顺序相同
        self.assertEqual(first, [s.qid for s in iter_samples("textvqa", "validation", limit=5)])
        self.assertEqual(first[0], "34602")   # 钉住第一题的 id:读取顺序一变这里就报错

    @unittest.skipUnless(_downloaded("realworldqa"), "RealWorldQA not downloaded")
    def test_realworldqa_sample_shape(self) -> None:
        # 没装 pyarrow 时跳过
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        from vlmdeploy.tasks import iter_samples

        # 读第一题
        sample = next(iter_samples("realworldqa", "test", limit=1))
        # 参考答案只有一个
        self.assertEqual(len(sample.answers), 1)
        # 题干以答题格式要求结尾
        self.assertTrue(sample.prompt.endswith("nothing else."))

    def test_calib_and_dev_share_no_image(self) -> None:
        # 没装 pyarrow 时跳过
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        from vlmdeploy.tasks import iter_split

        # 数据纪律:校准集与决策集不能有任何一张共同的图
        calib = {s.image_id for s in iter_split("calib")}
        dev = {s.image_id for s in iter_split("dev")}
        self.assertTrue(calib and dev)   # 两边都非空,否则下面的"无交集"是空话
        # 两个集合没有交集
        self.assertFalse(calib & dev)


# 直接运行本文件时执行全部用例
if __name__ == "__main__":
    unittest.main()
