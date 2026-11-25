"""评测任务:数据读取、提示词、打分。三者放在一起,保证"同一道题在任何推理栈上问法和判法都一样"。

只依赖 pyarrow + Pillow(trtllm / ovgenai 两个环境都有)。图片以 JPEG/PNG 字节存在 parquet 里,
按需解码 —— 不一次性解码全部图片(WSL 只有 8 GB 内存)。

打分口径与 lmms-eval 对齐(Qwen2-VL 官方分数也按这类协议报告):
    TextVQA     EvalAI 的答案归一化 + VQA accuracy(对 10 个人工答案留一:min(其余 9 个里与预测相同的个数 / 3, 1) 取平均);
    RealWorldQA 选择题抽取选项字母、其余题数字词转阿拉伯数字后,忽略大小写和标点精确匹配。
P1 会用 HF BF16 复现官方分数来验证这套打分没有写错。
"""

from __future__ import annotations

import hashlib
import io
import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vlmdeploy.config import DATASETS, RAW_DIR

# lmms-eval 的 textvqa_val 提示词后缀;RealWorldQA 的题干自带答题格式要求,不再追加。
TEXTVQA_SUFFIX = "\nAnswer the question using a single word or phrase."


@dataclass(frozen=True)
class VqaSample:
    """一道题。

    Attributes:
        dataset: "textvqa" / "realworldqa"。
        qid: 稳定的题目 id(TextVQA 用 question_id;RealWorldQA 没有 id,用"分片名:行号")。
        question: 原始题干。
        answers: 参考答案(TextVQA 10 个人工答案;RealWorldQA 1 个)。
        image_bytes: 原图的编码字节。
        image_id: 图片 id(TextVQA 一张图平均有 1.5 道题;划分校准 / dev 必须按图片而不是按题,
            否则同一张图会同时出现在校准集和 dev 集里)。RealWorldQA 每题一图,与 qid 相同。
    """

    dataset: str
    qid: str
    question: str
    answers: tuple[str, ...]
    image_bytes: bytes
    image_id: str = ""

    def image(self):  # noqa: ANN201 —— 返回 PIL.Image,避免顶层导入 Pillow
        """解码成 RGB 的 PIL.Image。"""
        # Pillow 只在这里导入
        from PIL import Image

        # 从字节解码并统一转成 RGB
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")

    @property
    def prompt(self) -> str:
        """送给模型的文字部分(图片由各推理栈按聊天模板放在文字前面)。"""
        # TextVQA 的题干后面追加作答格式要求,其余数据集原样返回
        return self.question + TEXTVQA_SUFFIX if self.dataset == "textvqa" else self.question


# 返回某个数据集某个划分的分片文件,按文件名排序
def _shards(dataset: str, split: str) -> list[Path]:
    # 数据集的下载清单
    spec = DATASETS[dataset]
    # 分片所在的本地目录
    root = RAW_DIR / spec.local_dir
    # 按文件名排序,读取顺序固定(同一个 limit 在任何环境取到同样的题,靠的就是它)
    shards = sorted(root / f for f in spec.files if Path(f).name.startswith(f"{split}-"))
    # 没有找到分片时提示先下载
    if not shards:
        raise FileNotFoundError(f"no '{split}' shards for {dataset} under {root}; run scripts/01_download_data.py")
    return shards


# ---------------------------------------------------------------------------
# 数据划分:calib / dev / test 三者互不重叠(沿用项目 A 的纪律)
# ---------------------------------------------------------------------------
#   test  = TextVQA validation(5000 题)与 RealWorldQA(765 题):只用来报最终数字,不参与任何决策;
#   dev   = TextVQA train 第 1 个分片中约 3/4 的图片:做决策(选量化配方、选校准数据)用;
#   calib = 同一分片中其余约 1/4 的图片:只给量化校准用。
# 按 image_id 的 SHA-1 取模划分,而不是按行号:与读取顺序、环境、Python 的 hash 随机化都无关,
# 任何机器上同一张图永远落在同一侧。
# 图片 id 的哈希对它取模,余 0 的归 calib
CALIB_MODULO = 4
# 全部划分名
SPLITS = ("test", "rwqa", "dev", "calib")


def subset_of(image_id: str) -> str:
    """TextVQA train 分片里一张图属于 "calib" 还是 "dev"。"""
    # 计算图片 id 的 SHA-1 摘要
    digest = hashlib.sha1(image_id.encode("utf-8")).digest()
    # 摘要首字节对 CALIB_MODULO 取模,余 0 归 calib,其余归 dev
    return "calib" if digest[0] % CALIB_MODULO == 0 else "dev"


def iter_split(name: str, limit: int | None = None) -> Iterator[VqaSample]:
    """按评测口径的划分名读题。

    Args:
        name: "test"(TextVQA val)/ "rwqa"(RealWorldQA)/ "dev" / "calib"(见上方划分说明)。
        limit: 最多读多少题。

    Yields:
        VqaSample。
    """
    # test 划分:TextVQA 的 validation
    if name == "test":
        yield from iter_samples("textvqa", "validation", limit=limit)
        return
    # rwqa 划分:RealWorldQA 的 test
    if name == "rwqa":
        yield from iter_samples("realworldqa", "test", limit=limit)
        return
    # 其余只接受 dev 和 calib
    if name not in ("dev", "calib"):
        raise ValueError(f"unknown split {name!r}")
    # dev / calib 同读 train 分片,按图片归属过滤;limit 数的是过滤之后的题数,所以不能直接传给 iter_samples
    # 已产出的题数
    n = 0
    # 逐题读取 train 分片
    for sample in iter_samples("textvqa", "train"):
        # 图片不属于本划分的跳过
        if subset_of(sample.image_id) != name:
            continue
        # 产出样本并计数
        yield sample
        n += 1
        # 达到 limit 时结束
        if limit is not None and n >= limit:
            return


def iter_samples(dataset: str, split: str, limit: int | None = None) -> Iterator[VqaSample]:
    """按分片顺序逐题读取(顺序固定,同一个 limit 在任何环境都取到同样的题)。

    Args:
        dataset: "textvqa" 或 "realworldqa"。
        split: TextVQA 为 "validation"(测试集)或 "train"(只下了第 1 个分片,用于校准 / dev);
            RealWorldQA 只有 "test"。
        limit: 最多读多少题;None 读全部。

    Yields:
        VqaSample。
    """
    # pyarrow 只在这里导入
    import pyarrow.parquet as pq

    # 已产出的题数
    n = 0
    # 按分片顺序读取
    for shard in _shards(dataset, split):
        # 打开 parquet 文件
        pf = pq.ParquetFile(shard)
        # 只读用到的列,两个数据集的列名不同
        cols = (["question_id", "image_id", "question", "answers", "image"] if dataset == "textvqa"
                else ["question", "answer", "image"])
        # 当前分片内的行号
        row = 0
        # 只读用到的列、每批 64 行:parquet 是列存,不读的列不进内存,图片字节也不会一次全载入
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            # 逐行处理
            for rec in batch.to_pylist():
                # TextVQA:题目 id 和图片 id 取自数据
                if dataset == "textvqa":
                    # answers 转成元组:VqaSample 是 frozen 数据类,字段都用不可变类型才能哈希
                    sample = VqaSample(dataset, str(rec["question_id"]), rec["question"], tuple(rec["answers"]),
                                       rec["image"]["bytes"], rec["image_id"])
                else:
                    # RealWorldQA 没有 id,用分片名加行号
                    qid = f"{shard.stem}:{row}"
                    sample = VqaSample(dataset, qid, rec["question"], (rec["answer"],), rec["image"]["bytes"], qid)
                # 行号加一
                row += 1
                # 产出样本并计数
                yield sample
                n += 1
                # 达到 limit 时结束
                if limit is not None and n >= limit:
                    return


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
class EvalAIAnswerProcessor:
    r"""VQA / TextVQA 官方(EvalAI)答案归一化,逐行对应 lmms-eval 的同名类。

    有意保留了官方实现的两处怪癖,以便分数可比:
        - PERIOD_STRIP 的 "(?!<=\d)" 实为负向先行断言(原意应是后行断言),行为照抄;
        - 先按"是否含 '标点+空格' 或千分位逗号"决定标点是删除还是替换成空格。
    """

    # 缺撇号的缩写到标准写法的映射
    CONTRACTIONS = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
        "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
        "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
        "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
        "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
        "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
        "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
        "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
        "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
        "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
        "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
        "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
        "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
        "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
        "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
        "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
        "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
        "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
        "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
        "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
        "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
        "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
        "youve": "you've",
    }
    # 数字词到阿拉伯数字的映射
    NUMBER_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                  "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    # 归一化时去掉的冠词
    ARTICLES = ("a", "an", "the")
    # 句点的匹配规则,写法照抄官方实现
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    # 数字之间的千分位逗号
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    # 要处理的标点
    PUNCTUATIONS = (";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`",
                    ",", "?", "!")

    # 处理标点:按官方规则删除或替换成空格,再去掉句点
    def _punctuation(self, text: str) -> str:
        out = text
        # 逐个标点处理
        for p in self.PUNCTUATIONS:
            # 标点紧邻空格或文本含千分位逗号时直接删除,否则替换成空格
            if (p + " " in text or " " + p in text) or self.COMMA_STRIP.search(text) is not None:
                out = out.replace(p, "")
            else:
                out = out.replace(p, " ")
        # 去掉句点
        return self.PERIOD_STRIP.sub("", out)

    def _digit_article(self, text: str) -> str:
        # 转小写后逐词处理:数字词转阿拉伯数字、去冠词、还原缩写
        # 数字词转阿拉伯数字
        words = [self.NUMBER_MAP.get(w, w) for w in text.lower().split()]
        # 去掉冠词
        words = [w for w in words if w not in self.ARTICLES]
        # 还原缩写后用空格连接
        return " ".join(self.CONTRACTIONS.get(w, w) for w in words)

    def __call__(self, answer: str) -> str:
        """归一化一个答案字符串。"""
        # 转小写,去掉逗号和问号,把 's 与前面的词分开
        text = answer.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
        # 换行和制表符换成空格
        text = text.replace("\n", " ").replace("\t", " ").strip()
        # 先处理标点,再处理数字词、冠词和缩写
        return self._digit_article(self._punctuation(text))


# 模块级单例。别的模块用下面公开的 normalize_answer,不直接碰这个带下划线的私有名字
_EVALAI = EvalAIAnswerProcessor()


def normalize_answer(answer: str) -> str:
    """EvalAI 归一化后的答案;用来判断两个答案是否“归一化后相同”(冒烟测试的宽松比对)。"""
    return _EVALAI(answer)


def vqa_accuracy(prediction: str, answers: tuple[str, ...]) -> float:
    """VQA accuracy:对 10 个人工答案做留一,每次"其余 9 个里与预测相同的个数 / 3,封顶 1",取平均。

    Returns:
        0 ~ 1 之间的分数。4 个及以上标注者给出同一答案时,预测该答案得 1 分;恰好 3 个时得 0.9
        (留掉这 3 个之一的那 3 次只剩 2 个匹配)。
    """
    # 归一化预测答案
    pred = _EVALAI(prediction)
    # 归一化全部人工答案
    gts = [_EVALAI(a) for a in answers]
    # 每次留一得到的分数
    scores = []
    # 依次留掉一个人工答案
    for i in range(len(gts)):
        # 其余的人工答案
        others = gts[:i] + gts[i + 1:]
        # 与预测相同的个数除以 3,封顶为 1
        scores.append(min(1.0, sum(g == pred for g in others) / 3))
    # 取平均
    return statistics.mean(scores)


# _OPTION:独立的选项字母(两侧是词边界,单词里的 A–E 不算);_PUNCT:标点(既不是字母数字下划线、也不是空白的字符)
_OPTION = re.compile(r"\b([A-E])\b")
_PUNCT = re.compile(r"[^\w\s]")


def _plain(text: str) -> str:
    # RealWorldQA 非选择题用的简单归一化:标点换成空格、转小写、数字词转阿拉伯数字
    # 标点换成空格后转小写并分词
    words = _PUNCT.sub(" ", text.lower()).split()
    # 数字词转阿拉伯数字后用空格连接
    return " ".join(EvalAIAnswerProcessor.NUMBER_MAP.get(w, w) for w in words)


def realworldqa_correct(prediction: str, answer: str) -> float:
    """RealWorldQA 判分:参考答案是单个选项字母时,取预测里第一个独立的 A–E;否则归一化后精确匹配。

    Returns:
        1.0 或 0.0。
    """
    # 参考答案是单个选项字母时按选择题判分
    if re.fullmatch(r"[A-E]", answer.strip()):
        # 取预测里第一个独立的选项字母
        match = _OPTION.search(prediction.strip())
        # 找到且与参考答案相同才得分
        return float(match is not None and match.group(1) == answer.strip())
    # 其余题目归一化后精确匹配
    return float(_plain(prediction) == _plain(answer))


def score(sample: VqaSample, prediction: str) -> float:
    """按样本所属数据集选择判分函数。"""
    # TextVQA 用 VQA accuracy
    if sample.dataset == "textvqa":
        return vqa_accuracy(prediction, sample.answers)
    # RealWorldQA 用精确匹配,参考答案只有一个
    return realworldqa_correct(prediction, sample.answers[0])
