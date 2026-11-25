# 代码导读:逐文件、逐行读 vlm-deploy

这份导读带你把整个仓库读一遍:每个文件解决什么问题,每一段代码为什么这样写。
配合源码一起看 —— 左边开源码,右边开这份文档。

- **格式**:每个文件一节,节内是"行 / 代码 / 说明"三列的表。"行"是源码行号(或起止范围),
  "代码"列里反引号包住的片段是这几行里原样出现的文字,用来对位置。
- **行号有机器把关**:`make docs-check`(`tools/check_docs.py`)检查每一行表格里的片段确实出现在标注的行号范围内,
  代码增删行导致的过期能查出来;改了代码之后跑一遍,报错的就是要更新的导读行。它不检查"说明"一列写得对不对 ——
  那一列是对照源码手写的,另做过一轮独立的自动复核(抽查约 150 行),发现不符请以代码为准。
- 想先看结论与数据:[deployment-record.md](deployment-record.md);想看每个阶段的结果表:仓库根目录的 README。

---

## 0. 先建立全景

### 0.1 这个仓库在做什么

把同一个视觉语言模型(Qwen2-VL-2B)部署到两条推理栈上,量清楚"一次图文问答的时间花在哪",再逐段优化:

```
                    ┌──────────────── GPU 栈(conda 环境 trtllm)────────────────┐
图片 + 问题 ──预缩放──┤ TensorRT-LLM:视觉塔(HF 实现 或 TensorRT 引擎)→ 语言模型(BF16 / FP8 / AWQ-INT4) │──→ 答案
 (imaging.py)       └────────────────────────────────────────────────────────────┘
                    ┌──────────────── CPU 栈(conda 环境 ovgenai)───────────────┐
                    │ OpenVINO GenAI:视觉塔 IR(FP16 / INT8)→ 语言模型 IR(FP16 / INT8 / INT4)      │──→ 答案
                    └────────────────────────────────────────────────────────────┘
```

一次请求分三段,全仓库的实验都围绕这三段展开:

| 段 | 做什么 | 受什么限制 | 谁能加速它 |
|---|---|---|---|
| 视觉编码 | 视觉塔 32 层,处理图像块 | 算力 | 视觉塔引擎(GPU)、视觉塔 INT8(CPU) |
| prefill | 语言模型一次吃进"图像 token + 文字 token" | 算力 | FP8(GPU)、INT8 / INT4(CPU) |
| decode | 逐个生成 token,每步读一遍全部权重 | 显存 / 内存带宽 | 一切把权重变小的量化 |

首字延迟(TTFT)= 视觉编码 + prefill。

### 0.2 三个贯穿全仓库的概念

- **变体(variant)**:一个字符串指代"哪条栈 + 什么精度",如 `trtllm-fp8-img`、`ov-int4-vis8w`。
  评测、计时、演示都只认这个名字,由 `vlmdeploy/variants.py` 翻译成"用哪个后端、模型在哪个目录"。
- **档位(budget)**:图像最多占多少个 token。`t256 / t576 / t1024` 保持长宽比;`s256 / s576 / s1024` 是方图
  (给静态形状的视觉塔引擎用)。1 个图像 token = 28×28 像素。
- **划分(split)**:`test`(只报数)、`rwqa`(只报数)、`dev`(做决策)、`calib`(只给量化校准)。四者互不重叠。

### 0.3 目录与依赖方向

```
vlmdeploy/   公共库 —— 被所有脚本导入;自己不导入任何重依赖(torch / tensorrt_llm / openvino 都在函数内部才 import)
scripts/     流水线 —— 编号即执行顺序;每个脚本读前面步骤的产物,把结果写进 results/
cpp/         CPU 栈的 C++ 程序 —— 直接调 OpenVINO GenAI 的 C++ 接口,不依赖 Python
tools/       独立小工具
tests/       单元测试(只测公共库里不需要 GPU / 模型的部分)
results/     机器可读结果 —— README 表格的唯一来源
```

数据流是单向的:`scripts/NN_*.py` → `results/*.json` → `scripts/90_report.py` → README 的表。

### 0.4 建议的阅读顺序

**半小时版**(抓主线):`config.py` → `imaging.py` → `variants.py` → `backends.py` → `10_eval.py` → `11_bench_gpu.py`。
读完就知道"一道题怎么被送进模型、怎么打分、怎么计时"。

**完整版**(本文的顺序):先读公共库(第 1 节),再读兼容补丁(第 2 节),然后按编号读流水线(第 3 节)、C++ 程序(第 4 节),
最后是工具、测试与构建文件(第 5、6 节)。

### 0.5 全仓库通用的几个写法

读代码前先知道这几条,后面就不会反复疑惑:

| 写法 | 为什么 |
|---|---|
| 每个文件第一行是长 docstring | 它是这个文件的设计说明:为什么这么做、用法、产物。读文件先读它 |
| `from __future__ import annotations` | 类型注解不在运行时求值,可以写 `int \| None` 这类新语法,也避免为了注解去 import 重依赖 |
| 脚本开头 `sys.path.insert(0, 仓库根目录)` | 脚本直接 `python scripts/xx.py` 运行,不要求先 `pip install -e .` |
| `import torch` 写在函数里而不是文件顶部 | 两个 conda 环境装的东西不同;顶部导入会让另一个环境连 `import` 这个文件都失败 |
| `# noqa: ANN001 —— 理由` | 这一行确实标不了类型(PIL、torch、第三方对象),显式豁免静态检查并写明原因 |
| `main()` 返回 `int`,末尾 `sys.exit(main())` | 返回值就是进程退出码,Makefile 靠它判断这一步成没成 |
| 评测、计时、压缩的结果 JSON 都带 `time` 字段 | 每个数字都能追溯到是哪次运行产生的 |

---

## 1. 公共库 `vlmdeploy/`

### 1.1 `vlmdeploy/__init__.py` —— 包的门面

| 行 | 代码 | 说明 |
|---|---|---|
| 1–6 | `导入即轻量` | 包级 docstring,立下全库最重要的一条规矩:顶层不导入重依赖。下载步骤只有标准库,也要能 `import vlmdeploy` |
| 9 | `__version__` | 版本号,与 `pyproject.toml` 的 `version` 保持一致(每个阶段一个次版本) |

### 1.2 `vlmdeploy/config.py` —— 单一事实来源

所有路径、模型常量、档位、数据清单都定义在这里。脚本里不写魔数:要改一个参数,只改这一处。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–5 | `单一事实来源` | 文件说明。关键一句:改动任一"协议"参数都会让已有的 `results/` 失效 |
| 7–11 | `from dataclasses import dataclass` | 只用标准库:`os` 读环境变量,`dataclass` 定义数据清单,`Path` 拼路径 |
| 17 | `REPO_ROOT` | 仓库根目录 = 本文件往上两级。用 `resolve()` 得到绝对路径,不依赖从哪个目录启动脚本 |
| 19–23 | `RESULTS_DIR` | 仓库内的几个目录:`data/raw` 放下载的 parquet(只读),`results/` 放机器可读结果,`logs/` 放日志 |
| 25–29 | `VLM_ARTIFACTS` | 大文件(模型、量化产物、IR)放仓库外,位置由环境变量决定,默认 D 盘。注释解释了为什么不放 WSL 家目录:虚拟盘只增不减 |
| 31–36 | `TRTLLM_CKPT_DIR` | 大文件目录下的四个子目录:HF 原始模型、安装包、OpenVINO IR、TensorRT-LLM 量化 checkpoint |
| 39 | `HF_ENDPOINT` | Hub 地址,默认走镜像;同样可用环境变量覆盖 |
| 45–47 | `MODEL_REPO` | 模型的仓库名与本地目录。全项目只有这一个模型 |
| 49–51 | `tie_word_embeddings` | 注释记录了逐项核对过的模型结构。最后一句(输入 embedding 与输出层共享权重)是第 2 节那个"输出全是感叹号"问题的伏笔 |
| 55–59 | `TOKEN_PIXELS` | 视觉塔按 14 像素切块,再 2×2 合并成一个 token,所以 1 个图像 token = 28×28 像素。单张图会被复制成 2 帧 |
| 61–63 | `IMAGE_MEAN` | 归一化用 CLIP 的均值方差,不是 ImageNet 的。注释提醒不能照抄:照抄别的项目的常数不会报错,只会让结果悄悄变差 |
| 65–70 | `DEFAULT_MAX_PIXELS` | 处理器默认的像素范围,上限高达 16384 个 token —— 所以评测必须自己显式控制 token 数 |
| 72–73 | `TOKEN_BUDGETS` | 三个保持长宽比的档位。字典的键是档位名,值是 token 上限 |
| 74–75 | `SQUARE_BUDGETS` | 三个方图档位,只给视觉塔引擎的端到端对比用(引擎是静态形状,需要所有图落在同一个网格上) |
| 82–96 | `class HubDataset` | 一个不可变的数据类(`frozen=True`),描述"Hub 上的一个数据集要下载哪些文件"。docstring 用 Google 风格的 `Attributes:` 逐个说明字段 |
| 100–113 | `"textvqa": HubDataset(` | TextVQA:验证集 3 个分片当测试集;训练集只取第 1 个分片,从中切出校准集与 dev 集 |
| 114–121 | `"realworldqa": HubDataset(` | RealWorldQA:765 道车载 / 真实场景的题,用官方仓库。许可写进记录,便于 README 引用 |

### 1.3 `vlmdeploy/imaging.py` —— 把图片预缩放到 token 预算内

**要解决的问题**:两条推理栈各有一份图像预处理(一个 Python,一个 C++)。只要处理器里真的发生缩放,两份实现的插值细节就可能不同,
"两栈精度差"里就混进了"预处理差"。办法是自己先按官方规则缩好,处理器再算一遍时尺寸不变,缩放变成恒等。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–14 | `缩放变成恒等` | 文件说明:为什么自己预缩放(第一段),以及 token 与像素的换算(第二段) |
| 18–20 | `import math` | 只依赖标准库与 `config`;PIL 在用到的函数里才导入 |
| 22 | `MAX_ASPECT_RATIO` | 长宽比上限 200,与官方实现一致,超过就拒绝 |
| 25–26 | `def smart_resize` | 官方的目标尺寸规则。参数默认值取自 `config`:倍数 28、处理器默认的像素范围 |
| 27–47 | `四舍六入五成双` | docstring。"注意"一段很关键:Python 的 `round` 是银行家舍入,这里必须和官方用同一个 `round`,否则恰好落在 .5 的尺寸会差一行 token |
| 49–50 | `raise ValueError` | 长宽比超限直接报错(与官方实现一致) |
| 52–53 | `h_bar = round(height / factor) * factor` | 第一步:高、宽各自舍入到 28 的整数倍 |
| 55–59 | `math.floor(height / beta / factor)` | 面积超过上限:按面积等比缩小(`beta` 是边长的缩放系数),再**向下**取整到 28 的倍数,保证不超预算;`max(factor, …)` 防止缩成 0 |
| 60–64 | `math.ceil(height * beta / factor)` | 面积低于下限:等比放大后**向上**取整 |
| 66 | `return h_bar, w_bar` | 返回 (高, 宽),注意顺序是先高后宽(PIL 的 `size` 是先宽后高,后面调用时要小心) |
| 69–79 | `def image_tokens` | 给定尺寸算图像 token 数 = (高 / 28) × (宽 / 28)。尺寸不是 28 的倍数就报错 —— 说明图片没经过预缩放,宁可崩也不要算出一个错的数 |
| 82–94 | `def budget_size` | 把"token 预算"换算成 `max_pixels`(预算 × 28²)再调 `smart_resize`。非方图的实际 token 数通常略小于预算 |
| 97 | `def resize_to_budget` | 真正缩放图片的入口。`noqa` 的理由:参数和返回值是 PIL 的图片对象,为了不在文件顶部 import PIL,这里不写类型 |
| 108–111 | `image.convert("RGB")` | 函数内才导入 PIL;不是 RGB 就转成 RGB(灰度图、带透明通道的图都会遇到) |
| 113–116 | `return image` | 算出目标尺寸;已经是目标尺寸就原样返回,**不重复插值**(缩好的图再过一遍这个函数不会变)。"处理器再算一遍是恒等"靠的是另一件事:输出尺寸是 28 的倍数且在像素范围内 |
| 118 | `Image.Resampling.BICUBIC` | 用双三次插值,与 HF 处理器用的 PIL 插值一致。注意 `resize` 的参数顺序是 (宽, 高) |
| 121–132 | `def square_side_tokens` | 方图每边多少个 token(256 → 16)。用整数开方 `isqrt` 再平方回去核对,预算不是完全平方数就报错 |
| 135–141 | `def square_to_budget` | 方图档位:中心裁成正方形再缩放。docstring 说明了用途与代价(裁掉了两侧内容,分数只能和方图档位比) |
| 146–153 | `image.crop(` | 取短边为边长,算出居中的裁剪框 `(左, 上, 右, 下)` |
| 155–157 | `square_side_tokens(budget_tokens) * TOKEN_PIXELS` | 目标边长 = 每边 token 数 × 28,缩放到这个正方形 |

### 1.4 `vlmdeploy/tasks.py` —— 取题、划分、提示词、打分

**要解决的问题**:保证"同一道题在任何推理栈上问法和判法都一样"。取题、提示词、打分放在同一个文件里,
换后端时这三样一个字都不会变,分数之差才能归到推理实现上。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–10 | `lmms-eval` | 文件说明:只依赖 pyarrow 与 Pillow;打分口径与 lmms-eval 对齐,P1 用复现官方分数来验证没写错 |
| 14–22 | `from collections.abc import Iterator` | 标准库加本包的 `config`。`hashlib` 用来做划分,`io` 用来从字节解码图片,`re` 用来做答案归一化 |
| 25 | `TEXTVQA_SUFFIX` | TextVQA 每道题后面追加的那句"用一个词或短语回答"。RealWorldQA 的题干自带格式要求,不加 |
| 28–47 | `class VqaSample` | 一道题的数据类。`image_bytes` 存的是**编码后的字节**而不是解码的图片 —— 几千道题的图全部解码会吃光 8 GB 内存 |
| 38–39 | `划分校准 / dev 必须按图片` | `image_id` 字段的说明:一张图平均 1.5 道题,按题划分会让同一张图同时出现在校准集和 dev 集 |
| 49–55 | `def image(self)` | 用到时才解码成 RGB 图片。PIL 在方法内导入 |
| 57–61 | `def prompt(self)` | 送给模型的文字部分。图片由各推理栈按聊天模板放在文字前面,这里不管 |
| 65–75 | `def _shards` | 找出某个数据集、某个划分对应的 parquet 分片,按文件名排序(顺序固定)。找不到就报错并提示先跑下载脚本 |
| 81–85 | `按 image_id 的 SHA-1 取模划分` | 划分规则的说明。用 SHA-1 而不是行号或 Python 的 `hash()`:与读取顺序、机器、进程都无关 |
| 87–89 | `CALIB_MODULO = 4` | 约 1/4 的图进校准集,其余进 dev;`SPLITS` 列出四个划分名 |
| 92–97 | `def subset_of` | 一张图属于 calib 还是 dev:取 SHA-1 的第一个字节模 4,等于 0 的进 calib |
| 100–109 | `def iter_split` | 按划分名读题的统一入口,评测 / 计时 / 校准都调它。是个生成器,逐题产出、不占内存 |
| 111–117 | `yield from iter_samples("textvqa", "validation"` | `test` 和 `rwqa` 直接对应数据集自带的划分,原样转发 |
| 119–134 | `if subset_of(sample.image_id) != name` | `dev` 和 `calib` 都从训练分片里读,用 `subset_of` 过滤;`limit` 数的是过滤之后的题数 |
| 137–148 | `def iter_samples` | 底层的读取函数:按分片顺序逐题读。docstring 强调顺序固定 —— 同一个 `limit` 在任何环境取到的都是同样的题 |
| 150–164 | `pf.iter_batches(batch_size=64, columns=cols)` | 每次只读 64 行、只读需要的列。parquet 是列存,不读的列不会进内存 |
| 166–171 | `tuple(rec["answers"])` | TextVQA:10 个人工答案转成元组(数据类是不可变的,元组可哈希) |
| 172–175 | `qid = f"{shard.stem}:{row}"` | RealWorldQA 没有题目 id,用"分片名:行号"造一个稳定的 id;每题一图,所以 `image_id` 也用它 |
| 177–183 | `if limit is not None and n >= limit` | 逐题产出;够数就结束 |
| 189–195 | `class EvalAIAnswerProcessor` | VQA 官方的答案归一化,逐行照抄 lmms-eval。docstring 记下了**有意保留的两处官方怪癖**:为了分数可比,连 bug 也照抄 |
| 198–227 | `CONTRACTIONS = {` | 缩写词表("dont" → "don't")。照抄官方 |
| 229–232 | `NUMBER_MAP` | 数字词转阿拉伯数字("two" → "2"),以及要去掉的冠词 |
| 234–239 | `PERIOD_STRIP` | 两个正则与标点表。第一个正则里的 `(?!<=\d)` 就是 docstring 说的那处官方笔误 |
| 242–252 | `def _punctuation` | 标点处理:视情况删除或替换成空格(判断条件也是照抄官方的) |
| 254–261 | `def _digit_article` | 转小写、数字词转数字、去冠词、还原缩写 |
| 263–270 | `def __call__` | 归一化的入口:先做基本清理,再过上面两步。实例可以像函数一样调用 |
| 274–279 | `def normalize_answer` | 模块级的归一化实例,以及对外公开的函数。别的模块用这个公开名字,不直接碰带下划线的私有实例 |
| 282–302 | `def vqa_accuracy` | VQA 准确率:对 10 个人工答案做留一 —— 每次拿掉 1 个,看剩下 9 个里有几个与预测相同,除以 3、封顶 1,最后取平均。4 个及以上标注者给出同一答案,预测它得满分;恰好 3 个时是 0.9(留掉这 3 个之一的那 3 次只剩 2 个匹配)|
| 306–315 | `def _plain` | RealWorldQA 用的简单归一化:标点换成空格(所以 a-b 会拆成两个词)、转小写、数字词转数字。两个正则一个匹配独立的选项字母,一个匹配标点 |
| 318–331 | `def realworldqa_correct` | 参考答案是单个字母时,取预测里第一个独立的 A–E 来比;否则归一化后精确匹配 |
| 334–340 | `def score` | 统一的打分入口:按题目所属数据集选判分函数。评测脚本只调这一个 |

### 1.5 `vlmdeploy/timing.py` —— 计时协议

**要解决的问题**:笔记本上测时间有两个噪声源 —— 升温降频(先测的占便宜)和偶发的慢样本。这个文件提供对付它们的三样工具,
且只用标准库(两个 conda 环境都要用,不想为它引入 numpy)。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–13 | `同一轮内配对相减` | 文件说明。最后一段是全项目计时的核心纪律:差值类指标要在同一轮内配对相减再取中位数,不能拿两个中位数相减 |
| 17–20 | `import random` | 只有标准库 |
| 23–43 | `def percentile` | 线性插值分位数,与 numpy 的默认算法一致(单元测试里有对照)。空样本直接报错 |
| 37–43 | `pos = (len(data) - 1) * q / 100` | 排序后,目标位置落在两个样本之间时按距离线性插值 |
| 46–51 | `def summarize` | 把一组样本压成 `{n, p50, p90, mean, min}`。所有计时结果都用这个格式落盘 |
| 54–68 | `def interleaved_schedule` | 交错轮次计划:每一轮包含全部测量项各一次,顺序随机打乱。种子固定,所以计划本身可复现 |
| 71–76 | `def round_spread` | 各轮中位数的相对极差 (最大 − 最小) / 中位数,衡量"这个数字稳不稳",小于 5% 算稳定 |
| 80–87 | `def ratio_ci` | 两组配对样本的中位数之比,以及它的 bootstrap 95% 置信区间。用途:报"快了几倍"时给出区间,免得把噪声当提速。**目前只有单元测试调用它**:README 里的加速比是两个 P50 相除,没有带区间 |
| 89–104 | `rng.choice(idx)` | 有放回地重抽 2000 次,每次算一个比值,取 2.5% 和 97.5% 分位作为区间。两组用**同一批下标**重抽,保持配对关系 |

### 1.6 `vlmdeploy/variants.py` —— 变体注册表

**要解决的问题**:评测、计时、演示都要知道"`trtllm-fp8-img` 是什么"。只在这里翻译一次。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–11 | `命名规则` | 文件说明里就是完整的变体清单与命名规则:`<栈>-<语言模型精度>[-<校准数据>]` |
| 13–17 | `from vlmdeploy.config import` | 只导入路径常量。后端类在下面的函数里才导入 |
| 20–21 | `VISION_ENGINE_DIR` | GPU 量化的四个后缀,以及视觉塔引擎所在的目录 |
| 24–27 | `def is_gpu` | 变体跑在 GPU 上吗。决定用哪个 conda 环境,也决定能不能与别的任务并行 |
| 30–41 | `def trtllm_checkpoint` | TensorRT-LLM 变体的模型目录:BF16(含视觉塔引擎变体)直接用原始 HF 目录;量化变体去量化 checkpoint 目录。不认识的名字报错,不会悄悄落到默认值 |
| 44–50 | `def ov_ir_dir` | OpenVINO 变体的 IR 目录:去掉 `ov-` 前缀就是子目录名 |
| 55 | `def make_answerer` | 按变体名构造答题后端,评测与演示都经它加载模型(两个计时脚本要拿逐请求的性能指标,是各自直接构造后端的,只借用这里的路径解析;两个冒烟脚本也各自构造,路径直接取自 `config`,不经过本模块)。返回类型不标注,是因为后端类要等到函数里才导入 |
| 56–61 | `vision_grid` | docstring 解释了 `vision_grid` 参数的来历:三个引擎同时加载在 8 GB 卡上放不下,所以只加载本次用到的那个网格 |
| 62–65 | `from vlmdeploy import backends` | 到这里才导入后端模块(它本身也是轻量的,真正的重依赖在各后端的构造函数里) |
| 68–71 | `VLMDEPLOY_VISION_TRT_GRIDS` | 用环境变量把"只加载哪个网格"传给 worker 进程;不需要时清掉,避免上一次调用的值残留 |
| 73–84 | `return backends.TrtllmAnswerer(` | 按前缀分发到三个后端类。`-vistrt` 结尾的变体才带上引擎目录 |

### 1.7 `vlmdeploy/backends.py` —— 三个后端的统一答题接口

**要解决的问题**:评测脚本只认 `answer(一批题) → 一批答案`。换后端、换精度都不改取题、提示词、打分中的任何一步。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–12 | `两栈对照能成立的前提` | 文件说明:统一接口的意义;三个后端的提示词各从哪来、如何核对一致;重依赖只在构造函数里导入 |
| 14–26 | `from vlmdeploy.smoke import chat_prompt` | 顶层只导入标准库和本包的轻量模块。`chat_prompt` 是 GPU 两个后端共用的提示词函数 |
| 30–33 | `MAX_NEW_TOKENS = 16` | 两个协议常量:最多生成 16 个 token(参考答案都是短语或选项字母);CPU 用 4 线程。冒烟脚本导入前者,冒烟、压缩、CPU 计时脚本导入后者 |
| 36–47 | `class Answerer` | 后端基类,是个数据类:变体名、加载耗时、附加信息 |
| 49–52 | `def answer(self, batch` | 对外的统一接口:调子类的 `_answer`,再把每个答案去掉首尾空白。去空白放在基类里,三个后端的行为保证一致 |
| 55–60 | `raise NotImplementedError` | 子类必须实现 `_answer`;`close` 默认什么都不做(只有 TensorRT-LLM 需要关 worker 进程) |
| 63–68 | `class HfAnswerer` | HF transformers 参考后端。docstring 解释为什么逐题、不做批量:批量要左填充,会引入"填充是否影响答案"这个新变量 |
| 71–75 | `import torch` | 构造函数里才导入 torch 与 transformers |
| 78–89 | `attn_implementation="sdpa"` | 加载处理器与模型:BF16、SDPA 注意力、放到 GPU、设为推理模式;记下加载耗时 |
| 92–99 | `self.processor(text=[chat_prompt(` | 逐题:用聊天模板生成提示词,连同图片交给处理器得到模型输入 |
| 101–105 | `ids[:, inputs["input_ids"].shape[1]:]` | 贪心生成;`generate` 返回的是"输入 + 输出"的完整序列,切掉输入部分再解码 |
| 107–114 | `self._torch.cuda.empty_cache()` | 每 50 题归还一次 CUDA 缓存。上面 4 行注释是 P1 踩的坑:原生分辨率下每张图序列长度不同,缓存只涨不跌,显存占满后被换到共享内存 |
| 118–128 | `class TrtllmAnswerer` | TensorRT-LLM 后端。docstring 说明四个参数;`checkpoint` 给原始目录就是 BF16,给量化目录就是对应精度 |
| 131–141 | `os.environ["VLMDEPLOY_VISION_TRT"]` | 视觉塔引擎目录通过环境变量传给 worker 进程(引擎是在 worker 里由 sitecustomize 加载的)。必须在构造 `LLM` 之前设置;不用引擎时清掉 |
| 143–145 | `from tensorrt_llm import LLM, SamplingParams` | 环境变量设好之后才导入 TensorRT-LLM |
| 148–155 | `free_gpu_memory_fraction=0.25` | 构造 `LLM`。4 行注释解释两个设置:KV cache 只给剩余显存的 25%(默认 90%,会挤掉视觉塔的空间);关闭 KV 块复用(与计时脚本保持同一配置) |
| 156–161 | `SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)` | 处理器只用它的聊天模板;`temperature=0` 即贪心解码;记下加载耗时和模型位置 |
| 164–169 | `"multi_modal_data": {"image": [img]}` | 整批题一次提交:每题是"提示词 + 图片"的字典,TensorRT-LLM 自己调度批处理 |
| 171–173 | `self.llm.shutdown()` | 关闭 TensorRT-LLM:结束 worker 进程、释放显存。评测脚本在 `finally` 里调它 |
| 176–182 | `class OpenVinoAnswerer` | OpenVINO GenAI 后端,逐题(GenAI 的 VLM 管线一次处理一个请求) |
| 185–203 | `ov_genai.VLMPipeline(str(ir_dir), "CPU", INFERENCE_NUM_THREADS=CPU_THREADS)` | 加载管线到 CPU、指定线程数;生成配置只设最大 token 数,贪心是默认 |
| 206–218 | `np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None])` | 逐题:PIL 图片转成 `[1, 高, 宽, 3]` 的 uint8 张量;只传问题文字,GenAI 自己套聊天模板 |

### 1.8 `vlmdeploy/smoke.py` —— 冒烟测试的公共部分与共用提示词

| 行 | 代码 | 说明 |
|---|---|---|
| 1–7 | `这条推理栈装好了、输出是对的吗` | 文件说明:冒烟测试只回答一个问题,判据是与 HF 参考逐题一致,不是绝对准确率 |
| 11–17 | `from vlmdeploy.tasks import` | 标准库 + 本包的轻量模块 |
| 20–29 | `def load_smoke_set` | 取 TextVQA 验证集前 N 题,图片预缩放到档位内。顺序固定 |
| 33 | `def chat_prompt` | **全项目共用的提示词函数**。GPU 两个后端、量化校准、计时、演示都调它 |
| 34–40 | `多一个 token` | docstring 记下为什么不用 TensorRT-LLM 自带的输入加载器:它在图片占位符后多插一个换行,只多一个 token,冒烟测试里答案就和 HF 不同了 |
| 42–44 | `apply_chat_template` | 构造"图在前、文字在后"的单轮对话,用模型自带的聊天模板渲染成字符串,末尾带 assistant 起始标记 |
| 47–61 | `def summarize` | 汇总一个后端的冒烟结果并落盘 |
| 63–72 | `"vqa_acc": round(score(sample, ans), 4)` | 逐题记录:题目、回答、图像 token 数、得分;再汇总平均分与各阶段耗时 |
| 74–87 | `agree_with_hf_raw` | 若 HF 参考结果已存在,逐题比对:一个是逐字相同,一个是归一化后相同(忽略大小写、标点) |
| 89–94 | `out.write_text(` | 写入 `results/smoke_<后端>_<档位>.json` |
| 97–109 | `def print_table` | 把逐题结果打成一张易读的表。`log` 参数默认是 `print`,脚本里传 `log.info` 就走日志 |

### 1.9 `vlmdeploy/sysinfo.py` 与 `vlmdeploy/logging_.py` —— 两个小工具

### `vlmdeploy/sysinfo.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–5 | `import subprocess` | 只依赖标准库 |
| 8–16 | `def gpu_memory_used_mb` | 整卡已用显存。docstring 提醒口径:包含桌面和其他进程,不是本进程的占用 |
| 18–23 | `nvidia-smi` | 调 `nvidia-smi` 查询,取第一行(第一块显卡)转成整数。带 30 秒超时 |
| 24–26 | `return None` | 没有 `nvidia-smi`、命令失败、输出不是数字都返回 `None` 而不是抛异常 —— 读数只是辅助信息,不应该让主流程崩 |

### `vlmdeploy/logging_.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–6 | `避免遮蔽标准库` | 文件名带下划线,是为了不和标准库的 `logging` 重名 |
| 14–16 | `_FORMAT` | 统一格式:`时间 级别首字母 [模块] 消息`;`_configured` 记录是否已初始化 |
| 19–37 | `def get_logger` | 首次调用时配置根 handler(输出到标准输出,便于和子进程的输出按时间拼在一起),之后直接返回带名字的 logger |

### 1.10 `vlmdeploy/fetch.py` 与 `vlmdeploy/hub.py` —— 可靠的大文件下载

**要解决的问题**:镜像单连接只有 3–4 MB/s,大文件中途断流后只能从头再来。这里把文件切块、多连接下载、每块独立续传,
并保证"下完的文件一定是对的"。

### `vlmdeploy/fetch.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–13 | `下完的文件一定是对的` | 文件说明:三处正确性保护(要求 206、超长的块重下、sha256 校验) |
| 17–28 | `import concurrent.futures as cf` | 全标准库:线程池、哈希、HTTP。两个常量是请求头和超时 |
| 31–32 | `class FetchError` | 自定义异常,专指"重试也解决不了"的错误(校验失败、服务器不支持分块) |
| 35–49 | `def head_size` | 用 HEAD 请求取文件大小;没有 `Content-Length` 就报错 |
| 52–62 | `def sha256_of` | 流式算 sha256,每次读 4 MB,内存占用恒定 |
| 65–73 | `def _fetch_range` | 下载一个字节区间到一个分块文件,支持块内续传 |
| 75–85 | `if have > expected` | 每次重试先看块文件已有多少字节:正好够就完成;比预期还多说明上次写坏了,整块删掉重下 |
| 87–89 | `"Range": f"bytes={start + have}-{end}"` | 从已有字节之后继续请求 —— 这就是块内续传 |
| 91–98 | `if resp.status != 206` | 必须是 206(部分内容)。服务器若忽略 Range 回 200 整文件,拼出来的文件是错的,所以直接判失败 |
| 99–102 | `time.sleep(min(2 + attempt, 15))` | `FetchError` 原样抛出(重试没用);其他异常(断流、超时)退避后重试,等待时间逐次增加、最多 15 秒 |
| 104 | `failed after` | 重试耗尽才放弃 |
| 107–125 | `def fetch` | 对外入口:多连接下载并校验。docstring 逐个说明参数 |
| 127–133 | `already complete` | 已存在且长度、sha256 都对就跳过;长度对但内容错的要删掉 —— 不能留着冒充成品 |
| 136–143 | `ranges = [` | 建分块目录,把文件切成固定大小的区间(闭区间,最后一块可能短) |
| 146–161 | `cf.ThreadPoolExecutor(max_workers=workers)` | 线程池并行下载各块;每完成 8 块打印一次进度与速度 |
| 164–172 | `size mismatch` | 按块序拼成临时文件,核对总长度 |
| 174–180 | `sha256 mismatch` | 校验 sha256;不对就删掉临时文件和全部分块(分块里有坏数据,不能留着续传) |
| 182–187 | `os.replace(tmp, out)` | 校验通过才原子地改名成正式文件 —— 中途失败不会留下"看起来完整"的半成品 |

### `vlmdeploy/hub.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–8 | `snapshot_download` | 文件说明:为什么不用官方的下载函数(单连接、依赖还没建好的环境) |
| 23–24 | `_SMALL_FILE_BYTES` | 小于 32 MB 的文件单连接直接下,分块没有意义 |
| 27–39 | `class RemoteFile` | 仓库里的一个文件:路径、大小、sha256。只有大文件(LFS)有 sha256 |
| 43–46 | `def _get_json` | 发 GET 请求并解析 JSON |
| 49–60 | `def list_files` | 调 Hub 的 tree 接口,递归列出仓库里的全部文件 |
| 62–79 | `lfs.get("oid")` | 只保留文件(跳过目录);大文件的真实大小和 sha256 在 `lfs` 字段里 |
| 82–87 | `def resolve_url` | 拼出文件的下载地址;数据集比模型多一段 `/datasets` |
| 90–107 | `def download_files` | 下载并校验一组文件,返回下载记录 |
| 109–127 | `remote.size >= _SMALL_FILE_BYTES` | 大文件多连接分块,小文件单连接整块;每个文件记下是否做了 sha256 校验和耗时 |

---

## 2. 兼容补丁与视觉塔引擎开关

### 2.1 `vlmdeploy/patches/site/sitecustomize.py`

**这是全仓库最"不寻常"的一个文件,也是 GPU 栈能跑起来的前提。**

Python 启动时会自动导入一个叫 `sitecustomize` 的模块(如果搜索路径上有的话)。Makefile 给本项目的 GPU 命令加上
`PYTHONPATH=vlmdeploy/patches/site`,于是每个 Python 进程 —— 包括 TensorRT-LLM 自己另起的 worker 进程 —— 启动时都会先执行这个文件。
它做两件事:

1. 修补 TensorRT-LLM 1.3.0rc18 跑 Qwen2-VL 的三处兼容问题(不改安装目录里的任何文件);
2. 提供一个开关,把视觉塔换成 TensorRT 引擎(P5)。

为什么不在脚本里打补丁:模型跑在 worker 进程里,脚本里的修改到不了那里。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–5 | `官方版本组合下这条路径直接跑不起来` | 文件说明的总起:三处问题都出在 TensorRT-LLM 自己锁定的 transformers 版本下 |
| 7–14 | `flash-attn 2 缺失` | 问题 1 的完整记录:现象、根因(视觉塔被写死用 flash-attn 2,而它不在 pip 依赖里)、修补办法、为什么 SDPA 可以替代 |
| 16–20 | `BaseModelOutputWithPooling` | 问题 2:transformers 5.x 把视觉塔的返回值从张量改成了一个带两个字段的对象,TensorRT-LLM 仍当张量用,`len()` 得到 2 |
| 22–31 | `输出层权重共享丢失` | 问题 3(最隐蔽):权重共享的标志在 5.x 的配置拆分里丢了,输出层没加载到权重、全是零,所有题都输出 `!!!!!!!!` |
| 33–36 | `VLMDEPLOY_VISION_TRT` | 第 4 项不是修补而是实验开关:设了这个环境变量才把视觉塔换成引擎 |
| 38–43 | `导入后钩子` | 为什么用 sitecustomize,以及一个克制的设计:不在启动时就 import 那些大包,只注册钩子,目标模块真的被导入时才打补丁 |
| 46–55 | `_MARK = "_vlmdeploy_patched"` | 只用标准库。`_MARK` 是打在类上的记号,防止同一个类被补两次 |
| 58–60 | `def _note` | 往标准错误打一行带进程号的提示。每个补丁生效时都会说一声 —— 日志里能看到"补丁在 worker 进程里确实生效了" |
| 64–69 | `def _flash_attn_2_available` | 问 transformers"flash-attn 2 能用吗"。装了的环境里不做任何改动 |
| 73–80 | `def _patch_hf_vision_init` | **补丁 1**:取到 HF 的视觉塔类;取不到或已经补过就返回;存下原来的 `__init__` |
| 83–89 | `config._attn_implementation = "sdpa"` | 新的 `__init__`:如果配置要求 flash-attn 2 而环境里没有,改成 SDPA,然后调原来的 `__init__`。签名用 `*args, **kwargs` 原样转发 |
| 92–93 | `cls.__init__ = __init__` | 把类的 `__init__` 换成新的,并打上记号。这就是"猴子补丁" |
| 97–107 | `def _patch_trtllm_vision_forward` | **补丁 2 + 引擎开关**:取到 TensorRT-LLM 的视觉塔包装类,存下原来的 `forward` 与 `load_weights`;`warned` 是让下面那条提示只打一次的标志 |
| 109–114 | `_VisionEngines.try_forward(multimodal_params)` | 新的 `forward` 第一步:先试引擎。引擎能处理这批请求就直接返回它的输出 |
| 116–123 | `getattr(e, "pooler_output", e)` | 否则走 PyTorch 视觉塔,并做补丁 2 的事:返回值若是 5.x 的那种对象,取出 `pooler_output`(就是 4.x 直接返回的张量);张量没有这个属性,原样保留 |
| 126–134 | `_VisionEngines.release_hf_tower(self)` | 新的 `load_weights`:权重加载完立刻加载引擎,并把 PyTorch 视觉塔挪出显存。注释解释了为什么不能等到第一次前向再加载(那一刻正好在估算 KV cache 容量) |
| 137–139 | `cls.load_weights = load_weights` | 替换两个方法,打记号 |
| 142–143 | `class _VisionEngines` | 引擎管理器。全是类方法和类属性 —— 每个进程里只有一份状态 |
| 145–151 | `预计算 embedding` | 设计说明:为什么在视觉塔的调用点替换,而不是离线预计算 embedding(那样首字延迟里就看不到视觉段了);没有对应网格的引擎就整批回退,"宁可慢,不能错" |
| 153–161 | `显存账` | 一段实测记录:引擎接进来之后 KV cache 从 478 个块掉到 12–39 个,原因是同一个视觉塔的两份拷贝同时在显存里。修复后的数字也记在这里,包括没有完全恢复、根因未证实的部分 |
| 166–170 | `_hf_released` | 类级状态:已加载的引擎表、几个"只提示一次"的标志、PyTorch 视觉塔是否已挪走及它原来在哪个设备 |
| 172–184 | `def _load` | 加载引擎(只做一次)。没设环境变量就返回空表 —— 这时整个引擎功能等于不存在 |
| 191–196 | `VLMDEPLOY_VISION_TRT_GRIDS` | 只加载指定网格的引擎。注释是实测:三个引擎全加载,worker 初始化直接断言失败 |
| 198–212 | `runtime.deserialize_cuda_engine(f.read_bytes())` | 遍历目录里的引擎文件,从文件名解析出网格 (高, 宽),反序列化引擎并各建一个执行上下文,以网格为键存起来 |
| 214–243 | `def release_hf_tower` | 把 PyTorch 视觉塔挪到 CPU。几个提前返回处理各种边界(没有视觉塔、没有参数、已经在 CPU) |
| 236–243 | `torch.cuda.empty_cache()` | 记下原设备、挪到 CPU、**归还缓存**。不归还的话显存仍被 PyTorch 分配器占着,TensorRT-LLM 估算时看不到这块空出来 |
| 245–267 | `def hf_forward` | 回退到 PyTorch 视觉塔。如果它已被挪到 CPU,先搬回 GPU,算完(`finally` 里)再挪走。慢,但结果正确 |
| 269–276 | `def try_forward` | 试着用引擎处理一批请求;没有引擎就返回 `None`(调用方会回退) |
| 280–295 | `image["image_grid_thw"]` | 逐个检查请求:有图、不是视频、单张图、且它的网格 (高, 宽) 有对应引擎。任何一条不满足就整批放弃 |
| 297–298 | `torch.cuda.current_stream()` | 取 PyTorch 当前的 CUDA 流。引擎在同一条流上执行,才能保证与模型前向的先后顺序 |
| 300–311 | `ctx.execute_async_v3(stream.cuda_stream)` | 逐张图:像素转成 FP16、预分配输出张量、把输入输出的显存地址告诉引擎、执行;输出再转成语言模型用的 BF16 |
| 313–317 | `return [torch.cat(outs, dim=0)]` | 首次成功时提示一声;把各图的输出拼起来,按 TensorRT-LLM 期望的格式(列表里一个张量)返回 |
| 321–328 | `def _patch_trtllm_tied_embeddings` | **补丁 3**:取到 TensorRT-LLM 的 Qwen2-VL 模型类 |
| 331–341 | `text.tie_word_embeddings = top.tie_word_embeddings` | 新的 `__init__`:如果 `text_config` 缺这个属性而总配置有,就复制过去(只补缺,不覆盖),再调原来的 `__init__`。这一行修好了"输出全是感叹号" |
| 344–345 | `setattr(cls, _MARK, True)` | 替换并打记号 |
| 349–351 | `def _patch_trtllm_qwen2vl` | 同一个模块上的两个补丁合成一个入口 |
| 355–358 | `_PATCHES = {` | 补丁表:模块名 → 导入后要执行的补丁函数 |
| 361–362 | `class _PostImportHook` | 导入钩子:挂在 Python 的导入机制上,等目标模块被导入 |
| 364–370 | `def find_spec` | Python 每次导入模块都会来问一遍。不是目标模块就返回 `None`,交给正常机制 |
| 371–376 | `sys.meta_path.remove(self)` | 是目标模块:临时把自己摘掉,用正常机制找到模块的规格,再把自己挂回去(`finally` 保证一定挂回)。不摘的话会无限递归 |
| 378–390 | `spec.loader.exec_module = exec_and_patch` | 把"执行模块代码"这一步包一层:先正常执行,紧接着打补丁。于是模块一导入完成,补丁就已经在了 |
| 393–397 | `sys.meta_path.insert(0, _PostImportHook())` | 文件末尾真正执行的部分:目标模块若已经导入(正常不会),直接补;然后把钩子挂到导入机制的最前面 |

---

## 3. 流水线 `scripts/`

编号就是执行顺序。每个脚本的结构都一样:长 docstring(设计说明)→ 导入 → 常量 → 若干函数 → `main()`。

### 3.1 `scripts/00_download_model.py` 与 `scripts/01_download_data.py` —— 下载

两个脚本只用标准库(通过 `vlmdeploy.hub`),任何环境都能跑,甚至在 conda 环境建好之前。

#### `scripts/00_download_model.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–11 | `逐文件校验长度与 sha256` | 用法与产物。权重本身不进仓库 |
| 21–24 | `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` | 把仓库根目录加进搜索路径,再导入本包 —— 所有脚本都是这个开头 |
| 30–41 | `list_files(HF_ENDPOINT, MODEL_REPO, kind="model")` | 列出模型仓库的全部文件,算总大小,逐个下载并校验 |
| 44–54 | `download_model.json` | 把文件清单、大小、sha256、耗时写进 `results/`,打印有多少个文件做了 sha256 校验 |
| 58–59 | `sys.exit(main())` | 标准入口 |

#### `scripts/01_download_data.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–15 | `下载什么、为什么` | 说明下哪些分片、各多大。清单定义在 `config.DATASETS` |
| 38–42 | `--only` | 可以只下指定的数据集(可列多个);默认全下。只写 `--only` 不带名字时得到空列表,什么都不下 |
| 47–57 | `raise SystemExit` | 从 Hub 的文件清单里挑出要的分片。**少了任何一个就立刻失败** —— 上游改了分片命名时,不能悄悄少下几个、让测试集变小 |
| 59–68 | `download_files(HF_ENDPOINT, spec.repo, files` | 下载到 `data/raw/<数据集>/`,记下许可与耗时 |
| 71–82 | `merged.update(summary)` | 结果并入已有的 `download_data.json`(只下一个数据集时不会丢掉另一个的记录) |

### 3.2 `scripts/02_smoke_gpu.py` 与 `scripts/03_smoke_cpu.py` —— P0 冒烟

回答"推理栈装对了吗"。同一批 8 道题,HF 参考、TensorRT-LLM、OpenVINO GenAI 各答一遍,逐题比对。

#### `scripts/02_smoke_gpu.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–14 | `两者的输入完全相同` | 文件说明:两个 GPU 后端的题、图、模板、解码设置完全相同,所以差异只来自推理实现 |
| 25–29 | `from vlmdeploy.backends import MAX_NEW_TOKENS` | 生成上限与显存读数都从公共库导入,不在脚本里另写一份 |
| 35–39 | `def run_hf` | HF 参考后端。torch 与 transformers 在函数里才导入 |
| 42–49 | `attn_implementation="sdpa"` | 加载处理器与模型(BF16、SDPA),记下加载耗时 |
| 52–70 | `model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)` | 逐题:生成提示词 → 处理器 → 贪心生成 → 切掉输入部分解码。提示词也存下来,供后面核对 |
| 73–77 | `torch_max_allocated_mb` | 附加信息:整卡显存、PyTorch 峰值显存、第一题的提示词原文 |
| 80–85 | `def run_trtllm` | TensorRT-LLM 后端 |
| 88–94 | `KvCacheConfig(free_gpu_memory_fraction=0.25)` | 构造 `LLM`;注释解释为什么 KV cache 只给 25% |
| 96–101 | `chat_prompt(processor, s.prompt)` | 提示词与 HF 出自同一个函数 —— 输入 token 逐个相同;贪心解码 |
| 103–113 | `llm.generate(inputs, sampling_params=params)` | 8 道题一次提交;取答案;记下显存;关闭 worker |
| 116–127 | `--backend` | 命令行参数:哪个后端、几道题、哪个档位 |
| 130–139 | `(run_hf if args.backend == "hf" else run_trtllm)(items)` | 取题 → 跑选定的后端 → 汇总(与 HF 参考比对)→ 打印表格 |
| 143–144 | `入口必须有这道保护` | TensorRT-LLM 用 spawn 另起 worker,worker 会重新导入这个脚本;没有 `if __name__` 保护就会无限递归地起进程 |

#### `scripts/03_smoke_cpu.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–18 | `为什么导出 FP16、而不是直接导 INT4` | 文件说明:所有 CPU 压缩配方都从同一份 FP16 IR 出发,配方之间的差异才只来自压缩本身;另记了导出的内存风险 |
| 30–38 | `FP16_DIR = OV_DIR / "fp16"` | 线程数与生成上限从 `backends` 导入;基础 IR 的目录 |
| 41–58 | `def export_fp16` | 调 `optimum-cli` 把 HF 模型导出成 OpenVINO IR(FP16 权重)。已存在就跳过 |
| 61–67 | `def hf_input_lengths` | 一个巧妙的核对办法:GenAI 自己套聊天模板,我们看不到它最终的提示词;但只要它报告的输入 token 数与 HF 逐题相同,就排除了"多一个换行"这类差异 |
| 69–83 | `int(enc["input_ids"].shape[1])` | 用 HF 处理器算出每道题的输入 token 数,作为标尺 |
| 86–96 | `def answer` | 取题,先算好 HF 的 token 数 |
| 98–105 | `config.max_new_tokens = MAX_NEW_TOKENS` | 加载管线(CPU、4 线程);生成上限与 GPU 冒烟相同 |
| 108–128 | `pm.get_ttft().mean` | 逐题生成;从管线自带的性能指标里取首字延迟、输入 token 数、视觉段耗时(`prepare_embeddings`:视觉编码 + embedding 拼接,不是纯视觉塔) |
| 131–138 | `input_tokens_match_hf` | 逐题比较输入 token 数;把设备、线程数、版本号等写进附加信息 |
| 140–148 | `print_table(result, log.info)` | 汇总并打印 |
| 151–168 | `--export-only` | 入口:总是先确保 IR 已导出;`--export-only` 时到此为止(Makefile 用 `/usr/bin/time` 包住它单独记录导出的峰值内存) |

### 3.3 `scripts/10_eval.py` —— 评测

**全项目最核心的脚本。** 任一变体 × 任一划分 × 任一档位,逐题推理、打分、汇总,可断点续跑。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–7 | `任一变体 × 任一划分 × 任一 token 档位` | 用法举例,对应 P1–P4 各阶段 |
| 9–13 | `native = 不预缩放` | 四个划分(其中 calib 只给量化校准,不在这里评测)、四类档位的含义。`--budget` 另外还接受方图档位 s256 / s576 / s1024(P5 用),这一段没有列出 |
| 15–20 | `任何中断都不应该从头再来` | 三个设计决定:逐题落盘可续跑;汇总带置信区间并用配对差比较量化;P1 的验收线 |
| 22–25 | `.jsonl` | 两个产物:逐题的 `.jsonl` 与汇总的 `.json` |
| 29–44 | `from vlmdeploy.variants import make_answerer` | 标准库 + 本包。本脚本的模型加载只经过 `make_answerer` |
| 47–53 | `OFFICIAL = {"test": 79.7, "rwqa": 62.9}` | 结果目录;官方分数与验收容差 1.5 分 |
| 54–56 | `CHUNK = {"trtllm": 32}` | 一次交给后端几道题:TensorRT-LLM 批量调度所以给 32;其他后端逐题,批大小只影响落盘频率 |
| 59–61 | `def tag` | 结果文件名的主干;试跑(带 `--limit`)时加 `_n<题数>` 后缀,不会覆盖正式结果 |
| 64–70 | `def load_rows` | 读逐题 JSONL;文件不存在返回空列表 |
| 73–76 | `done = {r["qid"] for r in load_rows(out)}` | `run` 的第一步:读出已经答完的题目 id —— 断点续跑的依据 |
| 77–83 | `todo_ids` | 先只收集待答的 id,不把题目(含图片字节)留在内存里。注释给了理由:图片字节合计约 1 GB。没有待答的题就直接返回 |
| 85–87 | `stream = (s for s in iter_split(split, limit=limit) if s.qid in pending)` | 一个惰性的生成器:推理时再按分片顺序流式读取待答的题 |
| 88–93 | `vision_grid=grid` | 方图档位算出网格边长(2 × √token 数),让视觉塔引擎变体只加载对应的那个引擎;然后加载模型 |
| 95–97 | `chunk = CHUNK.get(variant.split("-")[0], 8)` | 按栈名取批大小;开始计时 |
| 98–105 | `itertools.islice(stream, chunk)` | 以**追加**模式打开结果文件;每次从流里取一批题 |
| 107–120 | `square_to_budget(img, SQUARE_BUDGETS[budget])` | 解码图片并按档位预缩放:方图档位裁成方图;普通档位保持长宽比;`native` 不缩放,只算出处理器会产生多少 token |
| 122–125 | `answerer.answer(` | 没有题了就结束;否则整批交给后端 |
| 127–130 | `f.flush()` | 逐题写一行 JSON(题目 id、回答、得分、图像 token 数),**每批刷盘** —— 进程被杀也只丢最后一批 |
| 132–135 | `s/question` | 每 200 题左右报一次进度与平均耗时 |
| 136–140 | `answerer.close()` | `finally` 保证无论成败都关闭后端(释放显存、结束 worker);返回本次运行的耗时信息 |
| 143–148 | `def mean_ci` | 均值与 95% 置信半宽(1.96 × 标准差 / √n),乘 100 换成"分" |
| 151–163 | `def summarize` | 汇总:**只读 JSONL**,与推理进程解耦 —— 中断后续跑,汇总自然包含之前答完的题 |
| 165–174 | `within_ci95` | 只有"HF BF16 + 原生分辨率 + 全量"才与官方分数比较,同时给出两个判据(1.5 分线、是否在置信区间内)。注释解释了为什么不事后放宽验收线 |
| 175–179 | `ref_path` | 配对比较:找参照变体的逐题结果。优先找同样题数的,找不到就用它的全量结果(题目顺序固定,前 N 题是全量的子集) |
| 181–191 | `r["score"] - ref[r["qid"]]["score"]` | **配对差的核心**:同一道题上两个变体的得分相减,对这些差求均值与置信区间;另算答案逐字相同的比例 |
| 193–194 | `write_text(json.dumps(result` | 写汇总 JSON |
| 197–212 | `--paired-vs` | 命令行参数。`--budget` 的可选值由 `config` 里的两个档位字典拼出来 |
| 215–224 | `info = run(` | 入口:推理 → 汇总 → 打印一行结果(不含运行信息) |
| 228–229 | `if __name__ == "__main__":` | 同样需要这道保护(TensorRT-LLM 的 worker 会重新导入本脚本) |

### 3.4 `scripts/11_bench_gpu.py` —— GPU 计时:首字延迟分解

**要解决的问题**:视觉段没法直接计时(它在 worker 进程里,夹在请求中间)。办法是给每个带图请求配一个
"把图像 token 全换成普通文字 token、长度逐个相同"的纯文本请求:两者 prefill 计算量相同,TTFT 之差就是视觉段。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–6 | `首字延迟(TTFT)分解` | 用法:普通档位与方图档位(P5 引擎对比) |
| 8–12 | `显存带宽受限` | 一个请求在 GPU 栈上的四段:预处理、视觉编码、prefill、decode |
| 14–22 | `视觉段 ② = TTFT(带图) − TTFT(纯文本)` | 怎么拆:配对请求相减得视觉段;TTFT 取引擎侧时间戳;预处理用客户端墙钟减引擎时长估出;decode 强制生成固定个数的 token |
| 24–27 | `1 轮预热丢弃` | 计时纪律与产物 |
| 41–48 | `from vlmdeploy.timing import interleaved_schedule, round_spread, summarize` | 计时工具、显存读数、变体解析都来自公共库 |
| 51–56 | `NUM_IMAGES = 12` | 12 张 dev 图(只计时不打分,所以不碰 test)、5 轮、每次强制生成 64 个 token |
| 57–61 | `FILLER_TOKEN = 279` | 顶替图像占位符的普通 token(词表里的 " the")。prefill 的计算量只与长度有关,与内容无关。下一行是三种图像相关的特殊 token |
| 64–69 | `def build_workload` | 构造测量项。先把三种特殊 token 转成 id |
| 71–79 | `if sample.image_id in seen` | 从 dev 集取不同的图(同一张图的第二道题跳过) |
| 81–91 | `text_ids = [FILLER_TOKEN if t in special else t for t in ids]` | **关键一步**:每个档位下,用处理器得到带图请求的完整 token 序列,再把其中的图像特殊 token 逐个换成填充 token —— 得到一条长度完全相同的纯文本序列 |
| 93–95 | `if len(seen) >= NUM_IMAGES` | 够 12 张图就停 |
| 98–105 | `def one_request` | 同步提交一个请求,记下客户端墙钟 |
| 107–112 | `tm.first_token_time.total_seconds()` | 从引擎返回的性能指标里取三个时间戳:请求到达、首个 token、最后一个 token |
| 114–118 | `"ttft": (first - arrival) * 1000` | 引擎侧 TTFT、引擎侧总时长、客户端多出来的开销(预处理 + 通信)、decode 速度 = (生成数 − 1) / (末 − 首) |
| 121–134 | `--square` | 命令行参数:变体、轮数、是否方图档位、只测哪些档位 |
| 136–141 | `resize = square_to_budget if args.square else resize_to_budget` | 按参数选档位表与缩放函数 |
| 144–156 | `VLMDEPLOY_VISION_TRT_GRIDS` | 视觉塔引擎变体:设置环境变量(必须在拉起 `LLM` 之前),方图档位下只让 worker 加载用得到的网格 |
| 159–169 | `work = build_workload(processor, budgets, resize)` | 导入 TensorRT-LLM,构造全部测量项,打印各档位的输入长度 |
| 172–176 | `mem_before = gpu_memory_used_mb()` | 加载前记一次整卡显存 |
| 177–185 | `enable_block_reuse=False` | 构造 `LLM` 时**必须关闭 KV cache 块复用**。注释是实测证据:不关的话,纯文本 TTFT 不随长度变化,恒为 20 ms |
| 188–189 | `min_tokens=DECODE_TOKENS, ignore_eos=True` | 采样参数:强制生成恰好 64 个 token(忽略结束符)、贪心、要求返回性能指标 |
| 192–195 | `plan = interleaved_schedule(keys, args.rounds + 1, seed=0)` | 测量项 = 每个(图, 档位)× {带图, 纯文本};排出"轮数 + 1"轮的随机顺序 |
| 197–212 | `if r > 0` | 逐轮执行。第 0 轮是预热,结果丢弃(注释说明预热轮里发生了什么) |
| 213–215 | `llm.shutdown()` | 测完再记一次显存,关闭 worker |
| 218–221 | `"gpu_mem_mb": {"before_load": mem_before` | 结果的公共部分:变体、轮数、图数、加载耗时、三次显存读数 |
| 223–230 | `one_budget_per_process` | 一档一个进程时,把已有结果文件里其他档位的数据并进来,不覆盖 |
| 232–238 | `vqa = [m for i in idx for m in samples[(i, "vqa")]]` | 按档位汇总:取出这一档全部的带图样本与纯文本样本 |
| 239–241 | `a["ttft"] - b["ttft"]` | **配对相减**:同一张图、同一轮的带图请求与纯文本请求相减,得到一组视觉段样本 |
| 243 | `per_round` | 每一轮的中位数,用来算跨轮极差 |
| 245–257 | `"vision_share"` | 这一档的全部指标:带图 TTFT、纯文本 TTFT(= prefill)、视觉段、视觉占比、客户端开销、decode 速度、跨轮极差 |
| 259–263 | `log.info("%s: TTFT %.0f ms (vision %.0f = %.0f%%, prefill %.0f)` | 打印一行人能读的摘要 |
| 265–268 | `out_file.write_text(` | 写入 `results/bench/gpu_<变体>[_square].json` |

### 3.5 `scripts/12_quantize_gpu.py` —— GPU 量化语言模型

用 ModelOpt 把语言模型量化成 FP8 或 AWQ-INT4,导出成 TensorRT-LLM 能直接加载的 checkpoint。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–5 | `FP8(W8A8)或 AWQ-INT4(W4A16)` | 用法:方法 × 校准数据,共四种组合 |
| 7–11 | `为什么只量化语言模型` | TensorRT-LLM 加载时会主动去掉视觉塔的量化配置,量化了也用不上;输出层与输入 embedding 共享权重,也不量化 |
| 13–17 | `两种方法各自省的是什么` | 跑之前写下的预测:FP8 权重和激活都量化,prefill 与 decode 都可能变快;AWQ 只压权重,只省 decode 的带宽 |
| 19–23 | `已知混杂` | 两种校准数据的唯一差别是"有没有图像 token";并坦白一个混杂因素:纯文本样本的 token 总数少一个数量级 |
| 25–27 | `hf_quant_config.json` | 产物:量化 checkpoint 与量化记录 |
| 50–54 | `NUM_CALIB = {"img": 128, "txt": None}` | 校准档位;图文校准用 128 条,纯文本用全部(部分弥补 token 数的差距) |
| 55–57 | `AUX_FILES` | 导出只写配置与权重,TensorRT-LLM 还要从模型目录读处理器、分词器文件,需要另外拷过去 |
| 60–69 | `cfg["quant_cfg"]["*visual*"] = {"enable": False}` | 量化配置:取 ModelOpt 的默认配置(深拷贝,不改库里的原对象),再用通配符关掉视觉塔上的全部量化器 |
| 72–81 | `def restore_key_layout` | 一个为绕开兼容问题写的函数。docstring 说明:导出的 checkpoint 里视觉塔的键名多了 `model.` 前缀,TensorRT-LLM 找不到 |
| 82–91 | `struct.unpack("<Q", f.read(8))` | safetensors 文件的格式:前 8 字节是头的长度,接着是 JSON 头,然后是数据区。先读出 JSON 头 |
| 93–101 | `k.replace("model.visual.", "visual.", 1)` | 只改头里的键名;并核对原始 checkpoint 的每个键都还在,少了就报错 |
| 103–113 | `shutil.copyfileobj(f, g, length=64 << 20)` | 写新的头(按 8 字节对齐),然后把数据区**原样流式复制** —— 不需要把 3 GB 权重读进内存 |
| 114 | `os.replace(tmp, src)` | 原子替换原文件 |
| 115–128 | `exclude_modules` | 顺手把两个配置文件里带错误前缀的排除列表也改正(只为配置文件本身表意准确) |
| 130 | `return sum(` | 返回改名的键数 |
| 133–134 | `def calib_inputs` | 把 calib 划分变成模型输入。张量留在 CPU,前向时再搬到 GPU,避免 128 份像素张量同时占显存 |
| 136–142 | `resize_to_budget(sample.image(), TOKEN_BUDGETS[CALIB_BUDGET])` | 图文校准:图片缩到 t576,提示词用共用的 `chat_prompt` |
| 143–150 | `{"type": "text", "text": sample.prompt}` | 纯文本校准:同一批题目,对话里不放图片 |
| 153–171 | `def enabled_quantizers` | 量化之后的自检函数。按类型(`TensorQuantizer`)找启用的量化器,不按名字后缀:FP8 除了权重量化器还有激活侧的 `input_quantizer`,只看后缀会漏掉它。返回两份名单:启用的权重量化器(数量写进量化记录)、落在视觉塔或输出层上的(应当为空) |
| 174–195 | `already exported` | 入口:解析参数;产物已存在就跳过(`--force` 才重做) |
| 198–207 | `import modelopt.torch.quantization as mtq` | 导入 ModelOpt,加载 BF16 模型到 GPU |
| 209–213 | `n_tokens = sum(` | 准备校准数据,统计样本数与 token 总数(写进记录,解释上面的混杂因素) |
| 216–220 | `def forward_loop` | ModelOpt 的校准回调:把每条校准样本搬到 GPU 跑一遍前向 |
| 223–227 | `mtq.quantize(model, quant_config(args.method), forward_loop)` | 量化(FP8 是收集激活范围,AWQ 是搜索通道缩放),计时 |
| 228–234 | `quantizers leaked into vision tower or lm_head` | **自检**:量化器只应该出现在语言模型上。调上面的 `enabled_quantizers`,视觉塔或输出层上只要有启用的量化器(权重侧、激活侧都算)就报错,报错信息里带上前几个名字和总数 |
| 237–243 | `export_hf_checkpoint(model, export_dir=out_dir)` | 清掉旧目录,导出 HF 格式的量化 checkpoint |
| 245–251 | `restore_key_layout(out_dir)` | 拷贝辅助文件,把视觉塔键名改回原始布局 |
| 253–268 | `"weights_bytes": size` | 写量化记录:方法、校准样本数与 token 数、被量化的层数、耗时、权重体积、量化配置原文 |

### 3.6 `scripts/13_compress_cpu.py` —— CPU 栈压缩

用 NNCF 对 OpenVINO IR 做权重压缩或静态量化。分两步、分两个进程:先收集校准输入,再按配方压缩 —— 内存峰值不叠加。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–8 | `只改动配方涉及的子模型` | 用法;所有配方从同一份 FP16 IR 出发,其余子模型硬链接 |
| 10–14 | `language_model` | Qwen2-VL 的 IR 由四个子模型组成,各多大、怎么处理 |
| 16–21 | `int4-vis8w` | 五个配方的定义。前三个只压语言模型,后两个再压视觉塔(仅权重 / 权重与激活) |
| 23–30 | `不编译语言模型` | 校准数据怎么来:在 FP16 IR 上跑带图题,挂钩子截获真实输入。关键技巧 —— 语言模型不编译(编译要 6 GB 以上),只截获它的输入就中止 |
| 32–35 | `配方目录` | 三类产物 |
| 50–55 | `from vlmdeploy.backends import CPU_THREADS` | 线程数从公共库导入 |
| 58–68 | `NUM_CALIB = 64` | 目录、校准档位(t256,CPU 栈的主档位)、校准样本数、两个子模型的文件名主干 |
| 71–77 | `RECIPES: dict[str, dict] = {` | 配方表:每个配方说明语言模型怎么压、视觉塔怎么压(没有 `vision` 键就不动视觉塔) |
| 81–83 | `ru_maxrss` | 本进程的峰值内存。内存是这个脚本最紧张的资源,每一步都记 |
| 89–90 | `class _PrefillCaptured` | 一个专用异常:语言模型的输入已经抓到了,用它中止本次生成 |
| 93–98 | `def collect` | 第 1 步:收集校准输入 |
| 101–106 | `compile=False` | 加载模型但**不编译**(各子模型首次调用时才编译);准备两个列表存截获的输入 |
| 109–122 | `def merger_hook` | 给视觉塔挂钩子:记录输入(块特征、旋转位置编码),再调原来的前向。注意力掩码不存 —— 单张图时它是全零,压缩时按尺寸重建,省下 N² 的体积 |
| 125–139 | `raise _PrefillCaptured` | 给语言模型换一个前向:调 `prepare_inputs` 拿到真正会送进去的输入、记录、然后抛异常。语言模型从头到尾没被编译 |
| 142–155 | `except _PrefillCaptured` | 对每道校准题:缩放图片、生成提示词、调 `generate`;预期会被上面的异常打断,捕获后继续下一题 |
| 157–163 | `np.savez(` | 两类输入各存成一个 npz 文件,键名是"样本序号/字段名" |
| 166–182 | `def load_calib` | 读回 npz,按样本序号还原成字典列表 |
| 188–195 | `CompressWeightsMode.INT8_ASYM` | 语言模型 INT8:非对称、逐通道,不需要数据 |
| 197–200 | `group_size=128, ratio=1.0` | INT4:每 128 个权重一组。`ratio=1.0` 让全部可压的层都进 INT4 —— 默认会按敏感度留一部分在 INT8,混进来就分不清是谁的功劳 |
| 202–214 | `awq=True` | INT4 + AWQ:用校准数据找通道缩放。`transform` 只保留模型真正有的输入名 |
| 219–227 | `def compress_vision` | 视觉塔 INT8 仅权重:同样不需要数据 |
| 229–241 | `np.zeros((1, n, n), dtype=np.float32)` | 视觉塔静态量化:读回校准输入,按块数重建全零的注意力掩码,交给分批量化函数 |
| 246 | `STAT_NODES_PER_PASS = 40` | 每次统计收集最多挂 40 个目标节点 |
| 249–261 | `def quantize_in_batches` | **这个函数是为解决 OOM 写的。** docstring 是完整的排障记录:直接调 `nncf.quantize` 时一次挂 584 个统计点,只喂 1 条样本内存也涨到 7.4 GiB,两次带垮整个 WSL;以及为什么分批不改变结果 |
| 263–274 | `from nncf.quantization.algorithms.pipeline import collect_statistics` | 用到的是 NNCF 的内部函数 —— 把 `nncf.quantize` 的流水线拆开自己编排 |
| 276–283 | `nncf.ModelType.TRANSFORMER` | 与直接调用完全相同的参数:TRANSFORMER 模型类型(跳过对 INT8 不友好的位置)、MIXED 预设 |
| 285–296 | `batches = [names[i:i + STAT_NODES_PER_PASS]` | 逐步:建图、取这一步需要的全部统计点、按目标节点切成每批 40 个 |
| 298–304 | `collect_statistics(part, model, graph, dataset)` | 每批只挂这一部分统计点去收集。收集器是共享对象,统计值直接落回完整的统计点集合里 |
| 306–308 | `pipeline.run_step(step, points, model, graph)` | 统计收齐后,用完整的统计点执行这一步的算法 |
| 309–313 | `compress_quantize_weights_transformation(model)` | 收尾与 `nncf.quantize` 相同:把权重上的伪量化节点折叠成 INT8 常量,并把参数写进模型元信息 |
| 316–323 | `def link_or_copy` | 没改动的文件硬链接过去(不占额外空间),不支持硬链接时退回复制 |
| 326–347 | `def build` | 第 2 步:生成一个配方目录。先在临时目录里做 |
| 348–354 | `spec["lm"] != recipe` | 组合配方(int4 + 视觉塔)在 int4 配方的 IR 已经存在时直接链接过来用,不重复压缩;还没有的话不报错,走 else 分支现压一遍 |
| 355–365 | `ov.save_model(model, tmp / f"{LM}.xml", compress_to_fp16=True)` | 否则读入 FP16 语言模型、压缩、保存,记下耗时 |
| 367–377 | `compress_vision(core.read_model(FP16_DIR / f"{MERGER}.xml")` | 配方涉及视觉塔时同样处理 |
| 379–381 | `link_or_copy(f, tmp / f.name)` | 其余文件(另外两个子模型、分词器、配置)全部硬链接 |
| 383–385 | `tmp.rename(out)` | **整个目录写完才改名**:中途失败不会留下"看起来完整"的半成品 |
| 388–403 | `"peak_rss_gib"` | 写配方记录:各子模型体积、耗时、峰值内存 |
| 406–423 | `add_mutually_exclusive_group(required=True)` | 入口:`--collect` 与 `--recipe` 二选一 |

### 3.7 `scripts/14_bench_cpu.py` —— CPU 计时

| 行 | 代码 | 说明 |
|---|---|---|
| 1–4 | `首字延迟分解与 decode 速度` | 用法:一次传入全部要比较的配方 |
| 6–9 | `prepare_embeddings` | GenAI 的管线自带分段计时,不用自己拆:视觉段、TTFT、每 token 耗时 |
| 11–16 | `ABBA` | 计时纪律:每个配方要单独加载(两个同时驻留超过 8 GB),所以不能逐请求交错;退而求其次,按 A B C … 测一遍再按 … C B A 测一遍,漂移在两遍里方向相反 |
| 41–47 | `NUM_IMAGES = 6` | 6 张图、强制生成 32 个 token、t256 档 |
| 50–66 | `def workload` | 取 dev 集前 6 张不同的图,预缩放 |
| 69–81 | `def measure` | 加载一个配方并计时 |
| 83–87 | `config.ignore_eos = True` | 强制生成恰好 32 个 token |
| 89–97 | `[items[0], *items]` | 第一张图多跑一次当预热,丢弃 |
| 99–101 | `pm.get_prepare_embeddings_duration().mean` | 每个请求记下 TTFT、视觉段、每 token 耗时、输入 token 数 |
| 102–103 | `del pipe` | 显式释放管线 —— 下一个配方加载前必须把内存让出来 |
| 106–117 | `passes = [args.variants, list(reversed(args.variants))]` | ABBA 的两遍:正序一遍、倒序一遍 |
| 119–130 | `samples[v].append(measure(v, items))` | 按顺序逐个配方测量,每遍的样本分开存 |
| 133–137 | `allm = [m for run in samples[v] for m in run]` | 汇总时把两遍的样本合并 |
| 138–150 | `"ttft_pass_drift"` | 各项指标的 P50 / P90;prefill = TTFT − 视觉段;**ABBA 漂移** = 两遍中位数之差 / 中位数,是"这个数字可不可信"的证据 |
| 152–161 | `cpu.json` | 打印摘要,写入 `results/bench/cpu.json` |

### 3.8 `scripts/15_vision_trt.py` —— 视觉塔 TensorRT 引擎

把视觉塔单独导出、建成引擎,与原实现对拍并计时。这个脚本只管"孤立的视觉塔";把引擎接回 TensorRT-LLM 做端到端测量,
是 `sitecustomize` 的引擎开关加上 `11_bench_gpu.py --square` 的事。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–6 | `强类型 FP16 引擎` | 流程:导出 ONNX → 建引擎 → 对拍 → 计时。触发条件是 P2 证实视觉塔占 TTFT 超过 40% |
| 8–10 | `孤立的视觉塔` | 本脚本的边界,以及端到端部分由谁完成 |
| 12–23 | `静态形状` | 沿用项目 A 的四条做法,每条都写了理由:静态形状、位置编码常量折叠、强类型 FP16、确认注意力被融合 |
| 25–27 | `以 FP32 视觉塔` | 对拍的设计:以 FP32 为真值,**同时**量现状(BF16)和引擎的误差 —— 引擎只要不比现状差就可以替换 |
| 53–59 | `WORKSPACE = 2 << 30` | 16 张图对拍、5 轮计时、每轮每个实现连测 10 次;构建工作区上限 2 GiB(不设的话 TensorRT 会尝试吃满显存) |
| 62–63 | `class FixedGridVision` | 把视觉塔包成"只有一个输入"的模块 |
| 65–73 | `grid = torch.tensor([[1, side, side]]` | 网格(1 帧 × 边长 × 边长)在构造时固定成常量。导出时它不是输入,于是位置编码等只依赖网格的计算全被折叠成常量 |
| 76–89 | `getattr(out, "pooler_output", out)` | 包装模块的前向:调视觉塔,取合并后的 embedding(兼容 transformers 5.x 的返回类型) |
| 92–112 | `def square_inputs` | 取 16 张不同的 dev 图,裁成方图,经 HF 图像处理器得到像素张量 |
| 115–121 | `def export_onnx` | 导出入口 |
| 122–127 | `sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False` | 一个兼容处理:transformers 5.x 会给注意力传一个导出器不支持的参数。导出期间临时关掉那条分支(视觉塔里它数值上是同一个计算) |
| 128–133 | `finally` | 导出完(无论成败)恢复原函数 —— 临时修改一定要还原 |
| 137–145 | `dynamo=False` | 真正的导出:用 TorchScript 导出器、opset 18、开常量折叠;输入输出命名为 `pixel_values` / `embeds` |
| 148–160 | `NetworkDefinitionCreationFlag.STRONGLY_TYPED` | 建引擎:**强类型网络** —— 每一层的精度完全由 ONNX 里的类型决定,构建可复现 |
| 162–164 | `ONNX parse failed` | 解析 ONNX;失败时把解析器的全部错误拼进异常信息 |
| 166–170 | `ProfilingVerbosity.DETAILED` | 构建配置:工作区上限;保留详细的层信息(后面要查注意力有没有被融合) |
| 172–179 | `builder.build_serialized_network(network, config)` | 构建并落盘 |
| 181–186 | `create_engine_inspector()` | 反序列化引擎,用 inspector 导出每一层的信息 |
| 187–195 | `"fused_attention_layers": len(mha)` | 在层信息里找融合注意力的痕迹(kernel 名里带 mha),统计个数;返回耗时、体积、层数。耗时到这里才结算,除了构建还含写引擎文件、反序列化和导出层信息,比纯构建时间略长 |
| 198–203 | `class TrtRunner` | 最小的引擎执行器。docstring 解释为什么必须跑在 PyTorch 的当前 CUDA 流上:计时事件也记在这条流上,两者同流才能恰好框住引擎的 kernel |
| 206–220 | `self.out = torch.empty(out_shape, dtype=torch.float16, device="cuda")` | 反序列化引擎、建执行上下文、预分配输出张量 |
| 222–230 | `self.ctx.execute_async_v3(` | 执行一次:把输入输出的显存地址告诉引擎,在当前流上执行 |
| 233–256 | `torch.cuda.Event(enable_timing=True)` | GPU 计时:用 CUDA 事件框住一次调用,前后各同步一次;连测多次取中位数。量到的是 GPU 上 kernel 的时间,不含 Python 侧准备输入 |
| 259–276 | `grid = f"g{2 * side_tokens}x{2 * side_tokens}"` | 入口:算出网格名(合并前的块数),运行期就是按它查引擎 |
| 278–284 | `visual_bf16 = full.model.visual.to("cuda").eval()` | 加载整个模型,只留下视觉塔;准备对拍输入 |
| 287–297 | `truth = [` | **FP32 真值**:同一份权重升到 FP32 算一遍,结果存到 CPU,然后立刻释放(FP32 视觉塔 2.7 GB,不能和后面的副本同时驻留) |
| 300–308 | `copy.deepcopy(visual_bf16).half()` | 导出用 FP16 副本 —— 强类型下 ONNX 里的类型就是引擎里的类型 |
| 310–322 | `build = build_engine(onnx_path, engine_path)` | 导出、释放 FP16 副本、建引擎 |
| 325–330 | `def compare` | 两个张量的余弦相似度与相对误差(差的范数 / 基准的范数) |
| 333–344 | `stats["trt_vs_fp32"].append(compare(got, t32))` | 对每张图算三组比较:BF16 对 FP32、引擎对 FP32、引擎对 BF16 |
| 346–353 | `"rel_err_max"` | 汇总成最小余弦、最大相对误差等 |
| 356–361 | `fns = {` | 计时的两个对象:HF BF16 视觉塔、引擎 |
| 363–373 | `interleaved_schedule(list(fns), ROUNDS + 1, seed=0)` | 交错轮次(第 0 轮预热丢弃),每轮每个实现测 10 次取中位数 |
| 375–386 | `"speedup_p50"` | 写结果:构建信息、对拍、两者的 GPU 时间、加速比 |

### 3.9 `scripts/20_demo.py` —— 演示

一张图 + 一个问题,在任一部署变体上流式作答,打印首字延迟与生成速度。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–7 | `流式作答` | 用法举例:GPU 量化模型、视觉塔引擎、CPU 配方、自己的图片 |
| 9–14 | `单次观察值` | 与评测 / 计时脚本的关系:加载走评测用的同一个入口(演示的就是被评测的配置);这里的耗时只是单次观察,不是结论 |
| 16–19 | `口径不同` | 两条栈"首字延迟"的口径不同,所以结果里分开记 |
| 21–24 | `不覆盖这份记录` | 默认用 dev 集的题;只有全默认的运行才落盘,改了图片、问题、题数或生成上限的运行只打印 |
| 38–45 | `from vlmdeploy.variants import is_gpu, make_answerer` | 全部来自公共库 |
| 48–54 | `MIN_DECODE_TOKENS = 8` | 常量:演示 4 张图、预热 2 次、最多生成 48 个 token(评测是 16)、少于 8 个 token 不报 decode 速度 |
| 57–62 | `def write_piece` | 把一段新生成的文字立刻打到终端:不换行、立即刷新 —— 这就是"流式"的观感 |
| 65–78 | `def ask_trtllm` | 在 TensorRT-LLM 上流式回答。直接用后端对象里已经加载好的 `llm` 与 `processor` |
| 80–85 | `return_perf_metrics=True` | 采样参数(贪心、要性能指标);请求 = 共用的提示词函数 + 图片 |
| 87–89 | `t0 = time.perf_counter()` | 初始化几个变量,开始计时 |
| 91–99 | `generate_async(request, sampling_params=params, streaming=True)` | **流式生成**:每次迭代拿到一段新文字。记下首段到达的时刻(只在第一次赋值)和最近一段的时刻;把新增的文字交给 `echo` |
| 101–103 | `if first is None` | 结束计时。防御性处理:一个 token 都没生成时,不让后面的减法崩掉 |
| 105–109 | `result["decode_tps"] = round((n_out - 1) / (last - first), 1)` | 首字延迟 = 首段到达 − 开始(客户端墙钟);生成够长才算 decode 速度 |
| 110–117 | `engine_ttft_ms` | 若拿得到引擎侧时间戳,另记一个引擎侧 TTFT(与基准同口径);拿不到就只留客户端口径 |
| 120–126 | `def ask_openvino` | 在 OpenVINO GenAI 上流式回答 |
| 129–135 | `ov_genai.StreamingStatus.RUNNING` | 生成配置;`streamer` 回调:每生成一段文字被调一次,返回"继续" |
| 137–143 | `streamer=streamer` | 图片转张量,带着回调调用管线 |
| 145–152 | `pm.get_prepare_embeddings_duration().mean` | 耗时取管线自带的指标(与 CPU 计时脚本同一来源):首字延迟、视觉段、decode 速度 |
| 155–160 | `def load_items` | 准备演示用的(图片, 问题) |
| 162–169 | `Image.open(args.image).convert("RGB")` | 给了 `--image`:读用户的图片;没给问题就默认"用一句话描述这张图" |
| 171–184 | `"sample": None if args.question else sample` | 否则取 dev 集前几张不同的图。用原题时保留样本对象,后面可以对照人工答案打分;换了问题就不打分 |
| 187–204 | `--max-new-tokens` | 命令行参数 |
| 206–212 | `parser.error(` | 两条参数校验:只支持部署变体;视觉塔引擎变体只能配方图档位。用 `parser.error` 给出带用法的报错 |
| 215–222 | `item["img"] = resize(item["img"], tokens)` | 按档位选缩放函数,把每张图预缩放好 |
| 225–229 | `mem_before = gpu_memory_used_mb() if on_gpu else None` | 加载前记一次整卡显存。注释记录了一次没查明原因的异常 —— 有了显存读数,数字异常时至少能排除"显存被挤占" |
| 230–240 | `answerer = make_answerer(` | **经评测用的同一个入口加载模型**;按栈选回答函数;加载后再记一次显存 |
| 242–248 | `items[i % len(items)]` | 预热 2 次,结果丢弃,`echo` 传空函数(不显示)。依次取前两张图;只有一张图时同一张问两次 |
| 250–262 | `r = ask(answerer, item["question"], img, args.max_new_tokens, write_piece)` | 逐题:打印图片尺寸与 token 数、问题,然后流式打印回答 |
| 264–269 | `first token` | 打印这一题的首字延迟、token 数、decode 速度、总耗时 |
| 271–281 | `Counter(item["sample"].answers).most_common()` | 组装记录;dev 原题另外打分,人工答案按出现次数从多到少排 |
| 282–284 | `answerer.close()` | `finally`:无论成败都关闭后端 |
| 286–289 | `results not saved` | 不是全默认设置的运行到此结束,不写文件(README 的演示表引用那份记录,不能被临时演示覆盖) |
| 291–305 | `"ttft_source"` | 全默认的运行写入 `results/demo/<变体>_<档位>.json`,并记下预热次数、加载前后的显存、首字延迟的口径 |

### 3.10 `scripts/90_report.py` —— 从结果生成 README 的表

**要解决的问题**:文档里的数字与实测脱节。办法是 README 的表一律由脚本从 `results/` 生成,人不手改。
编号 90 是为了让它永远排在各阶段脚本之后。它只用标准库,也不导入 `vlmdeploy`。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–11 | `不允许手工改` | 用法与约定:README 里一对 `RESULTS:xxx:BEGIN / END` 标记之间的内容由本脚本覆盖 |
| 23–27 | `README = ROOT / "README.md"` | 三个路径 |
| 30–35 | `def load` | 读一个结果 JSON;**不存在就返回空字典** —— 全脚本的容错都建立在这上面(某项没跑,对应的行或小节就跳过) |
| 38–48 | `def peak_rss_gb` | 从 `/usr/bin/time -v` 的输出里用正则取峰值内存 |
| 51–58 | `def table` | 拼一张 Markdown 表:表头行、分隔行、数据行 |
| 61–69 | `def smoke` | P0 表:三个后端各一行 |
| 71–89 | `prompt_check` | 每个后端:取结果、组织"显存 / 内存"与"输入是否与 HF 一致"两格的文字 |
| 90–104 | `notes` | 没有任何数据就返回空串;否则表格后面附说明:OpenVINO 导出的峰值内存总是附,单次运行的 TTFT 观察值只在冒烟记录里有这个字段时才附 |
| 110–113 | `def ev` | 读一条评测汇总;文件名规则与 `10_eval.py` 的 `tag` 一致 |
| 116–135 | `def fmt_paired` | 三个格式化函数:分数 ± 置信半宽、配对差 ± 置信半宽、答案相同的比例。没有数据时都显示"—" |
| 138–158 | `def p1` | P1 表:复现分数、官方分数、差、两个判据是否通过 |
| 161–177 | `def p2` | P2 第一张表:三档的精度,外加原生分辨率一行 |
| 179–208 | `b["budgets"].items()` | P2 第二张表:各档的 TTFT 分解;后面附计时方法与显存说明 |
| 211–243 | `def p3` | P3 表:第一行是未量化的 BF16 基线,其后每个量化变体一行,列出体积、dev / test 精度与配对差、计时。读不到量化记录的行,体积一格写死为 4.42 —— 基线本来就没有记录,但缺记录的量化变体也会显示这个数 |
| 246–257 | `def p4` | P4 表。先从结果文件名里取出 CPU 评测用了多少题 |
| 259–282 | `m['ttft_pass_drift']` | 每个配方一行:两个子模型的体积、精度、配对差、分段耗时、ABBA 漂移 |
| 287–295 | `fp16 行的配对差相对 GPU BF16` | 表后的两条说明 |
| 298–326 | `def p5` | P5 第一部分:孤立视觉塔的计时与对拍(体积一列的单位是 MiB)|
| 328–348 | `ttft_vqa_ms` | P5 第二部分:接回 TensorRT-LLM 的端到端 TTFT,基线与引擎两个结果文件逐档对比 |
| 350–356 | `acc_b, acc_e` | P5 第三部分:精度一句话 |
| 359–367 | `def fmt_median` | 一组演示记录里某个耗时字段的中位数,按单位格式化 |
| 371–383 | `def demo_sort_key` | 演示表的行序:GPU 在前、CPU 在后;同一条栈内按档位;同一档位内基线排第一。返回一个元组,Python 按元组逐项比较 |
| 386–397 | `def demo` | P6 演示表:读 `results/demo/` 下的全部记录并排序。目录里有什么文件就列什么行 —— 换过演示的变体后,旧记录要手工删 |
| 399–412 | `fmt_median(r["rows"], "engine_ttft_ms", unit)` | 每次运行一行:首字延迟、引擎侧 TTFT、视觉段的中位数,平均得分,各题的回答,运行时刻 |
| 413–425 | `asked = {` | 各次运行问的是同一批题时才在表后列出题目与最常见的人工答案;否则只提示去看记录文件 |
| 427–433 | `warmups` | 表后的两条说明:这些是单次观察值(预热次数从记录里读)、两条栈首字延迟的口径不同 |
| 436–468 | `def cpp() -> str:` | CPP 小节:读 `cpp_parity_cpu.json`,Python 与 C++ 各一行(首字延迟分解、decode、两遍漂移),表后写明对拍的两条判据各有多少题相同 |
| 472 | `SECTIONS = {` | 小节名 → 生成函数。新增一节只要在这里登记,并在 README 里放一对标记 |
| 475–483 | `--only` | 命令行:可以只刷新指定的小节 |
| 486–491 | `if args.only and name not in args.only` | 逐节处理;不在 `--only` 里的跳过(原样保留) |
| 492–497 | `re.compile(rf"(<!-- RESULTS:{name}:BEGIN -->\n).*?(<!-- RESULTS:{name}:END -->)", re.S)` | 用正则找到这一节的一对标记;README 里没有这对标记就警告并跳过 |
| 499–501 | `pattern.sub(` | 生成内容(空则写"尚未生成"),替换标记之间的部分,标记本身保留 |
| 503–506 | `newline="\n"` | 写回 README,强制 LF 行尾(脚本可能在 Windows 侧跑) |

### 3.11 `scripts/16_cpp_parity_cpu.py` —— CPU 栈 C++ 程序的对拍与配对计时

回答的问题只有一个:C++ 程序和评测用的 Python 路径是不是同一回事。两边调的是同一个库,所以判据是"逐字相同"。

| 行 | 代码 | 说明 |
|---|---|---|
| 6–19 | `要回答的问题只有一个` | 文件头:三条核对(回答逐字相同、输入 token 数相同、配对计时),以及怎么保证两边喂进去的东西相同 |
| 20–54 | `计时纪律` | ABBA 交替、预热用最后一张图(KV cache 复用会让与预热相同的那个请求跳过 prefill)、强制生成固定个数的 token |
| 56–60 | `BINARY = REPO_ROOT` | C++ 程序的位置与四个常量;档位、图片数、生成 token 数都与 `14_bench_cpu.py` 相同 |
| 63–92 | `def prepare` | 取 dev 题、预缩放,把图片存成 PNG、提示词存成文本,两边共用 |
| 93–106 | `icc_profile=None` | 存 PNG 时不带色彩配置文件(不影响像素);写完读回来核对落盘的像素与内存里的相同 |
| 109–124 | `def run_cpp` | 写请求清单,调 C++ 程序,读回它每个请求一行的 JSON;线程数与生成上限显式传过去 |
| 127–134 | `def as_tensor` | PIL 图 → `[1,H,W,3]` uint8 张量,与 `backends.py` 里的写法相同 |
| 137–179 | `def run_python` | Python 侧:经 `make_answerer` 加载(评测的同一个入口),直接调它的管线以拿到分段计时 |
| 181–183 | `answerer.close()` | 答完显式释放:下一步要起 C++ 进程,两份模型不能同时占内存 |
| 186–195 | `def segments` | 把一个实现的样本汇总成各段的 P50 / P90;prefill = 首字延迟 − 视觉段 |
| 198–223 | `def main` | 入口:只接受 `ov-*` 变体;C++ 程序没构建时提示先 `make cpp-cpu` |
| 225–259 | `py = run_python(args.variant, items, bench=False)` | 第 1 部分:两边各答一遍,逐题比回答(只去首尾空白)和输入 token 数,并用同一个打分函数打分 |
| 261–284 | `order = ["python", "cpp", "cpp", "python"]` | 第 2 部分:按 ABBA 四遍计时,每遍各自加载、测完释放;记下每遍的首字延迟中位数 |
| 285–293 | `ttft_pass_drift` | 同一个实现前后两遍的中位数之差:这次测量自身的漂移幅度,用来判断两个实现的差值有没有意义 |
| 295–312 | `ttft_ratio_cpp_over_python` | 写结果文件;回答或输入长度有不一致时以非零退出码结束 |

---

## 4. C++ 程序 `cpp/`(CPU 栈)

只有 CPU 栈有 C++ 程序:OpenVINO GenAI 自带 C++ 接口;GPU 栈的 TensorRT-LLM 用的是 PyTorch 后端,模型跑在 Python 进程里。

### 4.1 `cpp/vlm_infer_ov.cpp` —— 直接调 GenAI 的 C++ 接口

| 行 | 代码 | 说明 |
|---|---|---|
| 1–41 | `与 vlmdeploy/backends.py 的 OpenVinoAnswerer 对应` | 文件头:与 Python 是同一个库、同一份 IR、同样的配置;图片怎么进来;两种用法 |
| 44–46 | `double MsSince` | 计时小工具:单调时钟上量毫秒 |
| 49–59 | `struct Options` | 命令行选项;默认的线程数和生成上限与 `backends.py` 的常量相同 |
| 62–94 | `Options ParseArgs` | 解析参数;单题和批量两种模式必须二选一 |
| 97–105 | `std::string ReadText` | 把整个文件读成字符串(提示词文件、请求清单) |
| 109–128 | `ov::Tensor LoadImage` | 读图 → 转 RGB → 逐行拷进 `[1,H,W,3]` 的 uint8 张量;不缩放 |
| 133–146 | `ov::genai::GenerationConfig MakeConfig` | 生成参数:从默认构造出发(贪心)。**不用管线自带的配置** —— IR 目录里的 `generation_config.json` 写的是采样 |
| 149–160 | `struct Answer` | 一个请求的结果:回答,加上管线自带的分段计时 |
| 163–186 | `Answer Ask(` | 回答一个问题:只传问题文字和图片,模板由管线自己套;取出首字延迟、视觉段、每 token 耗时、输入与输出 token 数 |
| 190–213 | `std::string JsonString` | 把回答写成合法的 JSON 字符串(转义引号、反斜杠、换行、控制字符) |
| 217–229 | `std::string JsonNumber` | 数字写成 JSON;NaN / 无穷大写成 null |
| 232–251 | `std::vector<Request> LoadRequests` | 读请求清单:每行是图片路径和提示词文件路径 |
| 254–286 | `void RunBatch` | 批量模式:预热(用最后一个请求,原因见注释里的 KV cache 说明),再逐个作答,每个请求写一行 JSON 并立即落盘 |
| 289–313 | `void RunSingle` | 单题模式:流式回调把文字一段一段打到终端,最后打印首字延迟与生成速度 |
| 316–346 | `int main` | 入口:设好分词器库的环境变量,加载管线,按模式分发;所有错误汇到一个 catch,非零退出 |

### 4.2 `cpp/CMakeLists.txt` —— 构建

| 行 | 代码 | 说明 |
|---|---|---|
| 1–33 | `GenAI 的头文件:pip 包里没有` | 文件头:五样依赖各从哪里来,其中 GenAI 的头文件轮子里没有 |
| 36–38 | `find_package(OpenVINO REQUIRED COMPONENTS Runtime)` | OpenVINO 与 OpenCV 用各自的 CMake 配置找 |
| 41–53 | `set(GENAI_INCLUDE_DIR` | GenAI 没有随轮子发 CMake 配置:头文件目录、库、分词器库三个路径显式给,并逐个检查 |
| 56–63 | `add_executable(vlm_infer_ov` | 目标与头文件路径、链接的库 |
| 69 | `--disable-new-dtags` | **用 RPATH 而不是 RUNPATH**:否则 GenAI 依赖的 TBB 会从系统目录加载,和 Python 用的不是同一个线程库 |
| 71 | `VLM_TOKENIZERS_LIB=` | 把分词器库的路径作为宏写进程序,`main()` 用它设置环境变量 |

---

## 5. 工具与测试

### 5.1 `tools/parallel_fetch.py` —— 下载器的命令行入口

| 行 | 代码 | 说明 |
|---|---|---|
| 1–8 | `预取 2.7 GB 的 TensorRT-LLM wheel` | 实现在 `vlmdeploy/fetch.py`,这里只是薄薄一层命令行。环境安装脚本用它预取安装包 |
| 21–38 | `--sha256` | 参数:地址、输出路径、可选的期望大小与 sha256、并发数、分块大小 |
| 40–48 | `except FetchError` | 调 `fetch`;失败时把原因打到标准错误并返回 1 —— 安装脚本靠退出码判断 |

### 5.2 `tools/probe_trtllm.py` —— TensorRT-LLM 分段排障探针

P0 排查"输出全是感叹号"时写的工具,留在仓库里:下次升级 TensorRT-LLM 再出问题,可以直接用。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–13 | `哪一段出了问题` | 思路:把语言模型、视觉塔、拼接三段拆开测。纯文字题也坏 → 先查语言模型 |
| 23 | `TLLM_WORKER_USE_SINGLE_PROCESS` | 让模型跑在当前进程里,异常栈才看得全。必须在导入 TensorRT-LLM **之前**设置,所以写在文件顶部 |
| 30 | `TEXT_PROMPTS` | 两道答案确定的纯文字题(首都、加法) |
| 36–51 | `--no-cuda-graph` | 一组"二分开关":注意力后端、CUDA graph、调度器、数据类型。每次只动一个,看问题跟着哪个开关走 |
| 54–64 | `logprobs=1 if args.mode == "logprobs" else None` | 按开关构造 `LLM`;`logprobs` 模式下打印首个 token 的对数概率(全零或 NaN 会在这里现形) |
| 66–75 | `default_multimodal_input_loader` | 纯文字模式只走语言模型;带图模式走完整路径 |
| 77–87 | `tokens={list(comp.token_ids)}` | 打印生成的 token id 与文字 —— 看 id 比看文字更直接("全是 0 号 token"一眼可见) |

### 5.3 `tools/check_docs.py` —— 给本文的行号把关

| 行 | 代码 | 说明 |
|---|---|---|
| 1–16 | `带行号的文档最容易过期` | 为什么需要它,以及本文遵守的格式约定 |
| 17–19 | `不是正确性证明` | 坦白它的能力边界:锚点滑出范围能查出;范围内的小幅漂移、"说明"一列写得对不对,都查不出 |
| 29–39 | `SOURCE_SUFFIXES` | 三个正则:带反引号路径的标题、以行号开头的表格行、反引号里的锚点片段;以及"看起来是源文件"的后缀表 |
| 42–53 | `def check` | 逐行扫描文档,维护"当前文件"及其源码 |
| 57–69 | `if path.is_file()` | 遇到标题:反引号里的东西若是仓库里确实存在的文件,就切换当前文件 |
| 70–74 | `does not exist` | 看起来是源文件却不存在(改名、删除):报错并清掉当前文件,免得后面的行被拿去和上一个文件比 |
| 76–86 | `row says` | 遇到表格行:解析行号范围;超出文件长度就记一条错误 |
| 88–96 | `if not any(anchor in text for text in block)` | 取出这段源码,要求"代码"列里的每个锚点都原样出现在其中(子串匹配) |
| 99–125 | `--coverage` | 入口:打印错误;`--coverage` 另外打印每个文件被导读覆盖了多少行;有错误时退出码为 1 |

### 5.4 `tests/` —— 单元测试

测试只覆盖公共库里**不需要 GPU、模型**的部分(缩放规则、打分、划分、计时工具、变体解析)。
用标准库的 `unittest`,两个 conda 环境都能跑。

#### `tests/test_imaging.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 1–4 | `处理器再算一次 smart_resize 时尺寸不变` | 这个模块要守住的核心性质 |
| 18–22 | `HAVE_PIL` | 探测有没有 Pillow;没有的环境里跳过用到它的用例 |
| 27–31 | `test_square_budgets_hit_exactly` | 边长 448 / 672 / 896 的方图在对应档位下不缩放,正好 256 / 576 / 1024 个 token |
| 33–38 | `test_non_square_undershoots_budget` | 非方图的实际 token 数小于预算(4:3 的图在 256 档只有 234 个)—— 所以评测要记每张图的实际 token 数 |
| 40–50 | `test_never_exceeds_budget_and_keeps_multiple_of_28` | 各种长宽比、各个档位:结果永远是 28 的倍数、永远不超预算 |
| 52–61 | `test_idempotent_under_processor_defaults` | **最关键的一条**:预缩放后的尺寸再过一遍 `smart_resize`,必须原样返回 |
| 63–68 | `test_bankers_rounding_matches_reference` | 钉住银行家舍入的行为(1.5 → 2,2.5 → 2) |
| 70–78 | `assertRaises(ValueError)` | 两个异常路径:极端长宽比、尺寸不是 28 的倍数 |
| 82–95 | `self.assertIs(resize_to_budget(out, 256), out)` | 缩放后的尺寸正确;已是目标尺寸时返回**同一个对象**(没有重复插值) |
| 98–119 | `smart_resize as ref` | 与 transformers 的官方实现在几千个尺寸上逐点对比 —— "逐行对应官方实现"这句话是被测试保证的 |

#### `tests/test_tasks.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 18–19 | `TEN = (` | 一组 10 个人工答案,下面几个用例共用 |
| 24–34 | `test_minority_answer_scores_partial` | VQA 准确率的三种情形:多数答案得满分;恰好 3 人给出的答案得部分分;只有 1 人给出的答案得很少的分。注释里写了手算过程 |
| 36–43 | `test_normalization` | 归一化:去冠词与句点、数字词转数字、还原缩写、去千分位逗号 |
| 45–47 | `test_wrong_answer_scores_zero` | 答错得 0 分 |
| 51–62 | `class RealWorldQaTest` | 选项字母的抽取、文字与数字的匹配 |
| 66–80 | `test_subset_is_deterministic_and_roughly_quarter` | 划分是确定的(同一个 id 两次结果相同),calib 的比例在 20%–30% 之间 |
| 84–86 | `def _downloaded` | 某个数据集的目录在不在。两个数据集可以分开下载,所以跳过条件各看各的目录 |
| 90–104 | `test_order_is_stable_and_complete` | 读取顺序稳定(连读两次前 5 题相同),并钉住第一题的 id。没下载 TextVQA 时整个类跳过 |
| 106–120 | `test_realworldqa_sample_shape` | RealWorldQA 的样本只有 1 个答案、提示词以 `nothing else.` 结尾。单独一个用例、单独的跳过条件:只下了 TextVQA 时它跳过,不连累上面那个用例 |
| 122–135 | `test_calib_and_dev_share_no_image` | **数据纪律的测试**:calib 与 dev 没有任何一张共同的图 |

#### `tests/test_timing_variants.py`

| 行 | 代码 | 说明 |
|---|---|---|
| 23–31 | `test_percentile_matches_numpy_linear` | 分位数与 numpy 的结果一致(期望值是用 numpy 算好写进来的) |
| 33–39 | `test_summarize_fields` | 汇总字段齐全、数值正确 |
| 41–48 | `test_schedule_is_reproducible_permutation` | 交错计划可复现(同一个种子两次相同),且每一轮都是全部测量项的一个排列 |
| 50–51 | `test_round_spread` | 跨轮极差的算法 |
| 53–65 | `test_ratio_ci_brackets_point_estimate` | 比值的置信区间包住点估计;两组样本长度不等时报错 |
| 69–79 | `test_paths` | 变体名到模型目录的映射;不认识的名字报错 |
| 81–85 | `test_device` | 哪些变体跑在 GPU 上 |
| 89–94 | `test_gpu_memory_is_int_or_none` | 显存读数要么是非负整数、要么是 `None`,不会抛异常 |
| 97–122 | `test_detects_stale_line_numbers` | 行号校验工具自己的测试:造一份小文档,一行对、一行行号过期、一行超出文件长度,期望恰好报 2 个错 |

---

## 6. 构建与环境

### 6.1 `Makefile` —— 流水线入口

所有命令都从这里进。它把"用哪个 conda 环境、带哪些环境变量"封装起来,脚本本身不关心这些。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–7 | `两个 conda 环境严格隔离` | 开头的注释交代了运行环境。最后一行解释为什么用 conda 的绝对路径:make 起的是非交互 shell,PATH 里没有 conda |
| 8–15 | `GPU := $(CONDA) run --no-capture-output -n trtllm python` | 四个解释器前缀:GPU 环境、CPU 环境、系统 python3(下载和报告只用标准库)、ruff |
| 17–19 | `export PYTHONUNBUFFERED := 1` | 导出给所有子进程的环境变量:Hub 镜像、大文件目录、关闭输出缓冲(日志实时可见) |
| 20–23 | `TRTLLM_ENV := PYTHONPATH=$(CURDIR)/vlmdeploy/patches/site` | **只给 GPU 命令加的前缀**:让 Python 启动时加载 `sitecustomize` 补丁。CPU 命令不带它 |
| 26–32 | `.PHONY:` | 冒烟测试的默认参数;声明这些目标不是文件名 |
| 35–46 | `env-trtllm:` | 建环境、下模型、下数据 |
| 49–55 | `check: lint docs-check test` | 质量检查入口:单元测试、静态检查、文档行号校验,以及把三者串起来的 `make check`(提交前用) |
| 57–69 | `smoke: smoke-hf smoke-trtllm ov-export smoke-ov` | P0 冒烟的四步。CPU 的两步用 `/usr/bin/time -v` 包住,记录峰值内存 |
| 71–72 | `$(if $(ONLY),--only $(ONLY))` | 报告:给了 `ONLY=` 才加 `--only` 参数 |
| 74–81 | `EVAL_ARGS =` | 评测的通用参数。用 `=` 而不是 `:=`,是延迟展开 —— 每次调用时才代入当时的变量值 |
| 83–85 | `$(if $(filter ov-%,$(VARIANT)),$(CPU),$(TRTLLM_ENV) $(GPU))` | **按变体名自动选环境**:`ov-` 开头用 CPU 环境,否则用 GPU 环境加补丁。末尾的 `< /dev/null` 是踩坑后加的(注释说明了原因) |
| 87–90 | `p1:` | P1:HF BF16 原生分辨率,两个数据集 |
| 92–103 | `$(foreach b,$(BUDGETS),` | P2:`foreach` 把三个档位展开成三条用 `&&` 串起来的命令,末尾的 `true` 收住最后一个 `&&`。任何一条失败,后面的都不跑 |
| 105–108 | `quant_args =` | P3 的一个小技巧:把 `fp8-img` 这样的名字用 make 自己的函数拆成 `--method fp8 --calib img` |
| 110–122 | `p3: p3-quantize p3-eval p3-bench` | P3 三步:量化 4 个 checkpoint;在 dev / test / rwqa 上评测并与 BF16 配对;计时 |
| 124–128 | `CPU_N ?= 200` | P4 的配方列表与评测题数。注释坦白了 200 题的分辨率 |
| 129–141 | `p4: p4-compress p4-eval p4-bench` | P4 三步:压缩(先收集校准输入,再逐配方)、评测、ABBA 计时 |
| 143–158 | `p5:` | P5:三个档位各建引擎;每个档位、每个配置各起一个进程计时;s576 档做全量精度配对 |
| 160–174 | `demo: demo-gpu demo-cpu` | P6:GPU 四个配置(t576 基线与 FP8、方图 s576 基线与视觉塔引擎)、CPU 两个配置各演示一遍 |
| 176–193 | `cpp-cpu: genai-headers` | CPU 栈的 C++ 程序:`genai-headers` 取头文件,`cpp-cpu` 构建(四个路径从 ovgenai 环境里查出来;`OV_SITE` 用 `=` 延迟展开,只有这几个目标才去调一次 conda),`parity-cpu` 跑对拍与配对计时 |

### 6.2 `pyproject.toml` —— 包信息与静态检查规则

| 行 | 代码 | 说明 |
|---|---|---|
| 2–4 | `[build-system]` | 用 setuptools 构建 |
| 7–14 | `dependencies = ["pyarrow>=15", "pillow>=10"]` | 包名、版本、Python 版本要求。基础依赖只有两个 —— 公共库就靠它们加标准库 |
| 16–23 | `[project.optional-dependencies]` | 三组可选依赖:GPU 栈、CPU 栈、开发工具。前两组互相隔离,分别装进两个 conda 环境 |
| 30–32 | `line-length = 120` | 行宽 120,目标 Python 3.12 |
| 34–36 | `select = ["E", "F", "W", "I", "UP", "B", "D", "ANN", "RUF100"]` | 启用的规则集:基本错误、导入排序、现代语法、常见 bug、**docstring(D)**、**类型注解(ANN)**、失效的 `noqa` |
| 37–41 | `ignore = [` | 有意不启用的几条,每条带理由(如中文 docstring 不要求英文句点) |
| 43–44 | `convention = "google"` | docstring 用 Google 风格(`Args:` / `Returns:` / `Raises:`) |
| 47–51 | `"tests/*" = ["D", "ANN"]` | 按目录豁免:脚本要先改搜索路径再导入;测试不强制 docstring 与注解 |

### 6.3 `envs/setup_trtllm.sh` 与 `envs/setup_ovgenai.sh` —— 环境安装

两个脚本都可重复执行:每一步先检查是否已完成。文件头的注释是"版本为什么这样锁"的完整记录。

#### `envs/setup_trtllm.sh`

| 行 | 代码 | 说明 |
|---|---|---|
| 2–12 | `版本为什么这样锁` | 版本约束是直接读安装包里的元数据得到的(只取几十 KB,不下载 2.7 GB 整包);torch 必须先装、从哪个源装、为什么选这个 RC |
| 14–15 | `--no-cache-dir` | 磁盘上的考虑:WSL 虚拟盘不会自动缩小,所以 pip 不留缓存 |
| 20 | `set -euo pipefail` | 任何一步出错立即停;用到未定义变量也报错;管道里任一环节失败都算失败 |
| 23–37 | `TRTLLM_SHA256=` | 版本号、安装包文件名及其 sha256、torch 的版本与下载源 |
| 46–52 | `--override-channels -c conda-forge` | 第 1 步:建环境(Python 3.12 + OpenMPI + mpi4py),只用 conda-forge 一个频道 |
| 54–62 | `torch.version.cuda` | 第 2 步:装 torch。检查条件同时看版本号和 CUDA 大版本,防止装成别的变体还以为对了 |
| 64–76 | `parallel_fetch.py` | 第 3 步:先用多连接下载器预取安装包并校验 sha256,再从本地文件安装 |
| 78–93 | `import tensorrt_llm` | 第 4 步:自检 —— 打印关键包的版本,并真的导入一次(MPI 或 CUDA 运行库缺失只在导入时暴露) |
| 95 | `pip freeze` | 把装好的精确版本记录下来 |

#### `envs/setup_ovgenai.sh`

| 行 | 代码 | 说明 |
|---|---|---|
| 2–17 | `transformers 锁 5.0.0` | 为什么单独建一个环境;三个 OpenVINO 包必须同版本号;其中一条解释了为什么 transformers 锁在 5.0.0 |
| 38–43 | `conda create -y -n "$ENV_NAME"` | 建环境 |
| 46–49 | `--index-url https://download.pytorch.org/whl/cpu` | 装 CPU 版 torch(导出只在 CPU 上跑,GPU 版会多下 3 GB 用不上的库) |
| 52–54 | `"openvino-genai==$OV_VERSION.0"` | 装 OpenVINO 全家、NNCF、optimum-intel、transformers,版本都锁定;末尾的 pyarrow、pillow 没锁版本 |
| 57–69 | `VLMPipeline available` | 自检并记录版本 |

### 6.4 `envs/fetch_genai_headers.sh` —— 取 GenAI 的 C++ 头文件

pip 的 `openvino-genai` 轮子只有库、没有头文件。这个脚本按已装库的提交号取同一个提交的头文件,放在仓库之外。

| 行 | 代码 | 说明 |
|---|---|---|
| 1–19 | `头文件必须与已装的库来自同一个提交` | 文件头:为什么需要这一步、为什么版本必须对上、为什么不校验压缩包的 sha256 |
| 22–33 | `GENAI_COMMIT=` | 钉死的提交号、目标目录、记号文件、用哪个 Python 查已装版本 |
| 36–41 | `installed=$(` | 第 1 步:已装库的版本号最后一段必须是这个提交号的开头,否则报错退出 |
| 44–48 | `if [ -f "$STAMP" ]` | 第 2 步:记号文件里的提交号对得上、头文件也在,就什么都不做 |
| 50–66 | `trap ` | 第 3 步:下载到临时目录(脚本结束时自动清掉),核对压缩包的顶层目录名带着完整提交号,只解出头文件和许可文件 |
| 69–76 | `mkdir -p "$DEST"` | 第 4 步:换上新的一份,写记号文件 |

---

## 7. 读完之后:几个值得带走的写法

把上面散落的做法归拢一下。它们不依赖这个模型,换一个部署项目也用得上。

| 做法 | 在哪 | 解决什么 |
|---|---|---|
| 单一事实来源 | `config.py`、`variants.py` | 一个参数只定义一次;改一处,处处生效 |
| 重依赖延迟导入 | `backends.py` 的各构造函数 | 一份代码在依赖不同的多个环境里都能 import |
| 统一接口隔离变化 | `backends.Answerer` | 换后端、换精度不动评测逻辑,分数之差才能归因 |
| 逐条落盘 + 断点续跑 | `10_eval.py` 的 `run` | 长任务被打断不用从头再来 |
| 汇总与推理解耦 | `10_eval.py` 的 `summarize` | 汇总只读落盘的记录,任何时候都能重算 |
| 配对比较 | `10_eval.py`、`11_bench_gpu.py` | 同一道题 / 同一轮内相减,把共同的噪声消掉 |
| 先写临时文件再改名 | `fetch.py`、`13_compress_cpu.py`、`12_quantize_gpu.py` | 中途失败不留下"看起来完整"的半成品 |
| 自检后再继续 | `12_quantize_gpu.py` 的量化器检查、`01_download_data.py` 的分片检查 | 假设不成立时立刻失败,而不是悄悄产出错的结果 |
| 回退路径"宁可慢,不能错" | `sitecustomize.py` 的 `hf_forward` | 优化路径不适用时保证结果正确 |
| 把排障记录写进 docstring | `quantize_in_batches`、`_VisionEngines` | 下一个读代码的人知道这段"奇怪的写法"是为了什么 |
| 表格由结果生成,行号由机器把关 | `90_report.py`、`check_docs.py` | 数字不会和实测脱节;行号过期能被发现(文字说明仍要人核对) |
