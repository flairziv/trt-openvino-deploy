# vlm-deploy

Qwen2-VL-2B-Instruct(视觉塔 0.67B + 语言模型 1.31B)在一台笔记本上的两栈部署:

- **GPU 栈**:TensorRT-LLM 1.3.0rc18(PyTorch 后端)+ ModelOpt 0.37,RTX 4060 Laptop 8 GB(Ada,sm_89)。
- **CPU 栈**:OpenVINO GenAI 2026.4.1 + NNCF 3.4,i9-14900HX(WSL2 分到 4 个 vCPU)。

这是 [depth-anything-deploy](../depth-anything-deploy)(项目 A)的续作。A 部署的是一个纯 ViT;
这里的 ViT 变成了 VLM 的视觉塔,外加一个语言模型。两个项目共用同一套方法:
真实引擎上验证、配对置信区间、交错计时、决策集与测试集分离、结果表只从 `results/*.json` 生成。

> 数字只代表这台笔记本:GPU 动态频率,CPU 跑在 WSL2 里且无法绑定物理 P 核。
> 方法可以迁移到 Jetson Orin 等端侧平台,数字不能。

---

## 结论速览

每一行的出处是下文对应小节的结果表;带"±"的是逐题配对差的 95% 置信区间。

| 问题 | 结论 | 数据 |
|---|---|---|
| 首字延迟花在哪 | 视觉塔。GPU 上占 65–78%,CPU 上占 67% | GPU t576:TTFT 278 ms = 视觉 207 + prefill 71;CPU t256:9.77 s = 6.53 + 3.26(P2、P4) |
| 量化语言模型帮哪一段 | 帮 decode,FP8 还帮 prefill;首字延迟几乎不动 | FP8(图文校准):decode 70 → 104 tok/s、prefill 71 → 41 ms、TTFT 278 → 256 ms,精度 −0.21 ± 0.44;AWQ-INT4:decode 146 tok/s,精度 −1.53 ± 0.70(P3) |
| 校准要不要带图 | 差别在噪声内,图文校准更省时间 | TextVQA val:FP8 75.51 对 75.42,AWQ 74.18 对 73.99(P3) |
| 怎么降首字延迟 | 把视觉塔换成 TensorRT FP16 引擎 | TTFT 290 → 177 ms(s576,1.64×),精度 +0.18 ± 0.29(5000 题)(P5) |
| CPU 上的杠杆 | 视觉塔 INT8 仅权重 + 语言模型 INT4 | TTFT 9.77 → 4.37 s,decode 11.5 → 34.2 tok/s,体积约 4.9 → 2.1 GB,精度 −1.90 ± 3.11(200 题,判不了)(P4) |
| 什么不值得做 | 视觉塔静态 INT8;只在千题规模上选型 | 静态 INT8 比仅权重多省 0.76 s,却掉 5–7 分(相对 fp16 −7.00 ± 5.35)(P4);AWQ 的 −1.5 分在 1344 题的 dev 上区间含 0(P3) |

- 选型:GPU 要首字延迟就上视觉塔引擎,要吞吐再叠 FP8 语言模型(两者的组合没有实测);CPU 用 `int4-vis8w`。
- 这些数字怎么来的、每一步的代价和踩过的坑:[docs/deployment-record.md](docs/deployment-record.md);
  逐文件逐行读代码:[docs/code-walkthrough.md](docs/code-walkthrough.md)。

---

## 要回答的问题

一个视觉问答请求分三段:**视觉编码**(算力受限)→ **prefill**(算力受限,与 token 数成正比)
→ **decode**(显存带宽受限,每生成 1 个 token 读一遍全部权重)。

两个可心算的事实决定了实验怎么设计:

- 1 个图像 token = 28×28 像素(14 像素切块,再 2×2 合并)。448² 的图是 256 个 token,
  但视觉塔内部处理的是合并前的 1024 个块 —— 每个图像 token 上视觉塔的计算量约是语言模型的 2 倍,
  还要加上随块数平方增长的注意力。
- 语言部分与 Qwen2.5-1.5B 结构相同(28 层 / 1536 维 / 2 个 KV 头),decode 速度可以用此前在同一张卡上
  测过的 71.6(FP16)/ 152.9(AWQ-INT4)tok/s 预测。KV cache 每个 token 只有 28 KB,
  1000 个 token 才 28 MB,不到权重的 1% —— 单路推理下 KV 量化没有收益,所以不做。

| 问题 | 怎么测 | 跑之前写下的预测 |
|---|---|---|
| 1. 首字延迟花在哪 | 3 档 token(256 / 576 / 1024),同一轮交错计时"带图请求"和"同长度纯文本请求",相减得视觉段 | TTFT 约 0.1 / 0.3 / 0.55 s,视觉塔占 60–75% |
| 2. 量化帮哪一段 | 语言模型 BF16 / FP8(W8A8)/ AWQ-INT4(W4A16) | decode ≈ 70 / 110 / 150 tok/s;FP8 还能加速 prefill,AWQ 不能;TTFT 几乎不变(视觉塔没量化) |
| 3. 校准要不要带图 | FP8 / AWQ 各用纯文本、图文两种校准数据 | 图像 token 的激活分布与文本不同,纯文本校准的 FP8 掉得更多 |
| 4. CPU 上的杠杆 | 视觉塔 FP16 / INT8 仅权重 / INT8 静态;语言模型 INT8 / INT4 / INT4+AWQ | 4 核下视觉塔每张图 4–5 s,静态 INT8 约减半;仅权重 INT8 是否也变快取决于 OpenVINO 的动态量化,由实验判定 |

条件触发:若问题 1 证实视觉塔占 TTFT 超过 40%,用项目 A 的方法(静态形状、常量折叠位置编码、强类型 FP16、
确认融合 MHA)给视觉塔建 TensorRT 引擎,经 `multi_modal_embeddings` 接回 TensorRT-LLM 做**端到端**计时
(TensorRT-LLM 对 Qwen2-VL 的视觉塔没做任何优化,源码注释原话是"Qwen2-VL 过时了,保留 HF 实现")。

---

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 环境、模型与数据下载、三方冒烟测试 | ✅ 2026-11-09 |
| P1 | 评测基座:数据划分、提示词、打分;HF BF16 复现官方分数(TextVQA 79.7 / RealWorldQA 62.9,差 ≤1.5 才继续) | ✅ 2026-11-10 |
| P2 | GPU BF16 × 3 档 token:精度、TTFT 分解、decode、显存 | ✅ 2026-11-10 |
| P3 | GPU 量化:FP8 / AWQ-INT4 × 两种校准 | ✅ 2026-11-11 |
| P4 | CPU 栈:导出、NNCF 压缩、视觉塔量化 | ✅ 2026-11-15 |
| P5 | (条件触发)视觉塔 TensorRT 引擎,端到端 TTFT | ✅ 2026-11-15 |
| P6 | 报告与演示:流式演示脚本、结论速览、部署记录与代码导读、静态检查 | ✅ 2026-11-16 |

> 2026-11-12 本机仓库目录丢失,源码与提交历史由开发记录重建;`results/` 里 P0–P3 的原始结果文件没有恢复。
> 下面 P0–P3 的表格是丢失前生成的内容,P4 / P5 的表格来自 2026-11-14 至 15 的完整运行。补齐 P0–P3 的结果文件之前,
> 刷新表格请用 `make report ONLY="P4 P5 DEMO CPP"`:不带 `ONLY` 会把没有结果文件的小节刷成“尚未生成”。

### P0 冒烟测试

TextVQA 验证集前 8 题,图片预缩放到 256 token 预算内,三个后端用同样的题、同样的图、同样的聊天模板、
贪心解码、最多 16 个新 token。判据是与 HF 参考**逐字**一致,而不是绝对准确率(8 道题没有统计意义)。

<!-- RESULTS:SMOKE:BEGIN -->
| 后端 | 精度 / 设备 | 8 题平均 VQA acc | 答案与 HF 逐字相同 | 输入与 HF 一致 | 加载 s | 峰值显存 / 内存 |
|---|---|---|---|---|---|---|
| HF transformers | BF16,GPU | 0.738 | 参考 | — | 37.5 | 整卡 5652 MB |
| TensorRT-LLM 1.3.0rc18 | BF16,GPU | 0.738 | 8/8 | 提示词文本相同 | 88.4 | 整卡 6651 MB |
| OpenVINO GenAI 2026.4.1 | FP16 权重,FP32 计算,CPU 4 线程 | 0.738 | 8/8 | 8/8 | 1.7 | 进程峰值 7.12 GiB |

- OpenVINO 导出(optimum-cli,FP16):进程峰值 6.86 GiB。
- OpenVINO 单次运行观察值(含首次调用开销,不是基准):TTFT 均值 12.2 s,其中视觉编码 + embedding 拼接 7.3 s。
<!-- RESULTS:SMOKE:END -->

三方 8/8 逐字一致,两条推理栈都装对了。达到这个结果之前修了四个问题,前三个见下文"环境"一节的兼容性表;
第四个是提示词:TensorRT-LLM 的 `default_multimodal_input_loader` 在图片占位符和文字之间多插了一个换行
(Qwen2-VL 官方模板里两者紧挨着),只多一个 token,8 题里就有 2 题答案与 HF 不同(其中 1 题从对变错)。
现在 GPU 两个后端用同一个函数生成提示词(`vlmdeploy/smoke.py` 的 `chat_prompt`);OpenVINO GenAI 自己套模板,
用"输入 token 数与 HF 逐题相同"来核对。

### P1 评测基座:复现官方分数

HF transformers BF16,原生分辨率(处理器默认像素范围,与官方评测相同),贪心解码,最多 16 个新 token。
提示词与打分口径见 `vlmdeploy/tasks.py`(与 lmms-eval 对齐)。验收线:|复现 − 官方| ≤ 1.5 分。

<!-- RESULTS:P1:BEGIN -->
| 数据集 | 题数 | 平均图像 token | 复现(95% CI) | 官方 | 差 | 绝对差 ≤ 1.5 | 绝对差 ≤ CI 半宽 |
|---|---|---|---|---|---|---|---|
| TextVQA val | 5000 | 985 | 79.00 ± 1.05 | 79.7 | -0.70 | ✅ | ✅ |
| RealWorldQA | 765 | 1736 | 61.31 ± 3.45 | 62.9 | -1.59 | ❌ | ✅ |
<!-- RESULTS:P1:END -->

- **评测基座验证通过**。TextVQA 差 −0.70,在 1.5 分线和置信区间内;RealWorldQA 差 −1.59,超出预设的 1.5 分线,
  但它只有 765 题,95% 置信半宽 ±3.45 —— 1.5 分线比这个数据集的统计噪声还窄,所以另报"差是否在置信区间内"
  (两个判据都保留,不事后改线)。为排除打分写错,逐题核对过:438 道选项题全部答成单个字母,抽取无歧义;
  106 道答错的自由作答题全是真错(yes/no 反、计数错),不是格式问题。
- 剩下的 0.7–1.6 分差距方向一致(都偏低),可能来自生成参数或像素上限与官方评测不同,不再深究:
  后面所有阶段比较的都是同一基座下的相对差,不依赖与官方分数逐位对齐。
- 工程坑:原生分辨率下每张图的序列长度都不同,PyTorch 缓存分配器只涨不跌,约 700 题后 8 GB 显存占满、
  被 Windows 换到共享内存,每题从 0.7 s 变慢到 4–7 s(GPU 利用率仍显示 100%)。每 50 题归还一次缓存解决
  (`vlmdeploy/backends.py`)。

### P2 GPU BF16:精度与首字延迟分解

<!-- RESULTS:P2:BEGIN -->
| 档位 | TextVQA 平均图像 token | TextVQA val | RealWorldQA |
|---|---|---|---|
| t256 | 241 | 69.43 ± 1.21 | 52.94 ± 3.54 |
| t576 | 550 | 75.72 ± 1.12 | 57.25 ± 3.51 |
| t1024 | 936 | 79.27 ± 1.04 | 59.22 ± 3.48 |
| 原生(HF,P1) | 985 | 79.00 ± 1.05 | 61.31 ± 3.45 |

| 档位 | 输入 token | TTFT ms | 视觉段 ms | prefill ms | 视觉占比 | 预处理+通信 ms | decode tok/s | 跨轮极差 |
|---|---|---|---|---|---|---|---|---|
| t256 | 286 | 120 | 78 | 42 | 65% | 39 | 70.5 | 1.3% |
| t576 | 590 | 278 | 207 | 71 | 74% | 43 | 70.3 | 0.4% |
| t1024 | 928 | 484 | 377 | 107 | 78% | 50 | 70.1 | 1.6% |

- 计时:12 张 dev 图 × 5 轮交错,P50;decode 强制生成 64 个 token。视觉段 = 同轮配对的(带图 TTFT − 同长度纯文本 TTFT)。
- 显存(nvidia-smi 整卡口径,含 Windows 桌面占用):加载前 486 MB → 加载后 6133 MB → 测完 6153 MB。
<!-- RESULTS:P2:END -->

- **精度随 token 数单调上升,t1024 已追平原生分辨率**:TextVQA 从 t256 到 t576 涨 6.3 分、再到 t1024 涨 3.6 分;
  t1024(平均 936 token)与原生(985 token,HF 参考)差 0.27 分,在置信区间内。RealWorldQA 的图是 1080p 车载画面,
  原生要 1736 个 token,t1024 仍差 2.1 分 —— 档位该按"图的信息密度"选,不是越省越好。
- **首字延迟的大头是视觉塔**:TTFT 120 / 278 / 484 ms,视觉段占 65 / 74 / 78%,与跑之前的预测(0.1 / 0.3 / 0.55 s、
  60–75%)一致。每个图像 token 上视觉塔耗时 0.30–0.37 ms,语言模型 prefill 每 token 只要 0.12–0.15 ms(约 2.5–3 倍,
  且随 token 数增长更快:视觉塔内部处理 4 倍的块,注意力随块数平方增长)。
  → **只量化语言模型,TTFT 最多省 26%**(t576 档 prefill 71 ms / 278 ms);这触发了 P5(视觉塔 TensorRT 引擎)。
- **引擎外还有 40–50 ms**:带图请求在客户端多出 39–50 ms,纯文本请求只有 2 ms —— 这是 HF 图像处理器(Python 里
  缩放、归一化、切块)加上把像素张量送进 worker 进程的开销,不在引擎时间戳里,却在用户等待的路径上。
  t256 档它相当于 TTFT 的三分之一,端侧要把预处理挪进 C++ / GPU。
- **decode 70 tok/s,与图无关**:三档都是 70.1–70.5 tok/s,与此前同一张卡上测 Qwen2.5-1.5B 的 71.6 tok/s 吻合 ——
  decode 每步读一遍语言模型权重,是显存带宽受限,KV cache 每 token 只有 28 KB,1000 个 token 也不改变这一点。
- 显存:TensorRT-LLM 加载后整卡 6.1 GB(BF16 权重 4.4 GB + KV 池 + CUDA graph);计时跨轮极差 0.4–1.6%,数字稳定。

### P3 GPU 量化:FP8 / AWQ-INT4 × 校准数据

<!-- RESULTS:P3:BEGIN -->
| 变体 | 权重 GB | dev | dev 配对差 | TextVQA val | val 配对差 | 答案与 BF16 相同 | RealWorldQA 配对差 | TTFT ms | prefill ms | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 4.42 | 80.59 ± 2.00 | — | 75.72 ± 1.12 | — | — | — | 278 | 71 | 70.3 |
| fp8-txt | 3.11 | 80.41 ± 2.00 | -0.18 ± 0.88 | 75.42 ± 1.12 | -0.29 ± 0.45 | 85.7% | +1.44 ± 1.68 | 255 | 41 | 104.4 |
| fp8-img | 3.11 | 80.31 ± 2.00 | -0.28 ± 0.87 | 75.51 ± 1.12 | -0.21 ± 0.44 | 88.2% | +0.92 ± 1.68 | 256 | 41 | 104.4 |
| awq-txt | 2.49 | 79.73 ± 2.02 | -0.86 ± 1.34 | 73.99 ± 1.14 | -1.73 ± 0.70 | 69.3% | -0.65 ± 2.28 | 287 | 73 | 146.4 |
| awq-img | 2.49 | 79.81 ± 2.03 | -0.78 ± 1.36 | 74.18 ± 1.14 | -1.53 ± 0.70 | 71.9% | -1.31 ± 2.54 | 290 | 72 | 146.6 |

- 全部在 t576 档;配对差 = 同一题上(量化 − BF16)的平均分差 ± 95% CI,单位:分。
- 计时同 P2(12 张 dev 图 × 5 轮交错,P50);prefill = 同长度纯文本请求的 TTFT。
<!-- RESULTS:P3:END -->

<!-- CONCLUSION:P3 -->

- **FP8 与 BF16 分不出**:TextVQA val 配对差 −0.29 ± 0.45 / −0.21 ± 0.44,dev −0.18 / −0.28,RealWorldQA +1.44 / +0.92
  (± 1.68),区间全部含 0;逐题答案与 BF16 相同的占 86–88%。
- **AWQ-INT4 是真掉点,而且 dev 集看不出来**:val 上 −1.73 ± 0.70 / −1.53 ± 0.70,区间不含 0,答案相同率只剩 69–72%。
  同样两个模型在 dev(1344 题)上是 −0.86 ± 1.34 / −0.78 ± 1.36,区间含 0;RealWorldQA(765 题,± 2.3–2.5)也含 0。
  → **1.5 分量级的损失要 5000 题才显著**,只在千题规模的 dev 上做选型会把 AWQ 判成"无损"。
- **图文校准 vs 纯文本校准:差别在噪声内**:val 上 FP8 75.51 vs 75.42、AWQ 74.18 vs 73.99,差 0.09 / 0.19 分,远小于
  ± 1.1 的区间。图文校准的答案相同率略高(88.2 vs 85.7%、71.9 vs 69.3%),方向一致但不足以下结论。
  代价上图文校准反而更省:AWQ 图文 128 条(75k token)搜索 510 s,纯文本 387 条(14k token)1155 s。
- **FP8 的 prefill 和 decode 都快,AWQ 只快 decode**:FP8 的 prefill 71 → 41 ms(1.74×)、decode 70.3 → 104.4 tok/s(1.49×);
  AWQ 的 decode 到 146.5 tok/s(2.08×),prefill 却没动,而且随 token 数变差 —— t256 档 37 vs 42 ms 略快、t576 档持平、
  t1024 档 112 vs 107 ms 略慢。这与"decode 受显存带宽限制(权重越小越快)、prefill 受算力限制(INT4 权重要先反量化
  再算)"的解释一致;没有做逐层剖析,只说一致。
- **TTFT 几乎不动,印证 P2 的预测**:FP8 278 → 255 ms(−8%),AWQ 287–290 ms(略升)。视觉塔没量化,视觉段
  207 → 214–218 ms 原地不动,量化后它占 TTFT 的 84%(t576)/ 86%(t1024)→ **要降首字延迟只能动视觉塔**(P5)。
- 体积与显存:权重 4.42 → 3.11 → 2.49 GB(视觉塔与 lm_head 不量化,所以不是减半 / 减到四分之一);加载后整卡
  6133 → 5607 → 5180 MB。
- **选型**:要首字延迟和精度,选 FP8(精度无损、prefill 与 decode 都快);只有显存或长输出吞吐吃紧时才选 AWQ-INT4,
  代价是 1.5–1.7 分。校准数据用哪种都行,图文校准更快。

### P4 CPU 栈:NNCF 压缩配方

OpenVINO GenAI,CPU 4 线程,t256 档。所有配方从同一份 FP16 IR 出发,只替换配方涉及的子模型:
`int8` / `int4` / `int4awq` 只压语言模型;`int4-vis8w` / `int4-vis8s` 在 int4 的基础上再把视觉塔压成 INT8
(仅权重 / 权重与激活都量化)。校准数据取自 calib 划分的 64 道带图题,与 dev / test 不重叠。

<!-- RESULTS:P4:BEGIN -->
| 配方 | 语言模型 GB | 视觉塔 GB | TextVQA(n=200) | 配对差 | TTFT s | 视觉段 s | prefill s | decode tok/s | ABBA 漂移 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 3.09 | 1.33 | 63.75 ± 6.42 | -0.50 ± 0.74 | 9.77 | 6.53 | 3.26 | 11.5 | -3.0% |
| int8 | 1.55 | 1.33 | 64.30 ± 6.40 | +0.55 ± 0.98 | 7.73 | 6.50 | 1.41 | 23.4 | +0.9% |
| int4 | 0.91 | 1.33 | 62.15 ± 6.48 | -1.60 ± 3.64 | 7.99 | 6.54 | 1.59 | 31.9 | +2.5% |
| int4awq | 0.91 | 1.33 | 62.65 ± 6.46 | -1.10 ± 3.22 | 7.56 | 6.23 | 1.53 | 34.1 | -2.0% |
| int4-vis8w | 0.91 | 0.67 | 61.85 ± 6.49 | -1.90 ± 3.11 | 4.37 | 2.90 | 1.50 | 34.2 | -3.6% |
| int4-vis8s | 0.91 | 0.67 | 56.75 ± 6.60 | -7.00 ± 5.35 | 3.61 | 2.14 | 1.52 | 34.2 | -3.0% |

- 精度:TextVQA val 前 200 题,t256;fp16 行的配对差相对 GPU BF16(同 200 题 64.25 ± 6.41),其余行相对 fp16。
- 计时:6 张 dev 图,配方按 ABBA 顺序各加载两次,P50;CPU 4 线程(WSL2 的 4 个 vCPU,不能绑核)。"ABBA 漂移"是两遍中位数之差,衡量这个数字的可信程度。
<!-- RESULTS:P4:END -->

<!-- CONCLUSION:P4 -->

- **CPU 上首字延迟的大头同样是视觉塔**:FP16 基线 TTFT 9.77 s = 视觉段 6.53 s(67%)+ prefill 3.26 s。跑之前预测视觉塔
  每张图 4–5 s,实测更慢。同一档位 GPU 的 TTFT 是 120 ms(P2),相差约 80 倍。
- **只压语言模型,TTFT 只降两成**:INT8 让 prefill 3.26 → 1.41 s(2.3×)、decode 11.5 → 23.4 tok/s(2.0×),INT4 的 decode 到
  31.9–34.1 tok/s(2.8–3.0×),但视觉段原地不动,TTFT 仍是 7.6–8.0 s。INT4 的 prefill(1.5–1.6 s)不比 INT8 快 ——
  与 GPU 上"低比特权重只帮 decode"的现象一致。
- **视觉塔 INT8 仅权重就能快 2.25 倍**(跑之前留给实验判定的问题):视觉段 6.53 → 2.90 s,TTFT 到 4.37 s,体积 1.33 → 0.67 GB,
  不需要校准数据,压缩 21 秒。这与 OpenVINO CPU 插件对 INT8 权重的全连接层做动态量化的行为一致,没有逐层剖析确认。
- **视觉塔静态 INT8 再快 0.76 s,但掉 5–7 分**:视觉段 2.14 s、TTFT 3.61 s;精度相对 fp16 −7.00 ± 5.35,相对同一语言模型的
  int4 是 −5.40 ± 5.18(由逐题结果另算),两个区间都不含 0,答案与 fp16 相同的只剩 49%。项目 A 里同样的 NNCF 配置
  (TRANSFORMER + SmoothQuant)在纯 ViT 上几乎无损,在这里不成立;没有做逐层敏感度分析或 accuracy control,原因未查。
- **语言模型:INT8 无损,INT4 在 200 题上判不了**:INT8 相对 fp16 +0.55 ± 0.98,答案相同 95.5%。INT4 是 −1.60 ± 3.64、
  加 AWQ 是 −1.10 ± 3.22,区间都含 0,但答案相同率只有 66% / 73% —— 方向和量级与 GPU 上 AWQ-INT4 的 −1.5 分(5000 题才显著)
  相近,200 题下不能说它无损。AWQ 相对不带 AWQ 的 int4 是 +0.50 ± 2.20(另算),分不出;代价是 371 s 和 6.97 GiB 的峰值内存。
- **两条栈的基线一致**:CPU 的 fp16 相对 GPU BF16 是 −0.50 ± 0.74,答案相同 94%,后面的差异可以归到压缩本身。
- **选型**:要速度选 int4-vis8w(TTFT 9.77 → 4.37 s,decode 11.5 → 34.2 tok/s,四个子模型合计约 4.9 → 2.1 GB);
  精度优先时语言模型用 INT8、视觉塔仍用 INT8 仅权重(这个组合没有实测)。视觉塔静态 INT8 不值得:多省 0.76 s,换来 5–7 分。
- **这组数字的分辨率有限**:精度只有 200 题(配对差的 95% CI 约 ±1–5 分),只能检出大的掉点;计时是 6 张图、每个配方
  加载两次,ABBA 漂移在 ±3.6% 以内。WSL2 的 4 个 vCPU 不能绑核,数字只代表这台机器。
- 工程坑:`nncf.quantize` 每一步把全部统计点一次挂到模型上再编译,视觉塔第 2 步有 584 个统计点,编译后的统计模型在 8 GB 的
  WSL 里放不下 —— 只喂 1 条样本,进程内存也在 15 秒内涨到 7.4 GiB,两次把整个 WSL 虚拟机带进 OOM,所以减校准样本没用。
  改为按目标节点分批收集统计(每批 40 个,流水线与参数不变),峰值 6.97 GiB,耗时 93 分钟
  (`scripts/13_compress_cpu.py` 的 `quantize_in_batches`)。

### P5 视觉塔 TensorRT 引擎

视觉塔单独导出成静态形状、强类型 FP16 的 TensorRT 引擎(每个网格一个),运行期由 `sitecustomize` 的第 4 个补丁
按 `image_grid_thw` 查表替换 TensorRT-LLM 里的 HF 视觉塔;语言模型仍是 BF16。静态形状要求固定网格,所以这一节
用**方图档位**(中心裁成方图再缩放到 s256 / s576 / s1024),与 P2 / P3 的保持长宽比档位(t\*)不是同一批输入。

<!-- RESULTS:P5:BEGIN -->
**孤立视觉塔**(GPU 时间,CUDA event,5 轮交错 × 每轮 10 次,P50):

| 档位(方图) | 合并前块数 | HF BF16 ms | TRT FP16 ms | 加速 | BF16 对 FP32 相对误差 | 引擎对 FP32 相对误差 | 引擎对 BF16 余弦(最小) | 融合注意力层 | 引擎 MiB |
|---|---|---|---|---|---|---|---|---|---|
| s256 | 1024 | 78.1 | 37.1 | 2.11× | 0.3868 | 0.2580 | 0.92997 | 32 | 1278 |
| s576 | 2304 | 209.8 | 102.1 | 2.06× | 0.3585 | 0.2343 | 0.92551 | 32 | 1278 |
| s1024 | 4096 | 452.4 | 212.0 | 2.13× | 0.3480 | 0.2210 | 0.93615 | 32 | 1279 |

- 相对误差 = 输出与 FP32 真值之差的范数 / 真值的范数,取 16 张 dev 图中的最大值;FP32 真值是同一份权重升到 FP32 的视觉塔。

**接回 TensorRT-LLM 的端到端 TTFT**(方图,12 张 dev 图 × 5 轮,P50;每个档位、每个配置各起一个进程先后测量 —— 8 GB 显存装不下三个引擎加 HF 模型):

| 档位 | TTFT ms(HF 视觉塔) | TTFT ms(引擎) | TTFT 加速 | 视觉段 ms(HF) | 视觉段 ms(引擎) | 视觉占比 |
|---|---|---|---|---|---|---|
| s256 | 118 | 83 | 1.42× | 77 | 40 | 65% → 48% |
| s576 | 290 | 177 | 1.64× | 217 | 104 | 75% → 59% |
| s1024 | 579 | 343 | 1.69× | 456 | 220 | 79% → 64% |

**精度**(TextVQA val 全量,方图 s576):HF 视觉塔 69.09 ± 1.22,引擎 69.27 ± 1.21;逐题配对差 +0.18 ± 0.29,答案逐字相同 94.5%。
<!-- RESULTS:P5:END -->

<!-- CONCLUSION:P5 -->

- **视觉塔换成引擎,首字延迟降到 0.59–0.70 倍**:TTFT 118 → 83 / 290 → 177 / 579 → 343 ms(1.42× / 1.64× / 1.69×)。
  对比 P3:只量化语言模型,FP8 的 TTFT 是 278 → 255 ms(−8%),AWQ-INT4 不降反升 —— 首字延迟的杠杆在视觉塔。
- **孤立测到的 2.1× 完整带到了端到端**:视觉段 77 → 40 / 217 → 104 / 456 → 220 ms(1.9–2.1×),与孤立引擎的
  37 / 102 / 212 ms 只差 2–8 ms;prefill(40–44 / 72 / 123 ms)和 decode(69–74 tok/s)不受影响。
- **视觉塔仍是 TTFT 里最大的一段**:占比 65 / 75 / 79% → 48 / 59 / 64%。s576 档剩下 104 ms 视觉 + 72 ms prefill;
  把 P3 的 FP8 语言模型(prefill 71 → 41 ms)叠上来应当还能再降,这个组合没有实测。
- **精度分不出**:TextVQA val 全量 5000 题,逐题配对差 +0.18 ± 0.29,区间含 0;答案逐字相同 94.5%。
- **FP16 引擎比现状的 BF16 更接近 FP32**:对 FP32 真值的最大相对误差,引擎 0.22–0.26,HF BF16 0.35–0.39。
  两者的绝对值都不小(32 层累积,最小余弦 0.93–0.97),但任务分数分不出 —— 判“能否替换”要看配对评测,
  不能只看特征误差。
- **32 层注意力全部被融合**:三个引擎各 32 个融合注意力层;加速来自哪些 kernel 没有逐层剖析。
- 显存与加载:加载后整卡 6.1–6.2 GB → 5.3–5.4 GB(引擎接管后 PyTorch 视觉塔移到 CPU);加载时间 144–158 s →
  185–206 s(各测一次)。计时跨轮极差:HF 视觉塔 3.4–3.7%,引擎 0.8–1.7%。
- **代价是静态形状**:
  - 每个网格一个引擎,各约 1.34 GB(1278 MiB,权重不共享);8 GB 显存放不下三个引擎加语言模型,worker 初始化时报
    `Impossible to fit in any sequence in kvCache`,所以运行期只加载用得到的网格。
  - 这里的方图是中心裁剪得到的,**裁剪本身掉 6.6 分**:同为 576 token,s576 的 HF 视觉塔基线 69.09,P2 保持长宽比的
    t576 是 75.72(TextVQA 的文字常在画面边缘)。这一节量的是“引擎替换”的收益,不是方图档位的精度。
    落地时应按相机的固定分辨率建对应网格的引擎(非方图只是另一个 `(h, w)`),而不是裁成方图。
- 工程坑:transformers 5.x 给 SDPA 传 `enable_gqa=True`,TorchScript 导出器直接断言失败,导出期间要关掉这条分支
  (视觉塔的 KV 组数恒为 1,数值不变);引擎必须跑在 PyTorch 的当前 CUDA stream 上,否则 CUDA event 量到的是
  主机等待时间。

### P6 演示

`scripts/20_demo.py`:一张图 + 一个问题,在任一部署变体上流式作答,现场打印首字延迟与生成速度。它经评测用的同一个入口
`variants.make_answerer` 加载,演示的就是前面被评测的那套配置(计时脚本为了拿逐请求的性能指标是各自构造后端的,参数相同)。
下表是 2026-11-16 一次完整 `make demo` 的记录,六个配置各起一个进程、依次运行,问同样 4 道 dev 题:
GPU 上是 t576 的 BF16 基线与 FP8 语言模型、方图 s576 的 BF16 基线与视觉塔引擎;CPU 上是 FP16 基线与推荐配方。

<!-- RESULTS:DEMO:BEGIN -->
| 变体 | 档位 | 题数 | 首字延迟(中位) | 引擎侧 TTFT(中位) | 视觉段(中位) | VQA acc 均值 | 回答(按题号) | 运行时刻 |
|---|---|---|---|---|---|---|---|---|
| trtllm-bf16 | t576 | 4 | 322 ms | 285 ms | — | 0.75 | Nokia / Propeller / Simon Clancy / 2011 | 11-16 01:22 |
| trtllm-fp8-img | t576 | 4 | 315 ms | 253 ms | — | 0.75 | nokia / Propeller / Simon Clancy / 2011 | 11-16 01:25 |
| trtllm-bf16 | s576 | 4 | 328 ms | 284 ms | — | 0.50 | nokia / airbus / qantas / 2011 | 11-16 01:27 |
| trtllm-bf16-vistrt | s576 | 4 | 221 ms | 177 ms | — | 0.50 | Nokia / propeller / qantas / 2011 | 11-16 01:30 |
| ov-fp16 | t256 | 4 | 9.51 s | — | 6.23 s | 0.75 | nokia / Airbus A320 / Simon Clancy / 2011 | 11-16 01:33 |
| ov-int4-vis8w | t256 | 4 | 4.49 s | — | 2.90 s | 0.75 | nokia / Propeller / Simon Clancy / 2011 | 11-16 01:34 |

题目(dev 集前 4 张不同的图,各取第 1 题):

1. what is the brand of phone?(最常见的人工答案:nokia、toshiba)
2. what type of plane is this?(最常见的人工答案:lape、cargo)
3. who is this copyrighted by?(最常见的人工答案:simon clancy、simon ciancy)
4. what year is shown in the photo?(最常见的人工答案:2011)

- 单次观察值:丢弃开头 2 次预热请求后每题只问一次,不是基准;结论性的数字见 P2–P5 的表。
- 首字延迟的口径:GPU 是客户端墙钟(含图像预处理与进程间通信),另列引擎侧时间戳(与 P2 / P5 同口径);CPU 是 OpenVINO GenAI 管线自带的计时(与 P4 同一来源),另列其中的视觉段。
<!-- RESULTS:DEMO:END -->

- **演示与基准对得上**(引擎侧 TTFT,括号里是 P2 / P3 / P5 的基准):t576 的 BF16 285 ms(278)、FP8 253 ms(256);
  方图 s576 的 BF16 284 ms(290)、视觉塔引擎 177 ms(177)。CPU 的 fp16 9.51 s(9.77)、int4-vis8w 4.49 s(4.37),
  视觉段 6.23 → 2.90 s(6.53 → 2.90)。
- **用户实际等的比引擎侧多 26–72 ms**:客户端墙钟减引擎侧时间戳,是主进程里的图像预处理加进程间通信(P2 的基准量到
  39–50 ms)。它把 FP8 省下的 32 ms 冲淡了:客户端看到的是 322 → 315 ms。4 道题、各问一次,这个量级的差别不能下结论 ——
  结论只认交错轮次的基准。
- **第 3 题答成 qantas 是裁剪造成的,不是引擎造成的**:这张图是竖图,版权字样在最底边,方图档位的中心裁剪把它裁掉了。
  同样用方图的 BF16 基线(HF 视觉塔)也答 qantas;保持长宽比的 t576 两个配置都答对(P5 已说明中心裁剪掉 6.6 分)。
  第 2 题各配置答 Propeller / airbus / Airbus A320,都不得分。
- 换成开放式问题(`--question "Describe this image in one sentence."`)能看到 decode 速度。2026-11-15 的两次试跑里,
  FP8 是 106–110 tok/s、CPU 的 int4-vis8w 是 35–37 tok/s,与 P3 的 104.4、P4 的 34.2 相近。这类运行只打印、不落盘,
  这两组数字出自当时的终端输出,仓库里没有对应的结果文件。
- 2026-11-15 第一次跑视觉塔引擎档时,4 道题的引擎侧 TTFT 是 427–558 ms;之后三次运行(含上表这次)都是
  158–178 ms,没有复现,原因未查明。那一次的记录已被后来的运行覆盖,只留在当时的终端输出里。脚本因此把加载前后的
  整卡显存一并写进结果文件。
- 用自己的图片:`<环境前缀> scripts/20_demo.py --variant ov-int4-vis8w --image a.jpg --question "..."`
  (环境前缀见 Makefile 的 `GPU` / `CPU` 变量;GPU 变体还要带 `PYTHONPATH=vlmdeploy/patches/site`)。

### C++ runtime(CPU 栈)

`cpp/vlm_infer_ov.cpp`:不依赖 Python 的 C++ 程序,直接调 OpenVINO GenAI 的 C++ 接口(`VLMPipeline`),
与 Python 侧的 `OpenVinoAnswerer` 是同一个库、同一份 IR、同样的配置(CPU、4 线程、贪心解码)。
支持单题流式输出和批量作答两种用法。`scripts/16_cpp_parity_cpu.py` 让两边答同一批 dev 题:
图片由脚本预缩放后存成 PNG(无损)交给 C++,两边送进管线的像素完全相同。

<!-- RESULTS:CPP:BEGIN -->
| 实现 | TTFT s | 视觉段 s | prefill s | decode tok/s | 两遍漂移 | 加载 s |
|---|---|---|---|---|---|---|
| Python(OpenVINO GenAI 的 Python 接口) | 4.05 | 2.67 | 1.42 | 35.1 | +3.7% | 2.7 |
| C++(`cpp/vlm_infer_ov`) | 4.08 | 2.67 | 1.40 | 36.4 | +0.3% | 2.2 |

- 变体 `ov-int4-vis8w`,档位 t256,4 线程;运行时刻 11-21 03:32。
- 对拍(dev 集前 16 题):回答逐字相同 16/16,输入 token 数相同 16/16;VQA acc 均值 Python 0.5625 / C++ 0.5625。
- 计时:6 张图,每个请求强制生成 32 个 token,顺序 python → cpp → cpp → python,每遍预热 1 次;表里是两遍合并后的 P50。C++ / Python 的 TTFT 之比 1.008。
<!-- RESULTS:CPP:END -->

- **C++ 与评测用的 Python 路径是同一回事**:回答逐字相同、输入 token 数逐题相同。
- **C++ 不提速,也不该提速**:首字延迟之比在两遍测量自身的漂移幅度之内。时间全花在视觉塔和语言模型的推理上,
  那是同一个库里的同一段代码;Python 这层只是把图和问题递进去。C++ 程序的价值是部署形态
  (不带 Python 解释器和 torch / transformers 这些包),不是速度。
- **GPU 栈没有对应的 C++ 程序**:TensorRT-LLM 这里用的是 PyTorch 后端,语言模型是 PyTorch 模块、生成循环在 Python 里,
  没有引擎文件可以交给 C++。要 C++ 得换一条路线:用纯 TensorRT 给语言模型建引擎并手写生成循环,或改用 TensorRT-LLM 的
  引擎后端(安装包里有它的运行库 `libtensorrt_llm.so`,但不带 C++ 头文件)。两条都没有做,也没有验证过可行性。
- 工程坑三处:
  - **轮子不带头文件**:pip 的 `openvino-genai` 只有库,没有 C++ 头文件(`openvino` 轮子是带的)。
    `envs/fetch_genai_headers.sh` 按已装库的提交号从官方仓库取同一个提交的头文件,放在仓库之外。
  - **线程库会被悄悄换掉**:GenAI 的库依赖 TBB,自己却没写搜索路径。按默认方式(RUNPATH)链接时,
    它加载的是系统的 `libtbb`,而 Python 用的是 `openvino` 轮子自带的那一份;改用 RPATH 后两边才是同一个线程库。
  - **预热不能用第 1 个请求**:管线会复用上一次请求的 KV cache。预热和第 1 个正式请求相同时,
    那个样本的 prefill 从约 1.5 s 变成 0.09 s,首字延迟就不是真实的了;预热改用另一张图。
- 用自己的图片:`cpp/build/vlm_infer_ov --model $VLM_ARTIFACTS/openvino/int4-vis8w --image a.png --question "..."`。
  C++ 程序自己不缩放图片,大图会按模型默认的上限处理,token 数和耗时都与评测档位不同。

---

## 环境

全部在 WSL Ubuntu 24.04 里原生安装,不用 Docker。两个 conda 环境严格隔离:

| 环境 | 用途 | 关键版本 | 安装 |
|---|---|---|---|
| `trtllm` | GPU 栈、HF 参考、ModelOpt 量化 | tensorrt_llm 1.3.0rc18 / torch 2.10.0+cu130 / TensorRT 10.15 / transformers 5.5.4 / modelopt 0.37 | `make env-trtllm` |
| `ovgenai` | CPU 栈、optimum-intel 导出、NNCF | openvino / openvino-genai / openvino-tokenizers 2026.4.1 / nncf 3.4.0 / optimum-intel 2.2.0 / transformers 5.0.0 / torch 2.10.0 CPU | `make env-ovgenai` |

装好后的精确版本在 `requirements-trtllm.txt` / `requirements-ovgenai.txt`。它们只是版本记录,不是安装入口
(TensorRT-LLM 那一行指向已删除的本地 wheel);重建环境用 `envs/*.sh`。两个环境的 transformers 版本不同是有意的:
TensorRT-LLM 在依赖里锁死 5.5.4;optimum-intel 2.2.0 的总依赖允许到 5.5.x,但它给 Qwen2-VL 的导出配置单独写了上限 5.0
(`Qwen2VLOpenVINOConfig.MAX_TRANSFORMERS_VERSION`),用 5.5.4 导出直接报错,所以 `ovgenai` 锁 5.0.0。

**版本为什么这样锁**:直接读 TensorRT-LLM wheel 里的 METADATA(用 HTTP Range 只取几十 KB,不下载 2.7 GB 整包)
得到它的依赖约束:torch ≥2.10.0 且 ≤2.11.0a0、TensorRT ~10.15.1、CUDA 13 运行库。
所以 torch 必须先从 PyTorch 官方 cu130 源装好,否则 pip 会按依赖从默认源挑一个别的 CUDA 变体。
驱动 581.80 对应 CUDA 13.0,正好咬合,不需要装系统级 CUDA 13。更新的 1.3 RC 要求 torch 2.14 预览版,原生装不上。

**原生安装要额外处理的事**:

1. **MPI**:`import tensorrt_llm` 就需要 MPI。本机没有免密 sudo,用 conda-forge 的 openmpi 4.1,不走 apt。
2. **TensorRT-LLM 1.3.0rc18 + transformers 5.5.4 跑 Qwen2-VL 的三处兼容问题**(5.5.4 正是 TensorRT-LLM 自己
   锁死的版本,所以官方组合下这条路径直接跑不起来)。`vlmdeploy/patches/site/sitecustomize.py` 在每个 Python 进程
   启动时修补,不改安装目录里的任何文件;Makefile 用 `PYTHONPATH` 只对本项目的 GPU 命令启用它。
   用 sitecustomize 是因为 TensorRT-LLM 单卡也会另起 worker 进程跑模型,脚本里的补丁到不了那里。

   | 现象 | 根因 | 修补 |
   |---|---|---|
   | 启动即报缺 FlashAttention2 | TensorRT-LLM 把 Qwen2-VL 视觉塔写死成 flash-attn 2,但它不在 pip 依赖里(容器镜像里预装) | 没有 flash-attn 时改用 PyTorch SDPA |
   | 预热时报 `Number of mm_embeds (2) does not match expected total` | transformers 5.x 的视觉塔改为返回 `BaseModelOutputWithPooling`,TensorRT-LLM 仍当张量用,`len()` 得到字段数 2 | 取 `pooler_output`(即 4.x 直接返回的张量) |
   | 不报错,但纯文字题、带图题一律输出 `!!!!!!!!` | **lm_head 全零**:模型的 lm_head 与输入 embedding 共享权重,checkpoint 里没有 `lm_head.weight`;transformers 5.x 把 `tie_word_embeddings` 只留在总配置上,TensorRT-LLM 用 `text_config` 建语言模型时读到默认值 False,lm_head 被当成独立权重又找不到,一直是零。logits 全为 0,贪心解码每步都选 0 号 token `!` | 建语言模型前把总配置的 `tie_word_embeddings` 复制到 `text_config` |

   第三处最隐蔽:没有任何报错,只能靠"输出固定是 0 号 token"这个形态推断。排障时先用 `tools/probe_trtllm.py --mode text`
   确认纯文字题也坏(排除视觉段),再逐个关掉 CUDA graph / overlap scheduler(都不是),最后顺着"0 号 token =
   logits 全零而非乱码"查到 lm_head。这和项目 A 里"误差形态本身就是线索"是同一个教训。
3. **磁盘**:WSL 虚拟盘在 E 盘且不会自动缩小,pip 一律 `--no-cache-dir`;大文件放 `$VLM_ARTIFACTS`
   (默认 `D:\vlm-artifacts`,普通 NTFS 目录,删掉即释放)。

---

## 快速开始

```bash
make env-trtllm env-ovgenai   # 两个环境(首次约 40 分钟,trtllm 约 16 GB、ovgenai 约 2 GB)
make model data               # 模型 4.43 GB + 数据 1.9 GB,多连接下载,逐文件校验 sha256
make test                     # 单元测试
make smoke                    # P0 冒烟:HF 参考 / TensorRT-LLM / OpenVINO GenAI 答同一批题
make p1                       # 评测基座:HF BF16 原生分辨率复现官方分数(约 1 小时)
make p2                       # GPU BF16:3 档 token 的精度 + TTFT 分解(约 2 小时)
make p3                       # GPU 量化:4 个 checkpoint × dev / test / RealWorldQA + 计时(约 4 小时)
make p5                       # 视觉塔 TensorRT 引擎:对拍、孤立计时、接回 TensorRT-LLM 的端到端 TTFT 与精度
make p4                       # CPU 栈:NNCF 压缩 5 个配方 + 精度 + ABBA 计时(必须在 GPU 任务全部结束后跑,见下)
make demo                     # P6 演示:GPU 四个配置 + CPU 两个配置,流式作答并打印首字延迟(约 15 分钟)
make cpp-cpu parity-cpu       # CPU 栈的 C++ 程序:取 GenAI 头文件 + 构建;与 Python 逐题对拍 + 配对计时(约 7 分钟)
make report                   # 用 results/ 回填本 README 的表格
```

- 每一步都可断点续跑:评测逐题落盘,重跑跳过已完成的题;量化 / 压缩的产物已存在就跳过(`--force` 重做)。
- 任意单项:`make eval VARIANT=trtllm-fp8-img SPLIT=dev BUDGET=t576 PAIRED=trtllm-bf16`。
- P4 不能与 GPU 任务并行:TensorRT-LLM 驻留时 WSL(8 GB)只剩约 3.6 GB,而 OpenVINO 加载与 NNCF 压缩要 5–7 GB。

---

## 数据

| 数据集 | 用途 | 规模 | 许可 |
|---|---|---|---|
| TextVQA val(划分名 `test`) | 测试集:只报最终数字,不参与任何决策(OCR 类,对 token 数最敏感) | 5000 题 | CC BY 4.0 |
| RealWorldQA(`rwqa`) | 测试集:车载 / 真实场景空间理解 | 765 题 | CC BY-ND 4.0 |
| TextVQA train 第 1 个分片 → `dev` | 决策集:选量化配方、选校准数据 | 1344 题 / 863 张图 | CC BY 4.0 |
| TextVQA train 第 1 个分片 → `calib` | 校准集:只给量化校准用 | 387 题 / 253 张图 | CC BY 4.0 |

dev 与 calib 按**图片**划分(TextVQA 一张图平均 1.5 道题,按题划分会让同一张图同时出现在两边),
规则是 image_id 的 SHA-1 首字节模 4(`vlmdeploy/tasks.py`):与读取顺序、机器、Python 的 hash 随机化都无关。

图片在送进任何推理栈之前,先按 Qwen2-VL 的规则预缩放到"28 的整数倍且不超过 token 预算"(`vlmdeploy/imaging.py`)。
两栈处理器再算一次时尺寸不变、不再插值,预处理天然一致。注意这个模型的均值方差是 CLIP 的,不是项目 A 的 ImageNet 值。

---

## 目录结构

```
vlmdeploy/          公共库(顶层不导入任何重依赖,两个环境和纯标准库步骤都能用)
  config.py         单一事实来源:路径、模型常量、token 档位(t* 保持长宽比 / s* 方图)、数据清单
  fetch.py hub.py   多连接分块下载 + Hub 文件清单与 sha256 校验
  imaging.py        Qwen2-VL 的 smart_resize、按 token 预算预缩放、方图档位
  tasks.py          数据读取、test / rwqa / dev / calib 划分、提示词、打分
  backends.py       三个推理后端的统一答题接口(HF / TensorRT-LLM / OpenVINO GenAI)
  variants.py       变体注册表:变体名 → 后端与模型位置(评测、计时、报告只认这里的名字)
  timing.py         计时协议:分位数、交错轮次、跨轮极差、配对比值的 bootstrap 区间
  smoke.py          P0 冒烟:取题、与 HF 参考逐题比对;共用的聊天模板 chat_prompt
  sysinfo.py        运行环境读数(整卡显存)
  patches/site/     只对本项目 GPU 命令生效的 sitecustomize:TensorRT-LLM 的 3 处兼容修补 + 视觉塔引擎开关
envs/               两个 conda 环境的安装脚本(版本选择的理由写在文件头)
scripts/            流水线各步,编号即执行顺序
  00–03             下载、P0 冒烟
  10_eval.py        评测(任一变体 × 划分 × 档位,断点续跑,配对差)
  11_bench_gpu.py   GPU 计时:TTFT 分解、decode、显存
  12_quantize_gpu.py ModelOpt 量化语言模型(FP8 / AWQ-INT4 × 两种校准)
  13_compress_cpu.py NNCF 压缩 OpenVINO IR(5 个配方)
  14_bench_cpu.py   CPU 计时(ABBA)
  15_vision_trt.py  视觉塔 TensorRT 引擎:导出、构建、对拍、孤立计时
  16_cpp_parity_cpu.py CPU 栈 C++ 程序与 Python 的逐题对拍 + 配对计时
  20_demo.py        演示:流式视觉问答 + 单次首字延迟
  90_report.py      从 results/ 回填 README 表格
cpp/                CPU 栈的纯 C++ 程序 vlm_infer_ov(OpenVINO GenAI 的 C++ 接口;GPU 栈没有,见"C++ runtime"一节)
tools/              命令行工具:多连接下载、TensorRT-LLM 分段排障探针、文档行号检查
tests/              单元测试(打分、划分、缩放、计时、变体注册表)
docs/               部署记录(每一步的数据与代价)、代码导读(逐文件逐行)
results/            机器可读结果(README 表格的唯一来源);eval/ 逐题、bench/ 计时、quantize/ compress/ 产物记录、demo/ 演示
```

---

## 开发约定

```bash
make test         # 单元测试(unittest,不需要 GPU / 模型;动了公共库要在 ovgenai 里再跑一遍,见下)
make lint         # ruff 静态检查,规则在 pyproject.toml;ruff 不在两个推理环境里:pip install ruff
make docs-check   # 检查 docs/code-walkthrough.md 里的行号有没有过期(改了代码后跑;不检查文字说明对不对)
make check        # 以上三项一起跑;提交前用(仓库没有持续集成)
```

- `vlmdeploy/` 顶层不导入重依赖(torch / tensorrt_llm / openvino),两个环境都要能 import;改完在另一个环境里再跑一遍测试:
  `~/miniconda3/bin/conda run -n ovgenai python -m unittest discover -s tests`。
- 协议参数(档位、题数、轮数、线程数、生成上限)只在 `vlmdeploy/config.py`、`vlmdeploy/backends.py` 和各脚本开头的常量里
  定义一次;改动任何一个都会让已有的 `results/` 失效。
- 函数都有 docstring 与类型注解(`ruff` 的 D / ANN 规则);确实标不了类型的地方(PIL、torch、第三方对象)用带理由的 `noqa`。
- 注释:每条语句或紧密相关的两三行上方写一句,说明这一步做什么;不举例、不解释语法、不用装饰符号。
  只改注释也会让行号变化,改完跑 `make docs-check`。
- 结果表不手改:`results/*.json` → `scripts/90_report.py` → README 的 `RESULTS` 标记之间。标记之外的结论是手写的,重跑后要人工核对。
- 提交信息用 Conventional Commits;每完成一个阶段更新 CHANGELOG 并打一个 tag。所有文件 LF 行尾(`.gitattributes`)。

---

## 许可

- 模型:Qwen2-VL-2B-Instruct 为 Apache-2.0。权重不随仓库分发。
- 数据:TextVQA(CC BY 4.0)、RealWorldQA(CC BY-ND 4.0),仅用于评测,不随仓库分发。
- 本仓库代码:许可待定。
