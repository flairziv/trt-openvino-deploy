# Changelog

本项目的所有重要变更记录在此文件。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),
版本号按阶段递增:每完成一个阶段(P0–P6)发一个次版本,结论性数字只在 README 的结果表里(由 `scripts/90_report.py` 生成)。

## [Unreleased]

### Added
- CPU 栈的 C++ 程序 `cpp/vlm_infer_ov.cpp`(`make cpp-cpu`):直接调 OpenVINO GenAI 的 C++ 接口,单题流式与批量两种用法。
  `envs/fetch_genai_headers.sh`(`make genai-headers`)按已装库的提交号取 GenAI 的头文件(pip 轮子不带)。
- `scripts/16_cpp_parity_cpu.py`(`make parity-cpu`):C++ 与 Python 逐题对拍(回答、输入 token 数)+ ABBA 配对计时,
  结果在 `results/cpp_parity_cpu.json`,README 的 "C++ runtime" 表由 `90_report.py` 新增的 `CPP` 小节生成。
  `ov-int4-vis8w` / t256 上实测:16 题回答逐字相同,首字延迟之比 1.008。GPU 栈没有对应的 C++ 程序。

### Changed
- 源码注释(2026-11-24):全部源文件补了逐行注释,每条语句或紧密相关的两三行上方一句,说明这一步做什么
  (只加注释,代码不变);`docs/code-walkthrough.md` 的行号随之更新。
- 源码注释:把 `docs/code-walkthrough.md` 里源码还没写明的说明补进对应代码处(34 个文件;只加注释,代码不变)。

### Fixed
- `scripts/12_quantize_gpu.py` 量化后的自检原先只核对权重量化器:FP8 激活侧的 `input_quantizer` 漏进视觉塔时发现不了。
  现在按类型核对全部启用的量化器(新函数 `enabled_quantizers`),报错信息带上漏出的量化器名。
  用脚本的量化配置在真实模型上核对过(只插入量化器、不校准):FP8、AWQ 都没有漏出的量化器;没有重新量化。
- `tests/test_tasks.py`:RealWorldQA 的断言拆成单独的用例并按自己的数据目录决定是否跳过;原先只下载了 TextVQA 时
  `test_order_is_stable_and_complete` 会报错而不是跳过。
- `docs/code-walkthrough.md` 15 处与代码不符的说明;`smoke.summarize` 与 `11_bench_gpu.py` 文件头里写的结果文件名。

## [0.7.0] - 2026-11-16

### Added
- P6 演示 `scripts/20_demo.py`(`make demo` / `demo-gpu` / `demo-cpu`):任一部署变体上流式视觉问答,打印首字延迟与
  生成速度;经评测用的同一个入口 `variants.make_answerer` 加载。记录在 `results/demo/`,README 的演示表由
  `90_report.py` 的 `DEMO` 小节生成。
- 文档:`docs/deployment-record.md`(每一步优化的数据、代价与踩过的坑)、`docs/code-walkthrough.md`(逐文件逐行导读);
  README 增加"结论速览"与"开发约定"。
- `tools/check_docs.py` + `make docs-check`:检查代码导读里的行号有没有过期(锚点是否还在标注的行号范围内;不检查文字说明)。
- `make lint`(以及合并了 lint / docs-check / test 的 `make check`):ruff 静态检查;规则集加入 docstring(D,Google 风格)、类型注解(ANN)与失效 `noqa` 检查(RUF100)。
- `vlmdeploy/sysinfo.py`:整卡显存读数(原先在两个脚本里各有一份);`tasks.normalize_answer` 公开的答案归一化入口。

### Fixed
- `tasks.vqa_accuracy` 的 docstring:留一法下 4 个及以上标注者答案相同才是满分,恰好 3 个是 0.9(代码没变,原先的说明写错了)。
- 文档与注释里视觉塔引擎的体积:1278 MiB 是 1.34 GB,不是 1.28 GB;`90_report.py` 的 P5 表头改为 MiB。

### Changed
- 41 个脚本 / 工具函数补上 docstring;超过 120 列的行折行(`90_report.py` 的输出逐字节不变)。
- `smoke.py` 不再跨模块导入 `tasks` 的私有名;`fetch` / `hub` / `smoke` 的 `log` 参数补上 `Callable` 注解。
- 冒烟与压缩脚本里重复的常量(生成上限 16、CPU 线程数 4)改为从 `vlmdeploy/backends.py` 导入,数值不变。
- `pyproject.toml` 与 `vlmdeploy.__version__` 的版本号对齐到 0.7.0(此前一直停在 0.1.0)。
- `.editorconfig` 去掉用不到的 C++ 小节(仓库里没有 C++ 源码)。

### Notes
- 演示的耗时是单次观察值,不是基准。2026-11-15 第一次跑视觉塔引擎档时引擎侧 TTFT 为 427–558 ms,之后三次运行为
  158–178 ms,未复现,原因未查明;那一次的记录已被覆盖,只留在当时的终端输出里。
- `results/demo/` 是 2026-11-16 一次完整 `make demo` 的记录(六个配置依次运行)。
- `00_download_model.py` 于 2026-11-16 重新执行:本地模型的两个权重分片与 Hub 公布的 sha256 一致(没有重新下载),
  记录在 `results/download_model.json`。
- `02_smoke_gpu.py`、`03_smoke_cpu.py`、`12_quantize_gpu.py` 在仓库重建后只做过静态检查、导入检查与 `--help`,
  没有重新执行(对应的 P0–P3 结果不重新生成)。

## [0.6.0] - 2026-11-15

### Results
- P4 CPU 栈(OpenVINO GenAI,4 线程,t256):FP16 基线 TTFT 9.77 s(视觉段 6.53 s,占 67%),decode 11.5 tok/s。
- 语言模型 INT8:相对 fp16 +0.55 ± 0.98(200 题),prefill 3.26 → 1.41 s,decode 23.4 tok/s;INT4 / INT4+AWQ:
  −1.60 ± 3.64 / −1.10 ± 3.22(区间含 0,200 题判不了),decode 31.9 / 34.1 tok/s。
- 视觉塔 INT8 仅权重:视觉段 6.53 → 2.90 s,TTFT 4.37 s;静态 INT8:视觉段 2.14 s、TTFT 3.61 s,但精度 −7.00 ± 5.35。
- CPU fp16 与 GPU BF16 在同 200 题上相差 −0.50 ± 0.74。

### Changed
- `13_compress_cpu.py`:视觉塔静态量化改为按目标节点分批收集统计(`quantize_in_batches`,每批 40 个节点),
  流水线与参数同 `nncf.quantize(model_type=TRANSFORMER, preset=MIXED)`。

### Fixed
- 视觉塔静态量化在 8 GB 的 WSL 上 OOM(两次带垮整个虚拟机):NNCF 一次挂上 584 个统计点后编译出的统计模型放不下,
  与校准样本数无关。分批后峰值 6.97 GiB。

## [0.5.0] - 2026-11-15

### Results
- P5 视觉塔 TensorRT FP16 引擎(方图档位 s256 / s576 / s1024):孤立视觉塔 2.11× / 2.06× / 2.13×;接回 TensorRT-LLM 后
  TTFT 118 → 83 / 290 → 177 / 579 → 343 ms(1.42× / 1.64× / 1.69×),视觉段占比 65 / 75 / 79% → 48 / 59 / 64%。
- 精度(TextVQA val 5000 题,s576):HF 视觉塔 69.09 ± 1.22,引擎 69.27 ± 1.21,逐题配对差 +0.18 ± 0.29,与基线分不出。
- 对 FP32 真值的最大相对误差:引擎 0.22–0.26,HF BF16 0.35–0.39。

### Added
- 视觉塔引擎接回 TensorRT-LLM:`sitecustomize` 第 4 个补丁(`VLMDEPLOY_VISION_TRT` 开关,按 `image_grid_thw` 查引擎,
  没有对应网格时回退 PyTorch 视觉塔);变体 `trtllm-bf16-vistrt`;评测与计时的方图档位 `s256` / `s576` / `s1024`。
- `VLMDEPLOY_VISION_TRT_GRIDS`:只加载指定网格的引擎(8 GB 显存放不下三个引擎加语言模型)。
- `11_bench_gpu.py --square --budgets`:一档一个进程计时,结果并入同一个 JSON。
- `90_report.py --only` / `make report ONLY=P5`:只刷新指定小节。

### Fixed
- 视觉塔 ONNX 导出在 transformers 5.x 下断言失败(SDPA 的 `enable_gqa`):导出期间关闭该分支。
- 引擎体积读取:`IHostMemory` 没有 `len()`,改用 `nbytes`。
- `make eval` 的标准输入改接 `/dev/null`:评测进程会读走标准输入,经 `bash -s` 连续提交多条评测时,
  第一条跑完后面的命令就没了(两次“只跑了基线”都是这个原因)。

### Notes
- 2026-11-12 本机仓库目录丢失,2026-11-14 由开发记录重建:源码与提交说明一致,提交哈希与丢失前不同。
  `results/` 里 P0–P3 的结果文件没有恢复,README 中这几节的表格是丢失前生成的内容;P5 的结果是重建后完整重跑的。
- `requirements-*.txt` 于 2026-11-14 从两个 conda 环境重新 `pip freeze`。

## [0.4.0] - 2026-11-11

### Results
- P3 GPU 量化(t576 档,相对 BF16 的逐题配对差):FP8 在 TextVQA val 上 −0.29 ± 0.45 / −0.21 ± 0.44(纯文本 / 图文校准),
  与 BF16 分不出;AWQ-INT4 −1.73 ± 0.70 / −1.53 ± 0.70,显著掉点,但在 1344 题的 dev 上区间含 0。
- 计时:FP8 prefill 71 → 41 ms、decode 70 → 104 tok/s;AWQ-INT4 decode 146 tok/s、prefill 不变;TTFT 278 → 255 / 287–290 ms,
  视觉段不动,占比升到 84%。权重 4.42 → 3.11 / 2.49 GB。图文校准与纯文本校准的差别在噪声内。

### Changed
- P4:校准输入收集不再编译语言模型(钩住 forward 抓到 prefill 输入后抛异常中断),压低内存峰值;
  CPU 评测题数 `CPU_N` 300 → 200(每题约 10 s,配对差 95% CI 约 ±3 分,只能检出大的掉点)。
- P5:视觉塔对拍改为以 FP32 为真值,分别报 BF16 / 引擎相对 FP32 的误差;引擎在 PyTorch 当前 stream 上执行;
  融合注意力的检测改为在整份层信息里搜索;`90_report.py` 的 P5 表随之调整。

### Fixed
- 量化 checkpoint 的视觉塔键名:ModelOpt 导出为 `model.visual.*`,TensorRT-LLM 要 `visual.*`,导出后改回。
- 长跑评测卡死在 MPI 上(`mca_btl_tcp_frag_recv: readv failed: Connection timed out`):运行前设
  `OMPI_MCA_btl_tcp_if_include=lo` 把父子进程通信钉在回环口。此后 2 小时续跑未再出现,根因未证实。

## [0.3.0] - 2026-11-10

### Results
- P2 GPU BF16:TextVQA t256 / t576 / t1024 = 69.43 / 75.72 / 79.27(t1024 追平原生分辨率);TTFT 120 / 278 / 484 ms,
  视觉塔占 65 / 74 / 78%;decode 70 tok/s 与档位无关;引擎外的图像预处理 + 通信 39–50 ms。

### Added
- P2:`scripts/11_bench_gpu.py` —— TTFT 分解(带图请求 − 同长度纯文本请求,同轮配对)、decode、显存。
- P3:`scripts/12_quantize_gpu.py` —— ModelOpt FP8 / AWQ-INT4 × 纯文本 / 图文校准,视觉塔与 lm_head 不量化。
- P4:`scripts/13_compress_cpu.py`(NNCF 五个配方,校准输入分进程收集)、`scripts/14_bench_cpu.py`(ABBA 计时)。
- P5:`scripts/15_vision_trt.py`(视觉塔强类型 FP16 引擎 + 对拍);sitecustomize 第 4 个补丁(环境变量开关)
  把引擎接回 TensorRT-LLM 做端到端计时;方图档位 `s256/s576/s1024`。
- `vlmdeploy/timing.py`(纯标准库的分位数、交错轮次、bootstrap 比值区间)、`variants.py`(变体注册表)。
- Makefile 目标 `p2` ~ `p5`;`90_report.py` 各阶段结果表。

### Fixed
- 计时时关闭 TensorRT-LLM 的 KV cache 块复用:否则重复请求命中前缀缓存,纯文本 prefill 恒为 20 ms(与长度无关)。

## [0.2.0] - 2026-11-10

### Added
- P1 评测基座 `scripts/10_eval.py`:逐题 JSONL、断点续跑、95% 置信区间、配对差、官方分数验收线。
- TextVQA train 分片按图片切成 calib / dev(SHA-1 取模),与测试集互不重叠。
- 三后端统一答题接口 `vlmdeploy/backends.py`。
- 版本管理:`.gitattributes`(统一 LF)、本 CHANGELOG。

### Results
- HF BF16 原生分辨率:TextVQA val 79.00 ± 1.05(官方 79.7);RealWorldQA 61.31 ± 3.45(官方 62.9,超出 1.5 分线、
  但在置信区间内;新增"是否在 CI 内"判据并列报告)。

### Fixed
- HF 后端每 50 题归还 CUDA 缓存:原生分辨率下缓存占满 8 GB 后溢出到共享内存,每题 0.7 s → 4–7 s。

## [0.1.0] - 2026-11-09

### Added
- P0:两个隔离的 conda 环境(`trtllm` / `ovgenai`)安装脚本与版本记录。
- 模型与数据的多连接下载 + 逐文件 sha256 校验。
- `vlmdeploy` 公共库:配置单一事实来源、Qwen2-VL smart_resize 预缩放、TextVQA / RealWorldQA 打分。
- TensorRT-LLM 1.3.0rc18 跑 Qwen2-VL 的三处兼容修补(`vlmdeploy/patches/site/sitecustomize.py`)。
- 三方冒烟:HF / TensorRT-LLM / OpenVINO GenAI 在 8 题上答案逐字一致。
