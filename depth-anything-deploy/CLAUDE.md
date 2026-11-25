# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Depth Anything V2-Small 双栈部署研究:GPU 栈 TensorRT 10.16(RTX 4060 Laptop)、CPU 栈 OpenVINO + NNCF(i9-14900HX / WSL2 4 vCPU)。两栈共用同一套 NYUv2 评测和同一个接受标准(`ACCEPT_ABSREL_DELTA = 0.001`),结论并排可比。代码、注释、文档均为中文,新增内容保持一致。

## 运行方式

全部在 **WSL** 中运行,入口是仓库根目录的 `Makefile`。两个 conda 环境严格隔离(不能合并:modelopt 与 NNCF 锁定的 onnx 版本冲突):

- `dl` —— torch / transformers / modelopt / tensorrt:TRT 栈脚本(00–07、09–13)
- `ov` —— openvino / nncf:OpenVINO 栈脚本(08,以及跑 `ov:*` 变体的 09/10)

Makefile 中 `DL` / `OV` 即 `conda run --no-capture-output -n dl python` / `-n ov python`。单独运行脚本用同样前缀,从仓库根目录执行 `python scripts/NN_xxx.py`。

shell 在 Windows 侧(Git Bash)时,命令要送进 WSL 再跑。Windows 侧没有 make;那里的 conda 没有 `dl`,却有一个同名但无关的旧 `ov` 环境(openvino 2023.3,没装 h5py),直接 `conda run -n ov` 会落进它:看到"缺 h5py"就是跑错了环境,不要去装包。WSL 的非交互 shell 里 conda 不在 PATH,要先 source:

```bash
wsl.exe -e bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && cd /mnt/e/code/VscodeProjects/vit-vlm-deploy/depth-anything-deploy && make test'
```

常用命令:

```bash
make all        # 全流程(需先按 README"快速开始"一次性下载数据,约 1.5 GB)
make test       # 单元测试(不需要 GPU / 模型 / 数据集)
make report     # 用 results/*.json 回填 README 表格
```

单步 target(标"独占"的不能与其他 GPU / CPU 负载并发):

- `make data export reference stats quantize` —— 数据、ONNX 导出、FP32 参考、激活统计、modelopt 量化(`dl`)
- `make sensitivity engines bench-gpu` —— **GPU 独占**(tactic 计时与延迟测量会被其他 GPU 负载污染);`bench-gpu` 按分辨率分批,避免一次加载全部引擎撑满 8 GB 显存
- `make ov bench-cpu` —— **CPU 独占**(WSL 只有 4 vCPU)
- `make eval-gpu eval-cpu` —— 测试集精度评测,不要求独占
- `make cpp parity` —— GPU 栈的 C++ runtime(`da_infer`,构建目录 `cpp/build`);`cpp` 需要 TensorRT OSS 头文件(pip 轮子不含头文件),路径由 Makefile 变量 `TRT_INCLUDE_DIR` 给出,默认值指向本机另一个项目的源码树,换位置时 `make cpp TRT_INCLUDE_DIR=...`;`parity` 依赖先构建,对拍分三层(预处理 / 同输入推理 / 端到端 AbsRel),只对 `trt:r518:fp16` 做,**GPU 独占**
- `make cpp-ov parity-ov` —— CPU 栈的 C++ runtime(`da_infer_ov`,构建目录 `cpp/build-ov`);OpenVINO 的头文件、库、CMake 配置都取自 `ov` 环境的 pip 包,不需要 TensorRT / CUDA。`parity-ov` 在 `ov` 环境里跑同一个 13 号脚本,变体由 Makefile 的 `CPP_OV_VARIANTS` 给出(默认 `ov:r518:fp32 ov:r518:int8`),**CPU 独占**。两个构建目录各只生成一个程序:`cpp/CMakeLists.txt` 找不到哪边的依赖就跳过哪个目标
- `make determinism anomaly profile` —— 复现构建不确定性 / fc2 异常 / 逐 kernel 剖析,同样要 **GPU 独占**(都在构建引擎或逐 kernel 计时)。`determinism` 和 `anomaly` 不在 `make all` 里,README 坑 2 的数字和 ANOMALY 表靠单独跑它们

单元测试是 `unittest`(不是 pytest),两个环境都能跑;`make test` 只跑 `dl`,动了共享模块要在 `ov` 里再跑一遍:

```bash
conda run --no-capture-output -n ov python -m unittest discover -s tests -v
```

跑单个测试(不要用 `-m unittest tests.test_core.xxx` 这种点分路径:`tests/` 没有 `__init__.py`,`dl` 里 ultralytics 自带的顶层 `tests` 包会遮住它,报 `No module named 'tests.test_core'`):

```bash
conda run --no-capture-output -n dl python tests/test_core.py VariantTest.test_round_trip -v
```

冒烟式跑单个脚本:只有 `09_evaluate.py` 有 `--limit`(结果不落盘);其余脚本没有只读模式,跑出结果就写进 `results/`,见下文"结果流向"。

```bash
conda run --no-capture-output -n dl python scripts/09_evaluate.py --variants trt:r518:fp16 --limit 20
```

没有 `make lint`;ruff 配置在 `pyproject.toml`(line-length 120、py311、中文 docstring 允许不以句号结尾),ruff 不在 requirements 里,需要时自行安装后 `ruff check .`。所有文件 LF 行尾(`.editorconfig`),从 Windows 侧写文件别带进 CRLF。

注释约定:源码已逐行加注释,每条语句或紧密相关的两三行上方一句,说明这一步做什么;不举例、不解释语法、不用装饰符号和 markdown 标记,标点用半角。文件头 docstring 和记录"为什么这么做"的原注释保留。新增或修改代码时按同样的写法补注释;只动注释时不要顺手改代码。

## 架构要点

### 变体契约

一切用变体字符串 `<backend>:<res>:<recipe>` 指代(如 `trt:r518:int8res`、`ov:r364:int8sq050`;后端取 `torch` / `ort` / `trt` / `ov`),`dadeploy/variants.py` 是解析与产物路径的唯一入口。07 / 09 / 10 / 11 的 `--variants` 吃这个格式(13 是单数 `--variant`);02–06、08 按 `--res` 工作(05 另有 `--mode`,08 另有 `--recipes`)。`Variant.parse` 不校验 trt 配方名:拼错了要到查 `TRT_RECIPES`(KeyError)或加载引擎(FileNotFoundError,提示"先运行 07")时才暴露。分辨率:r518 = 518×686(官方档,1813 token)、r364 = 364×476(884 token)、r266 只用于分辨率曲线。batch = 1、静态形状,没有动态 profile。

### 双环境与懒加载规则

`dadeploy/` 是两个环境共享的库,共享模块只允许依赖 numpy / cv2 / h5py / scipy;torch / tensorrt / onnxruntime / openvino / transformers 一律在 Runner 构造函数或函数内部懒加载。例外:`dadeploy/model.py`(torch)和 `dadeploy/trt_utils.py`(tensorrt)在模块顶层导入,只能在 `dl` 中用;`dadeploy/groups.py` 顶层导入 onnx(两个环境都装了,但只有 `dl` 的脚本用它);`dadeploy/trt_recipes.py` 是纯配置,两边都能读。在共享模块里加顶层后端 import 会破坏 `ov` 环境,而单测不导入 `runners` / `evaluation` / `data` / `reference`,改完用这条确认:

```bash
conda run --no-capture-output -n ov python -c "import dadeploy.runners, dadeploy.evaluation, dadeploy.data, dadeploy.reference, dadeploy.groups"
```

### config.py 是唯一事实来源

路径、分辨率、裁剪、深度范围、接受标准、计时轮数全部在 `dadeploy/config.py`,脚本里不允许出现魔数。**改动任一协议参数会使 `results/` 失效,必须整套重跑**,并在 README"评测协议"一节同步说明。

变体矩阵是例外,它在几处各有一份,新增配方或分辨率要全部同步:

- TRT 配方:`trt_recipes.TRT_RECIPES` → 产出对应 ONNX 的脚本(05 的 `QUANT_MODES` 或 06 的 `export_graphs`)→ 07 的 `MAIN_RECIPES` → Makefile 的 `TRT_RECIPES` → 12 的 `PRECISION_ORDER`;
- OV 配方:`variants.OV_RECIPES` → 08 的 `SQ_ALPHA` / `build_recipe` → Makefile 的 `OV_FULL` / `OV_R518` → 12 的 `PRECISION_ORDER`;
- 分辨率:`config.RESOLUTIONS`(高宽必须是 14 的整数倍)→ Makefile 的 `MAIN_RES` 和 `quantize` 里手写的档位 → 12 的 `RES_ORDER` / `RES_LABEL`(没登记的档位直接抛错),SENSITIVITY / ACTIVATION 两节的档位也写死在 12 里。

`12_report.py` 只用标准库、不 import `dadeploy`(任一环境都能跑),所以这些表是它自己的副本;漏登记的配方只是排到最后。README 里 GPU / CPU 结果表下面各有一张手写的配方说明表,也要补。

### 数据纪律(三个划分互不相交)

- `calib` 200 张:只用于激活范围校准;
- `val` 100 张:一切**决策**只看它(敏感度排序、回退选择、NNCF accuracy control);
- `test` 654 张:只用于最终报告,绝不在测试集上挑配置;
- `drift` 假划分 = test 每 6 张取 1(109 张):与落盘的 FP32 原始输出比 SQNR。

划分只由 `data/nyu/manifest.json` 定义(`calib` 与 `val` 的 h5 同在 `data/nyu/calib/` 目录下),一律通过 `dadeploy.data.split_names` / `iter_split` 取,不要按目录扫文件。

### 流水线 = scripts/00–13

编号大体是依赖顺序,每一步的输入是前几步的产物(ONNX → 量化图 → 引擎 / IR → 评测 `results/*.json` → README 表格);`make all` 先跑完 GPU 栈,再跑 CPU 栈(08 和 `ov:*` 的 09 / 10),12 最后。每个脚本开头的 docstring 写了"为什么这么做"和依赖前提,改脚本前先读。几处不显然的依赖:

- CPU 栈不自足:08 读 02 的 ONNX;`ov` 里没有 torch,09 用的 FP32 参考输出和 torch 基准都是 03 在 `dl` 里落盘的;
- `int8res` / `int8mix` 两张 ONNX 是 06 的产物,05 只出 `fp16` / `int8` / `fp8`。`06_sensitivity.py --export-only` 按已有的 `results/sensitivity_<res>.json` 重新导出这两张图,不重跑实验;
- 07 遇到缺失的 ONNX 只告警跳过,不会让整批失败(告警一律写"先运行 05",缺的是 int8res / int8mix 时该跑的是 06);
- 重跑不等于刷新:07 只在 ONNX 比引擎新时重建,08 只要 `.xml` 已存在就跳过,都要 `--force` 才强制,而 `make engines` / `make ov` 不带它。

### 统一 Runner 契约与评测主循环

四种后端(PyTorch / ORT / TRT / OpenVINO)实现同一个接口:`infer(x)` 收 `[1,3,H,W]` float32 预处理输出、返回 `[H,W]` float32 相对视差;`infer_timed` 另给分段耗时。所有变体(含 PyTorch 基准)的测试集评测都走 `dadeploy.evaluation.evaluate` 这一个主循环,新增后端只需实现该接口。06 / 08 在 val 上的决策循环和 13 的对拍是各自手写的,但复用同一套 `preprocess` / `postprocess` / `evaluate_prediction`,改评测口径时这几处要一起看。指标在**视差域**做尺度-平移最小二乘对齐后才算 AbsRel / δ1 / RMSE——跳过对齐指标没有意义。延迟数字一律来自 `10_benchmark.py` 的交错轮次(5 轮乱序、报 P50/P90、轮间极差 >10% 标 `~`;GPU 用 config 默认的预热 20 + 计时 100 次,CPU 的 `--warmup 3 --iters 15` 写在 Makefile 的 `bench-cpu` 里),C++ 表例外,来自 13 调用的 `da_infer --bench` / `da_infer_ov --bench`(单进程连续计时,没有交错轮次;CPU 两行的推理耗时与 CPU 结果表不在同一次运行里测,差几个百分点属于机器漂移);`06_sensitivity.py` 里的快速延迟只用于流程内部决策,不进任何表。

### 结果流向:results/*.json → README(单向)

README 表格由 `scripts/12_report.py` 从 `results/*.json` 生成,填在 `<!-- RESULTS:xxx:BEGIN/END -->` 标记之间。**不要手工编辑标记之间的内容**;改了协议或重跑后执行 `make report` 刷新。标记之外的正文(结论速览、关键结论展开、各个坑)引用的数字是手写的,`make report` 不会更新,重跑后要人工对一遍。

写 `results/` 时要知道:

- 表里有哪些行由 `results/` 里有什么决定(GPU / CPU 矩阵的行 = `results/eval/*__test.json` 的文件集合);
- `engines` / `bench_*` / `quantize` / `openvino` / `export` / `profile_trt` / `cpp_*` 这几个 json 是"读旧文件 → 按键更新 → 写回",删掉或改名的变体会留下旧条目并继续进表,要手工清;
- 除了 `09 --limit`,没有不落盘的模式:手工跑一次 10 / 11 / 13,对应变体已经进表的数字就被换掉。10 试跑时给 `--out` 一个 `results/` 之外的绝对路径。

### TRT 配方:强类型 vs 弱类型

配方表在 `dadeploy/trt_recipes.py`。除 fp32 / tf32 外,所有参与对比的引擎都以**强类型**构建:精度完全由 ONNX 张量类型(FP16 initializer / Cast / Q-DQ)决定,构建可复现。弱类型的 `BuilderFlag.FP16` 只是"允许"FP16,由 builder 按计时选精度——同一份 ONNX 连续构建 3 次有 1 次退回 TF32 且无融合 MHA(README 坑 2),对照实验不可接受。强类型网络不能再设精度 flag,builder 会直接报错。engine 与 timing cache 绑定 GPU 型号 + TRT 版本,换机器必须重建。`fp16w` 配方只为复现上述不确定性,不是对比项。

06 和 07 共用 `artifacts/engines/timing.cache`,`determinism` 和 fc2 探针故意不用它。每个引擎旁边有一份 IEngineInspector 导出的 `.layers.json`,11 靠它给 kernel 归类;"引擎里实际跑的精度"靠 `trt_utils._PRECISION_PATTERNS` 匹配 tactic 名里的精度标记,升级 TensorRT 后要复核。

### INT8 路线:共用校准 + 只拆 Q/DQ

int8 / int8res / int8mix 三张图共用 `05_quantize_modelopt.py` 的同一份 INT8 校准,实验只删 Q/DQ、从不重新校准,差异只来自"哪些位置量化"。`dadeploy/groups.py` 用正则把 Q/DQ 节点名映射到"第几个 block 的哪个子层"(`patch_embed`、`b00`–`b11` 的 `qkv` / `o_proj` / `fc1` / `fc2` / `residual`、`neck.*`、`head`),正则匹配 02 导出器生成的 `/model/backbone/...` 模块路径节点名(02 坚持用 `dynamo=False` 的 TorchScript 导出器就是为了这些名字);更换导出方式必须同步更新,否则 `find_qdq_pairs` 直接抛错。`bypass_pairs` 把消费者改接到 Q 的输入并删除 Q/DQ。

### 决策只信真实引擎

ORT 对 QDQ 图的假量化与真实 TRT 输出偏差差好几个 dB、逐图排序基本不相关(`OrtRunner` 因此拒绝静默回退到 CPU EP);另有 fc2 异常(仅拆 fc2 时,INT8 fc1 + FP16 fc2 在 TRT 上 SQNR 跌到约 3.5 dB,ORT 正常,根因未证实)。所以敏感度排序与回退选择全部在真实引擎上做:一次实验 = 拆 Q/DQ → 强类型构建(共享 timing cache)→ val 100 张评测。

### 会咬人的既有约束(改代码前先读 README"踩过的坑")

- 导出不得换成 onnxsim:它把 3D MatMul+Add 改写成 Reshape→Gemm→Reshape,破坏 TRT 的融合 MHA 与 modelopt 的模式匹配;位置编码插值用 onnx-graphsurgeon 迭代常量折叠到不动点。
- `ov.save_model` 必须显式 `compress_to_fp16=False`,否则"FP32 基准"会带入 FP16 权重舍入误差。
- modelopt AutoCast 的参考推理在 8 GB WSL 上会 OOM,默认关闭。
- ORT 找不到 pip 装的 cuDNN 时会静默退回 CPU EP:建 CUDA 会话之前(包括 modelopt 校准内部建的会话)必须先调 `runners.preload_ort_cuda()`。
- C++ 预处理(`cpp/da_common.h`,`da_infer` 与 `da_infer_ov` 共用)与 `dadeploy/preprocess.py` 逐步对应,归一化常量在 C++ 里另有一份;改任何一边都要 `make cpp parity` 和 `make cpp-ov parity-ov` 重新对拍。`cpp/ov_engine.h` 的三项编译配置同样要与 `runners.OvRunner` 保持一致,否则"同输入输出逐位相同"这一层会失效。
- 下载走 hf-mirror:Makefile 已 `export HF_ENDPOINT`,手工跑 00 要自己带上;这台机器的 WSL 访问 GitHub 不稳定。
- 所有数字只代表这台笔记本(GPU 动态频率、WSL 无法绑定物理核),方法可迁移到端侧平台,数字不可引用。
