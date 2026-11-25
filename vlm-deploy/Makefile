# 流水线入口(在 WSL Ubuntu 中运行,不用 Docker)。
# 两个 conda 环境严格隔离:
#   trtllm  = TensorRT-LLM 1.3.0rc18 / ModelOpt 0.37 / torch 2.10 + CUDA 13(GPU 栈)
#   ovgenai = OpenVINO GenAI 2026.4.1 / NNCF / optimum-intel / torch CPU(CPU 栈)
# 下载步骤只用标准库,用系统 python3,不依赖任何 conda 环境。
# 大文件(模型、量化产物、IR、安装包)在 $(VLM_ARTIFACTS),默认 D 盘普通目录,见 vlmdeploy/config.py。
# 用 conda 的绝对路径:make 起的是非交互 shell,conda init 写进 ~/.bashrc 的初始化不会执行,PATH 里没有 conda。
CONDA ?= $(HOME)/miniconda3/bin/conda
# 两个环境各自的 python 命令前缀
GPU := $(CONDA) run --no-capture-output -n trtllm python
CPU := $(CONDA) run --no-capture-output -n ovgenai python
# 系统 python3,用于只依赖标准库的步骤
PY  := python3
# ruff 的可执行文件名,可在命令行覆盖
RUFF ?= ruff
# 以下三项 export 给所有配方的子进程:Hub 镜像、大文件目录、关闭 Python 的输出缓冲(日志实时可见)。
export HF_ENDPOINT ?= https://hf-mirror.com
export VLM_ARTIFACTS ?= /mnt/d/vlm-artifacts
export PYTHONUNBUFFERED := 1
# TRT-LLM 的 Qwen2-VL 视觉塔被写死用 flash-attn 2,而 pip 不会装它;这个目录里的 sitecustomize.py
# 在每个 Python 进程(含 TRT-LLM 拉起的 worker)启动时把它改成 PyTorch SDPA,不改安装目录里的文件。
# 这个前缀只加在 GPU 命令前面,CPU 环境的命令不带它。
TRTLLM_ENV := PYTHONPATH=$(CURDIR)/vlmdeploy/patches/site

# 冒烟测试的题数和图像 token 档位
SMOKE_N ?= 8
SMOKE_BUDGET ?= t256

# 这些目标都不对应文件:声明成 .PHONY 后,不管有没有同名的文件或目录都照常执行。
.PHONY: env-trtllm env-ovgenai model data test lint docs-check check smoke-hf smoke-trtllm ov-export smoke-ov smoke report eval p1 \
        p2 p3 p3-quantize p3-eval p3-bench p4 p4-compress p4-eval p4-bench p5 demo demo-gpu demo-cpu \
        genai-headers cpp-cpu parity-cpu

# 建 GPU 栈的 conda 环境,安装日志写到 logs/
env-trtllm:            # 约 15 GB,首次 20~40 分钟;可重复执行
	mkdir -p logs && bash envs/setup_trtllm.sh 2>&1 | tee logs/env_trtllm.log
# 建 CPU 栈的 conda 环境,安装日志写到 logs/
env-ovgenai:           # 约 3 GB
	mkdir -p logs && bash envs/setup_ovgenai.sh 2>&1 | tee logs/env_ovgenai.log

# 下载模型
model:                 # Qwen2-VL-2B-Instruct,4.43 GB,逐文件校验
	$(PY) scripts/00_download_model.py
# 下载数据集
data:                  # TextVQA val + train 分片 + RealWorldQA,约 1.9 GB,逐文件校验
	$(PY) scripts/01_download_data.py

# 单元测试,在 trtllm 环境里运行
test:
	$(GPU) -m unittest discover -s tests -v
lint:                  # 静态检查(规则在 pyproject.toml);ruff 不在两个推理环境里,pip install ruff 后用
	$(RUFF) check .
docs-check:            # 核对 docs/code-walkthrough.md 里的行号与源码一致(只依赖标准库);改了代码后跑
	$(PY) tools/check_docs.py
check: lint docs-check test   # 提交前的全套检查(仓库没有持续集成,这一条就是它的本地替身)

# ---- P0 冒烟:同一批题,HF 参考 / TRT-LLM / OpenVINO GenAI 三方答案对照 ----
smoke-hf:              # HF transformers BF16 参考答案(GPU 独占)
	$(TRTLLM_ENV) $(GPU) scripts/02_smoke_gpu.py --backend hf --num $(SMOKE_N) --budget $(SMOKE_BUDGET)
smoke-trtllm:          # TensorRT-LLM BF16(GPU 独占),与 HF 答案逐题比对
	$(TRTLLM_ENV) $(GPU) scripts/02_smoke_gpu.py --backend trtllm --num $(SMOKE_N) --budget $(SMOKE_BUDGET)
ov-export:             # optimum-intel 导出 FP16 基础 IR(P4 的各种压缩都从它出发);/usr/bin/time 记录峰值内存
	/usr/bin/time -v -o results/ov_export_time.txt $(CPU) scripts/03_smoke_cpu.py --export-only
# 打印记录里的峰值内存和耗时
	grep -E "Maximum resident|Elapsed" results/ov_export_time.txt
smoke-ov:              # OpenVINO GenAI(CPU 4 线程),与 HF 答案逐题比对;同样记录峰值内存(WSL 只有 8 GB)
	/usr/bin/time -v -o results/smoke_ov_time.txt $(CPU) scripts/03_smoke_cpu.py --num $(SMOKE_N) --budget $(SMOKE_BUDGET)
# 冒烟全流程
smoke: smoke-hf smoke-trtllm ov-export smoke-ov

report:                # 用 results/ 回填 README 的表格(只依赖标准库);make report ONLY=P5 只刷新一节
	$(PY) scripts/90_report.py $(if $(ONLY),--only $(ONLY))

# ---- 评测(scripts/10_eval.py,可断点续跑,结果在 results/eval/)----
# 一般形式:make eval VARIANT=trtllm-bf16 SPLIT=test BUDGET=t576 [LIMIT=200] [PAIRED=trtllm-bf16]
# 评测参数的默认值
VARIANT ?= hf-bf16
SPLIT ?= test
BUDGET ?= t576
# 用 = 是延迟展开:到用它的地方才代入各变量当时的值;LIMIT / PAIRED 没给时,对应的参数整个不出现。
EVAL_ARGS = --variant $(VARIANT) --split $(SPLIT) --budget $(BUDGET) $(if $(LIMIT),--limit $(LIMIT)) $(if $(PAIRED),--paired-vs $(PAIRED))
# 按变体名自动选环境:ov- 开头的用 CPU 环境,其余用 GPU 环境并带上补丁前缀。
# 标准输入接 /dev/null:评测进程会读走标准输入,经 `bash -s <<EOF` 连续提交多条 make eval 时,后面的命令会被它吃掉。
eval:
	$(if $(filter ov-%,$(VARIANT)),$(CPU),$(TRTLLM_ENV) $(GPU)) scripts/10_eval.py $(EVAL_ARGS) < /dev/null

# P1:HF BF16 原生分辨率复现官方分数(TextVQA val 79.7 / RealWorldQA 62.9,差 ≤ 1.5 才进入 P2)
p1:
	$(MAKE) eval VARIANT=hf-bf16 SPLIT=test BUDGET=native
	$(MAKE) eval VARIANT=hf-bf16 SPLIT=rwqa BUDGET=native

# ---- P2:GPU BF16 —— 精度 × 3 档 token、TTFT 分解、decode、显存 ----
# foreach 把各档位展开成同一行里用 && 串起来的几条命令,末尾的 true 收住最后一个 &&;
# 任何一条失败,后面的都不跑。P3–P5 的 foreach 是同一种写法。
# P2 的三个图像 token 档位
BUDGETS := t256 t576 t1024
p2:
# 三个档位的 TextVQA 评测
	$(foreach b,$(BUDGETS),$(MAKE) eval VARIANT=trtllm-bf16 SPLIT=test BUDGET=$(b) &&) true
# 三个档位的 RealWorldQA 评测
	$(foreach b,$(BUDGETS),$(MAKE) eval VARIANT=trtllm-bf16 SPLIT=rwqa BUDGET=$(b) &&) true
# 首字延迟分解和 decode 计时
	$(TRTLLM_ENV) $(GPU) scripts/11_bench_gpu.py --variant trtllm-bf16

# ---- P3:GPU 量化 —— FP8 / AWQ-INT4 × 纯文本 / 图文校准;dev 集做决策,test 集报数 ----
# 变体名 fp8-img 拆成 --method fp8 --calib img(用 make 的 word/subst,避免 shell 参数展开)
GPU_QUANT := fp8-txt fp8-img awq-txt awq-img
quant_args = --method $(word 1,$(subst -, ,$(1))) --calib $(word 2,$(subst -, ,$(1)))
# 逐个变体量化
p3-quantize:
	$(foreach q,$(GPU_QUANT),$(TRTLLM_ENV) $(GPU) scripts/12_quantize_gpu.py $(call quant_args,$(q)) &&) true
# 先评 BF16 基线的 dev 集,再评各量化变体的 dev / test / rwqa,都与 BF16 逐题配对
p3-eval:
	$(MAKE) eval VARIANT=trtllm-bf16 SPLIT=dev BUDGET=t576
	$(foreach q,$(GPU_QUANT),$(MAKE) eval VARIANT=trtllm-$(q) SPLIT=dev BUDGET=t576 PAIRED=trtllm-bf16 &&) true
	$(foreach q,$(GPU_QUANT),$(MAKE) eval VARIANT=trtllm-$(q) SPLIT=test BUDGET=t576 PAIRED=trtllm-bf16 &&) true
	$(foreach q,$(GPU_QUANT),$(MAKE) eval VARIANT=trtllm-$(q) SPLIT=rwqa BUDGET=t576 PAIRED=trtllm-bf16 &&) true
# 逐个量化变体计时
p3-bench:
	$(foreach q,$(GPU_QUANT),$(TRTLLM_ENV) $(GPU) scripts/11_bench_gpu.py --variant trtllm-$(q) &&) true
# P3 全流程
p3: p3-quantize p3-eval p3-bench

# ---- P4:CPU 栈 —— NNCF 压缩配方、精度(test 前 CPU_N 题,t256)、ABBA 计时 ----
# CPU 栈的压缩配方
CPU_RECIPES := int8 int4 int4awq int4-vis8w int4-vis8s
# CPU 上每题约 10 s;200 题的逐题配对差 95% CI 约 ±3 分,只能检出大的掉点(README 注明)
CPU_N ?= 200
p4-compress:           # 两步分进程:先收集校准输入(峰值约 7 GB),再逐配方压缩
	/usr/bin/time -v -o results/compress_collect_time.txt $(CPU) scripts/13_compress_cpu.py --collect
	$(foreach r,$(CPU_RECIPES),$(CPU) scripts/13_compress_cpu.py --recipe $(r) &&) true
# GPU 基线、CPU 基线和各配方评同一批题,逐题配对
p4-eval:
	$(MAKE) eval VARIANT=trtllm-bf16 SPLIT=test BUDGET=t256 LIMIT=$(CPU_N)
	$(MAKE) eval VARIANT=ov-fp16 SPLIT=test BUDGET=t256 LIMIT=$(CPU_N) PAIRED=trtllm-bf16
	$(foreach r,$(CPU_RECIPES),$(MAKE) eval VARIANT=ov-$(r) SPLIT=test BUDGET=t256 LIMIT=$(CPU_N) PAIRED=ov-fp16 &&) true
# 基线和全部配方一起计时
p4-bench:
	$(CPU) scripts/14_bench_cpu.py --variants ov-fp16 $(addprefix ov-,$(CPU_RECIPES))
# P4 全流程
p4: p4-compress p4-eval p4-bench

# ---- P5(条件触发:视觉塔占 TTFT > 40%,P2 实测 65–78%):视觉塔 TensorRT FP16 引擎 ----
# 1) 孤立视觉塔:建引擎 + 与 HF BF16 对拍 + GPU 时间;2) 引擎接回 TensorRT-LLM:方图档位端到端 TTFT
#    (每个档位、每个配置各起一个进程:8 GB 卡装不下三个引擎 + HF 模型,见 variants.make_answerer);
# 3) 精度:方图 s576 档 test 全量,与同档 BF16 逐题配对。
# 三个方图档位
SQUARES := s256 s576 s1024
p5:
# 逐档导出、构建引擎、对拍并计时
	$(foreach b,$(SQUARES),$(TRTLLM_ENV) $(GPU) scripts/15_vision_trt.py --budget $(b) &&) true
# 删掉旧的方图计时结果,下面各档的结果会并入同名文件
	rm -f results/bench/gpu_trtllm-bf16_square.json results/bench/gpu_trtllm-bf16-vistrt_square.json
# 每个档位、每个配置各起一个进程计时
	$(foreach b,$(SQUARES),$(foreach v,trtllm-bf16 trtllm-bf16-vistrt,$(TRTLLM_ENV) $(GPU) scripts/11_bench_gpu.py --variant $(v) --square --budgets $(b) &&)) true
# 方图 s576 档的精度,引擎与 BF16 逐题配对
	$(MAKE) eval VARIANT=trtllm-bf16 SPLIT=test BUDGET=s576
	$(MAKE) eval VARIANT=trtllm-bf16-vistrt SPLIT=test BUDGET=s576 PAIRED=trtllm-bf16

# ---- P6:演示 —— 流式视觉问答 + 单次首字延迟(dev 集前 4 张图;结果在 results/demo/,README 的演示表由它生成)----
# 自己的图片:<环境前缀> scripts/20_demo.py --variant ov-int4-vis8w --image a.jpg --question "..."(不落盘)
# 演示用的量化变体,可在命令行覆盖
DEMO_GPU ?= trtllm-fp8-img
DEMO_CPU ?= ov-int4-vis8w
demo-gpu:              # GPU 独占,每个配置各起一个进程:t576 基线 → 量化语言模型;方图 s576 基线 → 视觉塔引擎
	$(TRTLLM_ENV) $(GPU) scripts/20_demo.py --variant trtllm-bf16 --budget t576 < /dev/null
	$(TRTLLM_ENV) $(GPU) scripts/20_demo.py --variant $(DEMO_GPU) --budget t576 < /dev/null
	$(TRTLLM_ENV) $(GPU) scripts/20_demo.py --variant trtllm-bf16 --budget s576 < /dev/null
	$(TRTLLM_ENV) $(GPU) scripts/20_demo.py --variant trtllm-bf16-vistrt --budget s576 < /dev/null
demo-cpu:              # 基线 → 推荐配方;必须在 GPU 任务结束后跑(内存,见 P4)
	$(CPU) scripts/20_demo.py --variant ov-fp16 --budget t256 < /dev/null
	$(CPU) scripts/20_demo.py --variant $(DEMO_CPU) --budget t256 < /dev/null
# 两条栈的演示依次运行
demo: demo-gpu demo-cpu

# ---- CPU 栈的 C++ 程序(cpp/vlm_infer_ov.cpp):与 Python 逐题对拍 ----
# GPU 栈没有 C++ 程序:TensorRT-LLM 走 PyTorch 后端,模型和生成循环都在 Python 进程里
# ovgenai 环境的 site-packages 目录;延迟展开,只有用到它的目标才调用 conda
OV_SITE = $(shell $(CPU) -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
# GenAI 的库文件;文件名带版本号,按通配符查找
GENAI_LIB = $(firstword $(wildcard $(OV_SITE)/openvino_genai/libopenvino_genai.so*))
# 对拍用的变体
CPP_VARIANT ?= ov-int4-vis8w
genai-headers:         # 按已装库的提交号取 GenAI 的 C++ 头文件(pip 包不带头文件)
	bash envs/fetch_genai_headers.sh
# 配置并编译 C++ 程序,依赖路径由上面的变量给出
cpp-cpu: genai-headers
	cmake -S cpp -B cpp/build -DOpenVINO_DIR=$(OV_SITE)/openvino/cmake \
	  -DGENAI_INCLUDE_DIR=$(VLM_ARTIFACTS)/genai-include -DGENAI_LIBRARY=$(GENAI_LIB) \
	  -DGENAI_TOKENIZERS=$(OV_SITE)/openvino_tokenizers/lib/libopenvino_tokenizers.so
	cmake --build cpp/build -j4
parity-cpu:            # CPU 独占,需先 make cpp-cpu;逐题对拍并做配对计时
	$(CPU) scripts/16_cpp_parity_cpu.py --variant $(CPP_VARIANT) < /dev/null
