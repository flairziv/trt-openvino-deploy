# 一键流水线(在 WSL 中运行)。两个 conda 环境隔离:dl = PyTorch / TensorRT / modelopt,ov = OpenVINO / NNCF。
# 产物全部可重新生成,`make all` 从模型下载跑到报告;每一步也可单独执行。
# dl 环境的 Python 解释器前缀
DL := conda run --no-capture-output -n dl python
# ov 环境的 Python 解释器前缀
OV := conda run --no-capture-output -n ov python
# Hub 镜像地址,导出给所有子进程;外部已设置时不覆盖
export HF_ENDPOINT ?= https://hf-mirror.com
# 关闭 Python 的输出缓冲,日志实时可见
export PYTHONUNBUFFERED := 1

# 主分辨率档位
MAIN_RES := r518 r364
# TensorRT 的全部配方
TRT_RECIPES := fp32 tf32 fp16 int8 int8res int8mix fp8
# r364 档的全部 OpenVINO 配方
OV_FULL := fp32 int8 int8sqoff int8sq050 int8sq075 int8ac
# r518 档只跑的 OpenVINO 配方
OV_R518 := fp32 int8 int8ac
# C++ OpenVINO 程序对拍与计时用的变体
CPP_OV_VARIANTS ?= ov:r518:fp32 ov:r518:int8
# TensorRT 头文件目录,pip 包不带头文件,换位置时在命令行覆盖
TRT_INCLUDE_DIR ?= /mnt/e/code/VscodeProjects/FosuVision/learning/trt_quantize/TensorRT-10.16/include

# 下列目标都不对应文件,始终执行
.PHONY: all test data export reference stats quantize sensitivity engines determinism anomaly \
        eval-gpu bench-gpu profile cpp parity ov eval-cpu bench-cpu cpp-ov parity-ov report

# 全流程:先 GPU 栈,再 CPU 栈,最后生成报告
all: data export reference stats quantize sensitivity engines eval-gpu bench-gpu profile cpp parity ov eval-cpu bench-cpu cpp-ov parity-ov report

# 单元测试
test:
	$(DL) -m unittest discover -s tests -v

# 下载模型并准备数据集
data:          # 需先按 README 下载 data/raw/
	$(DL) scripts/00_download_model.py
	$(DL) scripts/01_prepare_data.py
# 导出固定形状的 ONNX
export:
	$(DL) scripts/02_export_onnx.py
# 生成 PyTorch FP32 基准与参考输出
reference:
	$(DL) scripts/03_reference.py
# 统计 r518 档的激活分布
stats:
	$(DL) scripts/04_activation_stats.py --res r518
# 生成派生 ONNX:主档位出 INT8 和 FP8,三个档位都出 FP16
quantize:
	for r in $(MAIN_RES); do for m in int8 fp8; do $(DL) scripts/05_quantize_modelopt.py --res $$r --mode $$m; done; done
	for r in r518 r364 r266; do $(DL) scripts/05_quantize_modelopt.py --res $$r --mode fp16; done
# 敏感度分析,逐个主档位执行
sensitivity:   # GPU 独占
	for r in $(MAIN_RES); do $(DL) scripts/06_sensitivity.py --res $$r; done
# 构建默认集合的全部引擎
engines:       # GPU 独占
	$(DL) scripts/07_build_trt.py
# 可复现性实验:弱类型 FP16 重复构建 3 次
determinism:
	$(DL) scripts/07_build_trt.py --determinism r364 --repeat 3
# 复现 fc2 异常
anomaly:
	$(DL) tools/probe_fc2_anomaly.py
# GPU 栈测试集评测:主档位的全部配方,再加 r266 的 FP16
eval-gpu:
	for r in $(MAIN_RES); do for p in $(TRT_RECIPES); do $(DL) scripts/09_evaluate.py --variants trt:$$r:$$p; done; done
	$(DL) scripts/09_evaluate.py --variants trt:r266:fp16
# GPU 栈延迟基准
bench-gpu:     # GPU 独占;按分辨率分批,避免一次加载过多引擎撑满 8 GB 显存
	for r in $(MAIN_RES); do $(DL) scripts/10_benchmark.py --variants $$(for p in $(TRT_RECIPES); do echo trt:$$r:$$p; done); done
	$(DL) scripts/10_benchmark.py --variants trt:r266:fp16
# 逐 kernel 耗时剖析
profile:
	$(DL) scripts/11_profile_trt.py --variants trt:r518:fp16 trt:r518:int8 trt:r518:int8res trt:r518:fp8
# 构建 TensorRT 版 C++ 程序,libnvinfer 的路径取自 dl 环境的 pip 包
cpp:
	cmake -S cpp -B cpp/build -DTRT_INCLUDE_DIR=$(TRT_INCLUDE_DIR) -DCUDAToolkit_ROOT=/usr/local/cuda \
	  -DCUDA_CUDART=/usr/local/cuda/lib64/libcudart.so \
	  -DTRT_NVINFER=$$($(DL) -c "import tensorrt_libs,os;print(os.path.join(os.path.dirname(tensorrt_libs.__file__),'libnvinfer.so.10'))")
	cmake --build cpp/build -j4
# TensorRT 版 C++ 程序的对拍与基准
parity:
	$(DL) scripts/13_cpp_parity.py --variant trt:r518:fp16
# 构建 OpenVINO 版 C++ 程序:单独的构建目录,依赖取自 ov 环境的 pip 包
cpp-ov:
	cmake -S cpp -B cpp/build-ov \
	  -DOpenVINO_DIR=$$($(OV) -c "import openvino,os;print(os.path.join(os.path.dirname(openvino.__file__),'cmake'))")
	cmake --build cpp/build-ov -j4
# OpenVINO 版 C++ 程序的对拍与基准
parity-ov:     # CPU 独占(对拍之后紧接着计时)
	for v in $(CPP_OV_VARIANTS); do $(OV) scripts/13_cpp_parity.py --variant $$v; done
# 生成 OpenVINO IR:r364 跑全部配方,r518 只跑结论配方
ov:            # CPU 独占;SmoothQuant 扫描只在 r364 上做,r518 只跑结论配方
	$(OV) scripts/08_quantize_openvino.py --res r364 --recipes $(OV_FULL)
	$(OV) scripts/08_quantize_openvino.py --res r518 --recipes $(OV_R518)
# CPU 栈测试集评测
eval-cpu:
	for p in $(OV_FULL); do $(OV) scripts/09_evaluate.py --variants ov:r364:$$p; done
	for p in $(OV_R518); do $(OV) scripts/09_evaluate.py --variants ov:r518:$$p; done
# CPU 栈延迟基准
bench-cpu:     # CPU 独占;CPU 单帧耗时大,迭代数减少
	$(OV) scripts/10_benchmark.py --iters 15 --warmup 3 --variants $$(for p in $(OV_FULL); do echo ov:r364:$$p; done)
	$(OV) scripts/10_benchmark.py --iters 15 --warmup 3 --variants $$(for p in $(OV_R518); do echo ov:r518:$$p; done)
# 用 results 回填 README 的表格
report:
	$(DL) scripts/12_report.py
