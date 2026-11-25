#!/usr/bin/env bash
# 建 conda 环境 ovgenai:OpenVINO GenAI(CPU 推理栈)+ NNCF + optimum-intel(把 HF 模型导出成 OpenVINO IR)。
#
# 为什么不复用项目 A 的 `ov`:导出 Qwen2-VL 需要 torch + transformers + optimum-intel,装进 `ov` 会改动它
# 锁定的 onnx / numpy 版本,A 的结果就无法原样复现。环境隔离的代价只是多 ~3 GB。
#
# 版本:
#   - openvino / openvino-tokenizers / openvino-genai 三者必须是同一版本号:GenAI 和 tokenizers 的 C++ 库
#     按 OpenVINO 的具体版本编译,错一位就加载失败。统一锁 2026.4.1,与项目 A 的 openvino 一致。
#   - torch 用 CPU 版:导出只在 CPU 上跑,GPU 版会多下 3 GB 用不上的 CUDA 库。
#   - nncf 与项目 A 同为 3.4.0。
#   - transformers 锁 5.0.0:optimum-intel 2.2.0 的总依赖写的是 transformers<5.6,pip 会挑 5.5.4;
#     但它给 Qwen2-VL 的导出配置(optimum/exporters/openvino/model_configs.py 的 Qwen2VLOpenVINOConfig)
#     单独写了 MAX_TRANSFORMERS_VERSION = "5.0",导出时直接报 "Maximum required is 5.0, got: 5.5.4"。
#     5.0.0 是上限内最新的版本,配置结构(拆出 text_config)与 trtllm 环境的 5.5.4 一致;
#     optimum-intel 对 <5 和 >=5 都写了分支,4.57.x 也可以,但没必要退回 4.x。
#   - 装完的精确版本记录在 requirements-ovgenai.txt。
#
# 用法(WSL,仓库根目录):bash envs/setup_ovgenai.sh        可重复执行。
# 出错即停,用到未定义的变量报错,管道里任一环节失败都算失败
set -euo pipefail

# conda 环境名
ENV_NAME=ovgenai
# 仓库根目录
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# OpenVINO 的版本号,三个包共用
OV_VERSION=2026.4.1

# 打印带时间戳的步骤标题
step() { echo "=== [$(date +%H:%M:%S)] $*"; }

# 加载 conda 的 shell 函数,之后才能用 conda activate
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 环境不存在时创建,只用 conda-forge 频道
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  step "create env $ENV_NAME (python 3.12, conda-forge)"
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=3.12"
fi
# 激活环境
conda activate "$ENV_NAME"

# 没有装 torch 2.10.0 时安装 CPU 版
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.10.0') else 1)" 2>/dev/null; then
  step "install torch 2.10.0 (CPU build)"
  pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
fi

# 安装 OpenVINO 的三个包、NNCF、optimum-intel 和 transformers,版本固定
step "install OpenVINO $OV_VERSION stack + optimum-intel"
pip install --no-cache-dir "openvino==$OV_VERSION" "openvino-tokenizers==$OV_VERSION.0" "openvino-genai==$OV_VERSION.0" \
    "nncf==3.4.0" "optimum-intel==2.2.0" "transformers==5.0.0" pyarrow pillow

# 自检:打印各包的版本,确认 VLMPipeline 可用
step "verify"
python - <<'EOF'
import importlib.metadata as m
for pkg in ("openvino", "openvino-genai", "openvino-tokenizers", "nncf", "optimum-intel", "optimum", "transformers", "torch"):
    try:
        print(f"  {pkg:20s} {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"  {pkg:20s} -")
import openvino_genai
print("VLMPipeline available:", hasattr(openvino_genai, "VLMPipeline"))
EOF
# 把装好的精确版本记录到仓库
pip freeze > "$REPO/requirements-ovgenai.txt"
step "done"
