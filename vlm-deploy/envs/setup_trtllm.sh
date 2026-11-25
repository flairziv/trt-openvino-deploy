#!/usr/bin/env bash
# 建 conda 环境 trtllm:TensorRT-LLM 1.3.0rc18(PyTorch 后端)+ ModelOpt,WSL Ubuntu 原生安装,不用 Docker。
#
# 版本为什么这样锁(2026-11-09 用 HTTP Range 直接读 wheel 里的 METADATA 得到,未下载整包):
#   tensorrt_llm 1.3.0rc18 要求 torch>=2.10.0,<=2.11.0a0、tensorrt~=10.15.1、nvidia-modelopt~=0.37.0、
#   transformers==5.5.4,以及 CUDA 13 运行库(cuda-python>=13、nvidia-nccl-cu13)。
#   - torch 必须先于 TRT-LLM、从 PyTorch 官方 cu130 源装。若交给 pip 按依赖自己挑,会从默认源拿到另一个
#     CUDA 变体,和 TRT-LLM 的 cu13 库混装。驱动 581.80 对应 CUDA 13.0,与 cu130 正好咬合,不装系统 CUDA 13。
#   - 1.3 后续 RC 要求 torch 2.14 预览版,原生装不上;1.3.0rc18 也是我们读过源码、确认支持 Qwen2-VL 的版本。
#     装不上时的退路是稳定版 1.2.1(torch 2.9.1 + transformers 4.57.3)。
#   - MPI:import tensorrt_llm 时就需要 mpi4py + 一个 MPI 实现。本机没有免密 sudo,用 conda-forge 的
#     openmpi,不走 apt。锁 4.1 系列:与 Ubuntu 24.04 的 libopenmpi-dev(TRT-LLM 官方安装说明用的)同代。
#
# 磁盘:环境约 15 GB,落在 E:\WSL\Ubuntu 的虚拟盘里,而虚拟盘不会自动缩小 —— 所以 pip 一律 --no-cache-dir;
#       2.7 GB 的 wheel 先下到 $VLM_ARTIFACTS/wheels(D 盘普通目录,删掉即释放),冒烟测试通过后再删。
#
# 用法(WSL,仓库根目录):bash envs/setup_trtllm.sh
# 可重复执行:每一步先检查是否已完成,已完成就跳过。
# -e 任何一步出错立即停;-u 用到未定义的变量就报错;pipefail 管道里任一环节失败都算失败。
set -euo pipefail

# conda 环境名
ENV_NAME=trtllm
# 仓库根目录
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# 大文件目录,可用环境变量 VLM_ARTIFACTS 覆盖
ARTIFACTS="${VLM_ARTIFACTS:-/mnt/d/vlm-artifacts}"

# TensorRT-LLM 的版本号、wheel 文件名和 sha256
TRTLLM_VERSION=1.3.0rc18
TRTLLM_WHEEL="tensorrt_llm-${TRTLLM_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
TRTLLM_SHA256=aa79ed599be5b992a868e672f590202e89ffd70df66391c3859a10d84f2b669a   # pypi.nvidia.com 索引页公布值
# torch 的版本和下载源
TORCH_PKGS=(torch==2.10.0 torchvision==0.25.0)
TORCH_INDEX=https://download.pytorch.org/whl/cu130
# NVIDIA 的 pip 源
NVIDIA_INDEX=https://pypi.nvidia.com

# 打印带时间戳的步骤标题
step() { echo "=== [$(date +%H:%M:%S)] $*"; }

# 加载 conda 的 shell 函数,之后才能用 conda activate
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 1) 环境本体 + MPI。整个环境只用 conda-forge 一个频道,避免 defaults 与 conda-forge 的 C 运行库混装。
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  step "create env $ENV_NAME: python 3.12 + openmpi 4.1 + mpi4py (conda-forge)"
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=3.12" "openmpi=4.1" "mpi4py"
fi
# 激活环境
conda activate "$ENV_NAME"

# 2) torch 2.10.0 + cu130(见文件头)。检查条件同时看版本号和 CUDA 大版本,防止装成别的变体还以为对了。
if ! python - <<'EOF' 2>/dev/null
import sys, torch
sys.exit(0 if torch.__version__.startswith("2.10.0") and (torch.version.cuda or "").startswith("13") else 1)
EOF
then
  step "install ${TORCH_PKGS[*]} from $TORCH_INDEX"
  pip install --no-cache-dir "${TORCH_PKGS[@]}" --index-url "$TORCH_INDEX"
fi

# 3) TensorRT-LLM:先用多连接下载器预取 wheel 并校验 sha256,再本地安装。
#    依赖从默认源(清华镜像)+ NVIDIA 源解析;torch 已满足版本约束,pip 不会再动它。
if ! python -c "import importlib.metadata as m, sys; sys.exit(0 if m.version('tensorrt_llm') == '$TRTLLM_VERSION' else 1)" 2>/dev/null; then
  # 创建 wheel 的保存目录
  mkdir -p "$ARTIFACTS/wheels"
  step "fetch $TRTLLM_WHEEL -> $ARTIFACTS/wheels"
  # 多连接下载 wheel 并校验 sha256
  python "$REPO/tools/parallel_fetch.py" "$NVIDIA_INDEX/tensorrt-llm/$TRTLLM_WHEEL" \
      "$ARTIFACTS/wheels/$TRTLLM_WHEEL" --sha256 "$TRTLLM_SHA256" --workers 8
  step "pip install $TRTLLM_WHEEL"
  # 从本地 wheel 安装,依赖另从 NVIDIA 源解析
  pip install --no-cache-dir "$ARTIFACTS/wheels/$TRTLLM_WHEEL" --extra-index-url "$NVIDIA_INDEX"
fi

# 4) 自检:版本咬合是否如预期(torch 没被换掉、TRT 是 10.15、CUDA 13),并真正 import 一次 TRT-LLM ——
#    MPI 或 CUDA 运行库缺失只会在 import 时暴露。
step "verify"
python - <<'EOF'
import importlib.metadata as m
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
for pkg in ("tensorrt_llm", "tensorrt", "nvidia-modelopt", "transformers", "flashinfer-python",
            "flash-attn-4", "mpi4py", "triton"):
    try:
        print(f"  {pkg:18s} {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"  {pkg:18s} -")
import tensorrt_llm
print("import tensorrt_llm OK:", tensorrt_llm.__version__)
EOF
# 把装好的精确版本记录下来(pyproject.toml 的可选依赖只列了主要的几个包)。
pip freeze > "$REPO/requirements-trtllm.txt"
step "done (wheel kept in $ARTIFACTS/wheels until the GPU smoke test passes)"
