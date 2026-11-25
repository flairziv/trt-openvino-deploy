#!/usr/bin/env bash
# 下载 OpenVINO GenAI 的 C++ 头文件,存放到 $VLM_ARTIFACTS/genai-include。
# cpp/vlm_infer_ov.cpp 编译时需要这些头文件。
#
# pip 安装的 openvino-genai 包只有库 libopenvino_genai.so,没有 C++ 头文件,需要另外下载。
#
# 头文件必须与已装的库来自同一个提交:类的内存布局要与库里编译的一致,
# 版本不一致时不一定报错,可能读错内存。openvino_genai.__version__ 的最后一段是构建它的提交号,
# 脚本先核对这一段,再按完整提交号下载。
# 升级 openvino-genai 后,把 GENAI_COMMIT 换成新版本对应的完整提交号,重新运行本脚本。
#
# 不校验压缩包的 sha256:GitHub 按提交现场打包,压缩包的字节不保证每次相同。
# 内容由提交号确定,解压出的顶层目录名带有完整提交号,脚本据此核对。
#
# 头文件是 Apache-2.0 许可,放在仓库之外的 $VLM_ARTIFACTS 下,许可文件一并保留。
#
# 用法(WSL,仓库根目录):bash envs/fetch_genai_headers.sh        可重复执行,已取过就跳过。
# 出错即停,用到未定义的变量报错,管道里任一环节失败都算失败
set -euo pipefail

# 与已装的 openvino-genai 对应的完整提交号
GENAI_COMMIT=ab01d1771e4adaa0e1e08e9d630def170572506c
# 大文件目录,可用环境变量 VLM_ARTIFACTS 覆盖
ARTIFACTS="${VLM_ARTIFACTS:-/mnt/d/vlm-artifacts}"
# 头文件的存放目录
DEST="$ARTIFACTS/genai-include"
# 记号文件,内容是这份头文件对应的提交号
STAMP="$DEST/COMMIT"
# 用来查询已装版本的 Python,默认取 ovgenai 环境里的,Makefile 可以传入
PYTHON="${PYTHON:-$HOME/miniconda3/envs/ovgenai/bin/python}"

# 打印带时间戳的步骤标题
step() { echo "=== [$(date +%H:%M:%S)] $*"; }

# 第 1 步:核对已装的库由这个提交构建。取版本号的最后一段,完整提交号必须以它开头。
installed=$("$PYTHON" -c "import openvino_genai; print(openvino_genai.__version__.rsplit('-', 1)[-1])")
case "$GENAI_COMMIT" in
  "$installed"*) ;;
  *) echo "installed openvino-genai was built from commit $installed, but this script pins $GENAI_COMMIT" >&2
     exit 1 ;;
esac

# 第 2 步:已经下载过同一个提交的头文件时直接结束
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$GENAI_COMMIT" ] \
   && [ -f "$DEST/openvino/genai/visual_language/pipeline.hpp" ]; then
  step "GenAI headers for $installed already in $DEST"
  exit 0
fi

# 第 3 步:下载仓库在这个提交上的压缩包,只解出头文件目录和许可文件。
# 压缩包放在临时目录里,脚本结束时由 trap 清理。
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
step "download openvino.genai @ $installed"
curl -fsSL --retry 5 --retry-delay 3 --max-time 300 -o "$tmp/genai.tar.gz" \
    "https://github.com/openvinotoolkit/openvino.genai/archive/$GENAI_COMMIT.tar.gz"
# 压缩包的顶层目录名是 openvino.genai-<完整提交号>,不一致说明下载的不是这个提交。
# head 读完第一行就退出,tar 会因管道关闭而以非零状态结束,在 pipefail 下整条管道算失败,
# 所以用 || true 忽略这个状态,是否正确由下一行的比较判断。
top="openvino.genai-$GENAI_COMMIT"
first="$(tar -tzf "$tmp/genai.tar.gz" 2>/dev/null | head -1 || true)"
[ "$first" = "$top/" ] || { echo "unexpected archive layout: $first" >&2; exit 1; }
# 解压时去掉路径前面的几级目录,头文件落在 $tmp/out/openvino 下,许可文件落在 $tmp/out/LICENSE
mkdir -p "$tmp/out"
tar -xzf "$tmp/genai.tar.gz" -C "$tmp/out" --strip-components=4 "$top/src/cpp/include/openvino"
tar -xzf "$tmp/genai.tar.gz" -C "$tmp/out" --strip-components=1 "$top/LICENSE"

# 第 4 步:替换旧的头文件。先把旧目录移走,再放入新目录,最后写记号文件。
mkdir -p "$DEST"
if [ -d "$DEST/openvino" ]; then
  mv "$DEST/openvino" "$tmp/old"
fi
mv "$tmp/out/openvino" "$DEST/openvino"
mv "$tmp/out/LICENSE" "$DEST/LICENSE"
echo "$GENAI_COMMIT" > "$STAMP"
step "done: $(find "$DEST/openvino" -name '*.hpp' | wc -l) headers in $DEST"
