#!/usr/bin/env bash
# 从 12 个训练 tar 的正中间各取 40 MB 窗口,作为量化校准 / 精度控制的抽样来源。
# 每个窗口大约覆盖一个场景的 ~25 帧;12 个窗口 ≈ 12 个场景、~300 帧,场景多样性远好于只取 tar 头部。
# 任一命令出错即停止,使用未定义变量或管道出错也算出错
set -euo pipefail
# 切换到原始数据目录
cd "$(dirname "$0")/../data/raw"
# 训练集 tar 的下载地址前缀
BASE=https://hf-mirror.com/datasets/sayakpaul/nyu_depth_v2/resolve/main/data
# 12 个训练 tar 的字节数
SIZES=(3003340800 3003658240 3003586560 3002982400 3003443200 3003064320 3002992640 3003289600 3003443200 3003064320 3003217920 1098700800)
# 每个窗口的大小:40 MB
WIN=$((40 * 1024 * 1024))
# 逐个 tar 下载中段窗口
for i in "${!SIZES[@]}"; do
  # 6 位的 tar 序号
  idx=$(printf "%06d" "$i")
  off=$(( (SIZES[i] / 2) / 512 * 512 ))          # 对齐到 tar 块边界(512 B),便于后续扫描头部
  # 从偏移处下载一个窗口,保存为 train_window_<序号>.bin
  python3 ../../tools/parallel_fetch.py "$BASE/train-$idx.tar" "train_window_$idx.bin" \
      --offset "$off" --size "$WIN" --workers 5 --chunk-mb 8
done
# 全部窗口下载完成的标记
echo ALL_WINDOWS_DONE
