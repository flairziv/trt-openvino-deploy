// da_common.h
// 两个 C++ 推理程序(da_infer、da_infer_ov)共用的预处理、后处理、文件输出和耗时统计。
// 预处理与 dadeploy/preprocess.py 逐步对应;归一化常量须与 dadeploy/config.py 保持一致,
// 修改任何一边之后都要重新对拍(make cpp parity、make cpp-ov parity-ov)。
// 纯头文件,只依赖标准库和 OpenCV。

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace dadeploy {

// 单调时钟,用于测量时间间隔
using Clock = std::chrono::steady_clock;

// 返回从 start 到当前的毫秒数
inline double MsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ImageNet 归一化均值与标准差,RGB 顺序
inline constexpr std::array<float, 3> kMean = {0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, 3> kStd = {0.229f, 0.224f, 0.225f};

// 归一化查找表:像素值只有 256 种,预先算好每个通道每个取值的归一化结果。
// 计算顺序与 Python 侧相同,结果逐位一致。
class NormalizeLut {
 public:
  // 构造时填表
  NormalizeLut() {
    // 1/255,把像素值缩放到 0 到 1
    const float inv255 = 1.0f / 255.0f;
    // 逐通道填表
    for (int c = 0; c < 3; ++c) {
      // 标准差的倒数,与 Python 侧一样用乘法代替除法
      const float inv_std = 1.0f / kStd[c];
      // 逐个像素取值计算归一化结果
      for (int v = 0; v < 256; ++v) {
        table_[c][v] = (static_cast<float>(v) * inv255 - kMean[c]) * inv_std;
      }
    }
  }
  // 查表:返回指定通道、指定像素值的归一化结果
  float operator()(int channel, uint8_t value) const { return table_[channel][value]; }

 private:
  float table_[3][256];  // 下标依次是通道和像素值
};

// 预处理:BGR 原图转成 [1,3,H,W] 的 float32,写入 dst(容量不少于 3*H*W 个 float)。
// rgb 和 resized 是调用方提供的中间缓冲,循环调用时可以复用内存。
inline void Preprocess(const cv::Mat& bgr, int height, int width, const NormalizeLut& lut,
                       cv::Mat* rgb, cv::Mat* resized, float* dst) {
  // BGR 转 RGB,模型按 RGB 训练
  cv::cvtColor(bgr, *rgb, cv::COLOR_BGR2RGB);
  // 双三次插值缩放到模型输入尺寸,cv::Size 的参数顺序是(宽,高)
  cv::resize(*rgb, *resized, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
  // 输出按 R、G、B 三个平面依次存放,三个指针分别指向各平面的起点
  float* r = dst;
  float* g = dst + static_cast<ptrdiff_t>(height) * width;
  float* b = g + static_cast<ptrdiff_t>(height) * width;
  // 逐行逐像素查表归一化,同时完成 HWC 到 CHW 的重排
  for (int y = 0; y < height; ++y) {
    // 第 y 行的首地址,行内按 R G B 交错存放
    const uint8_t* src = resized->ptr<uint8_t>(y);
    for (int x = 0; x < width; ++x) {
      *r++ = lut(0, src[3 * x]);
      *g++ = lut(1, src[3 * x + 1]);
      *b++ = lut(2, src[3 * x + 2]);
    }
  }
}

// 后处理:把 [H,W] 的相对视差图双线性缩放回原图尺寸。
// raw 指向 height*width 个 float,size 是原图尺寸。
inline void Restore(const float* raw, int height, int width, cv::Size size, cv::Mat* restored) {
  // 用 cv::Mat 包装输出缓冲,不拷贝数据,也不接管内存
  const cv::Mat wrapped(height, width, CV_32F, const_cast<float*>(raw));
  // 双线性插值,与 Python 侧的后处理一致
  cv::resize(wrapped, *restored, size, 0, 0, cv::INTER_LINEAR);
}

// 把 elements 个 float 按原始二进制写入文件,供 Python 读回对拍
inline void WriteRaw(const std::string& path, const float* data, size_t elements) {
  // 以二进制方式打开输出文件
  std::ofstream file(path, std::ios::binary);
  // 打开失败时报错
  if (!file) throw std::runtime_error("cannot write " + path);
  // 写入全部字节
  file.write(reinterpret_cast<const char*>(data),
             static_cast<std::streamsize>(elements * sizeof(float)));
}

// 把相对视差图保存为伪彩色图片,仅供人工查看
inline void WriteColorized(const std::string& path, const cv::Mat& disparity) {
  cv::Mat gray;
  cv::Mat color;
  // 按这张图的最小值和最大值拉伸到 0 到 255
  cv::normalize(disparity, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
  // 套用伪彩色
  cv::applyColorMap(gray, color, cv::COLORMAP_INFERNO);
  // 写出图片,失败时报错
  if (!cv::imwrite(path, color)) throw std::runtime_error("cannot write " + path);
}

// 返回第 p 百分位:排序后按位置直接取样本,不做插值
inline double Percentile(std::vector<double> values, double p) {
  // 参数按值传入,排序不影响调用方的数据
  std::sort(values.begin(), values.end());
  // 目标下标,不超过最后一个元素
  const size_t idx = std::min(values.size() - 1, static_cast<size_t>(p / 100.0 * values.size()));
  return values[idx];
}

// 基准样本:分段名到各帧耗时(毫秒)的映射
using Samples = std::map<std::string, std::vector<double>>;

// 打印各分段的 P50 和 P90;json_path 非空时同时写出 JSON,供 scripts/13_cpp_parity.py 读取
inline void ReportBenchmark(const Samples& samples, int iters, const std::string& json_path) {
  // 手工拼接 JSON 文本
  std::string json = "{";
  // 逐个分段统计
  for (const auto& [name, values] : samples) {
    const double p50 = Percentile(values, 50);
    const double p90 = Percentile(values, 90);
    // 打印这一分段
    std::printf("%-12s P50 %8.3f ms   P90 %8.3f ms\n", name.c_str(), p50, p90);
    // 追加到 JSON
    json += "\"" + name + "\":{\"p50\":" + std::to_string(p50) + ",\"p90\":" + std::to_string(p90) +
            "},";
  }
  // 记录计时帧数
  json += "\"iters\":" + std::to_string(iters) + "}";
  // 由整帧耗时的中位数换算帧率
  std::printf("end-to-end FPS (P50): %.1f\n", 1000.0 / Percentile(samples.at("total"), 50));
  // 需要时写出 JSON 文件
  if (!json_path.empty()) std::ofstream(json_path) << json << "\n";
}

}  // namespace dadeploy
