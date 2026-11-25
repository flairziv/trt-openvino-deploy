// da_infer_ov.cpp
// Depth Anything V2-Small 的 C++ OpenVINO 端到端推理程序(CPU),不依赖 Python。
// 预处理、后处理、文件输出和耗时统计来自 da_common.h,与 TensorRT 版(da_infer.cpp)共用;
// 推理部分由 ov_engine.h 封装。
//
// 流程:读图、预处理(写入输入张量)、推理、缩放回原图尺寸、保存伪彩色深度图。
//
// 用法:
//   da_infer_ov --model <x.xml> --image <in.png> [--out depth.png] [--threads 4]
//   da_infer_ov --model <x.xml> --image <in.png> --bench 100 --warmup 5 [--json out.json]
//   da_infer_ov --model <x.xml> --image <in.png> --dump-input in.bin --dump-output out.bin
//
// --dump-input / --dump-output 把预处理结果和模型原始输出保存为 float32 二进制文件,
// 供 scripts/13_cpp_parity.py 与 Python 对拍。
// --threads 的默认值与 dadeploy/config.py 的 CPU_THREADS 相同,对拍脚本会显式传入。

#include <cstdio>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>

#include "da_common.h"
#include "ov_engine.h"

namespace {

// 引入共用头文件中的名字
using dadeploy::Clock;
using dadeploy::MsSince;
using dadeploy::NormalizeLut;
using dadeploy::OvEngine;
using dadeploy::Preprocess;
using dadeploy::Restore;

// 命令行选项
struct Options {
  std::string model;              // IR 的 .xml 路径
  std::string image;              // 输入图片路径
  std::string out = "depth.png";  // 伪彩色深度图的输出路径
  std::string dump_input;         // 非空时保存预处理结果
  std::string dump_output;        // 非空时保存模型原始输出
  std::string json;               // 非空时把基准结果写成 JSON
  int bench = 0;                  // 基准的计时帧数,0 表示只推理一次
  int warmup = 5;                 // 基准开头不计时的预热帧数
  int threads = 4;                // 推理线程数
};

// 解析命令行,选项都以"--名字 值"的形式成对出现
Options ParseArgs(int argc, char** argv) {
  Options opt;
  // 从第 1 个参数开始,每次处理一对
  for (int i = 1; i < argc; ++i) {
    // 选项名
    const std::string arg = argv[i];
    // 选项名后面必须跟一个值
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
    // 选项值
    const std::string value = argv[++i];
    // 按选项名写入对应的字段,未知选项报错
    if (arg == "--model") opt.model = value;
    else if (arg == "--image") opt.image = value;
    else if (arg == "--out") opt.out = value;
    else if (arg == "--dump-input") opt.dump_input = value;
    else if (arg == "--dump-output") opt.dump_output = value;
    else if (arg == "--json") opt.json = value;
    else if (arg == "--bench") opt.bench = std::stoi(value);
    else if (arg == "--warmup") opt.warmup = std::stoi(value);
    else if (arg == "--threads") opt.threads = std::stoi(value);
    else throw std::runtime_error("unknown option " + arg);
  }
  // 模型和图片是必填项
  if (opt.model.empty() || opt.image.empty()) {
    throw std::runtime_error("usage: da_infer_ov --model <x.xml> --image <in.png> [options]");
  }
  return opt;
}

// 基准:同一张图重复推理,分段统计预处理、推理、后处理和整帧耗时
void RunBenchmark(OvEngine& engine, const cv::Mat& bgr, const Options& opt) {
  // 模型输入尺寸
  const int h = engine.input_height();
  const int w = engine.input_width();
  // 归一化查找表
  const NormalizeLut lut;
  // 中间图像缓冲,各帧复用
  cv::Mat rgb;
  cv::Mat resized;
  cv::Mat restored;
  // 各分段的耗时样本
  dadeploy::Samples samples;

  // 预热帧和计时帧连续执行
  for (int i = 0; i < opt.warmup + opt.bench; ++i) {
    // 整帧计时起点
    const auto t0 = Clock::now();
    // 预处理,结果直接写入模型的输入张量
    Preprocess(bgr, h, w, lut, &rgb, &resized, engine.host_input());
    const double pre_ms = MsSince(t0);
    // 推理
    const double compute_ms = engine.Run();
    // 后处理:缩放回原图尺寸
    const auto t1 = Clock::now();
    Restore(engine.host_output(), h, w, bgr.size(), &restored);
    const double post_ms = MsSince(t1);
    // 整帧耗时
    const double total_ms = MsSince(t0);
    // 预热帧不计入统计
    if (i < opt.warmup) continue;
    // 记录各分段耗时
    samples["preprocess"].push_back(pre_ms);
    samples["compute"].push_back(compute_ms);
    samples["postprocess"].push_back(post_ms);
    samples["total"].push_back(total_ms);
  }
  // 打印统计结果,需要时写出 JSON
  dadeploy::ReportBenchmark(samples, opt.bench, opt.json);
}

// 单次推理:按需保存输入输出张量,并保存伪彩色深度图
void RunOnce(OvEngine& engine, const cv::Mat& bgr, const Options& opt) {
  // 模型输入尺寸
  const int h = engine.input_height();
  const int w = engine.input_width();
  // 归一化查找表
  const NormalizeLut lut;
  // 中间图像缓冲
  cv::Mat rgb;
  cv::Mat resized;
  // 预处理,结果直接写入模型的输入张量
  Preprocess(bgr, h, w, lut, &rgb, &resized, engine.host_input());
  // 推理之前保存输入张量
  if (!opt.dump_input.empty()) {
    dadeploy::WriteRaw(opt.dump_input, engine.host_input(), static_cast<size_t>(3) * h * w);
  }
  // 推理
  engine.Run();
  // 保存模型原始输出
  if (!opt.dump_output.empty()) {
    dadeploy::WriteRaw(opt.dump_output, engine.host_output(), engine.output_elements());
  }
  // 缩放回原图尺寸
  cv::Mat restored;
  Restore(engine.host_output(), h, w, bgr.size(), &restored);
  // 保存伪彩色深度图
  dadeploy::WriteColorized(opt.out, restored);
  // 打印输出路径和尺寸变化
  std::printf("wrote %s (%dx%d -> %dx%d)\n", opt.out.c_str(), h, w, bgr.rows, bgr.cols);
}

}  // namespace

int main(int argc, char** argv) {
  // 所有错误统一在这里捕获:打印原因并以非零退出码结束
  try {
    // 解析命令行
    const Options opt = ParseArgs(argc, argv);
    // 读取图片,统一为 3 通道 BGR
    const cv::Mat bgr = cv::imread(opt.image, cv::IMREAD_COLOR);
    // 读取失败时报错
    if (bgr.empty()) throw std::runtime_error("cannot read image " + opt.image);
    // 加载并编译模型
    OvEngine engine(opt.model, opt.threads);
    // 指定了 --bench 时跑基准,否则推理一次
    if (opt.bench > 0) {
      RunBenchmark(engine, bgr, opt);
    } else {
      RunOnce(engine, bgr, opt);
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    // 打印错误原因
    std::fprintf(stderr, "error: %s\n", e.what());
    return EXIT_FAILURE;
  }
}
