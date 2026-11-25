// ov_engine.h
// 单输入单输出 OpenVINO 模型的封装(纯头文件),接口与 trt_engine.h 的 TrtEngine 一致。
// 编译配置与 dadeploy/runners.py 的 OvRunner 相同,两边的输出可以逐值比较。
// 与 TensorRT 版的区别:在 CPU 上推理,没有主机与显卡之间的拷贝,耗时用墙钟测量。
// 约束:模型只有一个 [1,3,H,W] 的输入和一个含 H*W 个元素的输出,均为 FP32 静态形状;
// 不满足时构造函数抛异常。

#pragma once

#include <chrono>
#include <cstddef>
#include <fstream>
#include <openvino/openvino.hpp>
#include <stdexcept>
#include <string>

namespace dadeploy {

class OvEngine {
 public:
  // xml_path 是 IR 的 .xml 路径,threads 是推理线程数
  OvEngine(const std::string& xml_path, int threads) {
    // 文件不存在时给出明确的提示
    if (!std::ifstream(xml_path)) {
      throw std::runtime_error("cannot open " + xml_path +
                               " (run scripts/08_quantize_openvino.py)");
    }
    // 读取 IR 并编译到 CPU。三项配置与 OvRunner 相同:
    // 单路低延迟、指定线程数、浮点层按 FP32 计算
    compiled_ = core_.compile_model(xml_path, "CPU",
                                    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                    ov::inference_num_threads(threads),
                                    ov::hint::inference_precision(ov::element::f32));
    // 核对输入输出并记录尺寸
    CheckIo();
    // 创建推理请求,之后每帧复用
    request_ = compiled_.create_infer_request();
    // 取输入张量并记下数据指针,预处理结果直接写入这块内存
    input_ = request_.get_input_tensor();
    host_in_ = input_.data<float>();
  }

  // 禁止拷贝:一个对象独占一个推理请求
  OvEngine(const OvEngine&) = delete;
  OvEngine& operator=(const OvEngine&) = delete;

  // 模型输入的高和宽
  int input_height() const { return height_; }
  int input_width() const { return width_; }

  // 输入缓冲:调用方写入预处理结果后再调用 Run()
  float* host_input() { return host_in_; }
  // 输出缓冲:最近一次 Run() 的结果,未调用过 Run() 时为空指针
  const float* host_output() const { return host_out_; }
  // 输出的元素个数
  size_t output_elements() const { return out_elems_; }

  // 执行一次同步推理,返回耗时(毫秒)
  double Run() {
    // 计时起点
    const auto start = std::chrono::steady_clock::now();
    // 同步推理,返回时输出已就绪
    request_.infer();
    // 推理耗时
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    // 重新取输出张量的数据指针,这一步不计入耗时
    output_ = request_.get_output_tensor();
    host_out_ = output_.data<float>();
    return ms;
  }

 private:
  // 核对模型的输入输出是否符合约束,并记录尺寸
  void CheckIo() {
    // 必须恰好一个输入和一个输出
    if (compiled_.inputs().size() != 1 || compiled_.outputs().size() != 1) {
      throw std::runtime_error("expect exactly 1 input and 1 output");
    }
    // 取输入和输出的描述
    const ov::Output<const ov::Node> in = compiled_.input();
    const ov::Output<const ov::Node> out = compiled_.output();
    // 输入输出都必须是 FP32
    if (in.get_element_type() != ov::element::f32 || out.get_element_type() != ov::element::f32) {
      throw std::runtime_error("expect FP32 I/O");
    }
    // 不支持动态形状
    if (in.get_partial_shape().is_dynamic() || out.get_partial_shape().is_dynamic()) {
      throw std::runtime_error("expect static shapes");
    }
    // 输入形状必须是 [1,3,H,W]
    const ov::Shape in_shape = in.get_shape();
    if (in_shape.size() != 4 || in_shape[0] != 1 || in_shape[1] != 3) {
      throw std::runtime_error("expect input shape [1,3,H,W]");
    }
    // 记录输入的高和宽
    height_ = static_cast<int>(in_shape[2]);
    width_ = static_cast<int>(in_shape[3]);
    // 输出元素数必须等于 H*W,调用方按这个尺寸读取输出
    out_elems_ = ov::shape_size(out.get_shape());
    if (out_elems_ != static_cast<size_t>(height_) * static_cast<size_t>(width_)) {
      throw std::runtime_error("expect output with H*W elements");
    }
  }

  // 成员按声明顺序构造、逆序析构,core_ 最先构造、最后析构
  ov::Core core_;               // OpenVINO 入口对象
  ov::CompiledModel compiled_;  // 编译后的模型
  ov::InferRequest request_;    // 推理请求
  ov::Tensor input_;            // 输入张量
  ov::Tensor output_;           // 输出张量
  float* host_in_ = nullptr;         // 输入缓冲指针
  const float* host_out_ = nullptr;  // 输出缓冲指针
  int height_ = 0;                   // 输入高
  int width_ = 0;                    // 输入宽
  size_t out_elems_ = 0;             // 输出元素个数
};

}  // namespace dadeploy
