// trt_engine.h —— 单输入单输出 TensorRT 引擎的 RAII 封装(纯头文件)。
//
// 设计要点(承接 dcn-trt10 项目的 Nsight 归因结论):
//   1. host 侧输入 / 输出缓冲一次性用 cudaMallocHost 分配为锁页内存。dcn 项目每帧 new vector +
//      同步 cudaMemcpy,D2H 只有约 3 GB/s,反超推理成为后处理第一大块;这里从一开始规避。
//   2. H2D -> enqueueV3 -> D2H 全部排在同一条 stream 上,只在末尾同步一次,整链没有多余等待。
//   3. 用四个 cudaEvent 打点得到分段耗时,打点本身不引入同步。
//
// 假设:引擎 I/O 为 FP32 且只有一个输入、一个输出(modelopt / AutoCast 导出时保持 keep_io_types)。
// 不满足时构造函数直接抛异常,而不是静默算错。
// TensorRT C++ API 在 x86 与 Jetson 上一致,本封装可直接迁移。

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dadeploy {

// 检查 CUDA 调用的返回值,失败时抛出带文件名和行号的异常
inline void CheckCuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + file +
                             ":" + std::to_string(line));
  }
}
// 包装 CUDA 调用,自动带上调用处的文件名和行号
#define DADEPLOY_CUDA(call) ::dadeploy::CheckCuda((call), __FILE__, __LINE__)

// 一次推理的分段耗时(毫秒,GPU 时间线)。
struct GpuTiming {
  float h2d_ms = 0.f;  // 主机到显卡的拷贝
  float compute_ms = 0.f;  // 推理
  float d2h_ms = 0.f;  // 显卡到主机的拷贝
};

// TensorRT 日志回调:只输出警告及以上级别
class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
  }
};

// 单输入单输出 TensorRT 引擎的封装
class TrtEngine {
 public:
  // 反序列化失败通常意味着换了 GPU 或 TensorRT 版本:.engine 绑定两者,只能重新构建。
  explicit TrtEngine(const std::string& engine_path) {
    // 读入引擎文件
    const std::vector<char> blob = ReadFile(engine_path);
    // 创建运行时
    runtime_ = nvinfer1::createInferRuntime(logger_);
    // 反序列化引擎
    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    // 反序列化失败时报错
    if (engine_ == nullptr) {
      throw std::runtime_error("failed to deserialize engine (GPU model or TensorRT version changed?)");
    }
    // 创建执行上下文
    context_ = engine_->createExecutionContext();
    // 创建专用的 CUDA stream
    DADEPLOY_CUDA(cudaStreamCreate(&stream_));
    // 创建四个计时事件
    for (cudaEvent_t& event : events_) DADEPLOY_CUDA(cudaEventCreate(&event));
    // 查找输入输出张量并分配缓冲
    DiscoverIo();
  }

  // 先释放缓冲与 stream,最后才删 context / engine / runtime(后者依次依赖前者)。
  ~TrtEngine() {
    // 释放显存缓冲
    if (device_in_ != nullptr) cudaFree(device_in_);
    if (device_out_ != nullptr) cudaFree(device_out_);
    // 释放锁页内存缓冲
    if (host_in_ != nullptr) cudaFreeHost(host_in_);
    if (host_out_ != nullptr) cudaFreeHost(host_out_);
    // 销毁计时事件
    for (cudaEvent_t& event : events_) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    // 销毁 stream
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    // 依次释放执行上下文、引擎和运行时
    delete context_;
    delete engine_;
    delete runtime_;
  }

  // 禁止拷贝:对象独占 GPU 资源
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // 输入尺寸 [1, 3, H, W]。
  int input_height() const { return input_dims_.d[2]; }
  int input_width() const { return input_dims_.d[3]; }

  // 调用方把预处理结果直接写入这块锁页内存(零额外拷贝),再调用 Run()。
  float* host_input() { return host_in_; }
  // 输出缓冲,Run() 返回后可读
  const float* host_output() const { return host_out_; }
  // 输出的元素个数
  size_t output_elements() const { return out_elems_; }

  // 执行一次完整的 H2D -> 推理 -> D2H,返回分段耗时。
  GpuTiming Run() {
    // 三段都只入队不等待:GPU 按序执行,CPU 只在最后阻塞一次,不存在中途空转。
    // host 缓冲是锁页内存,async 拷贝走 DMA;若换成可分页内存,驱动会退化成先拷到内部缓冲。
    // 记录 H2D 开始
    DADEPLOY_CUDA(cudaEventRecord(events_[0], stream_));
    // 异步把输入从锁页内存拷到显存
    DADEPLOY_CUDA(cudaMemcpyAsync(device_in_, host_in_, in_elems_ * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_));
    // 记录 H2D 结束,同时也是推理开始
    DADEPLOY_CUDA(cudaEventRecord(events_[1], stream_));
    // 把推理排入 stream,返回 false 表示提交失败
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
    // 记录推理结束
    DADEPLOY_CUDA(cudaEventRecord(events_[2], stream_));
    // 异步把输出从显存拷回锁页内存
    DADEPLOY_CUDA(cudaMemcpyAsync(host_out_, device_out_, out_elems_ * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_));
    // 记录 D2H 结束
    DADEPLOY_CUDA(cudaEventRecord(events_[3], stream_));
    DADEPLOY_CUDA(cudaStreamSynchronize(stream_));   // 全链路唯一的同步点,返回后输出才可读

    // event 耗时取自 GPU 时间线,不受 CPU 调度影响;CPU 侧总耗时由调用方用墙钟另算。

    // 由相邻事件的时间差得到三段耗时
    GpuTiming timing;
    DADEPLOY_CUDA(cudaEventElapsedTime(&timing.h2d_ms, events_[0], events_[1]));
    DADEPLOY_CUDA(cudaEventElapsedTime(&timing.compute_ms, events_[1], events_[2]));
    DADEPLOY_CUDA(cudaEventElapsedTime(&timing.d2h_ms, events_[2], events_[3]));
    return timing;
  }

 private:
  // 把整个文件读入内存
  static std::vector<char> ReadFile(const std::string& path) {
    // 以二进制方式打开,并定位到文件末尾
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    // 打开失败时报错
    if (!file) throw std::runtime_error("cannot open " + path);
    // 当前位置即文件大小
    const std::streamsize size = file.tellg();
    // 回到文件开头
    file.seekg(0);
    // 按文件大小分配缓冲
    std::vector<char> blob(static_cast<size_t>(size));
    // 读入全部内容
    file.read(blob.data(), size);
    return blob;
  }

  // 计算张量的元素总数:各维大小相乘
  static size_t Volume(const nvinfer1::Dims& dims) {
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) volume *= static_cast<size_t>(dims.d[i]);
    return volume;
  }

  // 遍历引擎的输入输出张量:校验类型和形状,分配缓冲并绑定地址
  void DiscoverIo() {
    // 输入和输出张量的计数
    int inputs = 0;
    int outputs = 0;
    // 逐个张量处理
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      // 张量名
      const char* name = engine_->getIOTensorName(i);
      // 硬编码 FP32 读写:若引擎 I/O 是 FP16,按 float 拷贝会越界读写,所以这里必须拦下。
      if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(std::string("expect FP32 I/O, tensor ") + name + " is not");
      }
      // 张量形状
      const nvinfer1::Dims dims = engine_->getTensorShape(name);
      // 元素总数
      const size_t elems = Volume(dims);
      // 输入张量
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        ++inputs;
        // 输入形状必须是 [1,3,H,W]
        if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3) {
          throw std::runtime_error("expect input shape [1,3,H,W]");
        }
        // 记录输入形状和元素数
        input_dims_ = dims;
        in_elems_ = elems;
        // 分配显存缓冲
        DADEPLOY_CUDA(cudaMalloc(&device_in_, elems * sizeof(float)));
        // 分配锁页内存缓冲
        DADEPLOY_CUDA(cudaMallocHost(reinterpret_cast<void**>(&host_in_), elems * sizeof(float)));
        context_->setTensorAddress(name, device_in_);   // 地址只绑定一次,之后每帧复用,无分配开销
      } else {
        // 输出张量:同样分配两块缓冲并绑定地址
        ++outputs;
        out_elems_ = elems;
        DADEPLOY_CUDA(cudaMalloc(&device_out_, elems * sizeof(float)));
        DADEPLOY_CUDA(cudaMallocHost(reinterpret_cast<void**>(&host_out_), elems * sizeof(float)));
        context_->setTensorAddress(name, device_out_);
      }
    }
    // 必须恰好一个输入和一个输出
    if (inputs != 1 || outputs != 1) throw std::runtime_error("expect exactly 1 input and 1 output");
  }

  TrtLogger logger_;  // 日志回调
  nvinfer1::IRuntime* runtime_ = nullptr;  // 运行时
  nvinfer1::ICudaEngine* engine_ = nullptr;  // 引擎
  nvinfer1::IExecutionContext* context_ = nullptr;  // 执行上下文
  cudaStream_t stream_ = nullptr;  // 专用的 CUDA stream
  cudaEvent_t events_[4] = {nullptr, nullptr, nullptr, nullptr};  // 四个计时事件
  nvinfer1::Dims input_dims_{};  // 输入形状
  size_t in_elems_ = 0;  // 输入元素个数
  size_t out_elems_ = 0;  // 输出元素个数
  void* device_in_ = nullptr;  // 输入的显存缓冲
  void* device_out_ = nullptr;  // 输出的显存缓冲
  float* host_in_ = nullptr;  // 输入的锁页内存缓冲
  float* host_out_ = nullptr;  // 输出的锁页内存缓冲
};

}  // namespace dadeploy
