// vlm_infer_ov.cpp
// Qwen2-VL-2B 的 C++ 推理程序,基于 OpenVINO GenAI,在 CPU 上运行,不依赖 Python。
// 与 vlmdeploy/backends.py 的 OpenVinoAnswerer 对应:同一个库、同一份 IR、同样的配置。
// 同一张图和同一个问题的回答应当逐字相同,由 scripts/16_cpp_parity_cpu.py 核对。
//
// 图片用 OpenCV 读入并转成 RGB,按原尺寸交给管线,本程序不做缩放。
// 对拍时由对拍脚本把预缩放好的图存成 PNG 再交给本程序,两边送进管线的像素相同。
//
// 用法:
//   单题模式,流式打印回答:
//     vlm_infer_ov --model <IR 目录> --image a.png --question "What does the sign say?"
//   批量模式,模型只加载一次,每个请求写一行 JSON:
//     vlm_infer_ov --model <IR 目录> --requests list.tsv --out out.jsonl
//                  [--max-new-tokens 16] [--decode-tokens 32] [--warmup 1] [--threads 4]
//   list.tsv 每行依次是图片路径、制表符、提示词文件路径。提示词本身带换行,所以单独放文件。
//
// --threads 和 --max-new-tokens 的默认值与 vlmdeploy/backends.py 的 CPU_THREADS、
// MAX_NEW_TOKENS 相同。对拍脚本总是显式传这两个参数。

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "openvino/genai/visual_language/pipeline.hpp"

namespace {

// 单调时钟,用于测量时间间隔
using Clock = std::chrono::steady_clock;

// 返回从 start 到现在经过的毫秒数
double MsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// 命令行选项
struct Options {
  std::string model;             // IR 目录
  std::string image;             // 单题模式的图片路径
  std::string question;          // 单题模式的问题文字,不含聊天模板
  std::string requests;          // 批量模式的请求清单
  std::string out;               // 批量模式的结果文件,每个请求一行 JSON
  int max_new_tokens = 16;       // 生成 token 数的上限
  int decode_tokens = 0;         // 大于 0 时固定生成这么多个 token,计时用
  int warmup = 0;                // 批量模式的预热次数
  int threads = 4;               // 推理线程数
};

// 解析命令行。所有选项都是选项名后跟一个值。
Options ParseArgs(int argc, char** argv) {
  Options opt;
  // 逐对读取选项,argv[0] 是程序名
  for (int i = 1; i < argc; ++i) {
    // 选项名
    const std::string arg = argv[i];
    // 选项后面必须跟一个值
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
    // 取出选项的值
    const std::string value = argv[++i];
    // 按选项名写入对应的字段,未知的选项报错
    if (arg == "--model") opt.model = value;
    else if (arg == "--image") opt.image = value;
    else if (arg == "--question") opt.question = value;
    else if (arg == "--requests") opt.requests = value;
    else if (arg == "--out") opt.out = value;
    else if (arg == "--max-new-tokens") opt.max_new_tokens = std::stoi(value);
    else if (arg == "--decode-tokens") opt.decode_tokens = std::stoi(value);
    else if (arg == "--warmup") opt.warmup = std::stoi(value);
    else if (arg == "--threads") opt.threads = std::stoi(value);
    else throw std::runtime_error("unknown option " + arg);
  }
  // 单题模式需要 --image 和 --question,批量模式需要 --requests 和 --out
  const bool single = !opt.image.empty() && !opt.question.empty();
  const bool batch = !opt.requests.empty() && !opt.out.empty();
  // 必须给出模型目录,且两种模式只能选一种
  if (opt.model.empty() || single == batch) {
    throw std::runtime_error(
        "usage: vlm_infer_ov --model <ir_dir> (--image <img> --question <text> | "
        "--requests <list.tsv> --out <out.jsonl>) [options]");
  }
  return opt;
}

// 把整个文件读成一个字符串
std::string ReadText(const std::string& path) {
  // 以二进制方式打开,打不开时报错
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  // 把文件内容全部读入缓冲
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// 读入图片,返回管线需要的 [1, H, W, 3]、uint8、RGB 张量。
// 布局与 Python 侧的 ov.Tensor(np.asarray(img, dtype=np.uint8)[None]) 相同。
ov::Tensor LoadImage(const std::string& path) {
  // 读成 3 通道 8 位图,OpenCV 的通道顺序是 BGR
  const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  // 读不到时报错
  if (bgr.empty()) throw std::runtime_error("cannot read image " + path);
  // 转成 RGB。模型按 RGB 训练,漏掉这一步不会报错,但回答会变差。
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  // 分配张量,形状依次是批大小 1、高、宽、3 个通道
  ov::Tensor tensor(ov::element::u8, {1, static_cast<size_t>(rgb.rows), static_cast<size_t>(rgb.cols), 3});
  // 一行的字节数
  const size_t row_bytes = static_cast<size_t>(rgb.cols) * 3;
  // 张量数据区的起始地址
  uint8_t* dst = tensor.data<uint8_t>();
  // 逐行拷贝。cv::Mat 每行末尾可能有对齐用的填充字节,不能整块拷贝。
  for (int y = 0; y < rgb.rows; ++y) {
    std::memcpy(dst + static_cast<size_t>(y) * row_bytes, rgb.ptr<uint8_t>(y), row_bytes);
  }
  return tensor;
}

// 生成参数。从默认构造的 GenerationConfig 出发,即贪心解码,与 Python 侧相同。
// 不用管线自带的配置:它读自 IR 目录的 generation_config.json,其中 do_sample=true,
// 用它会使两边的解码方式不一致。
ov::genai::GenerationConfig MakeConfig(const Options& opt) {
  ov::genai::GenerationConfig config;
  if (opt.decode_tokens > 0) {
    // 计时模式:上下限相同并忽略结束符,每个请求都生成 decode_tokens 个 token,
    // 与 scripts/14_bench_cpu.py 的设置相同
    config.max_new_tokens = static_cast<size_t>(opt.decode_tokens);
    config.min_new_tokens = static_cast<size_t>(opt.decode_tokens);
    config.ignore_eos = true;
  } else {
    // 普通模式:只设生成上限
    config.max_new_tokens = static_cast<size_t>(opt.max_new_tokens);
  }
  return config;
}

// 一个请求的结果:回答和管线自带的分段计时
struct Answer {
  std::string text;          // 回答原文,没有去掉首尾空白
  double ttft_ms = 0;        // 首字延迟,从 generate 开始到第一个 token
  double vision_ms = 0;      // 视觉段耗时
  double tpot_ms = 0;        // 每个输出 token 的平均耗时
  size_t input_tokens = 0;   // 送进语言模型的序列长度
  size_t new_tokens = 0;     // 实际生成的 token 数
  double wall_ms = 0;        // 整个 generate 调用的墙钟耗时
};

// 流式回调的类型:管线每生成一段文字调用一次,返回值决定是否继续
using Streamer = std::function<ov::genai::StreamingStatus(std::string)>;

// 回答一个问题。streamer 为空时不做流式输出。
Answer Ask(ov::genai::VLMPipeline& pipe, const std::string& prompt, const ov::Tensor& image,
           const ov::genai::GenerationConfig& config, const Streamer& streamer) {
  // 没有回调时传 std::monostate,表示不做流式输出
  ov::genai::StreamerVariant sink = std::monostate{};
  if (streamer) sink = streamer;
  // 计时起点
  const auto t0 = Clock::now();
  // 只传问题文字和图片,聊天模板和图像占位符由管线添加,与 Python 侧的调用方式相同
  ov::genai::VLMDecodedResults result = pipe.generate(prompt, std::vector<ov::Tensor>{image}, config, sink);
  Answer answer;
  // 墙钟耗时
  answer.wall_ms = MsSince(t0);
  // 贪心解码只有一个候选,取第 0 个
  answer.text = result.texts.at(0);
  // 性能指标。取值函数没有声明为 const,所以 result 不能是 const。
  ov::genai::VLMPerfMetrics& pm = result.perf_metrics;
  // 各项指标返回均值和标准差,一次 generate 只有一个样本,取均值
  answer.ttft_ms = pm.get_ttft().mean;
  answer.vision_ms = pm.get_prepare_embeddings_duration().mean;
  answer.tpot_ms = pm.get_tpot().mean;
  answer.input_tokens = pm.get_num_input_tokens();
  answer.new_tokens = pm.get_num_generated_tokens();
  return answer;
}

// 把字符串写成带引号的 JSON 字符串。引号、反斜杠和控制字符需要转义,
// 非 ASCII 字符原样输出。
std::string JsonString(const std::string& text) {
  std::string out = "\"";
  // 逐字节处理
  for (const unsigned char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          // 其余控制字符按 4 位十六进制的 Unicode 转义输出
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          // 普通字符原样输出
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

// 把数值写成 JSON。NaN 和无穷大在 JSON 里不合法,写成 null。
// 只生成 1 个 token 时 TPOT 可能不是有限值。
std::string JsonNumber(double value) {
  if (!std::isfinite(value)) return "null";
  // 保留 3 位小数
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", value);
  return buf;
}

// 一个请求
struct Request {
  std::string image;    // 图片路径
  std::string prompt;   // 提示词,已从提示词文件读出
};

// 读请求清单。每行依次是图片路径、制表符、提示词文件路径,空行跳过。
std::vector<Request> LoadRequests(const std::string& path) {
  std::vector<Request> requests;
  // 整个清单读入后按行拆分
  std::istringstream lines(ReadText(path));
  std::string line;
  while (std::getline(lines, line)) {
    // 去掉 Windows 行尾的回车符
    if (!line.empty() && line.back() == '\r') line.pop_back();
    // 跳过空行
    if (line.empty()) continue;
    // 按制表符拆成两段,没有制表符时报错
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) throw std::runtime_error("bad request line (no tab): " + line);
    // 第一段是图片路径,第二段是提示词文件,读出其中的提示词
    requests.push_back({line.substr(0, tab), ReadText(line.substr(tab + 1))});
  }
  // 清单为空时报错
  if (requests.empty()) throw std::runtime_error("no requests in " + path);
  return requests;
}

// 批量模式:逐个回答清单里的请求,每个请求写一行 JSON
void RunBatch(ov::genai::VLMPipeline& pipe, const Options& opt, double load_s) {
  // 读入请求清单
  const std::vector<Request> requests = LoadRequests(opt.requests);
  // 生成参数
  const ov::genai::GenerationConfig config = MakeConfig(opt);
  // 打开结果文件,打不开时报错
  std::ofstream out(opt.out, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + opt.out);

  // 预热:用清单里最后一个请求跑 warmup 次,结果丢弃。
  // 不用第一个请求预热:管线会保留上一次请求的 KV cache,下一个请求开头相同的部分直接复用。
  // 用第一个请求预热时,紧接着的第一个正式请求与它完全相同,prefill 几乎被跳过,
  // 实测 272 个输入 token 的 prefill 从约 1500 ms 变成 90 ms。清单只有一个请求时无法避免。
  for (int i = 0; i < opt.warmup; ++i) {
    Ask(pipe, requests.back().prompt, LoadImage(requests.back().image), config, nullptr);
  }
  // 逐个处理正式请求
  for (size_t i = 0; i < requests.size(); ++i) {
    // 读图不计入耗时,计时从 generate 开始
    const ov::Tensor image = LoadImage(requests[i].image);
    const Answer a = Ask(pipe, requests[i].prompt, image, config, nullptr);
    // 每个请求写一行并立即刷新,中途被打断时已答完的题不会丢
    out << "{\"index\":" << i << ",\"answer\":" << JsonString(a.text)
        << ",\"ttft_ms\":" << JsonNumber(a.ttft_ms) << ",\"vision_ms\":" << JsonNumber(a.vision_ms)
        << ",\"tpot_ms\":" << JsonNumber(a.tpot_ms) << ",\"input_tokens\":" << a.input_tokens
        << ",\"new_tokens\":" << a.new_tokens << ",\"wall_ms\":" << JsonNumber(a.wall_ms)
        << ",\"load_s\":" << JsonNumber(load_s) << "}\n";
    out.flush();
    // 进度写到标准错误,不与结果文件混在一起
    std::fprintf(stderr, "[%zu/%zu] TTFT %.0f ms (vision %.0f), %zu tokens\n", i + 1, requests.size(),
                 a.ttft_ms, a.vision_ms, a.new_tokens);
  }
}

// 单题模式:流式打印回答,最后给出首字延迟和生成速度。数值是单次观察,不是基准。
void RunSingle(ov::genai::VLMPipeline& pipe, const Options& opt) {
  // 读入图片
  const ov::Tensor image = LoadImage(opt.image);
  // 打印图片尺寸和问题。张量形状是 {1, 高, 宽, 3}。
  std::printf("image %zux%zu\nQ: %s\nA: ", image.get_shape()[2], image.get_shape()[1], opt.question.c_str());
  std::fflush(stdout);
  // 流式回调:每收到一段文字立即打印并刷新,返回 RUNNING 表示继续生成
  const Streamer streamer = [](std::string piece) {
    std::cout << piece << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
  };
  // 作答
  const Answer a = Ask(pipe, opt.question, image, MakeConfig(opt), streamer);
  // 打印首字延迟、视觉段耗时和 token 数
  std::printf("\nfirst token %.0f ms (vision %.0f ms) | %zu input tokens | %zu new tokens", a.ttft_ms,
              a.vision_ms, a.input_tokens, a.new_tokens);
  // 生成的 token 太少时 TPOT 没有意义,不报速度
  if (a.new_tokens >= 8 && std::isfinite(a.tpot_ms) && a.tpot_ms > 0) {
    std::printf(" | decode %.1f tok/s", 1000.0 / a.tpot_ms);
  }
  // 打印总耗时
  std::printf(" | total %.0f ms\n", a.wall_ms);
}

}  // namespace

// 程序入口
int main(int argc, char** argv) {
  // 所有错误都在下面的 catch 里处理:打印原因并以非零退出码结束
  try {
    // 解析命令行
    const Options opt = ParseArgs(argc, argv);
#ifdef VLM_TOKENIZERS_LIB
    // GenAI 通过这个环境变量找到分词器库 libopenvino_tokenizers.so。
    // Python 里 import openvino_genai 时会自动设置,C++ 程序需要自己设置。
    // 路径在构建时由 CMake 写入,第三个参数 0 表示已有设置时不覆盖。
    setenv("OPENVINO_TOKENIZERS_PATH_GENAI", VLM_TOKENIZERS_LIB, 0);
#endif
    // 加载计时起点
    const auto t0 = Clock::now();
    // 读入 IR 目录里的各个子模型和分词器,编译到 CPU 上,并限定推理线程数
    ov::genai::VLMPipeline pipe(opt.model, "CPU", ov::inference_num_threads(opt.threads));
    // 加载耗时,只作记录,不计入延迟指标
    const double load_s = MsSince(t0) / 1000.0;
    std::fprintf(stderr, "loaded %s in %.1f s (%d threads)\n", opt.model.c_str(), load_s, opt.threads);
    // 给了请求清单时走批量模式,否则走单题模式
    if (!opt.requests.empty()) {
      RunBatch(pipe, opt, load_s);
    } else {
      RunSingle(pipe, opt);
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    // 打印错误原因
    std::fprintf(stderr, "error: %s\n", e.what());
    return EXIT_FAILURE;
  }
}
