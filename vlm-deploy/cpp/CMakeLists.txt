# vlm_infer_ov:Qwen2-VL-2B 的 C++ 推理程序,基于 OpenVINO GenAI,在 CPU 上运行。
#
# 依赖:
#   - OpenVINO 的头文件、库和 CMake 配置:取自 ovgenai 环境里 pip 安装的 openvino 包
#   - GenAI 的库 libopenvino_genai.so:取自 pip 安装的 openvino-genai 包
#   - GenAI 的头文件:pip 包里没有,由 envs/fetch_genai_headers.sh 按已装库的提交号下载到
#     $VLM_ARTIFACTS/genai-include,对应 make genai-headers
#   - 分词器库 libopenvino_tokenizers.so:取自 pip 安装的 openvino-tokenizers 包,运行时加载
#   - OpenCV:系统的 libopencv-dev,用于读图
#
# 构建(WSL,仓库根目录):
#   make cpp-cpu
# 等价于下面两条命令,四个路径由 Makefile 从 ovgenai 环境里查出:
#   cmake -S cpp -B cpp/build -DOpenVINO_DIR=<site-packages>/openvino/cmake \
#         -DGENAI_INCLUDE_DIR=<$VLM_ARTIFACTS/genai-include> \
#         -DGENAI_LIBRARY=<site-packages>/openvino_genai/libopenvino_genai.so.2641 \
#         -DGENAI_TOKENIZERS=<site-packages>/openvino_tokenizers/lib/libopenvino_tokenizers.so
#   cmake --build cpp/build -j4
#
# 运行时库的位置用 RPATH 写进了可执行文件,运行时不用设 LD_LIBRARY_PATH。
# 可执行文件因此绑定本机 ovgenai 环境的路径,环境换了位置或升级了 OpenVINO 后要重新构建。

# CMake 最低版本和项目名
cmake_minimum_required(VERSION 3.18)
project(vlm_infer_ov LANGUAGES CXX)

# 使用 C++17,不启用编译器扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # 不指定时按 Release 构建(开优化)
endif()

# OpenVINO 的 CMake 配置目录由 -DOpenVINO_DIR 给出
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
# OpenCV 只用读图和颜色转换相关的模块
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# GenAI 的 pip 包不带 CMake 配置,三个路径用缓存变量给出,可在命令行用 -D 覆盖
set(GENAI_INCLUDE_DIR "" CACHE PATH "GenAI 头文件目录(里面是 openvino/genai/...)")
set(GENAI_LIBRARY "" CACHE FILEPATH "libopenvino_genai.so 的完整路径")
set(GENAI_TOKENIZERS "" CACHE FILEPATH "libopenvino_tokenizers.so 的完整路径")
# 逐个检查三个路径,不存在时在配置阶段报错并给出处理办法
if(NOT EXISTS "${GENAI_INCLUDE_DIR}/openvino/genai/visual_language/pipeline.hpp")
  message(FATAL_ERROR "GENAI_INCLUDE_DIR 里没有 GenAI 头文件(${GENAI_INCLUDE_DIR});先运行 make genai-headers")
endif()
if(NOT EXISTS "${GENAI_LIBRARY}")
  message(FATAL_ERROR "找不到 GenAI 的库(${GENAI_LIBRARY});用 -DGENAI_LIBRARY=<libopenvino_genai.so> 指定")
endif()
if(NOT EXISTS "${GENAI_TOKENIZERS}")
  message(FATAL_ERROR "找不到分词器库(${GENAI_TOKENIZERS});用 -DGENAI_TOKENIZERS=<libopenvino_tokenizers.so> 指定")
endif()

# 可执行文件
add_executable(vlm_infer_ov vlm_infer_ov.cpp)
# 头文件目录。SYSTEM 表示第三方头文件,其自身的告警不计入本项目的告警输出。
target_include_directories(vlm_infer_ov SYSTEM PRIVATE ${GENAI_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
# 链接 GenAI、OpenVINO 和 OpenCV。openvino::runtime 是导入目标,自带头文件路径和编译宏。
# GenAI 的库按完整路径链接,CMake 会把库所在的目录写进可执行文件的 RPATH。
target_link_libraries(vlm_infer_ov PRIVATE ${GENAI_LIBRARY} openvino::runtime ${OpenCV_LIBS})
# 打开常用告警
target_compile_options(vlm_infer_ov PRIVATE -Wall -Wextra)
# 运行时库搜索路径写成 DT_RPATH,不用默认的 DT_RUNPATH。
# RUNPATH 只用于查找直接依赖的库,RPATH 还用于查找间接依赖的库。
# GenAI 的库依赖 TBB 且自身没有写搜索路径,用 RUNPATH 时会加载系统的 TBB,
# 而 Python 侧用的是 openvino 包自带的 TBB。线程库不同则计时不可比,所以改用 RPATH。
# 构建后可用 ldd 确认 libtbb 的位置。
target_link_options(vlm_infer_ov PRIVATE -Wl,--disable-new-dtags)
# 把分词器库的路径作为宏写进程序,main 用它设置环境变量 OPENVINO_TOKENIZERS_PATH_GENAI
target_compile_definitions(vlm_infer_ov PRIVATE VLM_TOKENIZERS_LIB="${GENAI_TOKENIZERS}")
