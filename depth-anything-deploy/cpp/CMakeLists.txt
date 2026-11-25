# 两个纯 C++ 端到端推理程序,共用同一份预处理 / 后处理(da_common.h):
#   da_infer     TensorRT(GPU)    da_infer.cpp    + trt_engine.h
#   da_infer_ov  OpenVINO(CPU)    da_infer_ov.cpp + ov_engine.h
# 每个程序只在它的依赖找得到时才生成,所以两个 conda 环境可以各建各的,互不要求对方的库:
# 只想建 CPU 版就不需要 TensorRT / CUDA,反过来也一样。
#
# 依赖:
#   - 两者共用:OpenCV(系统 libopencv-dev 即可)
#   - da_infer:TensorRT 头文件(TensorRT OSS 源码树的 include/,pip wheel 不带头文件)、
#     libnvinfer.so.10(pip 包 tensorrt_libs 提供)、CUDA Toolkit(cudart)
#   - da_infer_ov:OpenVINO 的头文件、库和 CMake 配置。pip 的 openvino 轮子里都带了
#     (site-packages/openvino 下的 include/、libs/、cmake/),不需要另装 OpenVINO 开发包
#
# 构建(WSL,仓库根目录):
#   make cpp          # da_infer,用 dl 环境里的 TensorRT 库,等价于:
#   cmake -S cpp -B cpp/build -DTRT_INCLUDE_DIR=<.../include> -DTRT_NVINFER=<.../libnvinfer.so.10>
#   make cpp-ov       # da_infer_ov,用 ov 环境里的 OpenVINO,等价于:
#   cmake -S cpp -B cpp/build-ov -DOpenVINO_DIR=<site-packages>/openvino/cmake
#   cmake --build <构建目录> -j
# 两个目标用两个构建目录(cpp/build 与 cpp/build-ov),互不覆盖对方的缓存。
#
# 运行时库的位置都用 RPATH 写进了可执行文件,直接运行即可,不用设 LD_LIBRARY_PATH。

# CMake 最低版本和项目名
cmake_minimum_required(VERSION 3.18)
project(da_infer LANGUAGES CXX)

# 使用 C++17,不启用编译器扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 没有指定构建类型时默认 Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # 基准必须开优化
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ---- da_infer(TensorRT)----
# 查找 TensorRT 的头文件目录和 libnvinfer,也可以在命令行用 -D 指定
find_path(TRT_INCLUDE_DIR NvInfer.h)
find_library(TRT_NVINFER NAMES nvinfer libnvinfer.so.10)
# 找到 TensorRT 时才生成 da_infer
if(TRT_INCLUDE_DIR AND TRT_NVINFER)
  # 查找 CUDA 运行库
  find_package(CUDAToolkit REQUIRED)
  # 可执行文件 da_infer
  add_executable(da_infer da_infer.cpp)
  # SYSTEM:第三方头文件自身的 deprecated 告警不混进本项目的 -Wall -Wextra 输出
  target_include_directories(da_infer SYSTEM PRIVATE ${TRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
  # 链接 TensorRT、CUDA 运行库和 OpenCV
  target_link_libraries(da_infer PRIVATE ${TRT_NVINFER} CUDA::cudart ${OpenCV_LIBS})
  # 打开常用告警
  target_compile_options(da_infer PRIVATE -Wall -Wextra)
  # 把 pip 包里 libnvinfer 所在目录写进 RPATH,运行时不用再设 LD_LIBRARY_PATH
  get_filename_component(TRT_LIB_DIR ${TRT_NVINFER} DIRECTORY)
  set_target_properties(da_infer PROPERTIES BUILD_RPATH ${TRT_LIB_DIR})
  # 标记已有可构建的目标
  set(DA_HAVE_TARGET ON)
else()
  # 没找到时提示该传哪些参数
  message(STATUS "TensorRT not found: skip da_infer (pass -DTRT_INCLUDE_DIR=<include> -DTRT_NVINFER=<libnvinfer.so.10>)")
endif()

# ---- da_infer_ov(OpenVINO)----
# 查找 OpenVINO,找不到时不报错,由文件末尾统一判断
# 位置由 -DOpenVINO_DIR 指定,取含 OpenVINOConfig.cmake 的目录
find_package(OpenVINO QUIET COMPONENTS Runtime)
# 找到 OpenVINO 时才生成 da_infer_ov
if(OpenVINO_FOUND)
  # 可执行文件 da_infer_ov
  add_executable(da_infer_ov da_infer_ov.cpp)
  # OpenCV 的头文件按系统头文件处理,其告警不计入本项目
  target_include_directories(da_infer_ov SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  # 链接 OpenVINO 的导入目标,头文件路径、库和编译宏随之带入
  target_link_libraries(da_infer_ov PRIVATE openvino::runtime ${OpenCV_LIBS})
  # 打开常用告警
  target_compile_options(da_infer_ov PRIVATE -Wall -Wextra)
  # 标记已有可构建的目标
  set(DA_HAVE_TARGET ON)
else()
  # 没找到时提示该传哪些参数
  message(STATUS "OpenVINO not found: skip da_infer_ov (pass -DOpenVINO_DIR=<site-packages>/openvino/cmake)")
endif()

# 两个程序的依赖都没找到时配置失败
if(NOT DA_HAVE_TARGET)
  message(FATAL_ERROR "TensorRT 和 OpenVINO 都没找到,没有可构建的目标;见本文件开头的构建说明")
endif()
