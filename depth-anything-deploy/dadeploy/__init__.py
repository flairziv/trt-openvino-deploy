"""dadeploy —— Depth Anything V2-Small 双栈部署工具包。

本包把"同一个 ViT 深度模型在两条推理栈上的精度 / 速度对照"所需的公共件集中在一处:

    config      全局常量:路径、分辨率档位、评测协议参数(单一事实来源)
    logging_    统一日志格式
    data        NYUv2 测试集 / 校准集读取
    preprocess  预处理与后处理(所有后端共用同一份实现,保证对照公平)
    metrics     尺度-平移对齐 + AbsRel / δ1 / RMSE,以及量化输出偏差指标
    variants    "后端:分辨率:精度" 变体字符串 → 产物路径 / Runner 的注册表
    runners     PyTorch / ONNX Runtime / TensorRT / OpenVINO 四种 Runner
    timing      分段计时与交错轮次统计

运行环境约束:
    TensorRT 相关代码只在 conda env `dl` 中可导入;OpenVINO 相关代码只在 env `ov` 中可导入。
    因此 runners 内各后端均为**惰性导入**,公共模块(data / preprocess / metrics)只依赖 numpy / cv2 / h5py。
"""

# 包版本号,与 pyproject.toml 的 version 一致
__version__ = "1.0.0"
