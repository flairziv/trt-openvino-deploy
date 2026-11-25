# 导读:从 YOLO 部署经验读懂这个 ViT 项目

这组文档写给做过 YOLO 在 TensorRT / OpenVINO 上部署优化、但没碰过 ViT 的读者。平台基础(引擎、IR、校准、基准测试)不重讲,只讲 ViT 带来的不同,以及每个结论在仓库里的出处。

## 阅读顺序

| 篇 | 内容 | 对应代码 |
|---|---|---|
| [01 模型结构](01_model.md) | 一张图怎样变成 token,一个 block 的 36 个节点,DPT 颈和头,与 YOLO 的对照 | `dadeploy/model.py` |
| [02 导出](02_export.md) | 固定形状、位置编码折叠、节点名为什么是接口 | `scripts/02_export_onnx.py` |
| [03 评测口径](03_evaluation.md) | 视差域对齐、配对 bootstrap、SQNR、三份数据各管什么 | `dadeploy/metrics.py`、`evaluation.py`、`data.py` |
| [04 TensorRT 构建](04_tensorrt.md) | 精度写在 ONNX 里、强类型构建、看引擎里实际跑的精度 | `dadeploy/trt_recipes.py`、`trt_utils.py`、`scripts/05`、`07` |
| [05 INT8 敏感度](05_int8_sensitivity.md) | Q/DQ 放在哪、残差流为什么难量化、真实引擎上的消融 | `dadeploy/groups.py`、`scripts/04`、`06` |
| [06 OpenVINO 对照](06_openvino.md) | NNCF 的 Transformer 模板、SmoothQuant、accuracy control | `scripts/08_quantize_openvino.py` |
| [07 推理、计时与 C++](07_inference_timing.md) | Runner、交错轮次基准、逐 kernel 剖析、C++ runtime | `dadeploy/runners.py`、`timing.py`、`scripts/10`、`11`、`13`、`cpp/` |
| [08 练习路线](08_practice.md) | 七步从零重写,每步有数字可对答案 | — |
| [09 没有闭环的地方](09_open_questions.md) | 项目没查清的问题和代码里的小瑕疵 | — |

只想知道结论,读仓库根目录 README 的"结论速览"就够。想自己动手,读完 01 后直接看 08,遇到不懂的再回查对应篇。

## 三个先建立的直觉

- **计算量在矩阵乘,不在卷积。** 折叠后的图有 518 个节点,432 个属于 12 个结构相同的 block;卷积只有 31 个。
- **"分辨率"变成"token 数"。** 518×686 按 14×14 切成 1813 个 token。降到 364×476(884 个 token)后,FP16 推理从 5.05 ms 降到 2.59 ms。
- **敏感点是一条数据通路,不是一个算子。** YOLOv8 上回退 DFL 解码链里的一个节点就收回大部分精度;这里要处理的是贯穿 12 个 block 的残差流。

## 关于文中的数字

- 数字取自 2026-11-08 那一轮 `results/*.json` 和 `artifacts/` 里的产物,只代表这台笔记本(RTX 4060 Laptop、WSL2 4 vCPU)。
- 文中的数字是手写的快照,重跑后不会自动更新。与根目录 README 的表格不一致时,以表格为准。
- 没有特别说明时,结构和形状指 518 档(`r518`)。
- 代码位置用"文件 + 函数名"指代,不写行号。
