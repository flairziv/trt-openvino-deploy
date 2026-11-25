# trt-openvino-deploy

同一台笔记本上的两个模型部署研究:一个纯 ViT,一个 VLM。两个项目用同一套方法,GPU 栈是 TensorRT / TensorRT-LLM,
CPU 栈是 OpenVINO + NNCF,结论可以并排比较。

| 项目 | 模型 | GPU 栈 | CPU 栈 |
|---|---|---|---|
| [depth-anything-deploy](depth-anything-deploy) | Depth Anything V2-Small(ViT-S/14 + DPT 头,约 25M 参数) | TensorRT 10.16 | OpenVINO 2026.4 + NNCF 3.4 |
| [vlm-deploy](vlm-deploy) | Qwen2-VL-2B-Instruct(视觉塔 0.67B + 语言模型 1.31B) | TensorRT-LLM 1.3.0rc18(PyTorch 后端)+ ModelOpt 0.37 | OpenVINO GenAI 2026.4.1 + NNCF 3.4 |

`vlm-deploy` 是 `depth-anything-deploy` 的续作:前者部署的纯 ViT,在后者里变成了 VLM 的视觉塔,外加一个语言模型。

> 数字只代表这台笔记本(i9-14900HX + RTX 4060 Laptop 8 GB,WSL2 分到 4 个 vCPU、8 GB 内存):GPU 动态频率,
> CPU 无法绑定物理核。方法可以迁移到 Jetson Orin 等端侧平台,数字不能。

---

## 共用的方法

- **决策只信真实引擎**:量化配方的选择都在真实的 TensorRT 引擎 / OpenVINO 模型上评测,不用仿真代替。
- **数据划分互不相交**:校准集只用于校准,决策集用于选配方,测试集只报最终数字。
- **交错计时**:各变体在同一轮里乱序各测一次,多轮后报 P50 / P90 和轮间极差,抵消笔记本的频率漂移。
- **配对比较**(vlm-deploy):精度差按同一道题上的分差计算,并给出 95% 置信区间。
- **结果表由脚本生成**:README 里的表格只从 `results/*.json` 生成,不手写。
- **C++ 与 Python 对拍**:C++ 程序的输出要与评测用的 Python 路径核对一致后,数字才引用。

---

## 主要结论

完整的结果表、实验设计和踩过的坑见各项目的 README,这里只列每个项目最主要的几条。

### depth-anything-deploy

| 问题 | 结论 |
|---|---|
| GPU 上选哪个配方 | FP16。精度与 FP32 在 654 张测试图上无法分辨,推理比 TF32 快约 5.8×(518 档) |
| INT8 为什么不比 FP16 快 | 线性层和卷积变快了,但融合注意力从 1.55 ms 涨到 2.51 ms,总时间打平;INT8 / FP8 买到的是体积,不是速度 |
| ViT 的 INT8 敏感点 | 残差流。只把残差流留在 FP16 就能收回大部分精度 |
| CPU 上的 INT8 | 推理快 2.0×(518 档)/ 2.25×(364 档),精度损失比 GPU INT8 小一个量级 |
| 仿真能否代替真实引擎 | 不能。同一张 INT8 图,ONNXRuntime 仿真与真实引擎的输出偏差差几个 dB,逐图排序也对不上 |
| C++ 端到端 | GPU 栈 6.57 ms / 152 FPS(518 档 FP16),同一份输入下与 Python 输出逐位相同 |

### vlm-deploy

| 问题 | 结论 |
|---|---|
| 首字延迟花在哪 | 视觉塔。GPU 上占 65–78%,CPU 上占 67% |
| 量化语言模型帮哪一段 | 帮 decode。FP8:70 → 104 tok/s,精度 −0.21 ± 0.44;AWQ-INT4:146 tok/s,精度 −1.53 ± 0.70。首字延迟几乎不动 |
| 校准要不要带图 | 差别在噪声内,图文校准更省时间 |
| 怎么降首字延迟 | 把视觉塔换成 TensorRT FP16 引擎:290 → 177 ms(1.64×),精度 +0.18 ± 0.29(5000 题) |
| CPU 上的杠杆 | 语言模型 INT4 + 视觉塔 INT8 仅权重:首字延迟 9.77 → 4.37 s,decode 11.5 → 34.2 tok/s,体积约 4.9 → 2.1 GB |
| C++ 程序 | 只有 CPU 栈有(OpenVINO GenAI);回答与 Python 逐字相同。GPU 栈的语言模型没有建引擎,所以没有 C++ 程序 |

---

## 目录结构

```
depth-anything-deploy/   Depth Anything V2-Small:TensorRT + OpenVINO
  dadeploy/              两个环境共用的库(配置、数据、预处理、各后端的 Runner、评测)
  scripts/               流水线 00–13:导出、量化、敏感度、建引擎、评测、计时、报告
  cpp/                   两栈的 C++ 程序(da_infer、da_infer_ov)
  docs/guide/            分章讲解
  results/               机器可读结果,README 表格的来源
vlm-deploy/              Qwen2-VL-2B:TensorRT-LLM + OpenVINO GenAI
  vlmdeploy/             共用库(配置、图像预算、题目、计时、各后端)
  scripts/               流水线:下载、冒烟、评测、计时、量化、压缩、视觉塔引擎、演示、报告
  cpp/                   CPU 栈的 C++ 程序(vlm_infer_ov)
  docs/                  部署记录、逐文件代码导读
  results/               机器可读结果
```

---

## 怎么运行

两个项目各自独立,入口都是项目目录下的 `Makefile`,全部在 WSL 中运行。环境搭建、数据下载和单步命令见各自的 README:

- [depth-anything-deploy/README.md](depth-anything-deploy/README.md)(conda 环境 `dl` / `ov`)
- [vlm-deploy/README.md](vlm-deploy/README.md)(conda 环境 `trtllm` / `ovgenai`)

每个项目都用两个互相隔离的 conda 环境:GPU 栈和 CPU 栈锁定的依赖版本互相冲突,不能装在一起。

仓库不包含模型权重、数据集,以及导出和量化的产物(ONNX、引擎、OpenVINO IR、量化 checkpoint)。
这些都按各项目 README 的步骤下载或重新生成;引擎绑定 GPU 型号和 TensorRT 版本,换机器必须重建。

想读懂原理和代码的话:

- `depth-anything-deploy` 有分章的讲解 [docs/guide](depth-anything-deploy/docs/guide/README.md)(模型、导出、评测、TensorRT、INT8 敏感度、OpenVINO、计时)。
- `vlm-deploy` 有逐文件的代码导读 [docs/code-walkthrough.md](vlm-deploy/docs/code-walkthrough.md)
  和部署记录 [docs/deployment-record.md](vlm-deploy/docs/deployment-record.md)。
- 两个项目的源码都带逐行注释。

---

## 许可

- 本仓库代码:MIT,见 [LICENSE](LICENSE)。
- 模型:Depth Anything V2-Small 与 Qwen2-VL-2B-Instruct 均为 Apache-2.0,权重不随仓库分发。
- 数据:NYU Depth V2、TextVQA(CC BY 4.0)、RealWorldQA(CC BY-ND 4.0),仅用于评测,不随仓库分发。
