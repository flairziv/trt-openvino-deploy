# 01 模型结构

Depth Anything V2-Small = DINOv2 ViT-S/14 主干 + DPT 解码器,共约 24.9M 参数:切 patch 和位置编码 0.92M,12 个 block 21.3M,DPT 颈 2.63M,头 0.03M。

类比 YOLO:ViT 主干相当于 backbone,DPT 的颈相当于 FPN,DPT 的头相当于 Detect 头。差别在三处:主干不下采样,金字塔是后造的,头里没有解码。

## 从图像到 token

```
pixel_values            [1, 3, 518, 686]
Conv k=14 s=14          [1, 384, 37, 49]     每个 14×14 的 patch 变成一个 384 维向量
Reshape + Transpose     [1, 1813, 384]       37×49 = 1813 个 token
Concat(CLS)             [1, 1814, 384]       前面拼一个可学习的 CLS token
Add(位置编码)           [1, 1814, 384]       加一个 [1, 1814, 384] 的常量
```

- 切 patch 只有这一个卷积,后面没有 MLP。它等价于"把每个 patch 的 14×14×3 个数展平后过一个线性层"。
- 出来的是特征向量,不是 logits。
- 位置编码和 token 数绑定,所以每档分辨率各有一份 ONNX 和引擎,输入高宽必须是 14 的倍数(`config.Resolution` 会检查)。

## 一个 block

12 个 block 结构相同,进出都是 `[1, 1814, 384]`,每个在 ONNX 里是 36 个节点:

```
x  ─ norm1 ─ q/k/v ─ Q·Kᵀ ─ softmax ─ ·V ─ o_proj ─ ×γ1 ─ (+ x)  = x'
x' ─ norm2 ─ fc1 ─ GELU ─ fc2 ─ ×γ2 ─ (+ x') = 下一个 block 的 x
```

| 步骤 | 输出形状 | 说明 |
|---|---|---|
| norm1(LayerNorm) | [1, 1814, 384] | 只归一化送进注意力的这一份,`x` 本身不变 |
| q / k / v 三个线性层 | [1, 1814, 384] | ONNX 里是 MatMul + Add |
| 拆成 6 个头 | [1, 6, 1814, 64] | Reshape + Transpose |
| Q·Kᵀ | [1, 6, 1814, 1814] | 每个头一张 1814×1814 的注意力表,计算量 ∝ token² |
| softmax,再乘 V | [1, 6, 1814, 64] | |
| 合并头 + o_proj | [1, 1814, 384] | |
| ×γ1(LayerScale)再加回 x | [1, 1814, 384] | γ 是逐通道的可学习乘子 |
| norm2 → fc1 | [1, 1814, 1536] | MLP 把通道放大 4 倍 |
| GELU → fc2 | [1, 1814, 384] | GELU 在 ONNX 里展开成 Div / Erf / Add / Mul / Mul |
| ×γ2 再加回 x' | [1, 1814, 384] | |

**残差流**就是图里的 `x`:一条 `[1, 1814, 384]` 的主路,每个 block 往上加两次增量。它自己从不被归一化,LayerNorm 只作用在分支的输入上。这条通路是 INT8 量化的头号问题,见 [05](05_int8_sensitivity.md)。

和 CNN 相比,部署时要记住的三点:

- 归一化是 LayerNorm,没有 BN。它在运行时对每个 token 的 384 个通道求均值方差,不能折进权重。
- 没有下采样,12 个 block 都是 1814 个 token,也就没有现成的 P3 / P4 / P5。
- 线性层在 ONNX 里是 3D 的 MatMul + Add,不是 Gemm。TensorRT 的注意力融合和 modelopt 的量化模式匹配都认这个形态。

## DPT:31 个卷积在哪

DPT(Dense Prediction Transformer)把 token 变回一张逐像素的图。

| 位置 | 卷积数 | 做什么 | 类比 YOLO |
|---|---|---|---|
| patch embedding | 1 | 切 patch | 主干的 stem |
| reassemble | 9,另有 2 个转置卷积 | 取第 3、6、9、12 个 block 的输出,各过一个 LayerNorm(四处共用同一组权重),去掉 CLS,摆回 37×49 的特征图,再缩放成四层 | 主干给出的多尺度特征 |
| fusion | 18 | 从最粗一层往上,逐级上采样、相加、过残差卷积 | FPN 自顶向下的一路 |
| head | 3 | 得到每个像素一个值 | Detect 头 |

reassemble 造出的四层:

| 抽头 | 1×1 投影后通道 | 缩放方式 | 特征图 |
|---|---|---|---|
| block 3 输出 | 48 | 转置卷积 k=4 s=4 | 148×196 |
| block 6 输出 | 96 | 转置卷积 k=2 s=2 | 74×98 |
| block 9 输出 | 192 | 不缩放 | 37×49 |
| block 12 输出 | 384 | 卷积 k=3 s=2 | 19×25 |

四层各过一个 3×3 卷积统一成 64 通道。fusion 用 4 个双线性 Resize 逐级放大到 296×392。头部是:3×3 卷积(64→32)→ Resize 到 518×686 → 3×3 卷积 + ReLU → 1×1 卷积(32→1)+ ReLU → 去掉通道维,输出 `[1, 518, 686]`。

## 和 YOLO 的差别

- **金字塔是造出来的。** YOLO 的主干逐级下采样,直接给出步长 8 / 16 / 32 的特征。这里四个抽头的步长都是 14,靠转置卷积和带步长的卷积造出四个尺度。
- **头里没有解码。** DFL 把每条边的 16 个 bin 做 softmax 求期望,再结合 anchor 和步长换算成框。DPT 的头直接给出每个像素的值,之后没有 NMS。
- **输出是相对视差。** 越大越近,没有绝对尺度,评测前要逐图对齐,见 [03](03_evaluation.md)。
- **耗时分布。** 颈和头参数很少,但卷积合计占 FP16 引擎剖析耗时的 20.8%,因为头部在 518×686 全分辨率上算卷积。

## 对应代码

- `dadeploy/model.py`:`DepthOnly` 把 HF 模型包成只返回 `predicted_depth` 的 Module;`disable_tf32` 保证 PyTorch 参考是严格 FP32。
- 模型权重在 `artifacts/hf/Depth-Anything-V2-Small-hf/`,`config.json` 里能看到 `out_indices = [3, 6, 9, 12]`、`neck_hidden_sizes = [48, 96, 192, 384]`。
- 想自己看图:用 Netron 打开 `artifacts/onnx/da2s_r518.onnx`,搜 `layer.0`。
