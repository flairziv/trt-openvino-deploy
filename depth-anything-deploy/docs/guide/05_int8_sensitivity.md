# 05 INT8 敏感度

项目的核心。对应 `dadeploy/groups.py`、`scripts/04_activation_stats.py`、`scripts/06_sensitivity.py`、`tools/probe_fc2_anomaly.py`。

## Q/DQ 放在哪

modelopt 在 518 档的图里插了 249 对 Q/DQ:105 对在权重上(104 对是每个输出通道一个 scale),144 对在激活上(每个张量一个 scale)。

`groups.find_qdq_pairs` 把每一对归到一个分组,依据是"DQ 的输出喂给谁":为了谁量化,就归谁。共 64 个分组,按子层类型汇总:

| 子层类型 | 分组数 | 激活 Q/DQ | 权重 Q/DQ |
|---|---|---|---|
| patch_embed | 1 | 1 | 1 |
| qkv | 12 | 12 | 36 |
| o_proj | 12 | 12 | 12 |
| fc1 | 12 | 12 | 12 |
| fc2 | 12 | 12 | 12 |
| residual | 12 | 48 | 0 |
| neck.reassemble | 1 | 11 | 11 |
| neck.fusion | 1 | 32 | 18 |
| head | 1 | 4 | 3 |

注意力内部的两个矩阵乘在 INT8 模式下不量化(24 个里 0 个)。

**residual 这一类最特别**:它不带权重,量化的是残差流本身。每个 block 4 处:

| 被量化的张量 | 喂给谁 |
|---|---|
| 上一个 block 的输出 | norm1 |
| ×γ1 的输出(注意力分支) | 第一个残差相加 |
| 第一个残差相加的输出 | norm2 |
| ×γ2 的输出(MLP 分支) | 第二个残差相加 |

跳连那一路本身没有经过 Q/DQ(block 0 的第二个相加例外)。被量化的是 LayerNorm 读到的残差流,和两条分支写回之前的输出。

modelopt 这么做是为了让 TensorRT 把"线性层 + 残差相加"融成一个 INT8 kernel。对 CNN 的 Conv + Add 这是合理的,对 ViT 就是问题所在。

## 残差流为什么难量化

`scripts/04_activation_stats.py` 在 PyTorch FP32 模型上挂 hook,对每类被量化的张量统计(518 档,calib 前 16 张):

| 张量 | 通道离群比 中位 | 单张量 INT8 SQNR(max 校准) | SQNR(99.99% 截断) |
|---|---|---|---|
| q/k/v 输入(norm1 输出) | 4.7 | 30.9 | 32.5 |
| o_proj 输入 | 3.4 | 33.4 | 34.6 |
| fc1 输入(norm2 输出) | 4.4 | 30.5 | 33.7 |
| fc2 输入(GELU 输出) | 7.6 | 23.3 | 26.3 |
| 残差流(注意力相加后) | 6.9 | 26.4 | 30.2 |
| 残差流(block 输出) | 11.1 | 22.9 | 27.9 |

- "通道离群比" = 各通道 absmax 的最大值 / 中位数。
- block 输出处的残差流离群最重,而且集中在固定通道:278 号通道在全部 12 个 block 里都排前三。
- LayerNorm 的输出(前三行)被归一化过,好量化得多。

激活量化每个张量只有一个 scale,要么迁就那几个大通道、牺牲其余通道的分辨率,要么截断它们。

一个具体例子:第 10 个 block 的输出(`layer.9/Add_1`)在图里的 scale 是 0.107,量程约 ±13.6;而这个张量 99.99% 的数值在 ±1.87 以内,只用到 ±127 格里的约 ±17 格。

## 为什么必须在真实引擎上做

最便宜的办法是用 ONNX Runtime 跑 Q/DQ 图做仿真,不用构建引擎。项目试过,不行:

- 同一张 INT8 图,ORT 仿真的 SQNR 约 15 dB(val 前 8 张),真实引擎是 21.5 dB(val 100 张);README 坑 4 还记录逐图排序基本不相关。
- ORT 曾静默退回 CPU,整轮结果作废。`OrtRunner` 现在拒绝回退。
- fc2 异常(见下)只在真实引擎上出现。

所以 06 的每个实验都是:拆 Q/DQ → 强类型构建引擎 → val 100 张评测。

## 怎么"回退一层"

显式量化下没有"给某层指定精度"的开关,回退就是改图。`groups.bypass_pairs`:

```python
rewire = {p.target: p.source for p in pairs}     # DQ 输出名 → Q 输入名
drop = {p.q for p in pairs} | {p.dq for p in pairs}
kept = [n for n in new.graph.node if n.name not in drop]
for node in kept:
    for idx, name in enumerate(node.input):
        if name in rewire:
            node.input[idx] = rewire[name]
```

下游改读量化前的张量,Q/DQ 两个节点删掉。其余 Q/DQ 的 scale 不变、不重新校准,所以实验之间的差异只来自"这些位置量化与否"。

## 06 的流程

`scripts/06_sensitivity.py`,目标线是 val AbsRel ≤ FP32 + 0.001:

1. **基线和天花板**:全 INT8;拆掉全部 Q/DQ(等价 FP16)。
2. **类型级留一法**:每次只把一类子层(12 个 block 的同名位置)拆回 FP16。
3. **贪心累加**:按改善从大到小依次回退,直到达标。
4. **逐块回收**:在已回退的分组里,按 04 的统计从"最好量化"到"最难量化"逐个 block 试着改回 INT8,仍达标就保留。

`EngineBench.run` 把一次实验封装成一个函数调用,共用 timing cache,每档 23 次实验:518 档 432 秒,364 档 329 秒。

518 档的结果(val,FP32 = 0.0607,目标 ≤ 0.0617):

| 配置 | AbsRel | SQNR dB |
|---|---|---|
| 全部 INT8 | 0.0638 | 21.5 |
| 全部 FP16 | 0.0607 | 54.1 |
| 仅 residual 回退 | 0.0616 | 26.4 |
| 仅 patch_embed 回退 | 0.0634 | 21.2 |
| 仅 head 回退 | 0.0635 | 21.5 |
| 仅 fc1 回退 | 0.0635 | 21.8 |
| 仅 neck.fusion 回退 | 0.0635 | 21.5 |
| 仅 neck.reassemble 回退 | 0.0638 | 21.5 |
| 仅 qkv 回退 | 0.0641 | 21.3 |
| 仅 o_proj 回退 | 0.0646 | 21.0 |
| 仅 fc2 回退 | 0.2081 | 4.4 |

- 只回退 residual 一类就达标,贪心在第一步停下。
- 逐块回收后,518 档 10 个 block 的残差流留 FP16(第 7、10 个 block 改回 INT8),364 档 5 个。

## 两张产物图

- `da2s_<res>.int8res.onnx`:规则化配置,12 个 block 的残差流全部留 FP16。
- `da2s_<res>.int8mix.onnx`:数据驱动配置,逐块回收后的最终选择。

到测试集上两者几乎一样(518 档 Δ AbsRel +0.0012 对 +0.0013;364 档 +0.0016 对 +0.0020),而且都没达到 +0.001 的接受线。逐块回收那一步的差异接近 val 100 张的噪声,在 val 上省下的几个 FP16 block 在测试集上没有兑现。交付应选 int8res:规则更简单,不依赖某一份 val 集。

`06_sensitivity.py --export-only` 可以按已有的 `results/sensitivity_<res>.json` 重新导出这两张图,不重跑实验。

## fc2 异常

留一法里"仅 fc2 回退"不是变好,而是崩掉(AbsRel 0.2081)。`tools/probe_fc2_anomaly.py` 在 364 档、不用 timing cache 复现:

| 拆掉的 Q/DQ | ORT CUDA SQNR dB | TensorRT SQNR dB |
|---|---|---|
| fc2 的激活和权重 | 18.0 | 3.5 |
| 只拆 fc2 的激活 | 17.1 | 3.9 |
| 只拆 fc2 的权重 | 17.1 | 21.4 |
| fc1 和 fc2 都拆 | 17.0 | 22.7 |

触发条件是"INT8 的 fc1 + FP16 的 fc2 激活"这个组合;ORT 执行同一张图是正常的。根因没有证实。它说明了一件事:只看仿真,会把一个部署后输出崩坏的配置当成合格。
