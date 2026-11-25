"""P3 GPU 量化:ModelOpt 把语言模型量化成 FP8(W8A8)或 AWQ-INT4(W4A16),校准数据为纯文本或图文,导出给 TensorRT-LLM。

用法(env: trtllm,带 PYTHONPATH=vlmdeploy/patches/site;Makefile 已封装):
    python scripts/12_quantize_gpu.py --method fp8 --calib img
    python scripts/12_quantize_gpu.py --method awq --calib txt

为什么只量化语言模型:
    TensorRT-LLM 对 Qwen2-VL 的视觉塔沿用 HF 实现,加载时会主动把量化配置从视觉塔上去掉
    (modeling_qwen2vl.py 里 "Re-setting QuantConfig to exclude vision encoder weights"),量化了也不会被用上。
    视觉塔的加速另走 P5(条件触发)。lm_head 也保持 BF16:它与输入 embedding 共享权重(tie_word_embeddings),
    且 ModelOpt 默认配置就排除 lm_head。

两种方法各自省的是什么(跑之前写下的预测,见 README "要回答的问题" 第 2 条):
    FP8(W8A8,per-tensor):权重与激活都是 FP8,Ada 的 FP8 tensor core 直接算 —— prefill(算力受限)和
        decode(带宽受限)都可能变快;激活量化需要校准激活范围(amax),对校准数据敏感。
    AWQ-INT4(W4A16,group 128):只压权重,计算时反量化回 BF16 —— 只省 decode 的带宽,prefill 不变;
        校准数据用来找"按激活幅度保护重要通道"的缩放,同样对校准数据敏感。

两种校准数据(问题 3:"校准要不要带图"):
    img = calib 划分的题目 + 图片(预缩放到 t576),语言模型看到的是 576 个图像 token + 文字 token;
    txt = 同一批题目、去掉图片,只剩文字。两者唯一的差别就是"有没有图像 token"。
    已知混杂:txt 每个样本只有几十个 token,总 token 数比 img 少一个数量级。FP8 的 amax 取最大值,
    样本少只会让 amax 偏小(更容易截断图像 token 的离群值)—— 这正是要检验的机制,结论里会说明。

产物:
    $VLM_ARTIFACTS/trtllm/<method>-<calib>/     HF 格式的量化 checkpoint(含 hf_quant_config.json)+ 处理器 / 分词器文件
    results/quantize/<method>-<calib>.json      量化记录(配置、校准样本数与 token 数、耗时、体积)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 把仓库根目录加入模块搜索路径,保证能导入 vlmdeploy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlmdeploy.config import MODEL_DIR, RESULTS_DIR, TOKEN_BUDGETS, TRTLLM_CKPT_DIR
from vlmdeploy.imaging import resize_to_budget
from vlmdeploy.logging_ import get_logger
from vlmdeploy.smoke import chat_prompt
from vlmdeploy.tasks import iter_split

# 模块级 logger
log = get_logger("quantize_gpu")
# 图文校准时图片缩放到的档位
CALIB_BUDGET = "t576"
# 各校准方式使用的题数,None 表示全部
NUM_CALIB = {"img": 128, "txt": None}   # txt 用全部 calib 题(约 430 题)以部分弥补 token 数的差距
# 处理器 / 分词器文件:save_pretrained 只写配置与权重,TensorRT-LLM 的 LLM API 还要从模型目录读这些
AUX_FILES = ("preprocessor_config.json", "chat_template.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "generation_config.json")


def quant_config(method: str) -> dict:
    """ModelOpt 量化配置:在默认配置上关掉视觉塔的全部量化器。"""
    # modelopt 只在这里导入
    import modelopt.torch.quantization as mtq

    # 深拷贝:下面要往配置里加条目,不能改到 ModelOpt 库里的默认配置对象
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG if method == "fp8" else mtq.INT4_AWQ_CFG)
    # 通配符按模块名匹配;Qwen2-VL 的视觉塔模块名里都有 "visual"(transformers 5.x 为 model.visual.*)
    cfg["quant_cfg"]["*visual*"] = {"enable": False}
    return cfg


def restore_key_layout(out_dir: Path) -> int:
    """把导出 checkpoint 的视觉塔键名改回原始布局,并核对原始 checkpoint 的每个键都在。返回改名的键数。

    为什么要改:transformers 5.x 加载旧格式 checkpoint 时把 `visual.*` 映射成 `model.visual.*`,保存时语言模型那一半
    映射回了 `model.layers.*`,视觉塔却没有映射回去。TensorRT-LLM 的视觉塔加载器按原始布局取 `visual.` 前缀
    (`filter_weights("visual", ...)`),于是报 "Missing key(s) ... patch_embed.proj.weight"。

    做法:safetensors 文件 = 8 字节头长度 + JSON 头 + 数据区,JSON 头里的偏移量相对数据区起点,
    所以只改头里的键名、数据区原样流式复制即可,不需要把 3 GB 权重读进内存(WSL 只有 8 GB)。
    """
    import struct

    # 导出的权重文件
    src = out_dir / "model.safetensors"
    # 以二进制方式读取
    with src.open("rb") as f:
        # 前 8 字节是 JSON 头的长度
        (n,) = struct.unpack("<Q", f.read(8))
        # 读出并解析 JSON 头
        header = json.loads(f.read(n))
        # 把 model.visual. 前缀改回 visual.,其余键不变
        renamed = {(k.replace("model.visual.", "visual.", 1) if k.startswith("model.visual.") else k): v
                   for k, v in header.items()}
        # 原始 checkpoint 的全部键名
        expected = set(json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"])
        # 改名后仍然缺少的键
        missing = expected - set(renamed)
        # 有缺失时报错
        if missing:
            raise RuntimeError(f"exported checkpoint lacks {len(missing)} original keys, e.g. {sorted(missing)[:3]}")
        # 序列化新的 JSON 头,不带多余空格
        blob = json.dumps(renamed, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)   # 与 safetensors 自己的序列化一致:头部按 8 字节对齐(用空格填充)
        # 先写到临时文件
        tmp = out_dir / "model.safetensors.tmp"
        with tmp.open("wb") as g:
            # 写出头长度
            g.write(struct.pack("<Q", len(blob)))
            # 写出 JSON 头
            g.write(blob)
            # 数据区原样流式复制,每次 64 MB
            shutil.copyfileobj(f, g, length=64 << 20)
    os.replace(tmp, src)  # 新文件完整写好之后才原子替换原文件
    # 量化配置里的排除列表同样带着错误前缀("model.layers.visual*",由 ModelOpt 的层池命名拼出);
    # TensorRT-LLM 加载 Qwen2-VL 时会无条件去掉视觉塔的量化配置,这里改正只为了配置文件本身表意准确。
    for name, key in (("hf_quant_config.json", ("quantization", "exclude_modules")),
                      ("config.json", ("quantization_config", "ignore"))):
        # 配置文件路径
        path = out_dir / name
        # 读入配置
        cfg = json.loads(path.read_text())
        # 排除列表所在的节点
        node = cfg[key[0]]
        # 把错误的前缀改成 visual*
        node[key[1]] = ["visual*" if m == "model.layers.visual*" else m for m in node.get(key[1], [])]
        # 写回
        path.write_text(json.dumps(cfg, indent=4))
    # 返回改名的键数
    return sum(1 for k in header if k.startswith("model.visual."))


def calib_inputs(processor, mode: str) -> list[dict]:  # noqa: ANN001
    """把 calib 划分变成模型输入(张量留在 CPU,前向时再搬到 GPU,避免 128 份像素张量同时占显存)。"""
    # 各样本的模型输入
    inputs = []
    # 逐题读取 calib 划分
    for sample in iter_split("calib", limit=NUM_CALIB[mode]):
        if mode == "img":
            # 图文:图片预缩放到校准档位,提示词用各后端共用的 chat_prompt(图在前、文字在后)
            img = resize_to_budget(sample.image(), TOKEN_BUDGETS[CALIB_BUDGET])
            enc = processor(text=[chat_prompt(processor, sample.prompt)], images=[img], return_tensors="pt")
        else:
            # 纯文本:同一道题、同一个聊天模板,只是 content 里不放图片项
            messages = [{"role": "user", "content": [{"type": "text", "text": sample.prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = processor(text=[text], return_tensors="pt")
        # 转成普通字典后保存
        inputs.append(dict(enc))
    return inputs


def enabled_quantizers(model) -> tuple[list[str], list[str]]:  # noqa: ANN001 —— torch.nn.Module
    """量化之后的自检:列出启用的量化器,找出落在不该量化的位置上的。

    按类型找(TensorQuantizer),不按名字后缀:FP8 是 W8A8,除了 weight_quantizer 还有激活侧的
    input_quantizer,只查前者的话,激活量化器漏进视觉塔时发现不了。

    Returns:
        (启用的权重量化器名, 落在视觉塔或 lm_head 上的启用量化器名)。后者非空说明排除规则有漏。
    """
    # modelopt 只在这里导入
    from modelopt.torch.quantization.nn import TensorQuantizer

    # 全部启用的量化器的模块名
    enabled = [n for n, mod in model.named_modules() if isinstance(mod, TensorQuantizer) and mod.is_enabled]
    # 其中的权重量化器
    weights = [n for n in enabled if n.endswith("weight_quantizer")]
    # 落在视觉塔或 lm_head 上的量化器
    leaked = [n for n in enabled if "visual" in n or "lm_head" in n]
    return weights, leaked


def main() -> int:
    """入口:量化语言模型并导出 TensorRT-LLM 可加载的 checkpoint。"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="P3 ModelOpt 量化语言模型")
    # 量化方法,必填
    parser.add_argument("--method", choices=("fp8", "awq"), required=True)
    # 校准数据,必填
    parser.add_argument("--calib", choices=("img", "txt"), required=True)
    # 已有产物时也重新量化
    parser.add_argument("--force", action="store_true", help="已有产物也重新量化")
    # 解析参数
    args = parser.parse_args()
    # 产物名:方法加校准数据
    name = f"{args.method}-{args.calib}"
    # 量化 checkpoint 的输出目录
    out_dir = TRTLLM_CKPT_DIR / name
    # 量化记录文件
    record_path = RESULTS_DIR / "quantize" / f"{name}.json"
    # 记录和权重都已存在且没有指定 --force 时跳过
    if record_path.exists() and (out_dir / "model.safetensors").exists() and not args.force:
        log.info("%s already exported (%s); use --force to redo", name, out_dir)
        return 0

    # 重依赖在确定要量化之后才导入
    import modelopt.torch.quantization as mtq
    import torch
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    # 加载 BF16 模型到 GPU,切到推理模式
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to("cuda").eval()
    # 准备校准输入
    data = calib_inputs(processor, args.calib)
    # token 总数写进量化记录:txt 校准的总 token 数比 img 少一个数量级(文件头的"已知混杂"),留下数字备查
    n_tokens = sum(int(d["input_ids"].shape[1]) for d in data)
    # 打印校准样本数和 token 总数
    log.info("calibration: %d samples, %d tokens (%s)", len(data), n_tokens, args.calib)

    # ModelOpt 的校准回调:mtq.quantize 把插好量化器的模型传进来,这里把每条校准样本搬到 GPU 跑一遍前向
    def forward_loop(m) -> None:  # noqa: ANN001
        # 关闭梯度记录
        with torch.no_grad():
            for d in data:
                m(**{k: v.to("cuda") for k, v in d.items()})

    # 量化计时起点
    t0 = time.time()
    # 插入量化器并校准
    model = mtq.quantize(model, quant_config(args.method), forward_loop)
    # 量化耗时
    quant_s = time.time() - t0
    # 核对:量化器只出现在语言模型上(权重侧、激活侧都查)
    enabled, leaked = enabled_quantizers(model)
    # 有量化器落在不该量化的位置时报错
    if leaked:
        raise RuntimeError(f"quantizers leaked into vision tower or lm_head: {leaked[:5]} ({len(leaked)} in total)")
    # 打印量化耗时和量化的线性层数
    log.info("quantized in %.0fs; %d linear layers quantized", quant_s, len(enabled))

    # 输出目录已存在时先删除
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # 导出计时起点
    t0 = time.time()
    # 导出 HF 格式的量化 checkpoint
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=out_dir)
    # 补齐处理器和分词器文件,已有的不覆盖
    for f in AUX_FILES:
        if (MODEL_DIR / f).exists() and not (out_dir / f).exists():
            shutil.copy2(MODEL_DIR / f, out_dir / f)
    # 把视觉塔的键名改回原始布局
    n_renamed = restore_key_layout(out_dir)
    # 打印改名的张量数
    log.info("restored original key layout for %d vision-tower tensors", n_renamed)
    # 权重文件的总字节数
    size = sum(p.stat().st_size for p in out_dir.glob("*.safetensors"))
    # 量化记录:方法、校准数据、样本数、token 数、层数、耗时、目录、体积、时间和量化配置
    record = {"name": name, "method": args.method, "calib": args.calib, "calib_samples": len(data),
              "calib_tokens": n_tokens, "calib_budget": CALIB_BUDGET if args.calib == "img" else None,
              "quantized_linears": len(enabled), "quantize_seconds": round(quant_s, 1),
              "export_seconds": round(time.time() - t0, 1), "checkpoint": str(out_dir),
              "weights_bytes": size, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "hf_quant_config": json.loads((out_dir / "hf_quant_config.json").read_text())}
    # 记录目录
    rec_dir = RESULTS_DIR / "quantize"
    # 创建记录目录
    rec_dir.mkdir(parents=True, exist_ok=True)
    # 写出量化记录
    (rec_dir / f"{name}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    # 打印输出目录和权重体积
    log.info("exported %s (%.2f GB weights)", out_dir, size / 1e9)
    return 0


# 脚本入口,main 的返回值作为进程退出码
if __name__ == "__main__":
    sys.exit(main())
