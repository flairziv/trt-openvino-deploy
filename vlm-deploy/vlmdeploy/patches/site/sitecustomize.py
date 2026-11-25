"""本项目 GPU 命令专用的 sitecustomize:修补 TensorRT-LLM 1.3.0rc18 跑 Qwen2-VL 的三处兼容问题。

三处问题都出在 Qwen2-VL 这条"沿用 HF 实现"的路径上(源码注释原话:Qwen2-VL 过时了,保留 HF 实现;
Qwen2.5-VL 是 TensorRT-LLM 自己重写的),而且都发生在 tensorrt_llm 1.3.0rc18 自己在依赖里锁死的
transformers 5.5.4 下 —— 也就是说,官方版本组合下这条路径直接跑不起来:

1. flash-attn 2 缺失 —— tensorrt_llm/_torch/models/modeling_qwen2vl.py 的 Qwen2VisionModelBase.__init__
   无条件把视觉塔的 _attn_implementation 设成 "flash_attention_2"(注释:为了避免 OOM),再构造 HF 的
   Qwen2VisionTransformerPretrainedModel;transformers 构造时检查 flash_attn 包,没装就 ImportError。
   flash_attn 2 不在 TensorRT-LLM 的 pip 依赖里(NVIDIA 容器镜像里预装,原生 pip 安装没有),
   在 4 核 8 GB 的 WSL 里从源码编译要几个小时。
   修补:包一层 HF 视觉塔类的 __init__,flash_attn 不可用时把要求的实现改成 "sdpa"。
   SDPA 是 PyTorch 自带算子,单张图(无块对角掩码)时同样走 FlashAttention 类内核;"避免 OOM"防的是
   eager 实现显式构造 N×N 注意力矩阵,SDPA 没有这个问题。装了 flash-attn 2 的环境里不做任何改动。

2. 视觉塔返回值类型 —— transformers 5.x 把 Qwen2VisionTransformerPretrainedModel.forward 的返回值从
   "合并后的 embedding 张量"改成了 BaseModelOutputWithPooling(last_hidden_state=合并前, pooler_output=合并后)。
   TensorRT-LLM 的 Qwen2VisionModelBase.forward 仍把返回值当张量放进 embedding 列表,下游 len() 得到的是
   字段数 2,启动预热时报 "Number of mm_embeds (2) does not match expected total (...)"。
   修补:包一层 Qwen2VisionModelBase.forward,把这类对象换成 pooler_output(即 transformers 4.x 直接返回的张量)。

3. 输出层权重共享丢失(最隐蔽:不报错,只是输出全错)—— Qwen2-VL-2B 的 lm_head 与输入 embedding 共享权重
   (config.json 顶层 tie_word_embeddings=true),checkpoint 里只有 model.embed_tokens.weight、没有 lm_head.weight。
   transformers 5.x 把配置拆成总配置 + text_config,而 tie_word_embeddings 只留在总配置上,text_config 没有
   这个属性。TensorRT-LLM 的 Qwen2VLModelBase.__init__ 用 text_config 构建语言模型,三处判断
   (modeling_utils.py 里构造 lm_head 与跳过加载的两处、checkpoints/hf/weight_mapper.py 一处)都是
   getattr(config, 'tie_word_embeddings', False) —— 拿到默认值 False:lm_head 被当成独立权重,
   checkpoint 里又找不到,于是一直是初始化时的全零。logits 全为 0,贪心解码每一步都选 0 号 token "!"。
   现象:纯文字题和带图题一律输出 "!!!!!!!!";补上之后纯文字题答 "Paris" / "3"(tools/probe_trtllm.py --mode text)。
   修补:包一层 Qwen2VLModelBase.__init__,在它深拷贝并拆出 text_config 之前,把总配置的 tie_word_embeddings
   复制到 text_config(只在 text_config 缺这个属性时补,不覆盖已有值)。

4.(可选,P5)视觉塔换成 TensorRT 引擎 —— 不是兼容修补,是实验开关:只在设置了环境变量
   VLMDEPLOY_VISION_TRT=<引擎目录> 时生效,详见下方 _VisionEngines。引擎接管之后,权重加载完就把
   PyTorch 视觉塔的权重挪到 CPU:引擎(约 1.34 GB)与视觉塔(约 1.3 GB)是同一个东西的两份拷贝,
   两份都留在显存里会把 8 GB 卡上的 KV cache 挤到装不下一条序列(实测数据见 _VisionEngines)。

为什么用 sitecustomize,而不是在脚本里打补丁:
    TensorRT-LLM 单卡默认也会另起 worker 进程跑模型(实测日志 "using MpiPoolSession to spawn MPI processes"),
    脚本里的猴子补丁到不了那个进程。sitecustomize 在每个 Python 进程启动时执行,worker 继承 PYTHONPATH 后同样生效。
    为了不拖慢无关进程,这里只注册"导入后钩子":目标模块真的被导入时才打补丁。不改安装目录里的任何文件,
    只对 Makefile 里设置了 PYTHONPATH 的本项目命令生效。升级 TensorRT-LLM / transformers 后应逐条复核,
    删掉不再需要的修补。
"""

from __future__ import annotations

# 顶层只用标准库:每个 Python 进程启动时都要执行这个文件,tensorrt / torch 这些大包留到函数内部、真的用到时再导入
import importlib.abc
import importlib.util
import os
import sys

# 打在被补过的类上的记号:同一个类不会被补第二次
_MARK = "_vlmdeploy_patched"


def _note(msg: str) -> None:
    # 写到 stderr 并带上进程号:每个补丁生效时都说一声,日志里能确认补丁在 worker 进程里确实生效了
    print(f"[vlmdeploy pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


# flash-attn 2 是否可用,transformers 不可导入时视为不可用
def _flash_attn_2_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available
    except ImportError:
        return False
    return bool(is_flash_attn_2_available())


# 补丁 1(文件开头的第 1 处):构造 HF 视觉塔时,flash-attn 2 不可用就把注意力实现改成 sdpa
def _patch_hf_vision_init(module) -> None:  # noqa: ANN001 —— transformers.models.qwen2_vl.modeling_qwen2_vl
    # 取出要修补的类,模块里没有该类时为 None
    cls = getattr(module, "Qwen2VisionTransformerPretrainedModel", None)
    # 类不存在或已经打过补丁时直接返回
    if cls is None or getattr(cls, _MARK, False):
        return
    # 保存原来的构造函数
    original_init = cls.__init__

    # 新的构造函数:必要时把注意力实现改成 sdpa,再调用原函数
    def __init__(self, config, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003 —— 签名跟随被替换的原函数
        # 要求 flash-attn 2 而环境里没有时改用 sdpa
        if getattr(config, "_attn_implementation", None) == "flash_attention_2" and not _flash_attn_2_available():
            config._attn_implementation = "sdpa"
            _note("Qwen2-VL vision tower: flash_attention_2 unavailable, using sdpa")
        # 调用原来的构造函数
        original_init(self, config, *args, **kwargs)

    # 替换类的构造函数并打上记号
    cls.__init__ = __init__
    setattr(cls, _MARK, True)


# 补丁 2(第 2 处)+ 视觉塔引擎开关(第 4 处):TensorRT-LLM 视觉塔包装类的 forward 与 load_weights 各包一层
def _patch_trtllm_vision_forward(module) -> None:  # noqa: ANN001 —— tensorrt_llm._torch.models.modeling_qwen2vl
    # 取出要修补的类
    cls = getattr(module, "Qwen2VisionModelBase", None)
    # 类不存在或已经打过补丁时直接返回
    if cls is None or getattr(cls, _MARK, False):
        return
    # 保存原来的两个方法
    original_forward = cls.forward
    original_load_weights = cls.load_weights
    # 是否已经提示过,用列表保存以便内层函数修改
    warned = []

    def forward(self, multimodal_params):  # noqa: ANN001, ANN202
        # 先试引擎:它能处理这一批就直接用它的输出;返回 None(没启用引擎,或这一批它处理不了)才走 PyTorch 视觉塔
        engine_out = _VisionEngines.try_forward(multimodal_params)
        # 引擎给出了结果就直接返回
        if engine_out is not None:
            return engine_out
        # 否则走 PyTorch 视觉塔
        embeds = _VisionEngines.hf_forward(self, original_forward, multimodal_params)
        # 补丁 2 落在这一行:transformers 5.x 返回的对象换成其中的 pooler_output(4.x 直接返回的那个张量)
        fixed = [getattr(e, "pooler_output", e) for e in embeds]   # 张量没有 pooler_output 属性,原样保留
        # 有元素被替换过时提示一次
        if not warned and any(f is not e for f, e in zip(fixed, embeds, strict=True)):
            warned.append(True)
            _note("Qwen2-VL vision tower returned BaseModelOutputWithPooling (transformers 5.x); using pooler_output")
        return fixed

    # 新的权重加载函数:加载完成后启用视觉塔引擎
    def load_weights(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        # 先按原方式加载权重
        result = original_load_weights(self, *args, **kwargs)
        # 权重加载完立刻拉起引擎,并把 PyTorch 视觉塔挪出显存。
        # 引擎若等第一次前向才惰性加载,那一刻正落在 TensorRT-LLM 估算 KV cache 容量的预热里,
        # 两份权重一起被算进峰值显存 → 留给 KV 的预算只剩个位数 block(见 _VisionEngines 的显存账)。
        if os.environ.get("VLMDEPLOY_VISION_TRT") and _VisionEngines._load():
            _VisionEngines.release_hf_tower(self)
        return result

    # 替换类的两个方法并打上记号
    cls.forward = forward
    cls.load_weights = load_weights
    setattr(cls, _MARK, True)


class _VisionEngines:
    """第 4 处(可选,P5):视觉塔换成 TensorRT 引擎。只在设置了环境变量 VLMDEPLOY_VISION_TRT=<引擎目录> 时启用。

    为什么在这里接、而不是用 TensorRT-LLM 的"预计算 embedding"输入:
        预计算意味着视觉段在请求之外跑,TTFT 里就看不到它 —— 端到端对比会变成"少算了一段"。
        在视觉塔的调用点替换,引擎和 HF 实现处于完全相同的位置(同一个 worker 进程、同一次 context 前向),
        TTFT 的差就只来自视觉塔本身。必须放在 sitecustomize 里,因为视觉塔跑在 TensorRT-LLM 另起的 worker 进程中。

    引擎是静态形状:目录里的 vision_g{H}x{W}.engine 对应合并前 H×W 个块的单张图(scripts/15_vision_trt.py 生成)。
    一批请求里只要有一张图的网格没有对应引擎,整批回退 HF 实现 —— 宁可慢,不能错。

    显存账(8 GB 卡实测;根因未证实,但账目对得上):baseline 的 KV cache 拿到 478 blocks / 15296 tokens
    (约 0.41 GiB);引擎接进来而 PyTorch 视觉塔仍留在显存时只剩 39 / 32 / 12 blocks = 1248 / 1024 / 384 tokens
    (s256 / s576 / s1024)。s1024 的提示词 1063 tokens 超过 384,max_seq_len 被压到 384,worker 报
    "default_max_tokens (-679) must be greater than 0" 起不来。多出来的约 1.6 GB 与"引擎约 1.34 GB +
    PyTorch 视觉塔约 1.3 GB 同时驻留"同量级 —— 同一个视觉塔的两份拷贝。所以权重一加载完就把 PyTorch
    那份挪到 CPU(release_hf_tower),只在某个网格真的没有引擎时才临时搬回来(hf_forward)。
    修复后实测 KV cache = 117 / 105 / 89 blocks(= 3744 / 3360 / 2848 tokens),三档 max_seq_len 都保持 2048,
    三档都能起。但没有回到 baseline 的 478 blocks:KV 估算的虚拟预热用的图像网格在引擎目录里不存在,
    于是走了 hf_forward 回退路径、PyTorch 塔被临时搬回 GPU 并计入峰值(根因未证实)。
    """

    # 状态全放在类属性上,方法全是 classmethod,从不实例化:每个进程里只有一份。
    # _engines:(高, 宽) 网格 -> (引擎, 执行上下文),None 表示还没加载过;两个 _announced 标志让提示只打一次。
    _engines: dict | None = None
    _announced = False
    _fallback_announced = False
    _hf_released = False       # PyTorch 视觉塔的权重是否已挪到 CPU
    _hf_device = None          # 挪走前它所在的设备(回退时要搬回去)

    @classmethod
    # 加载引擎目录下的引擎,只加载一次,返回网格到引擎的映射
    def _load(cls) -> dict:
        # 已加载过则直接返回
        if cls._engines is not None:
            return cls._engines
        # 只加载一次:先置成空表,之后的调用都从上面直接返回。没设环境变量就停在空表,整个引擎功能等于不存在
        cls._engines = {}
        # 引擎目录来自环境变量
        root = os.environ.get("VLMDEPLOY_VISION_TRT")
        # 没有设置时返回空表
        if not root:
            return cls._engines
        # 以下依赖只在需要加载引擎时导入
        import re
        from pathlib import Path

        import tensorrt as trt

        # 只加载用得到的网格,由 VLMDEPLOY_VISION_TRT_GRIDS 指定;不设置时加载目录里的全部引擎。
        #   实测(8 GB 显存):三个静态引擎各约 1.34 GB,全部常驻再加上语言模型(约 4.4 GB;当时 PyTorch 视觉塔
        #   的权重也还留在显存里没释放),剩余显存的 25% 放不下一条序列,worker 初始化直接断言失败:
        #   "Impossible to fit in any sequence in kvCache"。静态形状引擎的代价就是每个分辨率一份权重。
        # 解析要加载的网格集合,为空表示不过滤
        only = {tuple(map(int, g.split("x"))) for g in os.environ.get("VLMDEPLOY_VISION_TRT_GRIDS", "").split(",") if g}
        # 创建 TensorRT 运行时
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        # 按文件名顺序遍历引擎文件
        for f in sorted(Path(root).glob("vision_g*x*.engine")):
            # 从文件名里取出网格的高和宽
            h, w = map(int, re.fullmatch(r"vision_g(\d+)x(\d+)\.engine", f.name).groups())
            # 不在指定集合里的跳过
            if only and (h, w) not in only:
                continue
            # 反序列化引擎
            engine = runtime.deserialize_cuda_engine(f.read_bytes())
            # 为引擎创建执行上下文,按网格登记
            cls._engines[(h, w)] = (engine, engine.create_execution_context())
        # 打印已加载的网格
        _note(f"vision TensorRT engines loaded from {root}: grids {sorted(cls._engines)}")
        return cls._engines

    @classmethod
    def release_hf_tower(cls, module) -> None:  # noqa: ANN001 —— Qwen2VisionModelBase 实例
        """把 PyTorch 视觉塔的权重挪到 CPU:引擎接管之后,它在显存里就是一份多余的拷贝。"""
        # 已经挪过则直接返回
        if cls._hf_released:
            return
        import torch

        # 取出 PyTorch 视觉塔
        visual = getattr(module, "visual", None)
        # 没有视觉塔时直接返回
        if visual is None:
            return
        # 用第一个参数所在的设备代表整个视觉塔;视觉塔没有参数(StopIteration)或已经在 CPU 上,就没什么可挪的
        try:
            device = next(visual.parameters()).device
        except StopIteration:
            return
        # 已经在 CPU 上则不用处理
        if device.type == "cpu":
            return
        # 记下原来的设备,回退时要搬回去
        cls._hf_device = device
        # 把视觉塔的权重挪到 CPU
        visual.to("cpu")
        torch.cuda.empty_cache()   # 不归还缓存的话,显存仍被 PyTorch 分配器占着,KV 估算看不到这块空出来
        # 标记已释放
        cls._hf_released = True
        # 打印提示
        _note(f"vision TensorRT engine takes over; PyTorch vision tower moved from {device} to cpu")

    @classmethod
    def hf_forward(cls, module, original_forward, multimodal_params):  # noqa: ANN001, ANN206
        """回退到 PyTorch 视觉塔。权重若已被挪到 CPU,先搬回 GPU,用完再挪走 —— 慢,但结果正确。"""
        # 取出 PyTorch 视觉塔
        visual = getattr(module, "visual", None)
        # 权重已被挪到 CPU 时,需要临时搬回原设备
        moved = cls._hf_released and visual is not None and cls._hf_device is not None
        if moved:
            # 回退路径只提示一次
            if not cls._fallback_announced:
                cls._fallback_announced = True
                _note("an image grid has no TensorRT engine; falling back to the PyTorch vision tower (slow path)")
            # 把视觉塔搬回原设备
            visual.to(cls._hf_device)
        # 执行原来的前向,结束后无论成败都把视觉塔挪回 CPU
        try:
            return original_forward(module, multimodal_params)
        finally:
            if moved:
                import torch

                visual.to("cpu")
                torch.cuda.empty_cache()

    @classmethod
    def try_forward(cls, multimodal_params):  # noqa: ANN001, ANN206
        # 返回 None 表示这一批不归引擎处理(没有引擎,或下面任何一条检查没过),调用方回退到 PyTorch 视觉塔
        # 取已加载的引擎表
        engines = cls._load()
        # 没有引擎时不处理
        if not engines:
            return None
        import torch

        # 每个请求对应的引擎和输入
        items = []
        # 逐个请求检查:有图、不是视频、只有一张图(image_grid_thw 恰好 3 个数且 t == 1)、它的 (h, w) 网格有对应引擎。
        # 任何一条不满足就整批放弃。
        for p in multimodal_params:
            # 取该请求的图像输入
            image = p.multimodal_data.get("image")
            # 没有图像或带视频的请求不处理
            if image is None or p.multimodal_data.get("video") is not None:
                return None
            # 取图像网格的帧数、高和宽
            t, h, w = (int(v) for v in image["image_grid_thw"].reshape(-1, 3)[0].tolist())
            # 不是单张图或没有对应网格的引擎时不处理
            if image["image_grid_thw"].numel() != 3 or t != 1 or (h, w) not in engines:
                return None
            # 记录该请求用的引擎和像素输入
            items.append((engines[(h, w)], image["pixel_values"]))
        # 各请求的输出
        outs = []
        stream = torch.cuda.current_stream()   # 与 TensorRT-LLM 的模型前向同一条流,保证先后顺序
        # 逐个请求执行引擎
        for (engine, ctx), pixels in items:
            # 输入移到 GPU 并转成 FP16,保证内存连续
            x = pixels.to(device="cuda", dtype=torch.float16).contiguous()
            # 按引擎的输出形状预分配输出张量
            y = torch.empty(tuple(engine.get_tensor_shape("embeds")), dtype=torch.float16, device="cuda")
            # 按张量名把输入、输出两块显存的地址交给引擎,它直接在上面读写;y 按引擎的静态输出形状预分配
            ctx.set_tensor_address("pixel_values", x.data_ptr())
            ctx.set_tensor_address("embeds", y.data_ptr())
            # 在当前 stream 上执行,失败时报错
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("vision engine execution failed")
            outs.append(y.to(torch.bfloat16))   # 语言模型侧是 BF16
        # 首次使用引擎时提示一次
        if not cls._announced:
            cls._announced = True
            _note("Qwen2-VL vision tower: running TensorRT engine")
        # 各图的输出按顺序拼成一个张量,再包进列表:TensorRT-LLM 期望的返回格式就是"列表里一个张量"
        return [torch.cat(outs, dim=0)]


# 补丁 3(第 3 处):把总配置的 tie_word_embeddings 补进 text_config,修的就是"输出全是 !"那个问题
def _patch_trtllm_tied_embeddings(module) -> None:  # noqa: ANN001 —— tensorrt_llm._torch.models.modeling_qwen2vl
    # 取出要修补的类
    cls = getattr(module, "Qwen2VLModelBase", None)
    # 类不存在或已经打过补丁时直接返回
    if cls is None or getattr(cls, _MARK, False):
        return
    # 保存原来的构造函数
    original_init = cls.__init__

    # 新的构造函数:先补上 tie_word_embeddings,再调用原函数
    def __init__(self, model_config, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003 —— 同上
        # 总配置
        top = model_config.pretrained_config
        # 语言模型的配置
        text = getattr(top, "text_config", None)
        # 语言模型配置缺少该属性而总配置有时才补
        if text is not None and not hasattr(text, "tie_word_embeddings") and hasattr(top, "tie_word_embeddings"):
            text.tie_word_embeddings = top.tie_word_embeddings   # 在 __init__ 深拷贝并拆出 text_config 之前补上
            _note(f"text_config lacks tie_word_embeddings; copied from top-level config ({top.tie_word_embeddings})")
        # 调用原来的构造函数
        original_init(self, model_config, *args, **kwargs)

    # 替换类的构造函数并打上记号
    cls.__init__ = __init__
    setattr(cls, _MARK, True)


# 补丁 2、3 改的两个类在同一个模块里,而 _PATCHES 每个模块只登记一个函数,所以合成一个入口
def _patch_trtllm_qwen2vl(module) -> None:  # noqa: ANN001
    _patch_trtllm_vision_forward(module)
    _patch_trtllm_tied_embeddings(module)


# 补丁表:模块名 -> 该模块导入完成后要执行的补丁函数
_PATCHES = {
    "transformers.models.qwen2_vl.modeling_qwen2_vl": _patch_hf_vision_init,
    "tensorrt_llm._torch.models.modeling_qwen2vl": _patch_trtllm_qwen2vl,
}


class _PostImportHook(importlib.abc.MetaPathFinder):
    """目标模块被导入时,先让正常机制加载它,再在 exec_module 之后打补丁。"""

    def find_spec(self, fullname, path, target=None):  # noqa: ANN001, ANN202 —— 返回 ModuleSpec 或 None
        # Python 每导入一个模块都会先来问一遍;不是目标模块就返回 None,交给 sys.meta_path 上后面的正常机制
        # 查这个模块有没有登记补丁
        patch = _PATCHES.get(fullname)
        # 没有登记的模块不处理
        if patch is None:
            return None
        sys.meta_path.remove(self)   # 临时摘掉自己,避免下面的 find_spec 递归回来
        # 用正常机制查找模块的规格
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)   # 放在 finally 里:查找抛异常也一定挂回去
        # 找不到模块或没有加载器时原样返回
        if spec is None or spec.loader is None:
            return spec
        # 保存原来的模块执行函数
        exec_module = spec.loader.exec_module

        # 先执行模块代码,再打补丁
        def exec_and_patch(module) -> None:  # noqa: ANN001
            exec_module(module)
            patch(module)

        # 用包装后的函数替换加载器的执行函数
        spec.loader.exec_module = exec_and_patch
        return spec


for _name, _patch in _PATCHES.items():   # sitecustomize 先于一切用户代码,正常情况下这里一个都不会命中
    # 模块已经导入时直接打补丁
    if _name in sys.modules:
        _patch(sys.modules[_name])
sys.meta_path.insert(0, _PostImportHook())   # 挂到最前面,先于正常的查找机制被问到
