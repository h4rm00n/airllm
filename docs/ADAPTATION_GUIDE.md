# AirLLM 模型适配指南

## 目录

1. [架构概览](#1-架构概览)
2. [FP8 量化适配经验总结](#2-fp8-量化适配经验总结)
3. [NVFP4 量化格式适配方案](#3-nvfp4-量化格式适配方案)
4. [Qwen3_5_MoE 模型架构适配方案](#4-qwen3_5_moe-模型架构适配方案)
5. [关键文件与修改清单](#5-关键文件与修改清单)
6. [测试验证流程](#6-测试验证流程)

---

## 1. 架构概览

### 1.1 模型加载流程

```
AutoModel.from_pretrained()
  ├─> get_module_class()          # 根据 config.architectures 分派模型类
  ├─> import_module("airllm")     # 动态导入
  └─> ModelClass.__init__()
        ├─> super().__init__()
        │     ├─> set_layer_names_dict()     # 定义模型结构路径
        │     ├─> find_or_create_local_splitted_path()  # 拆分/定位 safetensors
        │     ├─> _adjust_config_for_model()           # 调整配置
        │     ├─> get_tokenizer()
        │     └─> init_model()               # 创建空 meta 模型 + 移动 buffer
        └─> _is_from_multimodal_ckpt / _is_vl_model 等标志位
```

### 1.2 逐层推理流程

```
forward()
  ├─> del self.model; init_model()     # 每轮推理重建空模型
  ├─> for each layer in layers:
  │     ├─> load_layer_to_cpu()        # 从磁盘加载 safetensors
  │     ├─> move_layer_to_device()     # 反量化 + 移动到 GPU
  │     ├─> layer(seq, ...)            # 执行该层前向
  │     └─> layer.to("meta"); clean_memory()  # 卸载该层权重
  ├─> final_norm(seq)
  └─> lm_head(seq) → logits
```

### 1.3 类继承关系

```
AirLLMBaseModel (GenerationMixin)         ← airllm_base.py (665行)
├── AirLLMLlama2                          ← 基础 Llama 架构
├── AirLLMQWen / AirLLMQWen2             ← Qwen1/2
├── AirLLMQWen3_5                         ← Qwen3.5 纯文本 (含 layer_types 分派)
│   └── _is_from_multimodal_ckpt 标志
├── AirLLMQWen3_5VL                       ← Qwen3.5 多模态 (含 FP8 反量化、视觉编码器)
│   ├── _dequantize_fp8_blockwise()      # FP8 块级反量化
│   ├── _run_vision_encoder()            # 视觉编码器
│   └── _inject_image_embeddings()       # 图像嵌入注入
├── AirLLMMixtral                         ← Mixtral MoE
└── ...其他架构
```

---

## 2. FP8 量化适配经验总结

### 2.1 量化格式说明

**config.json 中的 quantization_config：**
```json
{
  "quantization_config": {
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": ["model.language_model.embed_tokens", ...]
  }
}
```

- 权重以 `torch.float8_e4m3fn` 存储（范围 -448 ~ 448）
- 每个 128×128 块对应一个 `torch.bfloat16` 缩放因子
- 缩放因子存储在 `weight_scale_inv` 键中（虽名为 inv 但实际是直接乘的因子）
- `modules_to_not_convert` 中的模块不量化，保留原始 dtype（通常是 BF16）

### 2.2 反量化公式

来自 HuggingFace `Fp8Dequantize`：

```python
quantized = weight.to(scales.dtype)          # FP8 → BF16
reshaped  = weight.reshape(-1, H//128, 128, W//128, 128)
expanded  = scales.reshape(-1, H//128, W//128).unsqueeze(-1).unsqueeze(2)
result    = (reshaped * expanded).reshape(H, W)    # BF16 精度
```

与 AirLLM 当前实现的等价版本（`_dequantize_fp8_blockwise`）：

```python
# airllm_qwen3_5_vl.py:412-418
@staticmethod
def _dequantize_fp8_blockwise(weight, scale, block_size=128):
    rows, cols = weight.shape
    sr, sc = scale.shape
    w = weight.reshape(sr, block_size, sc, block_size)
    s = scale.unsqueeze(1).unsqueeze(3)
    return (w.float() * s.float()).reshape(rows, cols).to(torch.bfloat16)
```

两种实现数值**完全相同**（已验证 424 个权重矩阵全部匹配）。

### 2.3 关键 Bug 教训

**Bug**: `fp8_scale_keys` 构建时使用了错误的 `replace`：

```python
# 错误（会丢掉 .weight 后缀）
fp8_scale_keys = {k.replace('.weight_scale_inv', '') for k in state_dict
                  if k.endswith('.weight_scale_inv')}
# in_proj_qkv.weight_scale_inv → in_proj_qkv （少了 .weight！）

# 正确（只去掉 _scale_inv 后缀）
fp8_scale_keys = {k.replace('_scale_inv', '') for k in state_dict
                  if k.endswith('.weight_scale_inv')}
# in_proj_qkv.weight_scale_inv → in_proj_qkv.weight
```

**影响**: `param_name in fp8_scale_keys` 永远为 False，所有 FP8 权重都未被反量化，直接用原始 FP8 值参与计算，导致输出乱码。

### 2.4 安全 compute dtype 机制

```python
# airllm_base.py:53-60
@staticmethod
def _resolve_safe_compute_dtype(dtype):
    """FP8 类型 → BF16，其他类型原样返回"""
    _fp8_names = ('float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnez')
    for name in _fp8_names:
        if dtype == getattr(torch, name, None):
            return torch.bfloat16
    return dtype
```

用于 `set_module_tensor_to_device` 的 `dtype` 参数，确保 FP8 推理时权重放置在兼容的计算 dtype 上。

### 2.5 适配清单

为新增量化格式时需要的修改：

| 位置 | 修改内容 |
|------|---------|
| `auto_model.py:get_module_class()` | 无（通过 config 的 quantization_config 自动路由） |
| `airllm_base.py:_resolve_safe_compute_dtype()` | 添加新量化类型的 dtype 映射 |
| `airllm_base.py:__init__()` | 无（已通过 `_safe_compute_dtype` 机制覆盖） |
| 具体模型类的 `move_layer_to_device()` | **核心**：检测新量化格式 + 反量化逻辑 |
| 具体模型类的 `_dequantize_xxx()` | 新增反量化函数 |

---

## 3. NVFP4 量化格式适配方案

### 3.1 NVFP4 格式说明

NVFP4（NVIDIA FP4）是 NVIDIA 推出的 4-bit 浮点格式，与传统的 NF4（Normal Float 4）、INT4 不同：

```
NVFP4 格式特点:
- 每 16 个权重共享一个缩放因子（block size = 16）
- 权重以 4-bit 存储
- 缩放因子通常为 FP16/BF16
- 存储键命名风格：weight → weight_nvfp4, scale → weight_scale
```

### 3.2 适配步骤

#### Step 1: 更新 dtype 映射

```python
# airllm_base.py:_resolve_safe_compute_dtype()
# 无需修改：NVFP4 反量化后结果是 BF16/FP16，不需要特殊映射
# 但如果服务器传入 --dtype nvfp4，需要在 openai_server.py 添加别名：

_DTYPE_ALIASES = {
    "nvfp4": "bfloat16",      # NVFP4 反量化后为 BF16
    ...
}
```

#### Step 2: 添加反量化函数

在目标模型类（例如 `AirLLMQWen3_5VL` 或新增的专门类）中添加：

```python
@staticmethod
def _dequantize_nvfp4_blockwise(weight_packed, scale, block_size=16):
    """
    NVFP4 反量化：weight_packed 存储为 4-bit packed 格式

    Args:
        weight_packed: torch.Tensor, shape [rows, cols//2] (packed uint8)
        scale: torch.Tensor, shape [rows//block_size, cols//block_size]
        block_size: int, 默认 16

    Returns:
        torch.Tensor, shape [rows, cols], dtype=bfloat16
    """
    import torch  # NVFP4 需要自定义 unpack 逻辑或使用 nvidia 的 kernel
    # 方案 A: 使用 torch.float4_e2m1fn (PyTorch >= 2.4)
    # 方案 B: 手动 unpack + 乘 scale

    rows, cols_packed = weight_packed.shape
    cols = cols_packed * 2  # 4-bit → 每字节存 2 个权重

    # 假设 PyTorch 后续版本支持 float4
    # weight = weight_packed.view(torch.float4_e2m1fn)  # 待 PyTorch 支持

    # 手动 unpack (uint8 → 2×4bit → float)
    high = (weight_packed >> 4).float()      # 高位 4-bit
    low = (weight_packed & 0x0F).float()      # 低位 4-bit
    weight = torch.stack([low, high], dim=-1).flatten(-2, -1)

    # Block-wise multiply with scale
    sr, sc = scale.shape
    w = weight.reshape(sr, block_size, sc, block_size)
    s = scale.unsqueeze(1).unsqueeze(3)
    return (w * s).reshape(rows, cols).to(torch.bfloat16)
```

#### Step 3: 修改 `move_layer_to_device`

参照 FP8 的模式，在现有方法中增加 NVFP4 检测分支：

```python
def move_layer_to_device(self, state_dict):
    # ...现有的 FP8 逻辑...

    # 新增 NVFP4 检测
    nvfp4_weight_keys = {k.replace('_scale', '') for k in state_dict
                         if k.endswith('.weight_scale')}  # 注意：NVFP4 的 scale 键命名
    target_dtype = self._safe_compute_dtype

    for param_name in list(state_dict.keys()):
        if param_name.endswith('.weight_scale'):
            continue
        param = state_dict[param_name]
        scale_key = param_name + '_scale'

        # NVFP4 检测：可能以 packed uint8 存储
        is_nvfp4 = (param.dtype == torch.uint8 and
                    scale_key in state_dict and
                    param_name in nvfp4_weight_keys)

        if is_nvfp4:
            dequant = self._dequantize_nvfp4_blockwise(param, state_dict[scale_key])
            set_module_tensor_to_device(self.model, param_name, self.running_device,
                                        value=dequant, dtype=torch.bfloat16)
        # ...现有 FP8 + 非量化处理...
```

#### Step 4: 处理 safetensors 拆分

如果 NVFP4 模型的 safetensors 使用不同命名（如 `weight_nvfp4` 替代 `weight`），需要在 `utils.py:split_and_save_layers()` 中处理键名映射。

### 3.3 注意事项

1. **PyTorch 版本**: NVFP4 的 native 支持依赖 PyTorch >= 2.4（`torch.float4_e2m1fn`）。早期版本需要手动 unpack。
2. **Block Size**: NVFP4 的 block_size 通常为 16，与 FP8 的 128 不同。检查 `quantization_config.weight_block_size`。
3. **Scale 键命名**: 实测模型中的键名可能为 `weight_scale`（而非 `weight_scale_inv`）。
4. **性能**: 4-bit unpack 涉及位运算，可能在 CPU 上较慢。考虑在 GPU 上做 unpack 或预分解。

---

## 4. Qwen3_5_MoE 模型架构适配方案

### 4.1 MoE 架构特点

Qwen3.5-MoE 相比 Qwen3.5 的主要区别：

```
Qwen3.5 (Dense):
  layer: input_layernorm → attention → post_attention_layernorm → MLP

Qwen3.5-MoE (Sparse):
  layer: input_layernorm → attention → post_attention_layernorm → MoE MLP
  MoE MLP: router(gate) → {expert_0, expert_1, ..., expert_N} × weighted_sum
```

**关键差异**:
- MLP 层有多个 expert（通常 8/16/64 个）
- 有 router/gate 网络决定每个 token 走哪些 expert
- Expert 权重命名：`mlp.experts.0.down_proj.weight`, `mlp.experts.1.down_proj.weight` 等

### 4.2 适配步骤

#### Step 1: 模型分派

```python
# air_llm/airllm/auto_model.py:get_module_class()
# 新增 Qwen3_5MoE 的调派
if "Qwen3_5ForConditionalGeneration" in config.architectures[0]:
    if hasattr(config, 'text_config') and \
       config.text_config.get('model_type') == 'qwen3_5_moe':
        return "airllm", "AirLLMQWen3_5MoE"
    return "airllm", "AirLLMQWen3_5VL"
elif "Qwen3_5MoE" in config.architectures[0]:
    return "airllm", "AirLLMQWen3_5MoE"
```

#### Step 2: 创建模型类

```python
# air_llm/airllm/airllm_qwen3_5_moe.py

from .airllm_base import AirLLMBaseModel
from .airllm_qwen3_5 import AirLLMQWen3_5

class AirLLMQWen3_5MoE(AirLLMQWen3_5):
    """
    Qwen3.5 MoE 适配类

    继承 AirLLMQWen3_5 (复用其 layer_types 分派、DynamicCache 等)，
    仅覆盖 MoE 特有的行为。
    """

    def __init__(self, model_local_path_or_repo_id, *args, **kwargs):
        super().__init__(model_local_path_or_repo_id, *args, **kwargs)
        self._is_moe = True

    # MoE 的层名称可能与 Dense 不同
    # 如果 safetensors 中 expert 权重的命名与标准 MLP 一致，
    # 则 set_layer_names_dict 不需要修改；
    # 否则需要重写以处理 expert 权重

    def set_layers_from_layer_names(self):
        """MoE 可能有嵌套的 expert 模块"""
        # 如果 model.language_model.layers[i].mlp.experts[j].down_proj.weight
        # 需要递归地展开所有 expert 子层
        self.layers = []
        model_attr = self.model
        for attr_name in self._get_model_access_path("embed").split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self._get_model_access_path("layer_prefix").split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self._get_model_access_path("norm").split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        # MoE 不需要单独的 lm_head 层（与 text-only 类似）
```

#### Step 3: Expert 权重拆分

**关键问题**: MoE 的 expert 权重通常很大（每个 expert 有独立的 MLP 权重），需要决定拆分层级。

**策略 A（推荐）**: 将每个 Transformer Layer 的所有 expert 作为一个整体加载：

```
splitted_model/
  model.language_model.layers.0.safetensors  # 包含 router + 所有 expert
  model.language_model.layers.1.safetensors
  ...
```

**策略 B**: 将每个 expert 拆分为独立文件（文件数 = layers × experts，太多）：

```
splitted_model/
  model.language_model.layers.0.expert.0.safetensors
  model.language_model.layers.0.expert.1.safetensors
  ...
```

**策略 A 更简单**，只需修改 `split_and_save_layers` 中的权重分组逻辑，将整层的权重收集到一个文件中。

#### Step 4: 处理量化格式

如果 Qwen3_5_MoE 也使用了 FP8 量化，`move_layer_to_device` 中已支持（通过 FP8 scale key 检测）。但需要确保 `fp8_scale_keys` 构建能覆盖 expert 权重：

```python
# Expert 权重的键名示例：
# model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale_inv

# replace('_scale_inv', '') 会生成：
# model.language_model.layers.0.mlp.experts.0.down_proj.weight  ✓

# 匹配逻辑无需额外修改（已正确）
```

#### Step 5: 注册导出

```python
# air_llm/airllm/__init__.py
from .airllm_qwen3_5_moe import AirLLMQWen3_5MoE
```

### 4.3 注意事项

1. **Router 加载**: MoE 的 router/gate 网络必须有正确的权重，否则 token 分配错误会导致输出完全异常。
2. **共享 Expert**: Qwen3.5-MoE 可能有共享 expert（shared expert），权重命名可能不同。
3. **Expert 并行**: 当前 AirLLM 是逐层加载模式，不是 expert 并行。所有 expert 在同一 GPU 上顺序执行。
4. **VL-MoE**: 如果 MoE 也有多模态版本（`Qwen3_5VLMoeForConditionalGeneration`），需同时继承 VL 和 MoE 的适配逻辑。

---

## 5. 关键文件与修改清单

### 5.1 通用文件（所有适配都需要）

| 文件 | 作用 | 适配时需要 |
|------|------|-----------|
| `air_llm/airllm/auto_model.py` | 模型分派 | 添加新架构的判断 |
| `air_llm/airllm/__init__.py` | 导出模型类 | 添加新类 import |
| `air_llm/airllm/airllm_base.py` | 基类，核心推理循环 | 添加量化 dtype 映射 |
| `server/openai_server.py` | 服务器入口 | 添加 dtype 别名 |

### 5.2 新增/修改的模型类文件

| 文件 | 内容 | 说明 |
|------|------|-----|
| `airllm_qwen3_5_moe.py` | MoE 适配类 | 继承 `AirLLMQWen3_5`，覆盖 MoE 特有逻辑 |
| `airllm_qwen3_5_vl.py` | VL 适配类（已有） | 可能需要增加 NVFP4 分支 |

### 5.3 可复用的现有能力

| 能力 | 位置 | 说明 |
|------|------|-----|
| `layer_types` 分派 | `airllm_qwen3_5.py` / `.vl.py` forward | 区分 linear_attention / full_attention |
| `DynamicCache` | `airllm_qwen3_5.py` | API 风格 KV 缓存 |
| FP8 反量化 | `airllm_qwen3_5_vl.py:move_layer_to_device` | 可直接复用或扩展 |
| `_safe_compute_dtype` | `airllm_base.py` | 所有模型自动继承 |
| 视觉编码器按需加载 | `airllm_qwen3_5_vl.py` | VL-MoE 可直接复用 |
| safetensors 逐层加载 | `persist/safetensor_model_persister.py` | 通用，无需修改 |

---

## 6. 测试验证流程

### 6.1 本地验证（不启动服务器）

```python
# 1. 加载 HuggingFace 模型作为参考
import torch
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map='cuda:0', trust_remote_code=True,
    dtype=torch.bfloat16,  # FP8 模型需设置 quantization_config.dequantize=True
)

# 2. 用 AirLLM 加载
from airllm import AutoModel
airllm_model = AutoModel.from_pretrained(
    model_path, device='cuda:0', dtype='fp8',  # 或 'bf16'
)

# 3. 同 prompt 对比输出
prompt = "1+1="
# HF 输出
hf_out = hf_model.generate(**tokenizer(prompt, return_tensors='pt').to('cuda:0'), max_new_tokens=10)
# AirLLM 输出
airllm_out = airllm_model.generate(**tokenizer(prompt, return_tensors='pt').to('cuda:0'), max_new_tokens=10)

# 4. 逐层对比 hidden states（如有差异）
# 参考 /tmp/debug_airllm_compare2.py 中的方法：
# - HF 使用 register_forward_hook 捕获每层输出
# - AirLLM 等效执行逐层前向
# - 逐层 torch.allclose 比较
```

### 6.2 权重逐矩阵对比

```python
# 验证反量化正确性：比较 AirLLM 反量化后的权重与 HF 权重
# 参考 /tmp/debug_weights3.py 中的方法
import safetensors.torch
# 加载同一权重
layer_sd = safetensors.torch.load_file('splitted_model/layer_X.safetensors')
# AirLLM 反量化
dequant_airllm = _dequantize_fp8_blockwise(weight, scale)
# HF 模型中的权重
dequant_hf = hf_model.model.layers[X].mlp.down_proj.weight
# 比较
assert torch.allclose(dequant_airllm.float(), dequant_hf.float(), atol=1e-3)
```

### 6.3 常见问题排查

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| 输出乱码/重复 | 权重未正确反量化 | 逐矩阵对比权重 |
| NaN 错误 | compute dtype 不兼容 | 检查 `_safe_compute_dtype`，尝试 float32 |
| CUDA OOM | MoE expert 权重过大 | 将 expert 拆分为单独的文件 |
| KeyError 加载权重 | 键名映射错误 | 打印 `state_dict.keys()` 与实际模型属性对比 |
| 层数不匹配 | `set_layers_from_layer_names` 错误 | 打印 `self.layers` 数量与 config 对比 |

---

## 附录 A: 已适配的量化格式

| 格式 | 状态 | 文件位置 |
|------|------|---------|
| BF16/FP16 (原始) | ✅ 已支持 | `airllm_base.py` |
| FP8 (float8_e4m3fn) | ✅ 已支持 | `airllm_qwen3_5_vl.py:move_layer_to_device` |
| NF4 (bitsandbytes) | ✅ 已支持 (compression 参数) | `airllm/utils.py:uncompress_layer_state_dict` |
| 8-bit (bitsandbytes) | ✅ 已支持 (compression 参数) | `airllm/utils.py:uncompress_layer_state_dict` |
| NVFP4 | 🚧 待适配 | 参见本文第 3 节 |
| GPTQ | ❌ 未适配 | - |
| AWQ | ❌ 未适配 | - |

## 附录 B: 已适配的模型架构

| 架构 | config.architectures | 模型类 |
|------|---------------------|--------|
| Llama | `LlamaForCausalLM` | `AirLLMLlama2` |
| Qwen 1 | `QWenLMHeadModel` | `AirLLMQWen` |
| Qwen 2 | `Qwen2ForCausalLM` | `AirLLMQWen2` |
| Qwen 3.5 (text) | `Qwen3_5ForCausalLM` | `AirLLMQWen3_5` |
| Qwen 3.5 (VL) | `Qwen3_5ForConditionalGeneration` | `AirLLMQWen3_5VL` |
| Mixtral | `MixtralForCausalLM` | `AirLLMMixtral` |
| Mistral | `MistralForCausalLM` | `AirLLMMistral` |
| ChatGLM | `ChatGLMForConditionalGeneration` | `AirLLMChatGLM` |
| InternLM | `InternLMForCausalLM` | `AirLLMInternLM` |
| Baichuan | `BaichuanForCausalLM` | `AirLLMBaichuan` |
| Qwen 3.5 (MoE) | 🚧 待适配 | 参见本文第 4 节 |
