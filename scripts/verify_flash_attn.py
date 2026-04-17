#!/usr/bin/env python3
"""Verify whether Flash Attention is actually being used at runtime."""
import torch

# 1. Check installation
print("=== 1. flash-attn 安装检查 ===")
try:
    import flash_attn
    print(f"  flash-attn 版本: {flash_attn.__version__}")
    HAS_FA = True
except ImportError:
    print("  flash-attn 未安装!")
    HAS_FA = False

# 2. Load model
print("\n=== 2. 加载模型 ===")
from faster_qwen3_tts.utils import suppress_flash_attn_warning
with suppress_flash_attn_warning():
    from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "/app/models/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if HAS_FA else "eager",
)

talker = model.model.talker
config = talker.config

# 3. Check config
print(f"\n=== 3. Attention 配置 ===")
impl = getattr(config, "_attn_implementation", "not set")
print(f"  config._attn_implementation: {impl}")

# 4. Check actual layer type
layer0_attn = talker.model.layers[0].self_attn
cls_name = type(layer0_attn).__name__
print(f"  layer 0 attention class: {cls_name}")

is_flash = "flash" in cls_name.lower()
print(f"  Flash Attention in class name: {'YES' if is_flash else 'NO'}")

# 5. Check source code
import inspect
src = inspect.getsource(type(layer0_attn).forward)
has_flash_in_src = "flash_attn" in src or "flash_attention" in src
has_sdpa_in_src = "scaled_dot_product" in src
print(f"  'flash_attn' in forward source: {has_flash_in_src}")
print(f"  'scaled_dot_product' in forward source: {has_sdpa_in_src}")

# 6. Runtime profiling
print("\n=== 4. 运行时 CUDA Kernel 验证 ===")
from torch.profiler import profile, ProfilerActivity

dummy_emb = torch.randn(1, 10, 1024, device="cuda", dtype=torch.bfloat16)
dummy_mask = torch.ones(1, 10, device="cuda", dtype=torch.long)

with torch.no_grad():
    talker(inputs_embeds=dummy_emb, attention_mask=dummy_mask)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        talker(inputs_embeds=dummy_emb, attention_mask=dummy_mask)

flash_kernels = [e for e in prof.key_averages() if "flash" in e.key.lower()]
fmha_kernels = [e for e in prof.key_averages() if "fmha" in e.key.lower()]
sdpa_kernels = [e for e in prof.key_averages() if "efficient" in e.key.lower() or "mem_eff" in e.key.lower()]

def _time_ms(e):
    return getattr(e, "cuda_time_total", e.cpu_time_total) / 1000

if flash_kernels or fmha_kernels:
    print("  Flash Attention CUDA kernels detected!")
    for k in (flash_kernels + fmha_kernels)[:5]:
        print(f"    {k.key}: {_time_ms(k):.2f}ms")
elif sdpa_kernels:
    print("  Memory-efficient attention kernels (非 flash):")
    for k in sdpa_kernels[:5]:
        print(f"    {k.key}: {_time_ms(k):.2f}ms")
else:
    print("  No flash/fmha kernels found, checking all attention-related:")
    for k in prof.key_averages():
        if "attn" in k.key.lower():
            print(f"    {k.key}: {_time_ms(k):.2f}ms")

print("\n=== Top 10 kernels (by time) ===")
for e in sorted(prof.key_averages(), key=lambda x: _time_ms(x), reverse=True)[:10]:
    print(f"  {_time_ms(e):8.2f}ms  {e.key[:90]}")

# 7. Verdict
print("\n=== 结论 ===")
if HAS_FA and (flash_kernels or fmha_kernels or is_flash):
    print("  Flash Attention 已启用并在运行时使用")
elif HAS_FA:
    print("  flash-attn 已安装, 但运行时未检测到 flash kernel (可能 fallback 到 SDPA/eager)")
else:
    print("  flash-attn 未安装, 使用 eager attention")
