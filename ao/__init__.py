"""
torchao-based quantization for faster-qwen3-tts.

Provides W8A8 dynamic and SmoothQuant quantization for the talker
and code predictor transformers.
Optionally enables dp4a INT8 GEMV kernel for BS=1 acceleration.
Fused INT8 path: single-kernel BF16->INT8->dp4a->BF16 for maximum throughput.
"""
from .quantize import quantize_w8a8_dynamic, quantize_smoothquant
from .int8_gemv import enable_int8_gemv, disable_int8_gemv, is_enabled as is_gemv_enabled
from .fused_int8_linear import (
    FusedInt8Linear,
    convert_model_to_fused_int8,
    enable_fused_int8,
)

__all__ = [
    "quantize_w8a8_dynamic",
    "quantize_smoothquant",
    "enable_int8_gemv",
    "disable_int8_gemv",
    "is_gemv_enabled",
    "FusedInt8Linear",
    "convert_model_to_fused_int8",
    "enable_fused_int8",
]
