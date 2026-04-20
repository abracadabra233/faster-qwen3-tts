"""
Fused INT8 Linear layer for Jetson AGX Orin.

Replaces torchao's AffineQuantizedTensor-based nn.Linear with a single
fused CUDA kernel: BF16 input -> on-the-fly INT8 quant -> dp4a -> dequant -> BF16 output.

Eliminates:
  - Separate quantize/dequantize kernels
  - Weight transpose (.t().contiguous()) overhead
  - torchao tensor subclass dispatch overhead

Usage:
    from ao.fused_int8_linear import enable_fused_int8
    enable_fused_int8(faster_model)  # quantize + replace in one call
"""
import logging
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_GEMV_M_THRESHOLD = 16

# Backend strategies for M <= 16:
#   "v3_pipeline" — custom_quant(CUDA) -> v2 dp4a GEMV -> custom_dequant(CUDA)
#                   3 kernel calls per Linear; best end-to-end (~110 us in graph)
#   "v2_pipeline" — quantize(torch ops) -> v2 dp4a GEMV -> dequant(torch ops)
#                   10 kernel calls per Linear; ~146 us in graph
#   "fused"       — single fused kernel (BF16 -> quant -> dp4a -> dequant -> BF16)
#                   1 kernel call but Phase 1 overhead limits bandwidth (~185 us)
_DEFAULT_BACKEND = "v3_pipeline"


def _get_kernel():
    """Lazy-load the compiled CUDA extension."""
    import int8_gemv_cuda
    return int8_gemv_cuda


class FusedInt8Linear(nn.Module):
    """INT8 weight Linear using dp4a GEMV for M <= 16 (BS=1 decode).

    Stores INT8 weights in [N, K] layout (no transpose needed at runtime).

    Three backends available:
      - "v3_pipeline": custom CUDA quant -> v2 GEMV -> custom CUDA dequant (3 kernels, ~110us)
      - "v2_pipeline": torch quant -> v2 GEMV -> torch dequant (10 kernels, ~146us)
      - "fused": single fused kernel (~185us) — simpler but slower

    For M > 16 (prefill), falls back to dequantized BF16 matmul.
    """

    def __init__(self, weight_int8: torch.Tensor, weight_scale: torch.Tensor,
                 bias: torch.Tensor = None, backend: str = None):
        super().__init__()
        if backend is None:
            backend = _DEFAULT_BACKEND
        assert weight_int8.dtype == torch.int8
        assert weight_int8.dim() == 2
        assert backend in ("v3_pipeline", "v2_pipeline", "fused")
        self.register_buffer("weight_int8", weight_int8.contiguous())   # [N, K]
        self.register_buffer("weight_scale", weight_scale.float().contiguous())  # [N]
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias = None

        self.backend = backend
        self._weight_bf16_cache = None

    @property
    def out_features(self):
        return self.weight_int8.shape[0]

    @property
    def in_features(self):
        return self.weight_int8.shape[1]

    @torch.no_grad()
    def _get_weight_bf16(self):
        """Lazily dequantize weight to BF16 for M > 16 fallback."""
        if self._weight_bf16_cache is None:
            w = self.weight_int8.float() * self.weight_scale.unsqueeze(1)
            self._weight_bf16_cache = w.to(torch.bfloat16)
        return self._weight_bf16_cache

    def _forward_v3_pipeline(self, x: torch.Tensor) -> torch.Tensor:
        """custom CUDA quant -> v2 dp4a GEMV -> custom CUDA dequant. 3 kernels."""
        kernel = _get_kernel()
        x_int8, x_scale = kernel.quantize_bf16_to_int8(x.to(torch.bfloat16).contiguous())
        y_int32 = kernel.int8_gemv_v2(x_int8, self.weight_int8)
        return kernel.dequant_int32_to_bf16(y_int32, x_scale, self.weight_scale)

    def _forward_v2_pipeline(self, x: torch.Tensor) -> torch.Tensor:
        """quantize x (torch) -> v2 dp4a GEMV -> dequantize (torch). 10 kernels."""
        kernel = _get_kernel()
        x_f = x.float()
        absmax = x_f.abs().amax(dim=-1, keepdim=True)
        x_scale = absmax.clamp(min=1e-12) / 127.0
        x_int8 = (x_f / x_scale).round().clamp(-128, 127).to(torch.int8)

        y_int32 = kernel.int8_gemv_v2(x_int8, self.weight_int8)

        y = (y_int32.float() * x_scale * self.weight_scale.unsqueeze(0)).to(torch.bfloat16)
        return y

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Single fused kernel: BF16 -> quant -> dp4a -> dequant -> BF16."""
        return _get_kernel().fused_w8a8_gemv(
            x.to(torch.bfloat16).contiguous(),
            self.weight_int8,
            self.weight_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        M = x.shape[0]

        if M <= _GEMV_M_THRESHOLD and x.is_cuda:
            if self.backend == "v3_pipeline":
                y = self._forward_v3_pipeline(x)
            elif self.backend == "v2_pipeline":
                y = self._forward_v2_pipeline(x)
            else:
                y = self._forward_fused(x)
        else:
            y = torch.mm(x.to(torch.bfloat16), self._get_weight_bf16().T)

        if self.bias is not None:
            y = y + self.bias

        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], y.shape[-1])

        return y

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, backend='{self.backend}'")


def _unwrap_quantized_weight(weight):
    """Navigate torchao's wrapper hierarchy to find the AffineQuantizedTensor.

    torchao v0.12 structure:
      LinearActivationQuantizedTensor
        └── original_weight_tensor: AffineQuantizedTensor
              └── tensor_impl: PlainAQTTensorImpl
                    ├── data (or int_data): INT8 [N, K]
                    └── scale: [N] or [N, 1]
    """
    # Unwrap LinearActivationQuantizedTensor if present
    if hasattr(weight, "original_weight_tensor"):
        weight = weight.original_weight_tensor

    return weight


def _extract_int8_params(linear: nn.Linear):
    """Extract INT8 weight data and scale from a torchao-quantized Linear.

    Returns (int_data [N,K] int8, scale [N] float32, bias or None).
    Raises ValueError if the weight is not a quantized tensor.
    """
    weight = _unwrap_quantized_weight(linear.weight)

    tensor_impl = getattr(weight, "tensor_impl", None)
    if tensor_impl is None:
        tensor_impl = getattr(weight, "layout_tensor", None)
    if tensor_impl is None:
        raise ValueError(f"Weight is not a torchao quantized tensor: {type(weight)}")

    # int_data may be .int_data (callable or property) or .data
    int_data = None
    for attr in ("int_data", "data"):
        candidate = getattr(tensor_impl, attr, None)
        if candidate is None:
            continue
        if callable(candidate):
            candidate = candidate()
        if isinstance(candidate, torch.Tensor) and candidate.dtype == torch.int8:
            int_data = candidate
            break
    if int_data is None:
        raise ValueError(f"Cannot extract int_data from {type(tensor_impl)}")

    scale = getattr(tensor_impl, "scale", None)
    if callable(scale):
        scale = scale()
    if scale is None:
        raise ValueError(f"Cannot extract scale from {type(tensor_impl)}")

    if scale.dim() > 1:
        scale = scale.squeeze()

    bias = linear.bias
    if bias is not None:
        bias = bias.data.clone()

    return int_data.contiguous(), scale.float().contiguous(), bias


def convert_model_to_fused_int8(model: nn.Module) -> int:
    """Walk a torchao-quantized model and replace quantized Linears with FusedInt8Linear.

    Must be called AFTER torchao.quantize_() has been applied.
    Returns the number of modules replaced.
    """
    replacements = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight
        # Check for torchao quantized weight (may be wrapped in LinearActivationQuantizedTensor)
        unwrapped = _unwrap_quantized_weight(weight)
        if not (hasattr(unwrapped, "tensor_impl") or hasattr(unwrapped, "layout_tensor")):
            continue
        try:
            int_data, scale, bias = _extract_int8_params(module)
            fused = FusedInt8Linear(int_data, scale, bias)
            replacements[name] = fused
        except (ValueError, AttributeError) as e:
            logger.warning("Skipping %s: %s", name, e)

    for name, new_module in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    return len(replacements)


def enable_fused_int8(faster_model) -> dict:
    """Quantize with torchao W8A8 + replace Linears with FusedInt8Linear.

    Public API: one-call setup for fused INT8 inference.

    Args:
        faster_model: FasterQwen3TTS instance.

    Returns dict with timing, sizes, and replacement counts.
    """
    from torchao.quantization import quantize_

    from .quantize import _get_w8a8_config, _model_size_mb

    w8a8_config = _get_w8a8_config()

    m = faster_model.model.model
    talker_model = m.talker.model
    predictor_model = m.talker.code_predictor.model

    info = {
        "strategy": "fused_w8a8",
        "talker_before_mb": _model_size_mb(talker_model),
        "predictor_before_mb": _model_size_mb(predictor_model),
    }

    t0 = time.perf_counter()

    logger.info("Step 1/2: Applying torchao W8A8 quantization...")
    quantize_(talker_model, w8a8_config)
    quantize_(predictor_model, w8a8_config)
    t_quant = time.perf_counter() - t0
    info["quantize_time_s"] = t_quant
    logger.info("W8A8 quantization done in %.1fs", t_quant)

    logger.info("Step 2/2: Converting to FusedInt8Linear...")
    t1 = time.perf_counter()
    n_talker = convert_model_to_fused_int8(talker_model)
    n_predictor = convert_model_to_fused_int8(predictor_model)
    t_convert = time.perf_counter() - t1
    info["convert_time_s"] = t_convert
    info["n_fused_talker"] = n_talker
    info["n_fused_predictor"] = n_predictor

    info["total_time_s"] = time.perf_counter() - t0
    info["talker_after_mb"] = _model_size_mb(talker_model)
    info["predictor_after_mb"] = _model_size_mb(predictor_model)

    faster_model._warmed_up = False

    logger.info(
        "Fused INT8 done: talker %d layers (%.0f→%.0fMB), predictor %d layers (%.0f→%.0fMB) in %.1fs",
        n_talker, info["talker_before_mb"], info["talker_after_mb"],
        n_predictor, info["predictor_before_mb"], info["predictor_after_mb"],
        info["total_time_s"],
    )
    return info
