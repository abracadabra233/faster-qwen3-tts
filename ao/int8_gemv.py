"""
dp4a INT8 GEMV kernel integration for torchao on Jetson AGX Orin.

Monkey-patches torchao's safe_int_mm so that M<=16 (BS=1 decode)
uses our custom dp4a GEMV kernel instead of padding to M=32 and
calling torch._int_mm (which is 5x slower than BF16 for small M).

Usage:
    from ao.int8_gemv import enable_int8_gemv
    enable_int8_gemv()  # call once before quantization/generation
"""
import logging

import torch

logger = logging.getLogger(__name__)

_ENABLED = False
_ORIGINAL_SAFE_INT_MM = None

# Threshold: use dp4a GEMV for M <= this, original path for M > this
_GEMV_M_THRESHOLD = 16


def _get_kernel():
    """Lazy-load the compiled CUDA extension."""
    import int8_gemv_cuda
    return int8_gemv_cuda


def int8_gemv(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """INT8 GEMV: x[M,K] @ W[N,K]^T -> y[M,N] (int32).

    Uses dp4a intrinsic for efficient M=1 vector-matrix multiply.
    K must be a multiple of 4.
    """
    return _get_kernel().int8_gemv(x, W)


def _patched_safe_int_mm(input_tensor, mat2):
    """Drop-in replacement for torchao.kernel.intmm.safe_int_mm.

    For small M (<=16): use dp4a GEMV kernel
    For large M (>16): fall back to original safe_int_mm (torch._int_mm)

    safe_int_mm signature: (input[M,K], mat2[K,N]) -> [M,N]
    Our kernel signature:  (x[M,K], W[N,K]) -> [M,N]
    So we need to transpose mat2: [K,N] -> [N,K]
    """
    M = input_tensor.shape[0]

    if M <= _GEMV_M_THRESHOLD and input_tensor.is_cuda:
        W = mat2.t().contiguous()  # [K,N] -> [N,K]
        return _get_kernel().int8_gemv(input_tensor.contiguous(), W)

    return _ORIGINAL_SAFE_INT_MM(input_tensor, mat2)


def enable_int8_gemv():
    """Monkey-patch torchao to use dp4a GEMV for small-M INT8 matmul.

    Call this once before applying quantization or running inference.
    Safe to call multiple times (idempotent).
    """
    global _ENABLED, _ORIGINAL_SAFE_INT_MM

    if _ENABLED:
        logger.info("INT8 GEMV already enabled")
        return

    kernel = _get_kernel()
    logger.info("INT8 dp4a GEMV kernel loaded: %s", kernel)

    import torchao.kernel.intmm as intmm_module

    _ORIGINAL_SAFE_INT_MM = intmm_module.safe_int_mm
    intmm_module.safe_int_mm = _patched_safe_int_mm

    _ENABLED = True
    logger.info("INT8 GEMV enabled: M<=%d -> dp4a kernel, M>%d -> torch._int_mm",
                _GEMV_M_THRESHOLD, _GEMV_M_THRESHOLD)


def disable_int8_gemv():
    """Restore original torchao safe_int_mm."""
    global _ENABLED, _ORIGINAL_SAFE_INT_MM

    if not _ENABLED:
        return

    import torchao.kernel.intmm as intmm_module
    intmm_module.safe_int_mm = _ORIGINAL_SAFE_INT_MM
    _ORIGINAL_SAFE_INT_MM = None
    _ENABLED = False
    logger.info("INT8 GEMV disabled, restored original safe_int_mm")


def is_enabled() -> bool:
    return _ENABLED
