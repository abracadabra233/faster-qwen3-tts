/*
 * bindings.cpp — PyTorch C++ bindings for dp4a INT8 GEMV kernels.
 *
 * Exports:
 *   int8_gemv(x_int8, W_int8) -> int32                  — v1 kernel
 *   int8_gemv_v2(x_int8, W_int8) -> int32               — v2 with int4 vector loads
 *   fused_w8a8_gemv(x_bf16, W_int8, w_scale) -> bf16    — fully fused kernel
 *   quantize_bf16_to_int8(x_bf16) -> (x_int8, x_scale)  — v3 quant kernel
 *   dequant_int32_to_bf16(y, x_scale, w_scale) -> bf16  — v3 dequant kernel
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/* ── v1 launcher (int8_gemv.cu) ── */
extern "C" void launch_int8_gemv(
    const int8_t* x, const int8_t* W, int32_t* y,
    int M, int K, int N, cudaStream_t stream);

/* ── v2 + fused + v3 launchers (fused_int8_gemv.cu) ── */
extern "C" void launch_int8_gemv_v2(
    const int8_t* x, const int8_t* W, int32_t* y,
    int M, int K, int N, cudaStream_t stream);

extern "C" void launch_fused_w8a8_gemv(
    const __nv_bfloat16* x, const int8_t* W, const float* w_scale,
    __nv_bfloat16* y,
    int M, int K, int N, cudaStream_t stream);

extern "C" void launch_quantize_bf16_to_int8(
    const __nv_bfloat16* x, int8_t* x_q, float* x_scale,
    int M, int K, cudaStream_t stream);

extern "C" void launch_dequant_int32_to_bf16(
    const int32_t* y_int, const float* x_scale, const float* w_scale,
    __nv_bfloat16* y_out,
    int M, int N, cudaStream_t stream);

/* ═══════════ v1: original dp4a GEMV ═══════════ */

torch::Tensor int8_gemv(torch::Tensor x, torch::Tensor W) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8, got ", x.dtype());
    TORCH_CHECK(W.dtype() == torch::kInt8, "W must be int8, got ", W.dtype());
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2, "x and W must be 2D");

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = W.size(0);

    TORCH_CHECK(W.size(1) == K, "Dimension mismatch: x[", M, ",", K, "] vs W[", N, ",", W.size(1), "]");
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4 for dp4a, got K=", K);

    x = x.contiguous();
    W = W.contiguous();

    auto y = torch::empty({M, N}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(x.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_int8_gemv(x.data_ptr<int8_t>(), W.data_ptr<int8_t>(),
                     y.data_ptr<int32_t>(), M, K, N, stream);
    return y;
}

/* ═══════════ v2: vectorized dp4a GEMV ═══════════ */

torch::Tensor int8_gemv_v2(torch::Tensor x, torch::Tensor W) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8, got ", x.dtype());
    TORCH_CHECK(W.dtype() == torch::kInt8, "W must be int8, got ", W.dtype());
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2, "x and W must be 2D");

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = W.size(0);

    TORCH_CHECK(W.size(1) == K, "Dimension mismatch: x[", M, ",", K, "] vs W[", N, ",", W.size(1), "]");
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4 for dp4a, got K=", K);

    x = x.contiguous();
    W = W.contiguous();

    auto y = torch::empty({M, N}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(x.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_int8_gemv_v2(x.data_ptr<int8_t>(), W.data_ptr<int8_t>(),
                        y.data_ptr<int32_t>(), M, K, N, stream);
    return y;
}

/* ═══════════ fused: BF16 -> quant -> dp4a -> dequant -> BF16 ═══════════ */

torch::Tensor fused_w8a8_gemv(torch::Tensor x, torch::Tensor W, torch::Tensor w_scale) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && w_scale.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16, got ", x.dtype());
    TORCH_CHECK(W.dtype() == torch::kInt8, "W must be int8, got ", W.dtype());
    TORCH_CHECK(w_scale.dtype() == torch::kFloat32, "w_scale must be float32, got ", w_scale.dtype());
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2, "x and W must be 2D");
    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D");

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = W.size(0);

    TORCH_CHECK(W.size(1) == K, "Dimension mismatch: x[", M, ",", K, "] vs W[", N, ",", W.size(1), "]");
    TORCH_CHECK(w_scale.size(0) == N, "w_scale size mismatch: expected ", N, ", got ", w_scale.size(0));
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4 for dp4a, got K=", K);

    x = x.contiguous();
    W = W.contiguous();
    w_scale = w_scale.contiguous();

    auto y = torch::empty({M, N}, torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(x.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_w8a8_gemv(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        W.data_ptr<int8_t>(),
        w_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        M, K, N, stream);
    return y;
}

/* ═══════════ v3: standalone quant + dequant kernels ═══════════ */

std::tuple<torch::Tensor, torch::Tensor> quantize_bf16_to_int8(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16, got ", x.dtype());
    TORCH_CHECK(x.dim() == 2, "x must be 2D");

    const int M = x.size(0);
    const int K = x.size(1);

    x = x.contiguous();

    auto x_q = torch::empty({M, K}, torch::TensorOptions()
        .dtype(torch::kInt8).device(x.device()));
    auto x_scale = torch::empty({M}, torch::TensorOptions()
        .dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_quantize_bf16_to_int8(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        M, K, stream);
    return std::make_tuple(x_q, x_scale);
}

torch::Tensor dequant_int32_to_bf16(torch::Tensor y_int, torch::Tensor x_scale,
                                     torch::Tensor w_scale) {
    TORCH_CHECK(y_int.is_cuda() && x_scale.is_cuda() && w_scale.is_cuda(),
                "Inputs must be on CUDA");
    TORCH_CHECK(y_int.dtype() == torch::kInt32, "y must be int32, got ", y_int.dtype());
    TORCH_CHECK(x_scale.dtype() == torch::kFloat32, "x_scale must be float32");
    TORCH_CHECK(w_scale.dtype() == torch::kFloat32, "w_scale must be float32");
    TORCH_CHECK(y_int.dim() == 2, "y must be 2D");

    const int M = y_int.size(0);
    const int N = y_int.size(1);

    TORCH_CHECK(x_scale.size(0) == M, "x_scale size mismatch");
    TORCH_CHECK(w_scale.size(0) == N, "w_scale size mismatch");

    y_int = y_int.contiguous();
    x_scale = x_scale.contiguous();
    w_scale = w_scale.contiguous();

    auto y_out = torch::empty({M, N}, torch::TensorOptions()
        .dtype(torch::kBFloat16).device(y_int.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_dequant_int32_to_bf16(
        y_int.data_ptr<int32_t>(),
        x_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y_out.data_ptr()),
        M, N, stream);
    return y_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemv", &int8_gemv,
          "INT8 GEMV v1 using dp4a (x[M,K] @ W[N,K]^T -> y[M,N] int32)");
    m.def("int8_gemv_v2", &int8_gemv_v2,
          "INT8 GEMV v2 with int4 vector loads (x[M,K] @ W[N,K]^T -> y[M,N] int32)");
    m.def("fused_w8a8_gemv", &fused_w8a8_gemv,
          "Fused W8A8 GEMV: BF16 x + INT8 W + scale -> BF16 y (single kernel)");
    m.def("quantize_bf16_to_int8", &quantize_bf16_to_int8,
          "Per-token BF16 -> INT8 quantization (returns x_int8, x_scale)");
    m.def("dequant_int32_to_bf16", &dequant_int32_to_bf16,
          "Dequantize INT32 -> BF16 with per-token x_scale and per-channel w_scale");
}
