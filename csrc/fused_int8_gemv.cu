/*
 * fused_int8_gemv.cu — W8A8 GEMV kernels for Jetson AGX Orin (SM 8.7)
 *
 * Four kernels:
 *
 * 1. int8_gemv_v2:        INT8 GEMV with int4 (128-bit) vector loads
 * 2. fused_w8a8_gemv:     Fully-fused BF16 → INT8 → dp4a → BF16 (single kernel)
 * 3. quantize_bf16_to_int8: Standalone per-token BF16 → INT8 quantization
 * 4. dequant_int32_to_bf16: Standalone INT32 → BF16 dequantization
 *
 * v3 pipeline uses kernels 3 + 1 + 4 = 3 kernel calls per Linear
 * (vs 10 torch ops in v2_pipeline, or 1 slow fused kernel)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

/* ───────────── helpers ───────────── */

__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) { return __float2bfloat16(x); }

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* smem) {
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

/* ═══════════════════════════════════════════════════════════════════
 * Kernel 1: int8_gemv_v2 — optimised standalone INT8 GEMV
 *
 *   y[M,N] = x[M,K] (int8) @ W[N,K]^T (int8) → int32
 *
 * grid(N, M)  block(BLOCK_V2)
 * Uses int4 (128-bit) vector loads → 4 dp4a per load
 * ═══════════════════════════════════════════════════════════════════ */

constexpr int BLOCK_V2 = 128;

__global__ void int8_gemv_v2_kernel(
    const int8_t* __restrict__ x,   // [M, K]
    const int8_t* __restrict__ W,   // [N, K]
    int32_t* __restrict__ y,        // [M, N]
    const int K, const int N)
{
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;

    const int K16 = K >> 4;           // K / 16 (int4 loads)

    const int4* x_vec = reinterpret_cast<const int4*>(x + m * K);
    const int4* W_vec = reinterpret_cast<const int4*>(W + n * K);

    int32_t acc = 0;

    #pragma unroll 2
    for (int i = tid; i < K16; i += BLOCK_V2) {
        int4 xv = x_vec[i];
        int4 wv = W_vec[i];
        acc = __dp4a(xv.x, wv.x, acc);
        acc = __dp4a(xv.y, wv.y, acc);
        acc = __dp4a(xv.z, wv.z, acc);
        acc = __dp4a(xv.w, wv.w, acc);
    }

    /* handle remainder K%16 with int32 loads */
    const int K4 = K >> 2;
    const int k4_start = K16 * 4;     // first int32 index in the tail
    const int32_t* x_i32 = reinterpret_cast<const int32_t*>(x + m * K);
    const int32_t* W_i32 = reinterpret_cast<const int32_t*>(W + n * K);
    for (int k4 = k4_start + tid; k4 < K4; k4 += BLOCK_V2)
        acc = __dp4a(x_i32[k4], W_i32[k4], acc);

    /* reduction */
    __shared__ int32_t smem_i[BLOCK_V2 / 32];
    acc = warp_reduce_sum(acc);
    const int wid = tid / 32, lane = tid % 32;
    if (lane == 0) smem_i[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < BLOCK_V2 / 32) ? smem_i[lane] : 0;
        acc = warp_reduce_sum(acc);
        if (lane == 0) y[m * N + n] = acc;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Kernel 2: fused_w8a8_gemv — fully-fused BF16 → INT8 → dp4a → BF16
 *
 *   y[M,N] (bf16) = quant(x[M,K] bf16) ⊗ W[N,K] (int8) × scale[N]
 *
 * grid(N, M)  block(BLOCK_FUSED)
 *
 * Phase 1: quantise x[m,:] to INT8 in shared memory (per-token absmax)
 * Phase 2: dp4a dot-product with int4 vector loads
 * Phase 3: dequantise and write BF16 output
 * ═══════════════════════════════════════════════════════════════════ */

constexpr int BLOCK_FUSED = 128;

__global__ void fused_w8a8_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,   // [M, K]
    const int8_t*        __restrict__ W,   // [N, K]
    const float*         __restrict__ w_scale, // [N]
    __nv_bfloat16*       __restrict__ y,   // [M, N]
    const int K, const int N)
{
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    int8_t* x_q  = reinterpret_cast<int8_t*>(smem_raw);          // [K]
    float*  smem_f = reinterpret_cast<float*>(smem_raw + K);      // scratch (4 floats)

    /* ── Phase 1: quantise x[m,:] → x_q[] ── */

    /* 1a. find absmax (bf16→float, block reduce) */
    float local_max = 0.0f;
    const int K2 = K >> 1;
    const __nv_bfloat162* x_vec2 = reinterpret_cast<const __nv_bfloat162*>(x + m * K);

    #pragma unroll 4
    for (int i = tid; i < K2; i += BLOCK_FUSED) {
        __nv_bfloat162 v = x_vec2[i];
        float a = fabsf(__bfloat162float(v.x));
        float b = fabsf(__bfloat162float(v.y));
        local_max = fmaxf(local_max, fmaxf(a, b));
    }
    if ((K & 1) && tid == 0) {
        float v = fabsf(__bfloat162float(x[m * K + K - 1]));
        local_max = fmaxf(local_max, v);
    }

    local_max = block_reduce_max(local_max, smem_f);

    __shared__ float sh_act_scale;   // absmax / 127
    __shared__ float sh_quant_mult;  // 127 / absmax
    if (tid == 0) {
        float absmax = local_max;
        sh_act_scale  = absmax / 127.0f;
        sh_quant_mult = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
    }
    __syncthreads();

    float qm = sh_quant_mult;

    /* 1b. quantise to int8 in shared memory */
    #pragma unroll 4
    for (int i = tid; i < K2; i += BLOCK_FUSED) {
        __nv_bfloat162 v = x_vec2[i];
        float f0 = __bfloat162float(v.x);
        float f1 = __bfloat162float(v.y);
        x_q[i * 2]     = static_cast<int8_t>(__float2int_rn(f0 * qm));
        x_q[i * 2 + 1] = static_cast<int8_t>(__float2int_rn(f1 * qm));
    }
    if ((K & 1) && tid == 0) {
        float f = __bfloat162float(x[m * K + K - 1]);
        x_q[K - 1] = static_cast<int8_t>(__float2int_rn(f * qm));
    }
    __syncthreads();

    /* ── Phase 2: dp4a dot product with int4 vector loads ── */

    const int K16 = K >> 4;
    const int4* xq_vec = reinterpret_cast<const int4*>(x_q);
    const int4* W_vec  = reinterpret_cast<const int4*>(W + n * K);

    int32_t acc = 0;

    #pragma unroll 2
    for (int i = tid; i < K16; i += BLOCK_FUSED) {
        int4 xv = xq_vec[i];
        int4 wv = W_vec[i];
        acc = __dp4a(xv.x, wv.x, acc);
        acc = __dp4a(xv.y, wv.y, acc);
        acc = __dp4a(xv.z, wv.z, acc);
        acc = __dp4a(xv.w, wv.w, acc);
    }

    /* remainder (K%16) */
    const int K4 = K >> 2;
    const int k4_start = K16 * 4;
    const int32_t* xq_i32 = reinterpret_cast<const int32_t*>(x_q);
    const int32_t* W_i32  = reinterpret_cast<const int32_t*>(W + n * K);
    for (int k4 = k4_start + tid; k4 < K4; k4 += BLOCK_FUSED)
        acc = __dp4a(xq_i32[k4], W_i32[k4], acc);

    /* ── Phase 3: reduce, dequant, write BF16 ── */

    __shared__ int32_t smem_acc[BLOCK_FUSED / 32];
    acc = warp_reduce_sum(acc);
    const int wid = tid / 32, lane = tid % 32;
    if (lane == 0) smem_acc[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < BLOCK_FUSED / 32) ? smem_acc[lane] : 0;
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            float result = static_cast<float>(acc) * sh_act_scale * w_scale[n];
            y[m * N + n] = __float2bfloat16(result);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Kernel 3: quantize_bf16_to_int8 — standalone per-token INT8 quantization
 *
 *   x_bf16[M,K]  →  x_int8[M,K] + x_scale[M]
 *   x_scale[m] = absmax(x[m,:]) / 127
 *
 * grid(1, M)  block(BLOCK_QUANT)
 * Replaces 6 torch ops: .float(), .abs(), .amax(), /127, .round().clamp(), .to(int8)
 * ═══════════════════════════════════════════════════════════════════ */

constexpr int BLOCK_QUANT = 256;

__global__ void quantize_bf16_to_int8_kernel(
    const __nv_bfloat16* __restrict__ x,     // [M, K]
    int8_t*              __restrict__ x_q,    // [M, K]
    float*               __restrict__ x_scale,// [M]
    const int K)
{
    const int m = blockIdx.y;
    const int tid = threadIdx.x;

    const __nv_bfloat16* x_row = x + m * K;
    int8_t* xq_row = x_q + m * K;

    /* ── find absmax via BF16x2 vectorized loads ── */
    float local_max = 0.0f;
    const int K2 = K >> 1;
    const __nv_bfloat162* x_vec2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    #pragma unroll 4
    for (int i = tid; i < K2; i += BLOCK_QUANT) {
        __nv_bfloat162 v = x_vec2[i];
        float a = fabsf(__bfloat162float(v.x));
        float b = fabsf(__bfloat162float(v.y));
        local_max = fmaxf(local_max, fmaxf(a, b));
    }
    if ((K & 1) && tid == 0) {
        float v = fabsf(__bfloat162float(x_row[K - 1]));
        local_max = fmaxf(local_max, v);
    }

    __shared__ float smem_f[BLOCK_QUANT / 32];
    local_max = block_reduce_max(local_max, smem_f);

    __shared__ float sh_scale;
    __shared__ float sh_qmult;
    if (tid == 0) {
        float absmax = local_max;
        sh_scale = absmax / 127.0f;
        sh_qmult = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
        x_scale[m] = sh_scale;
    }
    __syncthreads();

    float qm = sh_qmult;

    /* ── quantize to int8, writing directly to global memory ── */
    #pragma unroll 4
    for (int i = tid; i < K2; i += BLOCK_QUANT) {
        __nv_bfloat162 v = x_vec2[i];
        float f0 = __bfloat162float(v.x);
        float f1 = __bfloat162float(v.y);
        xq_row[i * 2]     = static_cast<int8_t>(__float2int_rn(f0 * qm));
        xq_row[i * 2 + 1] = static_cast<int8_t>(__float2int_rn(f1 * qm));
    }
    if ((K & 1) && tid == 0) {
        float f = __bfloat162float(x_row[K - 1]);
        xq_row[K - 1] = static_cast<int8_t>(__float2int_rn(f * qm));
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Kernel 4: dequant_int32_to_bf16 — elementwise dequantize
 *
 *   y_bf16[M,N] = float(y_int32[M,N]) * x_scale[M] * w_scale[N]
 *
 * grid(ceil(N/BLOCK_DEQUANT), M)  block(BLOCK_DEQUANT)
 * Replaces 3 torch ops: .float(), * scale, .to(bf16)
 * ═══════════════════════════════════════════════════════════════════ */

constexpr int BLOCK_DEQUANT = 256;

__global__ void dequant_int32_to_bf16_kernel(
    const int32_t*       __restrict__ y_int,   // [M, N]
    const float*         __restrict__ x_scale, // [M]
    const float*         __restrict__ w_scale, // [N]
    __nv_bfloat16*       __restrict__ y_out,   // [M, N]
    const int N)
{
    const int m = blockIdx.y;
    const int n = blockIdx.x * BLOCK_DEQUANT + threadIdx.x;
    if (n >= N) return;

    float xs = x_scale[m];
    float val = static_cast<float>(y_int[m * N + n]) * xs * w_scale[n];
    y_out[m * N + n] = __float2bfloat16(val);
}

/* ═══════════════════════════════════════════════════════════════════
 * Launchers (extern "C" for binding)
 * ═══════════════════════════════════════════════════════════════════ */

extern "C" {

void launch_int8_gemv_v2(
    const int8_t* x, const int8_t* W, int32_t* y,
    int M, int K, int N, cudaStream_t stream)
{
    dim3 grid(N, M);
    dim3 block(BLOCK_V2);
    int8_gemv_v2_kernel<<<grid, block, 0, stream>>>(x, W, y, K, N);
}

void launch_fused_w8a8_gemv(
    const __nv_bfloat16* x, const int8_t* W, const float* w_scale,
    __nv_bfloat16* y,
    int M, int K, int N, cudaStream_t stream)
{
    dim3 grid(N, M);
    dim3 block(BLOCK_FUSED);
    int smem = K + 4 * sizeof(float);
    fused_w8a8_gemv_kernel<<<grid, block, smem, stream>>>(
        x, W, w_scale, y, K, N);
}

void launch_quantize_bf16_to_int8(
    const __nv_bfloat16* x, int8_t* x_q, float* x_scale,
    int M, int K, cudaStream_t stream)
{
    dim3 grid(1, M);
    dim3 block(BLOCK_QUANT);
    quantize_bf16_to_int8_kernel<<<grid, block, 0, stream>>>(
        x, x_q, x_scale, K);
}

void launch_dequant_int32_to_bf16(
    const int32_t* y_int, const float* x_scale, const float* w_scale,
    __nv_bfloat16* y_out,
    int M, int N, cudaStream_t stream)
{
    dim3 grid((N + BLOCK_DEQUANT - 1) / BLOCK_DEQUANT, M);
    dim3 block(BLOCK_DEQUANT);
    dequant_int32_to_bf16_kernel<<<grid, block, 0, stream>>>(
        y_int, x_scale, w_scale, y_out, N);
}

}  /* extern "C" */
