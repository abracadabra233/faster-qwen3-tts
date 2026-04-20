/*
 * int8_gemv.cu — dp4a-based INT8 GEMV for Jetson AGX Orin (SM 8.7)
 *
 * Computes y[M,N] = x[M,K] @ W[N,K]^T  (all INT8, output INT32)
 * using __dp4a intrinsic: 4x INT8 dot product per cycle.
 *
 * Designed for autoregressive decode (M=1..16) where torch._int_mm
 * requires M>16 and pads wastefully.
 *
 * Memory access pattern:
 *   - W is [N,K] row-major: each block reads one row contiguously (coalesced)
 *   - x is [M,K]: small, fits in L1/L2 cache, broadcast across blocks
 */

#include <cuda_runtime.h>
#include <cstdint>

constexpr int BLOCK_DIM = 256;

/*
 * Each block computes one element y[m][n].
 * grid = (N, M), block = (BLOCK_DIM,)
 * Threads cooperatively reduce along K using dp4a, then warp shuffle + smem.
 */
__global__ void int8_gemv_dp4a_kernel(
    const int8_t* __restrict__ x,    // [M, K]
    const int8_t* __restrict__ W,    // [N, K] row-major
    int32_t* __restrict__ y,         // [M, N]
    const int K,
    const int N)
{
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;

    const int K4 = K >> 2;  // K/4 (K must be multiple of 4)

    const int32_t* x_row = reinterpret_cast<const int32_t*>(x + m * K);
    const int32_t* W_row = reinterpret_cast<const int32_t*>(W + n * K);

    int32_t acc = 0;

    #pragma unroll 4
    for (int k4 = tid; k4 < K4; k4 += BLOCK_DIM) {
        acc = __dp4a(x_row[k4], W_row[k4], acc);
    }

    // Warp-level reduction (warp size = 32)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    // Block-level reduction across warps via shared memory
    __shared__ int32_t smem[BLOCK_DIM / 32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        smem[warp_id] = acc;
    }
    __syncthreads();

    // First warp reduces all partial sums
    if (warp_id == 0) {
        acc = (lane_id < (BLOCK_DIM / 32)) ? smem[lane_id] : 0;
        #pragma unroll
        for (int offset = (BLOCK_DIM / 64); offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane_id == 0) {
            y[m * N + n] = acc;
        }
    }
}

/*
 * Multi-row variant: each block handles multiple output rows (n)
 * using a wider grid for better SM utilization when N is small.
 * For the typical TTS case (N=1536..8960), the basic kernel is fine.
 */

extern "C" {

void launch_int8_gemv(
    const int8_t* x, const int8_t* W, int32_t* y,
    int M, int K, int N, cudaStream_t stream)
{
    dim3 grid(N, M);
    dim3 block(BLOCK_DIM);
    int8_gemv_dp4a_kernel<<<grid, block, 0, stream>>>(x, W, y, K, N);
}

}  // extern "C"
