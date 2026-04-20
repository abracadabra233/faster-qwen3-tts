# 第一版优化报告：dp4a INT8 GEMV Kernel

## 1. 背景与出发点

### 1.1 问题发现

在 Jetson AGX Orin 上使用 `torchao` 的 W8A8 动态量化（`int8_dynamic_activation_int8_weight`）后，
端到端推理速度比 BF16 基线 **慢 5 倍**（271 ms/step vs 52 ms/step）。

### 1.2 根因分析

`torchao` 的 INT8 矩阵乘法底层调用 `torch._int_mm`，该接口要求 M >= 16。
对于 BS=1 自回归解码（M=1），`torchao` 的 `safe_int_mm` 会将输入 **padding 到 M=32**，
然后调用 `torch._int_mm`。这导致：

- 实际计算量膨胀 32 倍（M=1 → M=32）
- `torch._int_mm` 在 Jetson Orin 上没有 Triton 优化，走的是未优化的 CUDA 路径
- 量化/反量化的额外 kernel 开销

### 1.3 优化思路

针对 BS=1 解码场景（M=1），自定义一个基于 NVIDIA `__dp4a` 指令的 GEMV kernel：

- `__dp4a`：一条指令完成 4 个 INT8 乘加运算，Ampere+ 架构原生支持
- GEMV（向量-矩阵乘）：M=1 时退化为向量点积，天然适合 dp4a
- 直接处理 `[N,K]` 布局的权重，无需 padding 或 transpose

## 2. 实现方案

### 2.1 Kernel 设计

```
int8_gemv_dp4a_kernel:
  输入: x[M,K] (int8), W[N,K] (int8, row-major)
  输出: y[M,N] (int32)
  Grid: (N, M), Block: (256)
  
  每个 block 计算一个输出元素 y[m][n]:
  1. 256 个线程协作遍历 K 维度，每次读取 4 个 int8 (int32 load)
  2. 调用 __dp4a 完成 4 元素点积累加
  3. Warp shuffle + shared memory 归约得到最终结果
```

### 2.2 集成方式

通过 monkey-patch `torchao.kernel.intmm.safe_int_mm`：
- M <= 16：拦截到自定义 dp4a GEMV kernel
- M > 16：回退到原始 `torch._int_mm`

### 2.3 文件结构

| 文件 | 说明 |
|------|------|
| `csrc/int8_gemv.cu` | dp4a GEMV CUDA kernel |
| `csrc/bindings.cpp` | PyTorch C++ 绑定 |
| `ao/int8_gemv.py` | monkey-patch safe_int_mm |
| `Dockerfile.jetson-ao-gemv` | 编译安装自定义算子 |

## 3. 实验结果

### 3.1 Kernel 微基准测试 (K=1536, N=8960, M=1)

| 方案 | 延迟 (us) | 带宽 (GB/s) | vs BF16 |
|------|-----------|-------------|---------|
| BF16 cuBLAS (M=1) | 186 | 148 | 1.00x |
| `torch._int_mm` (M=32 pad) | 435 | 32 | 0.43x (2.3x 慢) |
| **dp4a GEMV v1 (M=1)** | **190** | **72** | **0.98x** |

**分析**：dp4a v1 kernel 比 `torch._int_mm` 快 2.3 倍，接近 BF16 水平。
但只达到峰值带宽的 35%（72 / 204.8 GB/s），说明有较大优化空间。

### 3.2 端到端 TTS 推理

| 策略 | ms/step | RTF | vs BF16 |
|------|---------|-----|---------|
| BF16 基线 | 52.5 | 1.42x | 1.00x |
| torchao W8A8 (无 GEMV) | 271.5 | 0.29x | 0.19x (5.2x 慢) |
| **W8A8 + dp4a GEMV v1** | **82.2** | **0.91x** | **0.64x (1.57x 慢)** |

### 3.3 瓶颈分析

v1 比 BF16 慢 57% 的原因拆解：

| 开销来源 | 估算时间 | 说明 |
|----------|----------|------|
| `mat2.t().contiguous()` 转置 | ~15.5 ms | monkey-patch 中将 [K,N] 转为 [N,K]，被 CUDA Graph 捕获 |
| 独立 quant/dequant kernels | ~3 ms | torchao 的 choose_qparams + quantize + dequant |
| dp4a kernel 本身 | ~2 ms | 72 GB/s vs BF16 的 148 GB/s，带宽利用率低 |
| 非 GEMV 开销 | ~7.5 ms | attention、embedding、layer norm 等 |

**结论**：v1 解决了 `_int_mm` padding 的 5x 慢问题，但转置开销和低带宽利用率仍是瓶颈。

## 4. 局限性

1. **转置开销**：每次 GEMV 需要 `mat2.t().contiguous()`，即使在 CUDA Graph 中也执行
2. **带宽利用率低**：72 GB/s 仅为峰值 204.8 GB/s 的 35%，因为使用 32-bit load（一次读 4 字节）
3. **Kernel 数量多**：torchao dispatch chain 仍保留 5 个 kernel per Linear
4. **内存无节省**：monkey-patch 方式不改变权重存储，内存仍是 1650 MB
