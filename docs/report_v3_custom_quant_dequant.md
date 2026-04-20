# 第三版优化报告：Custom Quant/Dequant CUDA Kernels

## 1. 出发点：v2 的最后一个瓶颈

### 1.1 问题定位

v2 的 dp4a v2 kernel 本身已经很快（90 us, 153 GB/s, 峰值的 75%），
但端到端仍比 BF16 慢 18%。矛盾在于：

| 层级 | v2 INT8 | BF16 | v2 优势 |
|------|---------|------|---------|
| 纯 GEMV kernel | 90 us | 186 us | **2.08x 快** |
| v2_pipeline (in graph) | 146 us | 195 us | 1.33x 快 |
| 端到端 ms/step | 64.1 ms | 52.5 ms | **0.82x 慢** |

Kernel 层面 2x 快 → 端到端反而慢，损失在哪？

### 1.2 根因：torch 小 kernel 堆积

`v2_pipeline` 的 `_forward_v2_pipeline()` 展开为 10 个 CUDA kernel：

```
量化 x (6 个 kernel):
  ① x.float()           →  dtype cast
  ② .abs()              →  elementwise
  ③ .amax(dim=-1)       →  reduction
  ④ / 127.0             →  elementwise
  ⑤ .round().clamp()    →  elementwise
  ⑥ .to(torch.int8)     →  dtype cast

GEMV (1 个 kernel):
  ⑦ int8_gemv_v2()      →  dp4a kernel (90 us)

反量化 (3 个 kernel):
  ⑧ y.float()           →  dtype cast
  ⑨ * x_scale * w_scale →  elementwise
  ⑩ .to(torch.bfloat16) →  dtype cast
```

每个小 kernel 即使只有 5-10 us 的 GPU 执行时间，在 CUDA Graph 中也无法消除
（Graph 消除的是 CPU→GPU 的 launch 开销，不是 GPU 执行时间）。

231 个 Linear × 9 个小 kernel × ~6 us ≈ **12.5 ms 额外开销**。

### 1.3 优化思路

将 6 个量化 torch 操作合并为 **1 个自定义 CUDA kernel**，
将 3 个反量化 torch 操作合并为 **1 个自定义 CUDA kernel**。

每个 Linear 从 10 个 kernel 降为 **3 个 kernel**。

## 2. 实现方案

### 2.1 Kernel 1: `quantize_bf16_to_int8`

```
功能: BF16 x[M,K] → INT8 x_q[M,K] + float x_scale[M]
Grid: (1, M), Block: (256)

流程:
1. BF16x2 向量化加载 x[m,:] (一次读 2 个 BF16 = 4 字节)
2. Block-level reduce 求 absmax（warp_reduce_max + shared memory）
3. 计算 scale = absmax / 127, quant_mult = 127 / absmax
4. 向量化写出 INT8 量化结果

一个 kernel 替代: .float() + .abs() + .amax() + /127 + .round().clamp() + .to(int8)
```

### 2.2 Kernel 2: `dequant_int32_to_bf16`

```
功能: INT32 y[M,N] × float x_scale[M] × float w_scale[N] → BF16 y_out[M,N]
Grid: (ceil(N/256), M), Block: (256)

流程:
  y_out[m][n] = float(y[m][n]) * x_scale[m] * w_scale[n]
  → 转为 BF16 写出

一个 kernel 替代: .float() + * scale + .to(bf16)
```

### 2.3 v3_pipeline 集成

```python
def _forward_v3_pipeline(self, x):
    kernel = _get_kernel()
    x_int8, x_scale = kernel.quantize_bf16_to_int8(x)   # 1 kernel
    y_int32 = kernel.int8_gemv_v2(x_int8, self.weight_int8)  # 1 kernel
    y_bf16 = kernel.dequant_int32_to_bf16(y_int32, x_scale, self.weight_scale)  # 1 kernel
    return y_bf16
    # 3 kernels total, vs v2_pipeline 的 10 kernels
```

### 2.4 对比：三种 backend

| Backend | Kernel 数/Linear | 特点 |
|---------|-----------------|------|
| `v3_pipeline` (新默认) | **3** | custom quant + v2 GEMV + custom dequant |
| `v2_pipeline` | 10 | torch quant + v2 GEMV + torch dequant |
| `fused` | 1 | 单 kernel 全融合（但带宽低） |

## 3. 实验结果

### 3.1 Kernel 微基准测试 (K=1536, N=8960, M=1)

| 方案 | 延迟 (us) | 带宽 (GB/s) | vs BF16 |
|------|-----------|-------------|---------|
| BF16 cuBLAS | 180 | 153 | 1.00x |
| `torch._int_mm` (M=32 pad) | 398 | 35 | 0.45x (2.2x 慢) |
| dp4a v1 (第一版) | 187 | 73 | 0.96x |
| dp4a v2 (纯 kernel) | 85 | 162 | 2.12x 快 |
| Fused 单 kernel | 184 | 75 | 0.98x |
| **v3 pipeline (3 kernels)** | **97** | **142** | **1.85x 快** |

**v3 pipeline 开销仅比纯 v2 kernel 多 12 us**（97 - 85 = 12 us），
即 quantize (~6 us) + dequant (~6 us)，符合预期。

### 3.2 CUDA Graph 内延迟

| Pipeline | Graph replay 延迟 | vs BF16 |
|----------|-------------------|---------|
| BF16 | 195 us | 1.00x |
| v2_pipeline (10 kernels) | 146 us | 1.33x 快 |
| Fused 单 kernel | 207 us | 0.94x |
| **v3_pipeline (3 kernels)** | **105 us** | **1.86x 快** |

v3 在 CUDA Graph 中比 v2_pipeline 又快了 **40 us（28%）**，
正是消除了 7 个多余 torch 小 kernel 的收益。

### 3.3 端到端 TTS 推理

| 策略 | ms/step | RTF | 权重内存 | GPU 峰值 | vs BF16 |
|------|---------|-----|----------|----------|---------|
| BF16 基线 | 52.4 | 1.43x | 1650 MB | 2631 MB | 1.00x |
| torchao W8A8 (原始) | 271.5 | 0.29x | 1650 MB | 2264 MB | 0.19x |
| W8A8 + GEMV v1 | 82.2 | 0.91x | 1650 MB | 2280 MB | 0.64x |
| Fused W8A8 (v2_pipeline) | 64.1 | 1.17x | 660 MB | 2296 MB | 0.82x |
| **Fused W8A8 (v3_pipeline)** | **46.8** | **1.60x** | **660 MB** | **2264 MB** | **1.12x 快** |

## 4. 全版本演进总结

### 4.1 性能演进

```
端到端 ms/step:

  torchao W8A8 (原始)  ████████████████████████████████████████████████████  271.5 ms
  W8A8 + GEMV v1       ████████████████  82.2 ms
  Fused W8A8 (v2)      ████████████  64.1 ms
  BF16 基线            ██████████  52.4 ms
  Fused W8A8 (v3)      █████████  46.8 ms  ← INT8 首次超越 BF16
```

### 4.2 每版解决的核心问题

| 版本 | 核心问题 | 解决方案 | 提速 |
|------|----------|----------|------|
| v1 | `_int_mm` padding M=1→32 | dp4a GEMV kernel | 271→82 ms (3.3x) |
| v2 | 转置开销 + 低带宽 + torchao dispatch | int4 loads + FusedInt8Linear | 82→64 ms (1.28x) |
| v3 | torch 小 kernel 堆积 (10→3) | custom quant/dequant CUDA kernels | 64→47 ms (1.36x) |

### 4.3 累积优化效果

| 指标 | 原始 torchao | v3 最终 | 改进幅度 |
|------|-------------|---------|----------|
| 端到端延迟 | 271.5 ms | 46.8 ms | **5.8x 快** |
| vs BF16 | 0.19x (5.2x 慢) | **1.12x (12% 快)** | 翻转 |
| RTF | 0.29x | **1.60x** | 5.5x |
| 权重内存 | 1650 MB | **660 MB** | 2.5x 压缩 |
| GPU 峰值内存 | 2264 MB | **2264 MB** | -367 MB vs BF16 |

### 4.4 Kernel 数量演进

```
每个 Linear 层的 CUDA kernel 数量:

  torchao 原始 (safe_int_mm)    ████████████████  ~16 kernels (含 padding/dispatch)
  v1 (monkey-patch)             █████████████  ~13 kernels (含转置)
  v2 (v2_pipeline)              ██████████  10 kernels
  v2 (fused 单 kernel)          █  1 kernel (但慢)
  v3 (v3_pipeline)              ███  3 kernels  ← 最优平衡
```

## 5. 技术总结

### 5.1 关键认知

1. **CUDA Graph 不消除 GPU 执行时间**：只消除 CPU→GPU launch 开销。图中每个 kernel 仍需 GPU 时间。
2. **小 kernel 堆积是隐形杀手**：单个 torch 操作（如 `.abs()`）看似微不足道（~5 us），
   但 231 Linear × 9 ops = 2079 次调用，累计 12+ ms。
3. **INT8 的理论优势需要充分利用带宽才能兑现**：
   - 数据量减半（BF16→INT8）不自动等于 2x 快
   - 需要 vectorized loads（int4 = 128-bit）才能让内存子系统高效工作
   - v1 的 32-bit load 只有 72 GB/s（35%），v2 的 128-bit load 达到 153 GB/s（75%）
4. **融合不一定最优**：单个融合 kernel（fused）看似理想（1 kernel/Linear），
   但量化 x 的开销在每个 output column 都重复，导致计算瓶颈。
   3-kernel pipeline（quant 一次 + GEMV + dequant 一次）反而更快。

### 5.2 Jetson AGX Orin 平台特征

| 参数 | 值 |
|------|-----|
| GPU 架构 | Ampere (SM 8.7) |
| 内存带宽峰值 | 204.8 GB/s |
| v2 kernel 实际带宽 | 153 GB/s (75%) |
| v3 pipeline 等效带宽 | 142 GB/s (69%) |
| SM 数量 | 16 |
| `__dp4a` 支持 | 原生 |
| CUDA 版本 | 11.8 |
| PyTorch | 2.5.0a0 (JetPack 5) |

### 5.3 使用方式

```python
from ao.fused_int8_linear import enable_fused_int8

model = FasterQwen3TTS.from_pretrained(model_path, dtype="bfloat16")
info = enable_fused_int8(model)  # 自动量化 + 替换为 FusedInt8Linear (v3_pipeline)

# 推理 — 自动使用 v3 pipeline (3 custom CUDA kernels per Linear)
audio, sr = model.generate_voice_clone(text="...", ref_audio="ref.wav", ...)
```

Docker 构建与测试：
```bash
# 构建
docker build --network=host -f Dockerfile.jetson-ao-gemv -t faster-tts:jp5-ao-gemv .

# 端到端测试
docker compose run --rm v3-test-w8a8

# 完整 benchmark
docker compose run --rm v3-benchmark

# kernel 微测试
docker compose run --rm v3-micro-bench
docker compose run --rm v3-correctness
docker compose run --rm v3-cuda-graph
```
