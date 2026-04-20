# 第二版优化报告：Fused W8A8 GEMV + FusedInt8Linear

## 1. 出发点：v1 的三个瓶颈

v1 虽然解决了 `_int_mm` padding 问题（5x → 1.57x），但仍比 BF16 慢 57%。
通过性能分析定位到三个核心瓶颈：

### 1.1 转置开销（~15.5 ms/step）

`ao/int8_gemv.py` 中的 monkey-patch 代码：
```python
W = mat2.t().contiguous()  # [K,N] -> [N,K]，每个 Linear 都执行一次
```
这个转置操作被 CUDA Graph 捕获，每步 replay 都执行。
231 个 Linear × ~67 us/转置 ≈ 15.5 ms，占总耗时的 19%。

### 1.2 带宽利用率低（35% → 目标 75%）

v1 kernel 使用 `int32` load（32-bit，一次读 4 字节），导致内存事务效率低。
理论上 INT8 数据使用 `int4` load（128-bit，一次读 16 字节）可提升 4x 事务效率。

### 1.3 Kernel 数量多（5 个/Linear）

torchao 的 dispatch chain 在 CUDA Graph 内仍执行：
`choose_qparams → quantize → transpose → dp4a_gemv → dequant`

## 2. 优化方案

### 2.1 Tier 1：int4 向量化加载 + [N,K] 权重布局

新 kernel `int8_gemv_v2`：

```
改进点:
1. 权重直接存储为 [N,K] row-major，消除运行时转置
2. 使用 int4 (128-bit) 向量化 load，一次读 16 个 int8
3. 每次 int4 load 执行 4 次 __dp4a（16 元素/load）
4. Block size 从 256 降到 128（4 warps，减少归约开销）
```

### 2.2 Tier 2：完全融合 Kernel

新 kernel `fused_w8a8_gemv`：
```
BF16 输入 → 共享内存内量化为 INT8 → dp4a 点积 → 反量化 → BF16 输出
一个 kernel 替代原来的 5 个 kernel
```

### 2.3 Tier 3：FusedInt8Linear 模块

替换 torchao 的 monkey-patch 方案，直接替换 `nn.Linear`：

1. `torchao.quantize_()` 量化模型
2. 遍历模型，从 `AffineQuantizedTensor` 提取 INT8 权重和 scale
3. 替换为 `FusedInt8Linear`，直接调用自定义 kernel
4. 消除 torchao tensor subclass dispatch 开销

提供两个 backend：
- `v2_pipeline`：torch 量化 → v2 GEMV → torch 反量化（10 个 kernel）
- `fused`：单个融合 kernel（1 个 kernel）

## 3. 实验结果

### 3.1 Kernel 微基准测试 (K=1536, N=8960, M=1)

| 方案 | 延迟 (us) | 带宽 (GB/s) | vs BF16 |
|------|-----------|-------------|---------|
| BF16 cuBLAS | 186 | 148 | 1.00x |
| dp4a v1 (旧) | 190 | 72 | 0.98x |
| **dp4a v2 (int4 loads)** | **90** | **153** | **2.08x 快** |
| Fused 单 kernel | 185 | 74 | 1.01x |

**关键发现**：
- v2 standalone kernel 达到 153 GB/s（峰值的 **75%**），比 v1 快 **2.12x**
- INT8 权重只有 BF16 一半数据量，所以相同带宽下 v2 比 BF16 快 2x
- 但融合 kernel 因 Phase 1（量化 x）的开销，只达到 74 GB/s，未能超越 BF16

### 3.2 CUDA Graph 内延迟对比

| Pipeline | Graph replay 延迟 | vs BF16 |
|----------|-------------------|---------|
| BF16 | 195 us | 1.00x |
| v2_pipeline (10 torch ops) | 146 us | 1.33x 快 |
| Fused 单 kernel | 207 us | 0.94x |

v2_pipeline 在 CUDA Graph 中 1.33x 快于 BF16，但 10 个 torch 小 kernel 仍有累积开销。

### 3.3 端到端 TTS 推理

| 策略 | ms/step | RTF | 权重内存 | vs BF16 |
|------|---------|-----|----------|---------|
| BF16 基线 | 52.5 | 1.42x | 1650 MB | 1.00x |
| torchao W8A8 | 271.5 | 0.29x | 1650 MB | 0.19x |
| W8A8 + GEMV v1 | 82.2 | 0.91x | 1650 MB | 0.64x |
| **Fused W8A8 (v2_pipeline)** | **64.1** | **1.17x** | **660 MB** | **0.82x** |

### 3.4 改进效果总结

| 指标 | v1 → v2 改进 | 说明 |
|------|-------------|------|
| 单次 GEMV | 190 → 90 us | int4 向量化加载，2.12x |
| 端到端 | 82.2 → 64.1 ms | 消除转置 + FusedInt8Linear |
| 权重内存 | 1650 → 660 MB | INT8 存储 + 消除 torchao 包装 |
| GPU 峰值内存 | 2636 → 2296 MB | -340 MB |

## 4. 遗留问题

v2 端到端仍比 BF16 慢 18%（64.1 vs 52.5 ms）。分析原因：

**v2_pipeline 每个 Linear 需要 10 个 torch 小 kernel：**
```
量化 x (6 kernels):  .float() → .abs() → .amax() → /127 → .round().clamp() → .to(int8)
GEMV (1 kernel):      int8_gemv_v2
反量化 (3 kernels):   .float() → * scale → .to(bf16)
```

即使在 CUDA Graph 中消除了 CPU launch 开销，每个小 kernel 仍有 5-10 us 的 GPU 执行时间。
231 个 Linear × 9 个小 kernel × ~6 us ≈ **12.5 ms 额外开销**。

**这正是 v3 要解决的问题。**
