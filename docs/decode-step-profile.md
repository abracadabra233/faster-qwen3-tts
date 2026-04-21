# Decode Step 性能剖析报告

> 平台: Jetson AGX Orin, CUDA 11.8, PyTorch 2.5.0a0
> 模型: Qwen3-TTS-12Hz-0.6B-Base
> 方法: CUDA Event 精确计时，每步测量各组件 GPU 耗时
> 剖析范围: 跳过前 5 步 warmup，剖析连续 80 步 decode

## 1. 组件级耗时分解

### 1.1 BF16 基线

| 组件                       | 平均耗时     | 标准差  | 占比      |
| -------------------------- | ------------ | ------- | --------- |
| **Predictor graph replay** | **28.76 ms** | 0.07 ms | **54.2%** |
| **Talker graph replay**    | **21.71 ms** | 0.05 ms | **40.9%** |
| Sampling + clone           | 1.95 ms      | 0.11 ms | 3.7%      |
| Embedding + cat            | 0.46 ms      | 0.02 ms | 0.9%      |
| Build embed (misc)         | 0.11 ms      | 0.00 ms | 0.2%      |
| Codec head                 | 0.05 ms      | 0.00 ms | 0.1%      |
| **总计**                   | **53.11 ms** | 0.14 ms | **100%**  |

### 1.2 INT8 v3 (FusedInt8Linear, v3_pipeline)

| 组件                       | 平均耗时     | 标准差  | 占比      | vs BF16                 |
| -------------------------- | ------------ | ------- | --------- | ----------------------- |
| **Predictor graph replay** | **24.27 ms** | 0.05 ms | **50.8%** | -4.49 ms (15.6% 快)     |
| **Talker graph replay**    | **20.77 ms** | 0.04 ms | **43.5%** | -0.94 ms (4.3% 快)      |
| Sampling + clone           | 2.01 ms      | 0.27 ms | 4.2%      | +0.06 ms                |
| Embedding + cat            | 0.46 ms      | 0.04 ms | 1.0%      | 0                       |
| Build embed (misc)         | 0.11 ms      | 0.00 ms | 0.2%      | 0                       |
| Codec head                 | 0.05 ms      | 0.00 ms | 0.1%      | 0                       |
| **总计**                   | **47.73 ms** | 0.27 ms | **100%**  | **-5.38 ms (10.1% 快)** |

### 1.3 可视化

```
BF16 每步耗时分解 (53.11 ms):
┌─────────────────────────────────┬────────────────────────┬───┐
│     Predictor (28.76 ms)        │   Talker (21.71 ms)    │2.6│
│           54.2%                 │        40.9%           │4.9│
└─────────────────────────────────┴────────────────────────┴───┘
v3 INT8 每步耗时分解 (47.73 ms):
┌────────────────────────────┬─────────────────────────┬───┐
│    Predictor (24.27 ms)    │    Talker (20.77 ms)    │2.7│
│          50.8%             │        43.5%            │5.6│
└────────────────────────────┴─────────────────────────┴───┘
                                          graph 外 ────────┘
```

## 2. 关键发现

### 2.1 Predictor 是最大开销（54%），不是 Talker

直觉上 Talker 有 28 层，应该比 Predictor（5 层）慢。但 Predictor 内部有 **15 次循环**
（生成 15 个 codebook），每次循环跑 5 层 Transformer：

```
一步 decode 的完整计算拆解:
Talker:    1 次 forward × 28 层 = 28 层                → 21.71 ms
Predictor: (1 次 prefill + 14 次 decode) × 5 层 = 75 层  → 28.76 ms
                                                     ──────────
                                            等效 103 层   50.47 ms
```

Predictor 的等效层数（75）远大于 Talker（28），所以耗时更长。

### 2.2 INT8 收益主要来自 Predictor

| 组件          | BF16 → v3     | 节省        | 原因                                      |
| ------------- | ------------- | ----------- | ----------------------------------------- |
| **Predictor** | 28.76 → 24.27 | **4.49 ms** | 35 Linear × 15 次 = 525 次 INT8 GEMV 调用 |
| **Talker**    | 21.71 → 20.77 | **0.94 ms** | 196 Linear × 1 次 = 196 次 INT8 GEMV 调用 |

Predictor 的 INT8 收益是 Talker 的 4.8 倍，因为 Linear 调用次数是 525 vs 196 = 2.7 倍，
且 Predictor 每次循环的非 Linear 开销（attention, norm 等）更紧凑。

### 2.3 Graph 外开销极小（5%）

```
Graph 内:      50.47 ms (95.0%)  → 这是 GPU 执行时间，无法靠消除 Python 开销优化
Graph 外:       2.64 ms (5.0%)   → embedding lookup, codec_head, sampling, tensor ops
```

Graph 外的 2.64 ms 包括：

- Sampling (top-k + softmax + multinomial): 1.95 ms — 最大的 graph 外开销
- Embedding lookup + cat: 0.46 ms
- Build talker input embedding: 0.11 ms
- Codec head (Linear, 1024→3072): 0.05 ms

### 2.4 延迟不随序列长度增长

```
Step timing trend:
  step  5 (pos= 15): 53.00 ms
  step 84 (pos= 94): 53.29 ms   → 几乎恒定
```

这是因为 SDPA attention 使用 StaticCache（预分配固定大小），CUDA Graph 中的 kernel
不管 `seq_pos` 是 15 还是 94 都执行相同计算（多余位置被 mask 掉但仍参与计算）。

## 3. KV Cache 内存分析

### 3.1 静态分配

```
Talker StaticCache:
  28 层 × 2(K+V) × 8(heads) × 2048(max_seq) × 128(dim) × 2(BF16)
  = 28 × 8 MB = 224 MB
Predictor StaticCache:
  5 层 × 类似结构 (max_seq 较小)
  ≈ 10-20 MB
```

### 3.2 每步带宽消耗

| 指标                      | 值       |
| ------------------------- | -------- |
| 平均 seq position         | ~60      |
| 每步 KV 读取量            | 6.5 MB   |
| 理论带宽耗时 (204.8 GB/s) | 33 us/步 |
| 97 步总 KV 读取           | 650 MB   |
| 总带宽耗时                | 3.3 ms   |

### 3.3 KV Cache 量化收益

| 方案        | 静态内存 | 带宽省/步 | 整个合成省 |
| ----------- | -------- | --------- | ---------- |
| BF16 (当前) | 224 MB   | —         | —          |
| INT8        | 112 MB   | 17 us     | 1.7 ms     |
| INT4        | 56 MB    | 25 us     | 2.5 ms     |

**结论**: KV Cache 量化在当前序列长度下收益很小（1.7-2.5 ms/合成，约 3-5%）。
但如果 `max_seq_len` 增大或生成更长音频，收益会线性增长。

## 4. 权重内存对比

| 组件                  | BF16          | INT8 v3      | 压缩率        |
| --------------------- | ------------- | ------------ | ------------- |
| Talker transformer    | 1439.6 MB     | 599.6 MB     | 2.4x          |
| Predictor transformer | 210.0 MB      | 60.0 MB      | 3.5x          |
| Embeddings            | 6.0 MB        | 6.0 MB       | 1.0x          |
| Codec head            | 6.0 MB        | 6.0 MB       | 1.0x          |
| **总权重**            | **1661.6 MB** | **671.6 MB** | **2.5x**      |
| KV Cache              | 224 MB        | 224 MB       | 1.0x (未量化) |
| **GPU 峰值**          | **2491 MB**   | **1908 MB**  | **-583 MB**   |

## 5. 优化空间分析

### 5.1 时间预算表

基于 v3 的 47.73 ms/step：

```
已优化 (INT8 Linear):
  ├─ Predictor Linear (525 次)    →  已用 v3 pipeline, 收益已兑现
  └─ Talker Linear (196 次)       →  已用 v3 pipeline, 收益已兑现
可优化 (graph 内非 Linear):
  ├─ Attention (SDPA, ~103 次)    →  KV Cache INT8: 省 ~1.7 ms
  ├─ RMSNorm (~309 次)            →  融合进 quant/dequant: 省 ~0.7 ms
  ├─ Residual + Norm 融合          →  4-kernel → 1-kernel per transition: 省 ~1 ms
  └─ Quant 共享 (QKV)             →  省 56 个 quant kernel: 省 ~0.3 ms
可优化 (graph 外):
  ├─ Sampling (1.95 ms)           →  CUDA kernel 化: 省 ~1 ms
  └─ Embedding + cat (0.46 ms)    →  融合: 省 ~0.2 ms
不可优化:
  └─ GPU kernel 执行时间下限       →  受限于内存带宽和 SM 算力
```

### 5.2 优化方向性价比排序

| 排序 | 优化方向                               | 预估收益 | 难度 | 优先级 |
| ---- | -------------------------------------- | -------- | ---- | ------ |
| 1    | 算子融合 (dequant+residual+norm+quant) | ~1-2 ms  | 中   | 高     |
| 2    | KV Cache INT8                          | ~1.7 ms  | 中   | 中     |
| 3    | Sampling CUDA kernel 化                | ~1 ms    | 低   | 中     |
| 4    | Quant 共享 (QKV)                       | ~0.3 ms  | 中   | 低     |
| 5    | RMSNorm+Quant 融合                     | ~0.7 ms  | 低   | 低     |

### 5.3 理论下限估算

```
当前 v3:       47.73 ms/step
所有可优化项全部实施:
  - 算子融合:       -1.5 ms
  - KV Cache INT8:  -1.7 ms
  - Sampling 优化:  -1.0 ms
  - 其他融合:       -1.0 ms
                    ────────
目标:             ~42.5 ms/step (1.23x vs BF16)
真实下限 (纯 GPU 计算 + 带宽极限):
  - 权重读取: 671 MB @ 204.8 GB/s = 3.3 ms (v3 权重)
  - KV Cache: 6.5 MB @ 204.8 GB/s = 0.03 ms
  - 计算量:   ~3.5B FLOPs / 275 GFLOPS(Orin peak) = 12.7 ms
  - 理论极限: ~13-15 ms (受计算和带宽共同约束)

  当前 47.73 ms vs 理论 ~15 ms = 3.2x gap
  gap 主要来自: CUDA Graph 调度、kernel 间空闲、非最优 kernel 效率
```

## 6. 结论

1. **Predictor 是性能瓶颈的大头**（54%），因为 15 个 codebook 的循环生成导致等效 75 层
2. **INT8 v3 在 Predictor 上收益最大**（省 4.49 ms），因为 Linear 调用次数最多（525 次）
3. **Graph 外开销仅 5%**（2.6 ms），优化空间有限
4. **KV Cache 量化收益有限**（1.7 ms），因为当前序列长度短且 StaticCache 的 mask 机制
5. **延迟不随序列增长**，因为 StaticCache 预分配了固定大小
6. **下一步最有价值的优化是算子融合**（减少 graph 内 kernel 数量），而不是进一步量化
