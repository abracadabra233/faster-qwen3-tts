# Predictor 优化深度分析

## 1. 核心问题：15 次循环的固定结构

Predictor 每个 decode step 执行固定 15 次 model.forward()（1 prefill + 14 decode），
生成 15 个 codebook token。这个循环被捕获为**一个 CUDA Graph**。

```
当前 Predictor graph.replay() 内部结构:
step 0 (prefill, 2 tokens):
  small_to_mtp(input) → pred_model.forward(seq=2) → lm_head[0] → sample → tok[0]
step 1..14 (decode, 1 token each):
  codec_embed[i](tok) → small_to_mtp → pred_model.forward(seq=1) → lm_head[i+1] → sample → tok[i+1]
每次 pred_model.forward(seq=1) = 5 层 Transformer:
  × 5: RMSNorm → QKV Linear → Attention(SDPA) → O Linear → RMSNorm → Gate+Up → SiLU*mul → Down
```

## 2. Roofline 分析：Predictor 单步

### 2.1 权重流量（访存瓶颈主体）

```
Predictor 单层 Linear 权重 (INT8 v3):
  QKV fused:  1024 × 4096 × 1 = 4.0 MB
  O proj:     2048 × 1024 × 1 = 2.0 MB
  Gate:       1024 × 3072 × 1 = 3.0 MB
  Up:         1024 × 3072 × 1 = 3.0 MB
  Down:       3072 × 1024 × 1 = 3.0 MB
  ─────────────────────────────────
  单层 Linear 权重:  15.0 MB (INT8) / 30.0 MB (BF16)
5 层合计: 75 MB (INT8) / 150 MB (BF16)
```

### 2.2 15 次循环的总权重读取

```
每次 pred_model.forward() 读取全部 5 层权重: 75 MB (INT8)
15 次循环 = 15 × 75 MB = 1125 MB 权重读取（无 cache hit 情况）
在 204.8 GB/s 下:
  理论最小时间 = 1125 MB / 204.8 GB/s = 5.49 ms

实测: 24.27 ms
实际带宽利用率: 1125 / 24.27 = 46.4 GB/s (仅 22.6% 峰值!)
```

### 2.3 为什么带宽利用率这么低？

24.27 ms 中权重读取只占 5.49 ms（理论），剩余 ~18.8 ms 去哪了？

```
剩余开销拆解（估算）:
① 非 Linear 计算 (每次循环):
   - RMSNorm × 3:          ~9 us × 15 = 0.14 ms
   - SDPA attention:        ~10 us × 15 = 0.15 ms (seq 2→16, 很短)
   - RoPE:                  ~3 us × 15 = 0.05 ms
   - SiLU × mul:            ~2 us × 15 = 0.03 ms
   - Residual add × 2:     ~2 us × 15 = 0.03 ms
   小计: ~0.4 ms
② v3 pipeline quant/dequant:
   - 525 次 Linear × (quant ~6us + dequant ~6us) = 6.3 ms
③ Sampling (在 graph 内):
   - 15 次 lm_head Linear (60 MB 权重)
   - 15 次 top-k + softmax + multinomial: ~15 × 30us = 0.45 ms
④ CUDA kernel 调度/空闲:
   - 525 Linear × 3 kernels = 1575 GEMV kernels
   - 加上 ~500 个非 Linear kernels
   - ~2000+ 个 kernel 间的调度间隙: ~2-3 ms
⑤ small_to_mtp projection (graph 内):
   - 15 次 Linear (1024→1024): ~15 × 0.1 ms = 1.5 ms
⑥ codec_embed lookup (14 次): ~0.1 ms
估算合计: 5.49 + 0.4 + 6.3 + 0.45 + 2.5 + 1.5 + 0.1 ≈ 16.7 ms
实测: 24.27 ms
Gap: ~7.5 ms → 可能来自 L2 cache miss、kernel 效率低、SDPA 实际比估算慢
```

## 3. 优化方向

### 方案 A: 权重常驻 L2 Cache（Weight Pinning）

**核心思想**: Predictor 权重只有 60 MB (INT8)，Orin L2 cache 4 MB。
虽然放不下全部权重，但 15 次循环中，后续循环会部分命中之前 evict 的数据。
**问题**: CUDA 没有显式的 L2 pinning API（Hopper 有 `setAccessPolicy`，Ampere 没有）。
但可以通过减少其他 kernel 对 L2 的竞争来间接提升命中率。
**具体做法**: 将 15 次循环中的 Linear 权重读取尽量紧凑排列——
例如先跑所有 layer 0 的所有 15 个 token，再跑 layer 1 的所有 15 个 token。
这样每层权重只读一次。
→ 这就是**方案 B**。

### 方案 B: Batched Decode（15 token 一起跑）— 最有前景

**核心思想**: 不要逐个 token 跑 15 次 5 层 Transformer。
改为每层一次处理多个 token，利用 Causal Mask 保证自回归正确性。

```
当前 (15 次串行 decode):
  for cb in range(15):
    for layer in range(5):
      x = layer.forward(x, kv_cache, seq=1)  ← 每次读一遍权重
优化后 (打包为一次 prefill-like forward):
  # 构造 15-token 的 causal mask，每个 token 只能看到之前的 token
  x = [tok0, tok1, ..., tok14]  # 打包为 seq_len=15+2=17
  for layer in range(5):
    x = layer.forward(x, kv_cache=None, seq=17)  ← 权重只读一次！
```

**关键洞察**: 15 个 codebook 的生成虽然是自回归的（每个依赖前一个），
但如果我们用 **causal mask** 让 position i 只能 attend 到 position 0..i，
那么一次 forward pass 就可以 **并行计算所有 15 个 position 的 output**。
等等——**这不对**。每个 codebook token 的 input embedding 依赖于前一个 token 的
采样结果（`codec_embed[i](tok_prev)`），所以不能真正并行。
但有一个变体：**Speculative / Medusa 风格**。

### 方案 C: 减少循环次数（Speculative Predictor）

**思路**: 不跑 15 次循环，而是：

1. 用一个小 MLP head 从第一次 forward 的 hidden states 直接预测所有 15 个 codebook
2. 或者跑 2-3 次循环（每次预测 5-8 个 codebook），用 multi-token prediction
   **风险**: 需要训练额外的 head，且可能影响音质。

### 方案 D: 合并 quant/dequant + 权重读取优化 — 最实际

当前 v3 pipeline 每个 Linear 是 3 个 kernel:

```
quantize_bf16_to_int8 → int8_gemv_v2 → dequant_int32_to_bf16
```

Predictor 每步 525 个 Linear × 3 = **1575 个 GEMV 相关 kernel**。
优化思路：

**D1: 减少 kernel 数**

- 四合一融合: `dequant + residual + rmsnorm + quant` = 1 kernel
- 每个 Linear transition 从 4 kernel → 1 kernel
- Predictor: 525 × (4→1) = 省 ~1575 个 kernel
  **D2: 权重预加载**
- Predictor 权重只有 60 MB (INT8)
- 在跑 quant kernel 时，预加载下一个 Linear 的权重到 L2
- 利用 CUDA streams 或 `__prefetch_global_l2` 指令
  **D3: Persistent kernel（权重常驻寄存器/shared memory）**
- 对于小 Linear（如 1024×1024 的 small_to_mtp），权重可以 tile 到 shared memory
- 一个 persistent kernel 处理多次 forward 的同一层

## 4. 方案 B 可行性深入分析

回到 Batched Decode 方案，虽然自回归依赖使完全并行不可能，
但我们可以**改变计算顺序**来减少权重读取。

### 4.1 当前顺序 (token-major)

```
for tok_idx in 0..14:           ← 15 次循环
  for layer_idx in 0..4:        ← 5 层
    read weights[layer_idx]     ← 每层权重读 15 次
    compute(tok, weights)
```

权重读取: 5 层 × 15 次 = 75 次权重读取
假设无 L2 cache: 75 × 15 MB = 1125 MB

### 4.2 改为 layer-major 顺序

```
for layer_idx in 0..4:          ← 5 层
  read weights[layer_idx]       ← 每层权重只读 1 次!
  for tok_idx in 0..14:         ← 15 次循环
    compute(tok, weights)       ← 权重已在 L2/寄存器中
```

权重读取: 5 层 × 1 次 = 5 次权重读取
总量: 5 × 15 MB = 75 MB (vs 1125 MB, 15x 减少!)
**问题**: 需要手写 layer-by-layer forward，打破 HuggingFace 的 `model.forward()` 封装。
因为 HF 的 forward 是 `for layer in layers: x = layer(x)` 的 layer-serial 模式。

### 4.3 Layer-major 的理论收益

```
权重带宽: 75 MB / 204.8 GB/s = 0.37 ms (vs 当前 5.49 ms 理论)
节省: 5.49 - 0.37 = 5.12 ms (理论最大)
但实际上每次还有 attention 的 KV cache I/O:
  每次 attention 读 KV cache: 5 层 × 2 × 8 × seq × 128 × 2B
  seq 从 2 增长到 16
  总 KV 读取: Σ(seq=2..16) × 5层 × 2KB × 8heads × 128dim × 2B ≈ 2.8 MB → 微不足道
非 Linear 计算不变: ~0.4 ms
Quant/dequant: 525 × 12us = 6.3 ms (v3) → 仍然是大头
```

### 4.4 Layer-major + quant 融合的组合收益

```
Layer-major 权重读取:                    0.37 ms (vs 5.49 ms)
四合一融合 (dequant+res+norm+quant):     省 ~3 ms (vs 6.3 ms)
Sampling 不变:                           0.45 ms
非 Linear 计算:                          0.4 ms
small_to_mtp (15 次):                    1.5 ms
Kernel 调度 (大幅减少 kernel 数):        ~0.5 ms (vs 2.5 ms)
───────────────────────────────────────
理论目标:                                ~6-8 ms (vs 当前 24.27 ms)
```

**这意味着 Predictor 可以从 24.27 ms 降到 6-8 ms，节省 16-18 ms！**

## 5. 实施优先级

| 方案                     | 预估收益 | 难度 | 依赖                                   |
| ------------------------ | -------- | ---- | -------------------------------------- |
| **D1: 四合一算子融合**   | ~3 ms    | 中   | 需写 CUDA kernel + hook model forward  |
| **B: Layer-major 重排**  | ~5 ms    | 高   | 需手写 predictor forward，打破 HF 封装 |
| D2: 权重预加载           | ~1-2 ms  | 中   | CUDA prefetch 指令                     |
| C: Speculative predictor | ~10+ ms  | 极高 | 需要训练                               |

推荐路径: **D1 → B → D2**
D1（四合一融合）不需要改模型结构，只在现有 v3 pipeline 基础上融合 kernel。
B（layer-major）需要手写 predictor 的 forward pass，但收益最大。
