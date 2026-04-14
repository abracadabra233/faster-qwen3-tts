# Roofline Analysis: Faster-Qwen3-TTS 0.6B on Jetson AGX Orin

## 1. Hardware Specifications

**NVIDIA Jetson AGX Orin 64GB**

| Spec | Value |
|------|-------|
| GPU Architecture | Ampere (SM 8.7) |
| CUDA Cores | 2048 |
| Tensor Cores | 64 (3rd-gen) |
| Peak BF16 Tensor (dense) | 137.5 TOPS |
| Peak BF16 Tensor (sparse) | 275 TOPS |
| Peak FP32 CUDA | 5.3 TFLOPS |
| Peak FP16 CUDA | 10.6 TFLOPS |
| Memory | 64 GB LPDDR5 (unified CPU/GPU) |
| Memory Bandwidth | 204.8 GB/s |
| TDP | 15W - 60W (MAXN) |

**Roofline Ridge Point** (BF16 dense Tensor Cores):

\[
\text{Ridge Point} = \frac{\text{Peak Compute}}{\text{Peak Bandwidth}} = \frac{137.5 \times 10^{12}}{204.8 \times 10^{9}} \approx 671 \text{ FLOPs/byte}
\]

Any kernel with arithmetic intensity below 671 FLOPs/byte is **memory-bandwidth bound**; above it is **compute bound**. As we will show, single-token autoregressive decode sits at ~0.9 FLOPs/byte — orders of magnitude below the ridge.

---

## 2. Model Architecture (0.6B)

Source: `Qwen3-TTS-12Hz-0.6B-Base/config.json`

| Parameter | Talker | Code Predictor |
|-----------|--------|----------------|
| Layers | 28 | 5 |
| Hidden dim (H) | 1024 | 1024 |
| Num Q heads | 16 | 16 |
| Num KV heads | 8 (GQA, group=2) | 8 (GQA, group=2) |
| Head dim | 128 | 128 |
| FFN intermediate | 3072 | 3072 |
| Vocab size | 3072 (codec) | 2048 |
| RoPE | M-RoPE (3-axis: 24,20,20) | Standard 1D |
| Activation | SiLU | SiLU |
| Precision | BF16 | BF16 |

**Decode pipeline per step:**

```
┌──────────────────────────────────────────────────────────────┐
│                     One Decode Step                           │
│                                                              │
│  1. Talker forward (28 layers, 1 token)                      │
│     → 1 codec logit → sample first codebook token            │
│                                                              │
│  2. Predictor (5 layers):                                    │
│     → 1x prefill (2 tokens) + 14x decode (1 token each)     │
│     → 15 codebook tokens via 15 LM heads                    │
│                                                              │
│  3. Combine: 1 + 15 = 16 codec IDs → vocoder later          │
│                                                              │
│  Total: ~83.3 ms of audio per step (12 Hz codec)            │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. FLOPs and Memory Access per Decode Step

All analysis is for **batch_size=1, single-token decode** (the dominant regime). BF16 weights = 2 bytes per element.

### 3.1 Talker — Single Layer Breakdown

| Operation | Weight Shape | FLOPs | Weight Bytes | Notes |
|-----------|-------------|-------|--------------|-------|
| QKV projection | \[1024, 4096\] | 2 x 1024 x 4096 = 8.39M | 1024 x 4096 x 2 = 8 MB | Fused Q(2048)+K(1024)+V(1024)=4096 |
| QK-Norm (RMSNorm) | 2 x \[128\] | ~512 | 512 B | Per-head, negligible |
| RoPE | — | ~4K | 0 | In-place rotation |
| Attention (SDPA) | — | 2 x 16 x 1 x S x 128 | KV cache read: 2 x 8 x S x 128 x 2 | S = sequence length |
| O projection | \[2048, 1024\] | 2 x 2048 x 1024 = 4.19M | 2048 x 1024 x 2 = 4 MB | |
| RMSNorm (post-attn) | \[1024\] | ~2K | 2 KB | |
| FFN gate+up | \[1024, 6144\] | 2 x 1024 x 6144 = 12.58M | 1024 x 6144 x 2 = 12 MB | Fused gate(3072)+up(3072)=6144 |
| SiLU + element-wise mul | — | ~6K | 0 | |
| FFN down | \[3072, 1024\] | 2 x 3072 x 1024 = 6.29M | 3072 x 1024 x 2 = 6 MB | |
| RMSNorm (post-FFN) | \[1024\] | ~2K | 2 KB | |
| **Layer Total** | | **31.45M** | **30 MB** | |

> Note: Attention FLOPs and KV cache access scale with sequence length S. For typical S~100, attention adds ~0.4M FLOPs and ~0.4MB KV read per layer — small compared to weight-loading cost.

### 3.2 Talker — Full 28 Layers

| Metric | Value |
|--------|-------|
| Weight FLOPs (28 layers) | 28 x 31.45M = **880.6M** |
| Weight bytes (28 layers) | 28 x 30 MB = **840 MB** |
| Codec head (3072 logits) | 2 x 1024 x 3072 = 6.29M FLOPs, 6 MB |
| Embedding lookup | negligible |
| **Talker Total FLOPs** | **~887M** |
| **Talker Total Memory** | **~846 MB** |
| **Arithmetic Intensity** | 887M / 846MB = **1.05 FLOPs/byte** |

### 3.3 Predictor — Per Call and Total

The predictor runs **15 times** per decode step: 1 prefill (2 tokens) + 14 single-token decodes.

| Metric | Per Layer | 5 Layers | Notes |
|--------|-----------|----------|-------|
| Weight FLOPs (1 token) | 31.45M | 157.3M | Same layer structure as talker |
| Weight bytes (1 token) | 30 MB | 150 MB | |

| Call Type | Count | FLOPs per Call | Total FLOPs |
|-----------|-------|---------------|-------------|
| Prefill (2 tokens) | 1 | 2 x 157.3M = 314.6M | 314.6M |
| Single-token decode | 14 | 157.3M | 2,202.2M |
| LM heads (15 x \[1024, 2048\]) | 15 | 2 x 1024 x 2048 = 4.19M each | 62.9M |
| **Predictor Total FLOPs** | | | **~2,580M** |

Weight re-use: the same 5-layer weights (150 MB) are loaded 15 times from DRAM (no L2 re-use at this size on Orin's 4 MB L2):

| Metric | Value |
|--------|-------|
| Predictor weight bytes per call | 150 MB |
| Total memory traffic (15 calls) | 15 x 150 MB = **2,250 MB** |
| LM head weights (15 distinct) | 15 x 1024 x 2048 x 2 = **60 MB** |
| **Predictor Total Memory** | **~2,310 MB** |
| **Arithmetic Intensity** | 2,580M / 2,310MB = **1.12 FLOPs/byte** |

### 3.4 Full Decode Step Summary

| Component | FLOPs | Memory Traffic | Arithmetic Intensity |
|-----------|-------|---------------|---------------------|
| Talker (28 layers) | 887M | 846 MB | 1.05 FLOPs/byte |
| Predictor (5 layers x 15 calls) | 2,580M | 2,310 MB | 1.12 FLOPs/byte |
| Sampling + overhead | ~10M | ~1 MB | — |
| **Total per step** | **~3,477M** | **~3,157 MB** | **1.10 FLOPs/byte** |

---

## 4. Roofline Diagram

```
  Attainable
  TFLOPS
  (BF16)
     │
 137 ┤·····························································───── Peak BF16 Tensor (dense)
     │                                                          ╱
     │                                                        ╱
     │                                                      ╱
  10 ┤                                                    ╱
     │                                                  ╱
     │                                                ╱         ← Compute Bound
     │                                              ╱
   1 ┤                                            ╱
     │                                          ╱
     │          Memory Bound →                ╱
 0.2 ┤·····X·································╱
     │     ↑                               ╱
     │     │ Decode operating point       ╱
     │     │ (1.1 FLOPs/byte)           ╱
     │     │ = 0.225 TFLOPS            ╱
0.01 ┤     │                          ╱
     │                              ╱
     └────┬───┬───┬───┬───┬───┬───┬───┬───┬───────── Arithmetic Intensity
        0.1  1  10 100  1K  10K       671            (FLOPs/byte)
              ↑                        ↑
           Model                  Ridge Point
         (1.1 FLOPs/byte)     (671 FLOPs/byte)
```

The model operates at **1.1 FLOPs/byte**, roughly **610x below the ridge point**. Performance is entirely determined by memory bandwidth, not compute.

**Memory-bound theoretical throughput:**

\[
\text{Attainable} = \text{AI} \times \text{BW} = 1.1 \times 204.8 = 225 \text{ GFLOPS}
\]

This is only **0.16%** of the 137.5 TOPS BF16 peak — the Tensor Cores are almost entirely idle.

---

## 5. Theoretical vs Actual Latency

### 5.1 Theoretical Minimum (Memory-Bound)

The absolute lower bound is the time to stream all weights from DRAM once:

| Component | Weight Traffic | Theoretical Min Latency | Formula |
|-----------|---------------|------------------------|---------|
| Talker (28L) | 846 MB | 846 / 204.8 = **4.13 ms** | bytes / BW |
| Predictor (15 calls) | 2,310 MB | 2,310 / 204.8 = **11.28 ms** | bytes / BW |
| Overhead (sampling, etc.) | ~1 MB | ~0.005 ms | |
| **Total** | **3,157 MB** | **15.41 ms** | |

### 5.2 Measured Latency (CUDA Graph vs Baseline)

Source: README per-component breakdown table + `bench_results_Orin.json`

| Component | Baseline | CUDA Graph | Theoretical Min |
|-----------|----------|------------|-----------------|
| Talker (28L) | 75 ms | **12 ms** | 4.13 ms |
| Predictor (15 steps) | 190 ms | **26 ms** | 11.28 ms |
| Overhead | 65 ms | **16 ms** | ~0 ms |
| **Total per step** | **330 ms** | **54 ms** | **15.41 ms** |

### 5.3 Efficiency Analysis

| Metric | Baseline | CUDA Graph |
|--------|----------|------------|
| **Talker BW utilization** | 846 / (75e-3 x 204.8e3) = **5.5%** | 846 / (12e-3 x 204.8e3) = **34.4%** |
| **Predictor BW utilization** | 2310 / (190e-3 x 204.8e3) = **5.9%** | 2310 / (26e-3 x 204.8e3) = **43.4%** |
| **Overall BW utilization** | 3157 / (330e-3 x 204.8e3) = **4.7%** | 3157 / (54e-3 x 204.8e3) = **28.5%** |
| **Speedup** | 1x | **6.1x** |
| **Distance to theoretical** | 21.4x off | **3.5x off** |

```
  Bandwidth Utilization (% of 204.8 GB/s)
  100% ┤
      │
   80 ┤
      │
   60 ┤
      │  ┌────┐
   43 ┤  │Pred│ CUDA Graph
      │  │    │  ┌────┐
   34 ┤  │    │  │Talk│ CUDA Graph
   28 ┤──│────│──│────│── Overall CUDA Graph ──
      │  │    │  │    │
      │  │    │  │    │
      │  │    │  │    │
    6 ┤──│────│──│────│──┬────┬──── Baseline ──
    5 ┤  │    │  │    │  │Base│
      │  │    │  │    │  │line│
    0 ┤──┴────┴──┴────┴──┴────┴─────────────────
       Predictor  Talker  Overall
```

---

## 6. Where the Remaining 3.5x Gap Comes From

CUDA Graphs achieve 28.5% of theoretical memory bandwidth. The remaining 71.5% loss comes from:

### 6.1 KV Cache Access Overhead

The theoretical calculation only counts weight loading. During attention, the KV cache must also be read:

| Sequence Length S | KV Cache Read per Layer | 28 Layers Total |
|-------------------|------------------------|-----------------|
| 50 tokens | 2 x 8 x 50 x 128 x 2 = 0.2 MB | 5.6 MB |
| 100 tokens | 0.4 MB | 11.2 MB |
| 200 tokens | 0.8 MB | 22.4 MB |

At S=100, KV cache adds ~11 MB to the 846 MB weights — a small (~1.3%) correction for the talker, but more significant for the predictor's 15 repeated calls.

### 6.2 SDPA Kernel Efficiency

The SDPA (Scaled Dot-Product Attention) implementation on Orin without Flash Attention uses the `math` backend. This backend is not optimized for the GQA pattern (group=2) on SM 8.7 and achieves lower bandwidth than pure GEMM kernels.

### 6.3 Predictor Weight Re-loading (No L2 Reuse)

The predictor's 5-layer weights total 150 MB — far exceeding Orin's 4 MB L2 cache. Each of the 15 predictor calls reloads all weights from DRAM. If L2 could cache them, the 14 single-token decodes would be nearly free.

| Scenario | Predictor Memory Traffic |
|----------|------------------------|
| Current (no reuse) | 15 x 150 = 2,250 MB |
| If L2 cached (hypothetical) | 150 + 14 x ~0 = 150 MB |
| **Potential speedup** | **15x** for predictor |

### 6.4 Remaining Python/CUDA Overhead

Even with CUDA Graphs, 16 ms of "overhead" per step persists:
- Graph launch and synchronization latency
- Python interpreter between graph replays (predictor → talker sequencing)
- Embedding lookups and tensor concatenation (not captured in graph)
- Sampling (multinomial) on each of 15 predictor outputs + 1 talker output

### 6.5 Unified Memory Architecture

Jetson's unified memory means CPU and GPU share the 204.8 GB/s bandwidth. If the CPU is active (Python interpreter, audio codec), it competes for bandwidth with GPU weight loading.

---

## 7. RTF (Real-Time Factor) Derivation

The codec operates at **12 Hz** — each decode step produces 1/12 second = **83.33 ms** of audio.

| Metric | Formula | Value |
|--------|---------|-------|
| Audio per step | 1000 / 12 | 83.33 ms |
| Actual step time (CUDA Graph) | measured | 54 ms |
| **Theoretical RTF** | 83.33 / 54 | **1.54** |
| Theoretical step time (memory-bound) | calculated | 15.41 ms |
| **Upper-bound RTF** | 83.33 / 15.41 | **5.41** |
| **Measured RTF** (non-streaming) | bench_results_Orin.json | **1.437** |
| **Measured RTF** (README) | README table | **1.307** |

The measured RTF (1.307-1.437) is slightly below the per-step theoretical (1.54) because it includes **prefill latency** amortized over all steps. For a typical ~170-step generation, the ~500ms prefill adds ~3ms per step, reducing effective RTF.

### Streaming RTF by Chunk Size

| chunk_size | TTFA (ms) | RTF | Audio/Chunk (ms) | Overhead/Chunk |
|------------|-----------|-----|-------------------|----------------|
| 1 | 240 | 0.750 | 83 | 240 - 54 = 186 ms (graph capture amortization) |
| 2 | 266 | 1.042 | 167 | |
| 4 | 362 | 1.251 | 333 | |
| 8 | 556 | 1.384 | 667 | |
| 12 | 753 | 1.449 | 1000 | |
| Non-streaming | — | 1.57 | all | |

Smaller chunks have lower RTF because CUDA graph capture overhead is amortized over fewer steps. At chunk_size >= 2, the system is real-time (RTF > 1.0).

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────────┐
│                   Performance Summary                        │
│                                                             │
│  Theoretical peak (BF16):  137.5 TOPS                       │
│  Theoretical BW limit:     204.8 GB/s                       │
│  Model arithmetic intensity: 1.1 FLOPs/byte                 │
│  → Regime: DEEPLY MEMORY-BOUND (610x below ridge)           │
│                                                             │
│  ┌─────────────────────────────────────────────────┐        │
│  │         Latency per Decode Step                  │        │
│  │                                                  │        │
│  │  Theoretical min:    15.4 ms  ████                │        │
│  │  CUDA Graph actual:  54.0 ms  █████████████       │        │
│  │  Baseline actual:   330.0 ms  ████████████████████│        │
│  │                                        ████████  │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  CUDA Graph efficiency: 28.5% of memory BW                  │
│  CUDA Graph speedup:    6.1x over baseline                  │
│  RTF (real-time factor): 1.44 (faster than real-time)       │
│                                                             │
│  Key bottlenecks:                                           │
│    1. Predictor 15x weight reloads (no L2 reuse)            │
│    2. SDPA without Flash Attention on Orin                   │
│    3. Python overhead between graph replays                  │
│    4. Unified memory contention (CPU ↔ GPU)                 │
└─────────────────────────────────────────────────────────────┘
```

### Optimization Opportunities

| Optimization | Potential Impact | Difficulty |
|-------------|-----------------|------------|
| Weight quantization (INT8/INT4) | 2-4x less memory traffic → 2-4x RTF | Medium |
| Predictor weight caching (fuse 15 calls) | Up to 15x less predictor traffic | Hard |
| Flash Attention for Orin SM 8.7 | ~10-20% talker speedup | Medium |
| Full C++ inference (eliminate Python) | Remove 16ms overhead | Hard |
| Batch multiple requests | Higher arithmetic intensity → better utilization | Easy |
| JetPack 6 + newer PyTorch | Better SDPA kernels, potential TF32 paths | Easy |

---

*Analysis based on Qwen3-TTS-12Hz-0.6B-Base running on Jetson AGX Orin 64GB (MAXN mode), BF16 precision, CUDA Graphs enabled. Benchmark data from `bench_results_Orin.json` and faster-qwen3-tts README.*
