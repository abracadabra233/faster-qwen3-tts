#!/usr/bin/env python3
"""INT8 GEMV kernel correctness + performance tests.

Tests:
  1. correctness       — dp4a v1 GEMV vs torch reference
  2. correctness-v2    — dp4a v2 (int4 loads) vs torch reference
  3. correctness-fused — fused W8A8 GEMV (BF16->INT8->dp4a->BF16) vs torch reference
  4. benchmark         — BF16 mm vs torch._int_mm vs dp4a v1 vs dp4a v2 vs fused
  5. cuda-graph        — v1 CUDA Graph compatibility
  6. cuda-graph-fused  — fused kernel CUDA Graph compatibility
  7. integration       — enable_int8_gemv() + torchao W8A8 + generate
  8. integration-fused — enable_fused_int8() + TTS generate

Usage:
  python scripts/test_int8_gemv.py                          # all tests
  python scripts/test_int8_gemv.py --test correctness-fused # specific test
  python scripts/test_int8_gemv.py --test benchmark
"""
import argparse
import sys
import time

import torch


# ═══════════════════════════════════════════════════════════════════
# Correctness Tests
# ═══════════════════════════════════════════════════════════════════

def test_correctness():
    """Verify dp4a v1 GEMV matches reference INT8 matmul."""
    import int8_gemv_cuda

    print("=" * 60)
    print("  Correctness Test — dp4a v1")
    print("=" * 60)

    shapes = [
        (1, 1536, 8960),
        (1, 8960, 1536),
        (1, 1536, 1536),
        (1, 1024, 4096),
        (4, 1536, 8960),
        (16, 1536, 8960),
    ]

    all_pass = True
    for M, K, N in shapes:
        x = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        W = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)

        y_dp4a = int8_gemv_cuda.int8_gemv(x, W)
        y_ref = torch.mm(x.cpu().to(torch.int32), W.cpu().to(torch.int32).T).cuda()

        match = torch.equal(y_dp4a, y_ref)
        status = "PASS" if match else "FAIL"
        if not match:
            diff = (y_dp4a - y_ref).abs().max().item()
            print(f"  [{status}] M={M}, K={K}, N={N} — max diff: {diff}")
            all_pass = False
        else:
            print(f"  [{status}] M={M}, K={K}, N={N}")

    print()
    print(f"  {'All correctness tests PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


def test_correctness_v2():
    """Verify dp4a v2 GEMV (int4 loads) matches reference."""
    import int8_gemv_cuda

    print("=" * 60)
    print("  Correctness Test — dp4a v2 (int4 vector loads)")
    print("=" * 60)

    shapes = [
        (1, 1536, 8960),
        (1, 8960, 1536),
        (1, 1536, 1536),
        (1, 1024, 4096),
        (4, 1536, 8960),
        (16, 1536, 8960),
    ]

    all_pass = True
    for M, K, N in shapes:
        x = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        W = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)

        y_v2 = int8_gemv_cuda.int8_gemv_v2(x, W)
        y_ref = torch.mm(x.cpu().to(torch.int32), W.cpu().to(torch.int32).T).cuda()

        match = torch.equal(y_v2, y_ref)
        status = "PASS" if match else "FAIL"
        if not match:
            diff = (y_v2 - y_ref).abs().max().item()
            print(f"  [{status}] M={M}, K={K}, N={N} — max diff: {diff}")
            all_pass = False
        else:
            print(f"  [{status}] M={M}, K={K}, N={N}")

    print()
    print(f"  {'All v2 correctness tests PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


def test_correctness_fused():
    """Verify fused W8A8 GEMV (BF16 in, BF16 out) vs torch reference.

    Reference: y = (x_bf16 @ (W_int8 * scale).T) in float32.
    The fused kernel quantizes x on-the-fly, so there's quantization error.
    We use cosine similarity and normalized RMSE as error metrics, since
    near-zero output values cause relative error blow-up.
    """
    import int8_gemv_cuda

    print("=" * 60)
    print("  Correctness Test — Fused W8A8 GEMV")
    print("=" * 60)

    shapes = [
        (1, 1536, 8960),
        (1, 8960, 1536),
        (1, 1536, 1536),
        (1, 1024, 4096),
        (4, 1536, 8960),
    ]

    all_pass = True
    for M, K, N in shapes:
        x_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W_int8 = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
        w_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001

        y_fused = int8_gemv_cuda.fused_w8a8_gemv(x_bf16, W_int8, w_scale)

        W_deq = W_int8.float() * w_scale.unsqueeze(1)
        y_ref = torch.mm(x_bf16.float(), W_deq.T).to(torch.bfloat16)

        y_f = y_fused.float().flatten()
        y_r = y_ref.float().flatten()

        # Cosine similarity: robust to scale, ignores near-zero issues
        cos_sim = torch.nn.functional.cosine_similarity(y_f.unsqueeze(0), y_r.unsqueeze(0)).item()

        # Normalized RMSE: RMSE relative to output magnitude
        rmse = (y_f - y_r).pow(2).mean().sqrt().item()
        norm = y_r.pow(2).mean().sqrt().item()
        nrmse = rmse / max(norm, 1e-8)

        # INT8 per-token symmetric quant: cos_sim > 0.995, nrmse < 10%
        ok = cos_sim > 0.995 and nrmse < 0.10
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] M={M}, K={K}, N={N} — cos_sim: {cos_sim:.6f}, nrmse: {nrmse:.4f}")

    print()
    print(f"  {'All fused correctness tests PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


# ═══════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════

def test_benchmark():
    """Micro-benchmark: BF16 mm vs torch._int_mm vs dp4a v1 vs dp4a v2 vs fused."""
    import int8_gemv_cuda

    print("=" * 60)
    print("  Micro-Benchmark (K=1536, N=8960, M=1)")
    print("=" * 60)

    K, N = 1536, 8960
    N_WARMUP = 50
    N_ITER = 200

    results = {}

    # ── BF16 M=1 baseline ──
    x_bf16 = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    for _ in range(N_WARMUP):
        torch.mm(x_bf16, w_bf16.T)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        torch.mm(x_bf16, w_bf16.T)
    torch.cuda.synchronize()
    bf16_us = (time.perf_counter() - t0) / N_ITER * 1e6
    results["BF16 mm (M=1)"] = (bf16_us, 2)  # 2 bytes per element

    # ── torch._int_mm M=32 (pad) ──
    x_i8_32 = torch.randint(-128, 127, (32, K), device="cuda", dtype=torch.int8)
    w_i8_kn = torch.randint(-128, 127, (K, N), device="cuda", dtype=torch.int8)
    for _ in range(N_WARMUP):
        torch._int_mm(x_i8_32, w_i8_kn)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        torch._int_mm(x_i8_32, w_i8_kn)
    torch.cuda.synchronize()
    intmm_us = (time.perf_counter() - t0) / N_ITER * 1e6
    results["_int_mm (M=32 pad)"] = (intmm_us, 1)

    # ── dp4a v1 M=1 ──
    x_i8_1 = torch.randint(-128, 127, (1, K), device="cuda", dtype=torch.int8)
    W_nk = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
    for _ in range(N_WARMUP):
        int8_gemv_cuda.int8_gemv(x_i8_1, W_nk)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        int8_gemv_cuda.int8_gemv(x_i8_1, W_nk)
    torch.cuda.synchronize()
    dp4a_v1_us = (time.perf_counter() - t0) / N_ITER * 1e6
    results["dp4a v1 (M=1)"] = (dp4a_v1_us, 1)

    # ── dp4a v2 M=1 (int4 loads) ──
    for _ in range(N_WARMUP):
        int8_gemv_cuda.int8_gemv_v2(x_i8_1, W_nk)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        int8_gemv_cuda.int8_gemv_v2(x_i8_1, W_nk)
    torch.cuda.synchronize()
    dp4a_v2_us = (time.perf_counter() - t0) / N_ITER * 1e6
    results["dp4a v2 (M=1)"] = (dp4a_v2_us, 1)

    # ── Fused W8A8 GEMV M=1 ──
    x_bf16_1 = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    w_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001
    for _ in range(N_WARMUP):
        int8_gemv_cuda.fused_w8a8_gemv(x_bf16_1, W_nk, w_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        int8_gemv_cuda.fused_w8a8_gemv(x_bf16_1, W_nk, w_scale)
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - t0) / N_ITER * 1e6
    results["Fused W8A8 (M=1)"] = (fused_us, 1)

    # ── v3 pipeline: custom quant + v2 GEMV + custom dequant ──
    for _ in range(N_WARMUP):
        xq, xs = int8_gemv_cuda.quantize_bf16_to_int8(x_bf16_1)
        yi = int8_gemv_cuda.int8_gemv_v2(xq, W_nk)
        int8_gemv_cuda.dequant_int32_to_bf16(yi, xs, w_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        xq, xs = int8_gemv_cuda.quantize_bf16_to_int8(x_bf16_1)
        yi = int8_gemv_cuda.int8_gemv_v2(xq, W_nk)
        int8_gemv_cuda.dequant_int32_to_bf16(yi, xs, w_scale)
    torch.cuda.synchronize()
    v3_us = (time.perf_counter() - t0) / N_ITER * 1e6
    results["v3 pipeline (M=1)"] = (v3_us, 1)

    print()
    for name, (us, elem_bytes) in results.items():
        ratio = us / bf16_us
        data_bytes = K * N * elem_bytes
        bw = data_bytes / (us / 1e6) / 1e9
        print(f"  {name:25s}  {us:8.1f} us  {ratio:5.2f}x BF16  {bw:6.1f} GB/s")

    print()
    print(f"  v3 speedup vs BF16:            {bf16_us / v3_us:.2f}x")
    print(f"  v3 speedup vs fused:           {fused_us / v3_us:.2f}x")
    print(f"  v3 speedup vs _int_mm(pad):    {intmm_us / v3_us:.2f}x")
    print(f"  v2 speedup vs v1:              {dp4a_v1_us / dp4a_v2_us:.2f}x")
    return True


# ═══════════════════════════════════════════════════════════════════
# CUDA Graph Tests
# ═══════════════════════════════════════════════════════════════════

def test_cuda_graph():
    """Verify dp4a v1 GEMV works inside CUDA Graph capture/replay."""
    import int8_gemv_cuda

    print("=" * 60)
    print("  CUDA Graph Compatibility Test — v1")
    print("=" * 60)

    M, K, N = 1, 1536, 8960
    x = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
    W = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)

    x_buf = x.clone()
    y_buf = torch.empty(M, N, device="cuda", dtype=torch.int32)

    for _ in range(3):
        y_buf.copy_(int8_gemv_cuda.int8_gemv(x_buf, W))
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_buf.copy_(int8_gemv_cuda.int8_gemv(x_buf, W))

    print("  CUDA Graph captured successfully")

    x_new = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
    x_buf.copy_(x_new)
    graph.replay()
    torch.cuda.synchronize()

    y_ref = torch.mm(x_new.cpu().to(torch.int32), W.cpu().to(torch.int32).T).cuda()
    match = torch.equal(y_buf, y_ref)
    print(f"  Graph replay result correct: {match}")
    print()
    return match


def test_cuda_graph_fused():
    """Verify fused W8A8 GEMV works inside CUDA Graph capture/replay."""
    import int8_gemv_cuda

    print("=" * 60)
    print("  CUDA Graph Compatibility Test — Fused W8A8")
    print("=" * 60)

    M, K, N = 1, 1536, 8960
    x_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W_int8 = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
    w_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001

    x_buf = x_bf16.clone()
    y_buf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    for _ in range(3):
        y_buf.copy_(int8_gemv_cuda.fused_w8a8_gemv(x_buf, W_int8, w_scale))
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_buf.copy_(int8_gemv_cuda.fused_w8a8_gemv(x_buf, W_int8, w_scale))

    print("  CUDA Graph captured successfully")

    x_new = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    x_buf.copy_(x_new)
    graph.replay()
    torch.cuda.synchronize()

    # Reference
    W_deq = W_int8.float() * w_scale.unsqueeze(1)
    y_ref = torch.mm(x_new.float(), W_deq.T).to(torch.bfloat16)

    y_f = y_buf.float().flatten()
    y_r = y_ref.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(y_f.unsqueeze(0), y_r.unsqueeze(0)).item()
    ok = cos_sim > 0.995
    print(f"  Graph replay cosine similarity: {cos_sim:.6f} ({'OK' if ok else 'TOO LOW'})")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════

def test_integration():
    """End-to-end: enable_int8_gemv + W8A8 quantize + TTS generate."""
    import os

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_DIR)

    print("=" * 60)
    print("  Integration Test: dp4a GEMV + W8A8 + TTS")
    print("=" * 60)

    from ao.int8_gemv import enable_int8_gemv
    enable_int8_gemv()
    print("  INT8 GEMV monkey-patch enabled")

    from ao import quantize_w8a8_dynamic
    from faster_qwen3_tts import FasterQwen3TTS

    model_path = os.environ.get("MODEL_PATH", "/app/models/Qwen3-TTS-12Hz-0.6B-Base")
    print(f"  Loading model: {model_path}")
    m = FasterQwen3TTS.from_pretrained(model_path, dtype="bfloat16")

    print("  Applying W8A8 quantization...")
    info = quantize_w8a8_dynamic(m)
    print(f"  Quantized in {info['quantize_time_s']:.1f}s")

    ref_audio = os.path.join(PROJECT_DIR, "ref_audio.wav")
    print("  Generating audio...")
    t0 = time.perf_counter()
    audio_list, sr = m.generate_voice_clone(
        text="Hello, this is a dp4a INT8 GEMV test on Jetson.",
        language="English",
        ref_audio=ref_audio,
        ref_text="",
        xvec_only=True,
    )
    gen_time = time.perf_counter() - t0

    audio = audio_list[0]
    duration = len(audio) / sr
    rtf = duration / gen_time if gen_time > 0 else 0

    print(f"  Audio: {duration:.2f}s, Wall: {gen_time:.2f}s, RTF: {rtf:.2f}x")

    out_path = "/app/output/test_dp4a_gemv.wav"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import soundfile as sf
    sf.write(out_path, audio, sr)
    print(f"  Saved to {out_path}")
    print()
    return True


def test_integration_fused():
    """End-to-end: enable_fused_int8 + TTS generate."""
    import os

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_DIR)

    print("=" * 60)
    print("  Integration Test: Fused INT8 + TTS")
    print("=" * 60)

    from faster_qwen3_tts import FasterQwen3TTS

    model_path = os.environ.get("MODEL_PATH", "/app/models/Qwen3-TTS-12Hz-0.6B-Base")
    print(f"  Loading model: {model_path}")
    m = FasterQwen3TTS.from_pretrained(model_path, dtype="bfloat16")

    from ao.fused_int8_linear import enable_fused_int8
    print("  Applying Fused INT8 (W8A8 + FusedInt8Linear)...")
    info = enable_fused_int8(m)
    print(f"  Done in {info['total_time_s']:.1f}s")
    print(f"  Talker: {info['n_fused_talker']} layers, Predictor: {info['n_fused_predictor']} layers")

    ref_audio = os.path.join(PROJECT_DIR, "ref_audio.wav")
    print("  Generating audio...")
    t0 = time.perf_counter()
    audio_list, sr = m.generate_voice_clone(
        text="Hello, this is a fused INT8 GEMV test. One kernel per linear.",
        language="English",
        ref_audio=ref_audio,
        ref_text="",
        xvec_only=True,
    )
    gen_time = time.perf_counter() - t0

    audio = audio_list[0]
    duration = len(audio) / sr
    rtf = duration / gen_time if gen_time > 0 else 0

    print(f"  Audio: {duration:.2f}s, Wall: {gen_time:.2f}s, RTF: {rtf:.2f}x")

    out_path = "/app/output/test_fused_int8.wav"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import soundfile as sf
    sf.write(out_path, audio, sr)
    print(f"  Saved to {out_path}")
    print()
    return True


# ═══════════════════════════════════════════════════════════════════
# V3 Pipeline Tests
# ═══════════════════════════════════════════════════════════════════

def test_correctness_v3():
    """Verify v3 pipeline (custom quant + v2 GEMV + custom dequant) vs reference."""
    import int8_gemv_cuda

    print("=" * 60)
    print("  Correctness Test — v3 pipeline (custom quant/dequant)")
    print("=" * 60)

    shapes = [
        (1, 1536, 8960),
        (1, 8960, 1536),
        (1, 1536, 1536),
        (1, 1024, 4096),
        (4, 1536, 8960),
    ]

    all_pass = True
    for M, K, N in shapes:
        x_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W_int8 = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
        w_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001

        x_q, x_s = int8_gemv_cuda.quantize_bf16_to_int8(x_bf16)
        y_i32 = int8_gemv_cuda.int8_gemv_v2(x_q, W_int8)
        y_v3 = int8_gemv_cuda.dequant_int32_to_bf16(y_i32, x_s, w_scale)

        W_deq = W_int8.float() * w_scale.unsqueeze(1)
        y_ref = torch.mm(x_bf16.float(), W_deq.T).to(torch.bfloat16)

        y_f = y_v3.float().flatten()
        y_r = y_ref.float().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(y_f.unsqueeze(0), y_r.unsqueeze(0)).item()
        rmse = (y_f - y_r).pow(2).mean().sqrt().item()
        norm = y_r.pow(2).mean().sqrt().item()
        nrmse = rmse / max(norm, 1e-8)

        ok = cos_sim > 0.995 and nrmse < 0.10
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] M={M}, K={K}, N={N} — cos_sim: {cos_sim:.6f}, nrmse: {nrmse:.4f}")

    print()
    print(f"  {'All v3 correctness tests PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


def test_cuda_graph_v3():
    """Verify v3 pipeline works inside CUDA Graph capture/replay."""
    import int8_gemv_cuda

    print("=" * 60)
    print("  CUDA Graph Compatibility Test — v3 pipeline")
    print("=" * 60)

    M, K, N = 1, 1536, 8960
    x_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W_int8 = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
    w_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001

    x_buf = x_bf16.clone()
    y_buf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def v3_step():
        xq, xs = int8_gemv_cuda.quantize_bf16_to_int8(x_buf)
        yi = int8_gemv_cuda.int8_gemv_v2(xq, W_int8)
        y_buf.copy_(int8_gemv_cuda.dequant_int32_to_bf16(yi, xs, w_scale))

    for _ in range(3):
        v3_step()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        v3_step()
    print("  CUDA Graph captured (3 kernels)")

    x_new = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    x_buf.copy_(x_new)
    graph.replay()
    torch.cuda.synchronize()

    W_deq = W_int8.float() * w_scale.unsqueeze(1)
    y_ref = torch.mm(x_new.float(), W_deq.T).to(torch.bfloat16)

    y_f = y_buf.float().flatten()
    y_r = y_ref.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(y_f.unsqueeze(0), y_r.unsqueeze(0)).item()
    ok = cos_sim > 0.995
    print(f"  Graph replay cosine similarity: {cos_sim:.6f} ({'OK' if ok else 'TOO LOW'})")

    # Benchmark graph replay latency
    N_WARMUP, N_ITER = 50, 500
    for _ in range(N_WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        graph.replay()
    torch.cuda.synchronize()
    v3_graph_us = (time.perf_counter() - t0) / N_ITER * 1e6
    print(f"  v3 graph replay latency: {v3_graph_us:.1f} us")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test",
        choices=[
            "correctness", "correctness-v2", "correctness-fused", "correctness-v3",
            "benchmark",
            "cuda-graph", "cuda-graph-fused", "cuda-graph-v3",
            "integration", "integration-fused",
            "all",
        ],
        default="all",
    )
    args = p.parse_args()

    tests = {
        "correctness": test_correctness,
        "correctness-v2": test_correctness_v2,
        "correctness-fused": test_correctness_fused,
        "correctness-v3": test_correctness_v3,
        "benchmark": test_benchmark,
        "cuda-graph": test_cuda_graph,
        "cuda-graph-fused": test_cuda_graph_fused,
        "cuda-graph-v3": test_cuda_graph_v3,
        "integration": test_integration,
        "integration-fused": test_integration_fused,
    }

    if args.test == "all":
        run = list(tests.items())
    else:
        run = [(args.test, tests[args.test])]

    ok = True
    for name, fn in run:
        try:
            result = fn()
            if not result:
                ok = False
        except Exception as e:
            print(f"\n  [{name}] EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
