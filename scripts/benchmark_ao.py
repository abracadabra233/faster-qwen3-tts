#!/usr/bin/env python3
"""Benchmark BF16 vs W8A8 vs SmoothQuant vs W8A8+GEMV vs Fused W8A8 on Jetson.

Loads the model fresh for each strategy, runs N generations,
and prints a comparison table.

Usage:
  python scripts/benchmark_ao.py
  python scripts/benchmark_ao.py --runs 3 --strategies bf16 w8a8
  python scripts/benchmark_ao.py --runs 3 --strategies bf16 w8a8 w8a8_gemv fused_w8a8
"""
import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_ao")

BENCH_TEXT = (
    "Ladies and gentlemen, I have just been informed that this speech is being "
    "generated faster than I can speak it. The robots have officially won. "
    "Please remain calm."
)


def ensure_pcm_audio(path, target_sr=24000):
    info = sf.info(path)
    if info.subtype.startswith("PCM") and info.channels == 1 and info.samplerate == target_sr:
        return path
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample
        data = resample(data, int(len(data) * target_sr / sr)).astype(np.float32)
    out = os.path.join(PROJECT_DIR, "output", "ref_audio_pcm.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, data, target_sr, subtype="PCM_16")
    return out


def warmup_speaker_encoder(model_obj):
    dummy = np.random.randn(24000).astype(np.float32)
    tmp = os.path.join(tempfile.gettempdir(), "_warmup.wav")
    sf.write(tmp, dummy, 24000)
    try:
        model_obj.create_voice_clone_prompt(ref_audio=tmp, ref_text="", x_vector_only_mode=True)
    except Exception:
        pass
    os.remove(tmp)


def gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def run_benchmark(strategy, model_path, ref_audio, n_runs, alpha):
    """Load model, apply quantization, benchmark, return results dict."""
    from faster_qwen3_tts import FasterQwen3TTS

    torch.cuda.reset_peak_memory_stats()

    # --- Load ---
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        model_path, device="cuda", dtype=torch.bfloat16,
        attn_implementation="eager", max_seq_len=2048,
    )
    load_s = time.perf_counter() - t0

    warmup_speaker_encoder(model.model)
    mem_after_load = gpu_mem_mb()

    # --- Enable GEMV patch if needed ---
    use_gemv = strategy == "w8a8_gemv"
    if use_gemv:
        from ao.int8_gemv import enable_int8_gemv
        enable_int8_gemv()

    # --- Quantize ---
    quant_info = None
    quant_s = 0.0
    actual_quant = "w8a8" if strategy == "w8a8_gemv" else strategy
    if actual_quant == "w8a8":
        from ao import quantize_w8a8_dynamic
        t0 = time.perf_counter()
        quant_info = quantize_w8a8_dynamic(model)
        quant_s = time.perf_counter() - t0
    elif actual_quant == "fused_w8a8":
        from ao.fused_int8_linear import enable_fused_int8
        t0 = time.perf_counter()
        quant_info = enable_fused_int8(model)
        quant_s = time.perf_counter() - t0
    elif actual_quant == "smoothquant":
        from ao import quantize_smoothquant
        t0 = time.perf_counter()
        quant_info = quantize_smoothquant(
            model, ref_audio=ref_audio, alpha=alpha,
        )
        quant_s = time.perf_counter() - t0

    mem_after_quant = gpu_mem_mb()

    # --- Warmup run (triggers CUDA graph capture) ---
    logger.info("[%s] Warmup run (CUDA graph capture)...", strategy)
    model.generate_voice_clone(
        text=BENCH_TEXT[:50], language="English",
        ref_audio=ref_audio, ref_text="",
        max_new_tokens=20, xvec_only=True,
    )

    # --- Benchmark runs ---
    logger.info("[%s] Running %d benchmark iterations...", strategy, n_runs)
    timings = []
    audio_durations = []

    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        audio_list, sr = model.generate_voice_clone(
            text=BENCH_TEXT, language="English",
            ref_audio=ref_audio, ref_text="",
            xvec_only=True,
        )

        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        dur = len(audio_list[0]) / sr

        timings.append(wall)
        audio_durations.append(dur)
        logger.info("  run %d: %.2fs audio in %.2fs (RTF %.2f)", i, dur, wall, dur / wall)

    mem_peak = gpu_mem_mb()

    # Clean up GEMV patch to avoid cross-strategy contamination
    if use_gemv:
        from ao.int8_gemv import disable_int8_gemv
        disable_int8_gemv()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    avg_wall = np.mean(timings)
    avg_dur = np.mean(audio_durations)

    return {
        "strategy": strategy,
        "load_s": round(load_s, 2),
        "quantize_s": round(quant_s, 2),
        "mem_after_load_mb": round(mem_after_load, 0),
        "mem_after_quant_mb": round(mem_after_quant, 0),
        "mem_peak_mb": round(mem_peak, 0),
        "avg_wall_s": round(avg_wall, 3),
        "avg_audio_s": round(avg_dur, 2),
        "avg_rtf": round(avg_dur / avg_wall, 2) if avg_wall > 0 else 0,
        "quant_info": quant_info,
        "all_timings": [round(t, 3) for t in timings],
    }


def print_comparison(results):
    """Print a side-by-side comparison table."""
    strategies = [r["strategy"] for r in results]
    col_w = 14

    def row(label, key, fmt=".2f"):
        vals = []
        for r in results:
            v = r.get(key, "—")
            if isinstance(v, (int, float)):
                vals.append(f"{v:{fmt}}")
            else:
                vals.append(str(v))
        cells = "".join(f"{v:>{col_w}}" for v in vals)
        print(f"  {label:<28}{cells}")

    header = "".join(f"{s:>{col_w}}" for s in strategies)
    w = 28 + col_w * len(strategies)

    print(f"\n{'━' * w}")
    print(f"  {'COMPARISON':^{w - 4}}")
    print(f"{'━' * w}")
    print(f"  {'':28}{header}")
    print(f"  {'─' * (w - 4)}")

    row("Model load (s)", "load_s")
    row("Quantize (s)", "quantize_s")
    row("GPU mem after load (MB)", "mem_after_load_mb", ".0f")
    row("GPU mem after quant (MB)", "mem_after_quant_mb", ".0f")
    row("GPU mem peak (MB)", "mem_peak_mb", ".0f")
    print(f"  {'─' * (w - 4)}")
    row("Avg wall time (s)", "avg_wall_s", ".3f")
    row("Avg audio duration (s)", "avg_audio_s")
    row("Avg RTF (x realtime)", "avg_rtf")

    print(f"{'━' * w}")

    # Speedup relative to baseline
    if len(results) > 1:
        baseline = results[0]["avg_wall_s"]
        print(f"\n  Speedup vs {results[0]['strategy']}:")
        for r in results[1:]:
            speedup = baseline / r["avg_wall_s"] if r["avg_wall_s"] > 0 else 0
            print(f"    {r['strategy']:>14}: {speedup:.2f}x")
        print()


def main():
    p = argparse.ArgumentParser(description="Benchmark BF16 vs W8A8 vs SmoothQuant")
    p.add_argument("--model", default="/app/models/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--ref-audio", default=os.path.join(PROJECT_DIR, "ref_audio.wav"))
    p.add_argument("--runs", type=int, default=3, help="Benchmark iterations per strategy")
    p.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha")
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["bf16", "w8a8", "smoothquant"],
        choices=["bf16", "w8a8", "smoothquant", "w8a8_gemv", "fused_w8a8"],
        help="Strategies to benchmark (fused_w8a8 = single-kernel fused W8A8 GEMV)",
    )
    args = p.parse_args()

    print(f"\n{'╔' + '═' * 58 + '╗'}")
    print(f"{'║'} {'Jetson AO Benchmark':^56} {'║'}")
    print(f"{'╚' + '═' * 58 + '╝'}")
    print(f"  Model      : {args.model}")
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Torch      : {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Strategies : {', '.join(args.strategies)}")
    print(f"  Runs       : {args.runs}")

    ref_audio = ensure_pcm_audio(args.ref_audio)

    results = []
    for strategy in args.strategies:
        print(f"\n{'─' * 60}")
        print(f"  Benchmarking: {strategy}")
        print(f"{'─' * 60}")
        r = run_benchmark(strategy, args.model, ref_audio, args.runs, args.alpha)
        results.append(r)

    print_comparison(results)

    # Save results
    out_file = os.path.join(PROJECT_DIR, "output", "bench_ao_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    serializable = []
    for r in results:
        sr = dict(r)
        sr.pop("quant_info", None)
        serializable.append(sr)
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
