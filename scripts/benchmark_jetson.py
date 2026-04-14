#!/usr/bin/env python3
"""Jetson aarch64 benchmark wrapper — applies speaker encoder warmup + audio fix, then runs throughput benchmark."""
import os
import sys
import tempfile
import numpy as np
import soundfile as sf

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# --- Monkey-patch the throughput benchmark to use local model + workarounds ---

import importlib.util, types
import torch
from faster_qwen3_tts import FasterQwen3TTS

MODEL_SIZE = os.environ.get("MODEL_SIZE", "0.6B")
MODEL_PATH = os.environ.get("MODEL_PATH", f"/app/models/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base")
ref_audio_raw = os.path.join(PROJECT_DIR, "ref_audio.wav")

print(f"=== Jetson Benchmark (CUDA Graphs) ===")
print(f"Model: {MODEL_PATH}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

ref_audio_pcm = ensure_pcm_audio(ref_audio_raw)
print(f"Ref audio: {ref_audio_pcm}")

print("\nLoading model...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_PATH, device="cuda", dtype=torch.bfloat16,
    attn_implementation="eager", max_seq_len=2048,
)

print("Speaker encoder warmup (aarch64 workaround)...")
warmup_speaker_encoder(model.model)

text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."

import time, json

print("\nWarmup run...")
t0 = time.perf_counter()
audio_list, sr = model.generate_voice_clone(
    text=text[:50], language="English",
    ref_audio=ref_audio_pcm, ref_text=ref_text,
    max_new_tokens=20, xvec_only=True,
)
print(f"Warmup: {time.perf_counter() - t0:.2f}s")

# --- Throughput benchmark ---
RUNS = 5
CHUNK_SIZES = [4, 8, 12]

print(f"\n--- Throughput ({RUNS} runs) ---")
rtf_results = []
for i in range(RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio_list, sr = model.generate_voice_clone(
        text=text, language="English",
        ref_audio=ref_audio_pcm, ref_text=ref_text,
        xvec_only=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    audio_dur = len(audio_list[0]) / sr
    rtf = audio_dur / elapsed
    rtf_results.append(rtf)
    print(f"  run {i}: {elapsed:.3f}s, audio={audio_dur:.2f}s, RTF={rtf:.3f}")

mean_rtf = np.mean(rtf_results[1:])
print(f"\nMean RTF (excl warmup): {mean_rtf:.3f}")

# --- Streaming TTFA ---
print(f"\n--- Streaming TTFA (5 runs per chunk_size) ---")
ttfa_by_chunk = {}
for cs in CHUNK_SIZES:
    ttfas = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for chunk, chunk_sr, timing in model.generate_voice_clone_streaming(
            text=text, language="English",
            ref_audio=ref_audio_pcm, ref_text=ref_text,
            chunk_size=cs, xvec_only=True,
        ):
            if len(ttfas) < i + 1:
                torch.cuda.synchronize()
                ttfas.append((time.perf_counter() - t0) * 1000)
    mean_ttfa = np.mean(ttfas[1:]) if len(ttfas) > 1 else ttfas[0]
    ttfa_by_chunk[cs] = mean_ttfa
    print(f"  chunk_size={cs}: TTFA={mean_ttfa:.0f}ms")

# --- Save results ---
gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
results = {
    "gpu": gpu_name, "model": MODEL_SIZE, "dtype": "bf16",
    "rtf_mean": float(mean_rtf),
    "rtf_all": [float(r) for r in rtf_results],
    "ttfa_by_chunk_ms": {str(k): float(v) for k, v in ttfa_by_chunk.items()},
}
out_file = os.path.join(PROJECT_DIR, "output", f"bench_results_{gpu_name}.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_file}")

if audio_list:
    sample_file = os.path.join(PROJECT_DIR, "output", f"sample_{MODEL_SIZE}.wav")
    sf.write(sample_file, audio_list[0], sr)
    print(f"Sample saved to {sample_file}")
