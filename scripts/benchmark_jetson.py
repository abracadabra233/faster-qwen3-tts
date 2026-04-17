#!/usr/bin/env python3
"""Jetson aarch64 benchmark — detailed per-stage timing for TTS pipeline."""
import os
import sys
import tempfile
import time

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


# ---------------------------------------------------------------------------
# Detailed single-run profiler
# ---------------------------------------------------------------------------

def profile_generate(model, text, ref_audio, ref_text, xvec_only=True, max_new_tokens=2048):
    """Run generate_voice_clone with fine-grained per-stage timing.

    Returns (audio_arrays, sr, stage_timings_dict).
    """
    import torch

    stages = {}

    # --- Stage 1: Voice Clone Prompt (speaker embedding / x-vector) ---
    # We invalidate the cache to force re-computation so we can measure it.
    saved_cache = dict(model._voice_prompt_cache)
    model._voice_prompt_cache.clear()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    input_texts = [model.model._build_assistant_text(text)]
    input_ids = model.model._tokenize_texts(input_texts)

    vcp, ref_ids, using_icl = model._resolve_voice_clone_prompt(
        input_ids=input_ids,
        ref_audio=ref_audio,
        ref_text=ref_text,
        xvec_only=xvec_only,
        append_silence=True,
        voice_clone_prompt=None,
    )

    torch.cuda.synchronize()
    stages["voice_clone_prompt_ms"] = (time.perf_counter() - t0) * 1000

    # Restore cache for subsequent runs
    model._voice_prompt_cache = saved_cache

    # --- Stage 2: Input Preparation (talker embedding build) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    m = model.model.model
    tie, tam, tth, tpe = model._build_talker_inputs_local(
        m=m,
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=vcp,
        languages=["English"],
        speakers=None,
        non_streaming_mode=False,
        instruct_ids=[None],
    )

    if not model._warmed_up:
        model._warmup(tie.shape[1])

    talker = m.talker
    config = m.config.talker_config
    talker.rope_deltas = None

    ref_codes = None
    if using_icl and vcp.get("ref_code") and vcp["ref_code"][0] is not None:
        ref_codes = vcp["ref_code"][0]

    torch.cuda.synchronize()
    stages["input_prep_ms"] = (time.perf_counter() - t0) * 1000

    # --- Stage 3 + 4: Prefill + Decode (via fast_generate) ---
    from faster_qwen3_tts.generate import fast_generate

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        codec_ids, gen_timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=model.predictor_graph,
            talker_graph=model.talker_graph,
            max_new_tokens=max_new_tokens,
        )

    torch.cuda.synchronize()
    stages["prefill_ms"] = gen_timing["prefill_ms"]
    stages["decode_ms"] = gen_timing["decode_s"] * 1000
    stages["decode_steps"] = gen_timing["steps"]
    stages["ms_per_step"] = gen_timing["ms_per_step"]
    stages["steps_per_s"] = gen_timing["steps_per_s"]

    # --- Stage 5: Codec Decode (vocoder) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if codec_ids is not None:
        speech_tokenizer = m.speech_tokenizer
        if ref_codes is not None:
            codes_for_decode = torch.cat([ref_codes.to(codec_ids.device), codec_ids], dim=0)
        else:
            codes_for_decode = codec_ids
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_for_decode.unsqueeze(0)})

        ref_len = ref_codes.shape[0] if ref_codes is not None else 0
        total_len = codes_for_decode.shape[0]
        audio_arrays = []
        for a in audio_list:
            if hasattr(a, "cpu"):
                a = a.flatten().cpu().numpy()
            else:
                a = a.flatten() if hasattr(a, "flatten") else a
            if ref_len > 0:
                cut = int(ref_len / max(total_len, 1) * len(a))
                a = a[cut:]
            audio_arrays.append(a)
    else:
        audio_arrays = [np.zeros(1, dtype=np.float32)]
        sr = model.sample_rate

    torch.cuda.synchronize()
    stages["codec_decode_ms"] = (time.perf_counter() - t0) * 1000

    # --- Derived metrics ---
    stages["total_ms"] = (
        stages["voice_clone_prompt_ms"]
        + stages["input_prep_ms"]
        + stages["prefill_ms"]
        + stages["decode_ms"]
        + stages["codec_decode_ms"]
    )
    audio_dur = len(audio_arrays[0]) / sr if audio_arrays else 0
    stages["audio_duration_s"] = audio_dur
    stages["rtf"] = audio_dur / (stages["total_ms"] / 1000) if stages["total_ms"] > 0 else 0

    return audio_arrays, sr, stages


def profile_generate_cached(model, text, ref_audio, ref_text, xvec_only=True, max_new_tokens=2048):
    """Same as profile_generate but uses the cached voice clone prompt (warm path)."""
    import torch

    stages = {}

    # --- Stage 1: Voice Clone Prompt (cached lookup) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    input_texts = [model.model._build_assistant_text(text)]
    input_ids = model.model._tokenize_texts(input_texts)

    vcp, ref_ids, using_icl = model._resolve_voice_clone_prompt(
        input_ids=input_ids,
        ref_audio=ref_audio,
        ref_text=ref_text,
        xvec_only=xvec_only,
        append_silence=True,
        voice_clone_prompt=None,
    )

    torch.cuda.synchronize()
    stages["voice_clone_prompt_ms"] = (time.perf_counter() - t0) * 1000

    # --- Stage 2: Input Preparation ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    m = model.model.model
    tie, tam, tth, tpe = model._build_talker_inputs_local(
        m=m,
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=vcp,
        languages=["English"],
        speakers=None,
        non_streaming_mode=False,
        instruct_ids=[None],
    )

    talker = m.talker
    config = m.config.talker_config
    talker.rope_deltas = None

    ref_codes = None
    if using_icl and vcp.get("ref_code") and vcp["ref_code"][0] is not None:
        ref_codes = vcp["ref_code"][0]

    torch.cuda.synchronize()
    stages["input_prep_ms"] = (time.perf_counter() - t0) * 1000

    # --- Stage 3 + 4: Prefill + Decode ---
    from faster_qwen3_tts.generate import fast_generate

    torch.cuda.synchronize()

    with torch.inference_mode():
        codec_ids, gen_timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=model.predictor_graph,
            talker_graph=model.talker_graph,
            max_new_tokens=max_new_tokens,
        )

    torch.cuda.synchronize()
    stages["prefill_ms"] = gen_timing["prefill_ms"]
    stages["decode_ms"] = gen_timing["decode_s"] * 1000
    stages["decode_steps"] = gen_timing["steps"]
    stages["ms_per_step"] = gen_timing["ms_per_step"]
    stages["steps_per_s"] = gen_timing["steps_per_s"]

    # --- Stage 5: Codec Decode ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if codec_ids is not None:
        speech_tokenizer = m.speech_tokenizer
        if ref_codes is not None:
            codes_for_decode = torch.cat([ref_codes.to(codec_ids.device), codec_ids], dim=0)
        else:
            codes_for_decode = codec_ids
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_for_decode.unsqueeze(0)})

        ref_len = ref_codes.shape[0] if ref_codes is not None else 0
        total_len = codes_for_decode.shape[0]
        audio_arrays = []
        for a in audio_list:
            if hasattr(a, "cpu"):
                a = a.flatten().cpu().numpy()
            else:
                a = a.flatten() if hasattr(a, "flatten") else a
            if ref_len > 0:
                cut = int(ref_len / max(total_len, 1) * len(a))
                a = a[cut:]
            audio_arrays.append(a)
    else:
        audio_arrays = [np.zeros(1, dtype=np.float32)]
        sr = model.sample_rate

    torch.cuda.synchronize()
    stages["codec_decode_ms"] = (time.perf_counter() - t0) * 1000

    # --- Derived ---
    stages["total_ms"] = (
        stages["voice_clone_prompt_ms"]
        + stages["input_prep_ms"]
        + stages["prefill_ms"]
        + stages["decode_ms"]
        + stages["codec_decode_ms"]
    )
    audio_dur = len(audio_arrays[0]) / sr if audio_arrays else 0
    stages["audio_duration_s"] = audio_dur
    stages["rtf"] = audio_dur / (stages["total_ms"] / 1000) if stages["total_ms"] > 0 else 0

    return audio_arrays, sr, stages


def print_stage_table(stages, label=""):
    """Pretty-print a per-stage timing breakdown."""
    total = stages["total_ms"]
    rows = [
        ("Voice Clone Prompt", stages["voice_clone_prompt_ms"]),
        ("Input Preparation", stages["input_prep_ms"]),
        ("Prefill (Talker)", stages["prefill_ms"]),
        ("Decode (AR loop)", stages["decode_ms"]),
        ("Codec Decode (Vocoder)", stages["codec_decode_ms"]),
    ]
    width = 50
    print(f"\n{'─' * width}")
    if label:
        print(f"  {label}")
        print(f"{'─' * width}")
    print(f"  {'Stage':<26} {'Time':>8}  {'%':>6}")
    print(f"  {'─' * 26} {'─' * 8}  {'─' * 6}")
    for name, ms in rows:
        pct = ms / total * 100 if total > 0 else 0
        print(f"  {name:<26} {ms:>7.1f}ms {pct:>5.1f}%")
    print(f"  {'─' * 26} {'─' * 8}  {'─' * 6}")
    print(f"  {'TOTAL':<26} {total:>7.1f}ms {'100.0':>5}%")
    print(f"  Audio: {stages['audio_duration_s']:.2f}s | "
          f"Steps: {stages.get('decode_steps', 0)} | "
          f"{stages.get('ms_per_step', 0):.1f}ms/step | "
          f"RTF: {stages['rtf']:.3f}")
    print(f"{'─' * width}")


def print_avg_table(all_stages, label=""):
    """Print averaged timing from multiple runs."""
    keys = [
        "voice_clone_prompt_ms", "input_prep_ms", "prefill_ms",
        "decode_ms", "codec_decode_ms", "total_ms",
        "audio_duration_s", "decode_steps", "ms_per_step", "steps_per_s", "rtf",
    ]
    avg = {}
    n = len(all_stages)
    for k in keys:
        avg[k] = sum(s.get(k, 0) for s in all_stages) / n

    total = avg["total_ms"]
    rows = [
        ("Voice Clone Prompt", avg["voice_clone_prompt_ms"]),
        ("Input Preparation", avg["input_prep_ms"]),
        ("Prefill (Talker)", avg["prefill_ms"]),
        ("Decode (AR loop)", avg["decode_ms"]),
        ("Codec Decode (Vocoder)", avg["codec_decode_ms"]),
    ]
    width = 50
    print(f"\n{'━' * width}")
    if label:
        print(f"  {label}")
        print(f"{'━' * width}")
    print(f"  {'Stage':<26} {'Avg':>8}  {'%':>6}")
    print(f"  {'─' * 26} {'─' * 8}  {'─' * 6}")
    for name, ms in rows:
        pct = ms / total * 100 if total > 0 else 0
        print(f"  {name:<26} {ms:>7.1f}ms {pct:>5.1f}%")
    print(f"  {'─' * 26} {'─' * 8}  {'─' * 6}")
    print(f"  {'TOTAL':<26} {total:>7.1f}ms {'100.0':>5}%")
    print(f"  Audio: {avg['audio_duration_s']:.2f}s | "
          f"Steps: {avg['decode_steps']:.0f} | "
          f"{avg['ms_per_step']:.1f}ms/step | "
          f"RTF: {avg['rtf']:.3f}")
    print(f"{'━' * width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import torch
from faster_qwen3_tts import FasterQwen3TTS

MODEL_SIZE = os.environ.get("MODEL_SIZE", "0.6B")
MODEL_PATH = os.environ.get("MODEL_PATH", f"/app/models/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base")
ref_audio_raw = os.path.join(PROJECT_DIR, "ref_audio.wav")

print(f"╔══════════════════════════════════════════════════╗")
print(f"║  Jetson Benchmark — Detailed Stage Profiling     ║")
print(f"╚══════════════════════════════════════════════════╝")
print(f"  Model : {MODEL_PATH}")
print(f"  GPU   : {torch.cuda.get_device_name(0)}")
print(f"  Torch : {torch.__version__}, CUDA: {torch.version.cuda}")

ref_audio_pcm = ensure_pcm_audio(ref_audio_raw)
print(f"  Ref   : {ref_audio_pcm}")

print("\n[1/5] Loading model...")
t_load = time.perf_counter()
_attn_impl = "eager"
print(f"Using attn_implementation={_attn_impl} (flash_attention_2 incompatible with CUDA Graph capture)")

model = FasterQwen3TTS.from_pretrained(
    MODEL_PATH, device="cuda", dtype=torch.bfloat16,
    attn_implementation=_attn_impl, max_seq_len=2048,
)
t_load = time.perf_counter() - t_load
print(f"  Model loaded in {t_load:.1f}s")

print("\n[2/5] Speaker encoder warmup (aarch64 workaround)...")
t_spk = time.perf_counter()
warmup_speaker_encoder(model.model)
t_spk = time.perf_counter() - t_spk
print(f"  Speaker encoder warmup: {t_spk:.2f}s")

text = (
    "Ladies and gentlemen, I have just been informed that this speech is being "
    "generated faster than I can speak it. The robots have officially won. "
    "Please remain calm."
)
ref_text = (
    "I'm confused why some people have super short timelines, yet at the same "
    "time are bullish on scaling up reinforcement learning atop LLMs. If we're "
    "actually close to a human-like learner, then this whole approach of training "
    "on verifiable outcomes."
)

# --- Warmup run (triggers CUDA graph capture) ---
print("\n[3/5] Warmup run (CUDA graph capture)...")
t0 = time.perf_counter()
audio_list, sr = model.generate_voice_clone(
    text=text[:50], language="English",
    ref_audio=ref_audio_pcm, ref_text=ref_text,
    max_new_tokens=20, xvec_only=True,
)
print(f"  Warmup done in {time.perf_counter() - t0:.2f}s")

# --- Throughput benchmark with detailed per-stage timing ---
RUNS = 5
print(f"\n[4/5] Throughput Benchmark ({RUNS} runs, detailed staging)")

# Run 0: cold voice-clone-prompt (cache miss)
print(f"\n  Run 0 (cold — no voice prompt cache):")
audio_arrays, sr, stages = profile_generate(
    model, text, ref_audio_pcm, ref_text, xvec_only=True,
)
print_stage_table(stages, label="Run 0  [cold voice prompt]")

# Runs 1..N: warm path (voice prompt cached)
all_warm_stages = []
for i in range(1, RUNS):
    audio_arrays, sr, stages = profile_generate_cached(
        model, text, ref_audio_pcm, ref_text, xvec_only=True,
    )
    print_stage_table(stages, label=f"Run {i}  [cached voice prompt]")
    all_warm_stages.append(stages)

if all_warm_stages:
    print_avg_table(all_warm_stages, label=f"AVERAGE (runs 1-{RUNS - 1}, cached)")

# --- Streaming TTFA with per-stage breakdown ---
CHUNK_SIZES = [4, 8, 12]
print(f"\n[5/5] Streaming TTFA (5 runs per chunk_size)")

import json

ttfa_by_chunk = {}
for cs in CHUNK_SIZES:
    ttfas = []
    full_timings = []
    for i in range(5):
        torch.cuda.synchronize()

        # Measure prep time
        t_prep_start = time.perf_counter()
        input_texts = [model.model._build_assistant_text(text)]
        input_ids = model.model._tokenize_texts(input_texts)
        vcp, ref_ids, using_icl = model._resolve_voice_clone_prompt(
            input_ids=input_ids,
            ref_audio=ref_audio_pcm,
            ref_text=ref_text,
            xvec_only=True,
            append_silence=True,
            voice_clone_prompt=None,
        )
        torch.cuda.synchronize()
        prep_ms = (time.perf_counter() - t_prep_start) * 1000

        # Measure total TTFA via streaming
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first_chunk_timing = None
        total_chunks = 0
        total_decode_ms = 0
        for chunk, chunk_sr, timing in model.generate_voice_clone_streaming(
            text=text, language="English",
            ref_audio=ref_audio_pcm, ref_text=ref_text,
            chunk_size=cs, xvec_only=True,
        ):
            if first_chunk_timing is None:
                torch.cuda.synchronize()
                ttfa_wall = (time.perf_counter() - t0) * 1000
                first_chunk_timing = timing
                ttfas.append(ttfa_wall)
            total_chunks += 1
            total_decode_ms += timing.get("decode_ms", 0)

        if first_chunk_timing is not None:
            full_timings.append({
                "prep_ms": prep_ms,
                "ttfa_wall_ms": ttfas[-1],
                "prefill_ms": first_chunk_timing.get("prefill_ms", 0),
                "first_chunk_decode_ms": first_chunk_timing.get("decode_ms", 0),
                "total_chunks": total_chunks,
            })

    mean_ttfa = np.mean(ttfas[1:]) if len(ttfas) > 1 else (ttfas[0] if ttfas else 0)
    ttfa_by_chunk[cs] = mean_ttfa

    print(f"\n  chunk_size={cs}:")
    for idx, ft in enumerate(full_timings):
        tag = " (warmup)" if idx == 0 else ""
        print(
            f"    run {idx}{tag}: TTFA={ft['ttfa_wall_ms']:.0f}ms "
            f"(prep={ft['prep_ms']:.0f}ms, prefill={ft['prefill_ms']:.0f}ms, "
            f"1st_chunk_decode={ft['first_chunk_decode_ms']:.0f}ms) "
            f"chunks={ft['total_chunks']}"
        )
    print(f"    → Mean TTFA (excl warmup): {mean_ttfa:.0f}ms")

# --- Save results ---
gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
results = {
    "gpu": gpu_name,
    "model": MODEL_SIZE,
    "dtype": "bf16",
    "model_load_s": round(t_load, 2),
    "speaker_warmup_s": round(t_spk, 2),
    "throughput_runs": [],
    "ttfa_by_chunk_ms": {str(k): float(v) for k, v in ttfa_by_chunk.items()},
}

# Record all run stages
if all_warm_stages:
    for s in all_warm_stages:
        results["throughput_runs"].append({k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()})
    avg_rtf = np.mean([s["rtf"] for s in all_warm_stages])
    results["rtf_mean"] = float(round(avg_rtf, 3))

out_file = os.path.join(PROJECT_DIR, "output", f"bench_results_{gpu_name}.json")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_file}")

if audio_arrays:
    sample_file = os.path.join(PROJECT_DIR, "output", f"sample_{MODEL_SIZE}.wav")
    sf.write(sample_file, audio_arrays[0], sr)
    print(f"Sample saved to {sample_file}")
