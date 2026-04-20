#!/usr/bin/env python3
"""TTS test with optional W8A8 / SmoothQuant / Fused INT8 quantization.

Usage:
  python scripts/tts_test_ao.py --quantize w8a8                     # basic W8A8
  python scripts/tts_test_ao.py --quantize w8a8 --enable-gemv-patch # W8A8 + dp4a GEMV
  python scripts/tts_test_ao.py --quantize fused_w8a8               # Fused W8A8 (single kernel)
  python scripts/tts_test_ao.py --quantize smoothquant              # SmoothQuant
  python scripts/tts_test_ao.py                                     # BF16 baseline
"""
import argparse
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
logger = logging.getLogger("tts_test_ao")


def ensure_pcm_audio(path: str, target_sr: int = 24000) -> str:
    info = sf.info(path)
    if info.subtype.startswith("PCM") and info.channels == 1 and info.samplerate == target_sr:
        return path
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample as scipy_resample
        n_samples = int(len(data) * target_sr / sr)
        data = scipy_resample(data, n_samples).astype(np.float32)
    out_path = os.path.join(tempfile.gettempdir(), "ref_audio_pcm.wav")
    sf.write(out_path, data, target_sr, subtype="PCM_16")
    return out_path


def warmup_speaker_encoder(model):
    dummy = np.random.randn(24000).astype(np.float32)
    tmp = os.path.join(tempfile.gettempdir(), "_warmup.wav")
    sf.write(tmp, dummy, 24000)
    try:
        model.create_voice_clone_prompt(ref_audio=tmp, ref_text="", x_vector_only_mode=True)
    except Exception:
        pass
    os.remove(tmp)


def main():
    p = argparse.ArgumentParser(description="TTS test with torchao quantization")
    p.add_argument("--model", default="/app/models/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--ref-audio", default="ref_audio.wav")
    p.add_argument("--text", default="Hello, this is a test of the text to speech system.")
    p.add_argument("--language", default="English")
    p.add_argument("--output", default="/app/output/test_ao.wav")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--quantize",
        choices=["w8a8", "smoothquant", "fused_w8a8", "none"],
        default="none",
        help="Quantization strategy to apply",
    )
    p.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha (0-1)")
    p.add_argument("--enable-gemv-patch", action="store_true",
                   help="Enable dp4a INT8 GEMV kernel (M<=16) monkey-patch")
    p.add_argument("--backend", choices=["v3_pipeline", "v2_pipeline", "fused"],
                   default=None, help="FusedInt8Linear backend (for fused_w8a8)")
    args = p.parse_args()

    if args.enable_gemv_patch:
        from ao.int8_gemv import enable_int8_gemv
        enable_int8_gemv()
        print("  dp4a INT8 GEMV kernel enabled")

    from faster_qwen3_tts import FasterQwen3TTS

    gemv_label = "+GEMV" if args.enable_gemv_patch else ""
    backend_label = f"/{args.backend}" if args.backend else ""
    # --- Load model ---
    print(f"\n{'═' * 60}")
    print(f"  TTS Test — quantize={args.quantize}{gemv_label}{backend_label}")
    print(f"{'═' * 60}")
    print(f"  Model : {args.model}")
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  Torch : {torch.__version__}, CUDA: {torch.version.cuda}")

    t0 = time.perf_counter()
    m = FasterQwen3TTS.from_pretrained(args.model, dtype=args.dtype)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    print("\nWarming up speaker encoder...")
    warmup_speaker_encoder(m.model)

    ref_audio = ensure_pcm_audio(args.ref_audio, target_sr=m.model.model.speaker_encoder_sample_rate)

    # --- Apply quantization ---
    quant_info = None
    if args.quantize == "w8a8":
        print("\nApplying W8A8 dynamic quantization...")
        from ao import quantize_w8a8_dynamic
        quant_info = quantize_w8a8_dynamic(m)

    elif args.quantize == "fused_w8a8":
        if args.backend:
            import ao.fused_int8_linear as _fil
            _fil._DEFAULT_BACKEND = args.backend
            print(f"\nApplying Fused W8A8 (backend={args.backend})...")
        else:
            print("\nApplying Fused W8A8 (single-kernel INT8)...")
        from ao.fused_int8_linear import enable_fused_int8
        quant_info = enable_fused_int8(m)

    elif args.quantize == "smoothquant":
        print(f"\nApplying SmoothQuant (alpha={args.alpha})...")
        from ao import quantize_smoothquant
        quant_info = quantize_smoothquant(
            m,
            ref_audio=ref_audio,
            language=args.language,
            alpha=args.alpha,
        )

    if quant_info:
        print(f"\n  Quantization results:")
        print(f"    Strategy    : {quant_info['strategy']}")
        print(f"    Talker      : {quant_info['talker_before_mb']:.0f}MB → {quant_info['talker_after_mb']:.0f}MB")
        print(f"    Predictor   : {quant_info['predictor_before_mb']:.0f}MB → {quant_info['predictor_after_mb']:.0f}MB")
        total_before = quant_info["talker_before_mb"] + quant_info["predictor_before_mb"]
        total_after = quant_info["talker_after_mb"] + quant_info["predictor_after_mb"]
        print(f"    Compression : {total_before:.0f}MB → {total_after:.0f}MB ({total_after/total_before*100:.0f}%)")

    # --- Generate ---
    print(f'\nGenerating: "{args.text[:60]}..."')
    t0 = time.perf_counter()
    audio_list, sr = m.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=ref_audio,
        ref_text="",
        xvec_only=True,
    )
    gen_time = time.perf_counter() - t0

    audio = audio_list[0]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, audio, sr)

    duration = len(audio) / sr
    rtf = duration / gen_time if gen_time > 0 else 0
    print(f"\n  Audio     : {duration:.2f}s @ {sr}Hz")
    print(f"  Wall time : {gen_time:.2f}s")
    print(f"  RTF       : {rtf:.2f}x realtime")
    print(f"  Saved to  : {args.output}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
