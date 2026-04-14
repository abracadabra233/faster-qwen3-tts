#!/usr/bin/env python3
"""Quick TTS test with workarounds for Jetson aarch64 issues."""
import argparse
import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import torch


def ensure_pcm_audio(path: str, target_sr: int = 24000) -> str:
    """Convert audio to mono PCM at target_sr if needed (avoids torchaudio Resample NaN bug on aarch64)."""
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
    """ONNX Runtime speaker encoder on aarch64 returns NaN on first call — run a dummy call to flush it."""
    dummy = np.random.randn(24000).astype(np.float32)
    tmp = os.path.join(tempfile.gettempdir(), "_warmup.wav")
    sf.write(tmp, dummy, 24000)
    try:
        model.create_voice_clone_prompt(ref_audio=tmp, ref_text="", x_vector_only_mode=True)
    except Exception:
        pass
    os.remove(tmp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/app/models/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--ref-audio", default="ref_audio.wav")
    p.add_argument("--text", default="Hello, this is a test of the text to speech system.")
    p.add_argument("--language", default="English")
    p.add_argument("--output", default="/app/output/test_output.wav")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    from faster_qwen3_tts import FasterQwen3TTS

    print(f"Loading model: {args.model}")
    m = FasterQwen3TTS.from_pretrained(args.model, dtype=args.dtype)

    print("Warming up speaker encoder (aarch64 workaround)...")
    warmup_speaker_encoder(m.model)

    ref_audio = ensure_pcm_audio(args.ref_audio, target_sr=m.model.model.speaker_encoder_sample_rate)
    print(f"Reference audio: {ref_audio}")

    print(f"Generating: \"{args.text}\"")
    audio_list, sr = m.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=ref_audio,
        ref_text="",
        xvec_only=True,
    )

    audio = audio_list[0]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, audio, sr)
    duration = len(audio) / sr
    print(f"Done! {duration:.2f}s audio saved to {args.output}")


if __name__ == "__main__":
    main()
