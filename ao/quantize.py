"""
W8A8 quantization strategies for Qwen3-TTS on Jetson.

Two strategies:
  1. W8A8 Dynamic  — zero calibration, weights static INT8, activations per-token dynamic INT8
  2. SmoothQuant   — requires calibration, smooths activation outliers into weights before INT8

Both target the talker transformer (28 layers) and code predictor (5 layers).
Embeddings, codec head, speech tokenizer, and speaker encoder are left in BF16.
"""
import logging
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration texts — diverse lengths and languages for SmoothQuant
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATION_TEXTS = [
    "Hello, this is a test of the text to speech system running on NVIDIA Jetson.",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    (
        "Ladies and gentlemen, I have just been informed that this speech is being "
        "generated faster than I can speak it. The robots have officially won."
    ),
    "大家好，我是一个运行在英伟达杰特森平台上的语音合成系统，很高兴认识你。",
    (
        "Artificial intelligence is transforming every industry, from healthcare "
        "to transportation, education, and entertainment. The possibilities are endless."
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_w8a8_config():
    """Get W8A8 dynamic quantization config, handling torchao API changes."""
    try:
        from torchao.quantization import int8_dynamic_activation_int8_weight
        return int8_dynamic_activation_int8_weight()
    except ImportError:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        return Int8DynamicActivationInt8WeightConfig()


def _get_smoothquant_imports():
    """Import SmoothQuant APIs, trying prototype path (torchao v0.12)."""
    try:
        from torchao.prototype.smoothquant import (
            insert_smooth_quant_observer_ as insert_obs,
            SmoothQuantConfig as sq_config_cls,
        )
        return insert_obs, sq_config_cls
    except ImportError:
        pass

    from torchao.quantization.smoothquant import (
        insert_smooth_quant_observer_ as insert_obs,
        SmoothQuantConfig as sq_config_cls,
    )
    return insert_obs, sq_config_cls


def _model_size_mb(module: torch.nn.Module) -> float:
    """Parameter memory in MB (accounts for quantized dtypes)."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 * 1024)


def _run_calibration_prefill(faster_model, text, ref_audio, ref_text, language):
    """Run one talker prefill + a few predictor calls for calibration.

    Exercises all Linear layers in both transformers with real data
    so the SmoothQuant observers can record activation statistics.
    """
    m = faster_model.model.model
    talker = m.talker

    input_texts = [faster_model.model._build_assistant_text(text)]
    input_ids = faster_model.model._tokenize_texts(input_texts)

    vcp, ref_ids, _ = faster_model._resolve_voice_clone_prompt(
        input_ids=input_ids,
        ref_audio=ref_audio,
        ref_text=ref_text,
        xvec_only=True,
        append_silence=True,
        voice_clone_prompt=None,
    )

    tie, tam, tth, tpe = faster_model._build_talker_inputs_local(
        m=m,
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=vcp,
        languages=[language],
        speakers=None,
        non_streaming_mode=False,
        instruct_ids=[None],
    )

    # --- Talker prefill (exercises all 28 transformer layers) ---
    with torch.no_grad():
        talker.rope_deltas = None
        out = talker.forward(
            inputs_embeds=tie,
            attention_mask=tam,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=tth,
            tts_pad_embed=tpe,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )

    # --- Predictor calibration (exercises 5 transformer layers) ---
    past_hidden = out.past_hidden
    logits = out.logits[:, -1, :]
    token = logits.argmax(dim=-1)

    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()

    last_id_hidden = talker_codec_embed(token.unsqueeze(1))
    pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)  # [1, 2, H]

    with torch.no_grad():
        projected = predictor.small_to_mtp_projection(pred_input)
        predictor.model(
            inputs_embeds=projected,
            use_cache=False,
            return_dict=True,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_w8a8_dynamic(faster_model) -> dict:
    """Apply W8A8 dynamic quantization (no calibration needed).

    Replaces nn.Linear layers in the talker and predictor transformers
    with INT8 weight + dynamic INT8 activation variants.
    Uses torch._int_mm under the hood for Tensor Core INT8 GEMM.

    Returns dict with timing and size info.
    """
    from torchao.quantization import quantize_

    w8a8_config = _get_w8a8_config()

    m = faster_model.model.model
    talker_model = m.talker.model
    predictor_model = m.talker.code_predictor.model

    info = {
        "strategy": "w8a8_dynamic",
        "talker_before_mb": _model_size_mb(talker_model),
        "predictor_before_mb": _model_size_mb(predictor_model),
    }

    t0 = time.perf_counter()

    logger.info("Quantizing talker transformer (W8A8 dynamic)...")
    quantize_(talker_model, w8a8_config)

    logger.info("Quantizing code predictor (W8A8 dynamic)...")
    quantize_(predictor_model, w8a8_config)

    info["quantize_time_s"] = time.perf_counter() - t0
    info["talker_after_mb"] = _model_size_mb(talker_model)
    info["predictor_after_mb"] = _model_size_mb(predictor_model)

    # Force CUDA graph re-capture with quantized model
    faster_model._warmed_up = False

    logger.info(
        "W8A8 done: talker %.0fMB→%.0fMB, predictor %.0fMB→%.0fMB (%.1fs)",
        info["talker_before_mb"], info["talker_after_mb"],
        info["predictor_before_mb"], info["predictor_after_mb"],
        info["quantize_time_s"],
    )
    return info


def quantize_smoothquant(
    faster_model,
    ref_audio: str,
    ref_text: str = "",
    language: str = "English",
    alpha: float = 0.5,
    calibration_texts: Optional[List[str]] = None,
) -> dict:
    """Apply SmoothQuant W8A8 quantization with calibration.

    1. Inserts observers into the talker and predictor Linear layers.
    2. Runs calibration forward passes to record activation statistics.
    3. Computes per-channel smoothing scales (balances activation/weight ranges).
    4. Applies the smoothing + INT8 quantization.

    Args:
        faster_model: FasterQwen3TTS instance (not yet warmed up is ideal).
        ref_audio: Path to reference audio for voice cloning calibration.
        ref_text: Transcription of reference audio.
        language: Language for calibration texts.
        alpha: Smoothing strength (0 = all on activations, 1 = all on weights).
               Default 0.5 is a good starting point for most models.
        calibration_texts: Optional list of texts; defaults to built-in diverse set.

    Returns dict with timing and size info.
    """
    from torchao.quantization import quantize_

    insert_smooth_quant_observer_, SmoothQuantConfig = _get_smoothquant_imports()
    w8a8_config = _get_w8a8_config()

    if calibration_texts is None:
        calibration_texts = DEFAULT_CALIBRATION_TEXTS

    m = faster_model.model.model
    talker_model = m.talker.model
    predictor_model = m.talker.code_predictor.model

    info = {
        "strategy": "smoothquant",
        "alpha": alpha,
        "n_calibration_texts": len(calibration_texts),
        "talker_before_mb": _model_size_mb(talker_model),
        "predictor_before_mb": _model_size_mb(predictor_model),
    }

    t0 = time.perf_counter()

    # --- Step 1: Insert observers ---
    logger.info("Inserting SmoothQuant observers (alpha=%.2f)...", alpha)
    insert_smooth_quant_observer_(talker_model, alpha=alpha)
    insert_smooth_quant_observer_(predictor_model, alpha=alpha)

    # --- Step 2: Calibration ---
    logger.info("Running calibration with %d texts...", len(calibration_texts))
    for i, text in enumerate(calibration_texts):
        logger.info("  calibration [%d/%d]: %s...", i + 1, len(calibration_texts), text[:40])
        _run_calibration_prefill(faster_model, text, ref_audio, ref_text, language)

    t_calib = time.perf_counter() - t0
    info["calibration_time_s"] = t_calib
    logger.info("Calibration done in %.1fs", t_calib)

    # --- Step 3: Apply SmoothQuant ---
    t1 = time.perf_counter()
    logger.info("Applying SmoothQuant to talker...")
    quantize_(talker_model, SmoothQuantConfig())

    logger.info("Applying SmoothQuant to predictor...")
    quantize_(predictor_model, SmoothQuantConfig())

    info["quantize_time_s"] = time.perf_counter() - t1
    info["total_time_s"] = time.perf_counter() - t0
    info["talker_after_mb"] = _model_size_mb(talker_model)
    info["predictor_after_mb"] = _model_size_mb(predictor_model)

    # Force CUDA graph re-capture with quantized model
    faster_model._warmed_up = False

    logger.info(
        "SmoothQuant done: talker %.0fMB→%.0fMB, predictor %.0fMB→%.0fMB (%.1fs total)",
        info["talker_before_mb"], info["talker_after_mb"],
        info["predictor_before_mb"], info["predictor_after_mb"],
        info["total_time_s"],
    )
    return info
