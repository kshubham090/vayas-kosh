"""Whisper large-v3, zero-shot, via direct model calls (not the
transformers `pipeline()` abstraction).

Verified interface: `openai/whisper-large-v3` on the Hugging Face Hub,
WhisperForConditionalGeneration + WhisperProcessor.

CONFIRMED WORKING END-TO-END (2026-08-16): real run against a Common
Voice Hindi clip (common_voice_hi_26127974.mp3):
  gold: इंडिया टुडे ग्रुप ने प्रधानमंत्री को श्वेत पत्र सौंपा
  hyp:  इंडिया टूडे गुरुप ने प्रधान मंतरी को श्वेत पत्र सौपा।
More deviation than MMS/IndicConformer on this one clip -- plausible
given large-v3 is a general multilingual model, not Hindi-specialized.

REWRITTEN FROM pipeline() (2026-08-24) after three separate real
failures surfaced by actual runs on Kaggle's infra (P100 and T4, so not
GPU-specific -- a genuine fragility in the high-level API across
transformers/torch version combos):
1. Local dev machine: pipeline()'s file-path input mode shells out to a
   system `ffmpeg` binary that wasn't installed.
2. Kaggle + downgraded torch 2.7.1: pipeline()'s dict-input path imports
   `torchcodec`, whose compiled library didn't match that torch build --
   "Could not load libtorchcodec".
3. Kaggle + native torch 2.10.0: a genuine transformers pipeline bug --
   `processed.pop("num_frames")` with no default in
   automatic_speech_recognition.py's preprocess(), which only exists
   when audio arrives via dict input (not the file-path path) --
   KeyError: 'num_frames'.
Every other wrapper in this module (MMS, IndicConformer, IndicWhisper)
already calls its model directly instead of through pipeline() and none
of them hit any of this -- so this rewrite matches the pattern that was
already proven robust, rather than continuing to patch pipeline()'s
input-handling edge cases one at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vayas.audit.base import ASRSystem, default_device, load_audio_mono_16k

MODEL_ID = "openai/whisper-large-v3"


class WhisperSystem(ASRSystem):
    name = "whisper-large-v3"

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device or default_device()
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        # fp16 on CUDA: large-v3 is ~1.55B params (~3.1GB fp16 weights
        # alone), tight against this project's dev GPU's 4GB VRAM.
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._processor = WhisperProcessor.from_pretrained(MODEL_ID)
        self._model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(self._device)

    def transcribe(self, audio_path: Path, lang: str) -> str:
        import torch

        self._ensure_loaded()

        wav = load_audio_mono_16k(audio_path)
        inputs = self._processor(wav.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")
        # Match the model's dtype explicitly -- an IndicWhisper sibling
        # wrapper hit a real fp16/fp32 mismatch from relying on implicit
        # dtype resolution; casting here avoids the same class of bug.
        input_features = inputs.input_features.to(self._device, dtype=self._model.dtype)

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, language=lang, task="transcribe")
        return self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model, self._processor
        self._model = None
        self._processor = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
