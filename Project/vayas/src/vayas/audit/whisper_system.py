"""Whisper large-v3, zero-shot, via transformers' pipeline API.

Verified interface: `openai/whisper-large-v3` on the Hugging Face Hub,
loaded through transformers' automatic-speech-recognition pipeline.

Loads in fp16 on CUDA: large-v3 is ~1.55B params (~3.1GB fp16 weights
alone), which is tight against this project's dev GPU's 4GB VRAM — fp16
is not optional headroom here, it's required to fit at all.

CONFIRMED WORKING END-TO-END (2026-08-16): real run against the same
Common Voice Hindi clip as the other verified wrappers
(common_voice_hi_26127974.mp3):
  gold: इंडिया टुडे ग्रुप ने प्रधानमंत्री को श्वेत पत्र सौंपा
  hyp:  इंडिया टूडे गुरुप ने प्रधान मंतरी को श्वेत पत्र सौपा।
More deviation than MMS/IndicConformer on this one clip (गुरुप vs ग्रुप,
मंतरी vs मंत्री, dropped anusvara in सौपा vs सौंपा) — plausible given
large-v3 is a general multilingual model, not Hindi-specialized, unlike
those two. Not evidence of a bug; a real, differentiated result, which
is itself a good sign the harness produces genuine per-system signal
rather than identical or nonsense output everywhere.

The pipeline's file-path input mode shells out to a system `ffmpeg`
binary that isn't installed here (confirmed by a real failure) —
transcribe() passes a pre-decoded {"array", "sampling_rate"} dict via
load_audio_mono_16k() instead, bypassing that path entirely, same fix
class as the torchaudio/torchcodec issue in base.py.
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
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device=self._device,
            dtype=dtype,  # torch_dtype= is deprecated as of this transformers version
        )

    def transcribe(self, audio_path: Path, lang: str) -> str:
        self._ensure_loaded()
        # Pass a pre-decoded array, not the file path: the pipeline's
        # path-input mode shells out to a system `ffmpeg` binary that
        # isn't installed here (confirmed by a real failure) — reuse the
        # same soundfile-based decoder every other wrapper uses instead.
        wav = load_audio_mono_16k(audio_path)
        result = self._pipe(
            {"array": wav.squeeze(0).numpy(), "sampling_rate": 16_000},
            generate_kwargs={"language": lang},
        )
        return result["text"].strip()

    def unload(self) -> None:
        if self._pipe is None:
            return
        del self._pipe
        self._pipe = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
