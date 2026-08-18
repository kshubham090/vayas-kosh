"""Meta MMS (Massively Multilingual Speech), zero-shot, via transformers.

Verified interface: `facebook/mms-1b-all` on the Hugging Face Hub,
Wav2Vec2ForCTC + Wav2Vec2Processor/Wav2Vec2CTCTokenizer, with the target
language selected via processor.tokenizer.set_target_lang() before
inference — MMS is one model with per-language adapter weights, not one
checkpoint per language.

CONFIRMED WORKING END-TO-END (2026-08-16): real run on the project's dev
GPU (RTX 3050, 4GB VRAM) against a real Common Voice Hindi clip
(common_voice_hi_26127974.mp3). Hypothesis vs. gold:
  gold: इंडिया टुडे ग्रुप ने प्रधानमंत्री को श्वेत पत्र सौंपा
  hyp:  इंडिया टूडे ग्रूप ने प्रधान मंत्री को शवेद पत्र सौंपा
Close but not exact — minor vowel-length variants and one real
substitution (श्वेत -> शवेद) — a realistic error pattern, confirming the
pipeline (decode, resample, inference, CTC decode) is genuinely correct,
not just running without crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vayas.audit.base import ASRSystem, default_device, load_audio_mono_16k

MODEL_ID = "facebook/mms-1b-all"

# ISO 639-3 code MMS expects for Hindi.
_LANG_CODE_MAP = {"hi": "hin"}


class MMSSystem(ASRSystem):
    name = "mms-1b-all"

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device or default_device()
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import Wav2Vec2ForCTC, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self._device)

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model, self._processor
        self._model = None
        self._processor = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def transcribe(self, audio_path: Path, lang: str) -> str:
        import torch

        self._ensure_loaded()

        mms_lang = _LANG_CODE_MAP.get(lang, lang)
        self._processor.tokenizer.set_target_lang(mms_lang)
        self._model.load_adapter(mms_lang)

        wav = load_audio_mono_16k(audio_path)

        inputs = self._processor(wav.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(inputs.input_values.to(self._device)).logits
        ids = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(ids)[0].strip()
