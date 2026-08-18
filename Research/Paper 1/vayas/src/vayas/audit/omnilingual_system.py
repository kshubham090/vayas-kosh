"""Meta Omnilingual ASR, zero-shot, via the omnilingual-asr package.

Verified interface: `pip install omnilingual-asr`, then
omnilingual_asr.models.inference.pipeline.ASRInferencePipeline with a
model_card. Languages use "{code}_{script}" — Hindi is "hin_Deva".

Model card sizing: the 7B variant (facebook/omniASR-CTC-7B /
omniASR_LLM_7B_v2) needs ~14GB VRAM in fp16 and does not fit this
project's dev hardware (RTX 3050 laptop, 4GB VRAM). Defaulting to
omniASR_LLM_300M_v2 instead — confirmed to exist (facebookresearch/
omnilingual-asr cards/models/rc_models_v1.yaml) and small enough to fit.
The "LLM" family (not "CTC") is used specifically because CTC-variant
cards don't take a language-conditioning argument, and this wrapper's
transcribe() always passes one.

CONFIRMED BLOCKED ON WINDOWS (2026-08-16): `pip install omnilingual-asr`
fails to resolve — its core dependency `fairseq2` requires `fairseq2n`
(the native binary backend), which publishes wheels only for
manylinux2014_x86_64 and macosx_14_0_arm64, and only up to Python 3.11.
There is no Windows build at all, independent of Python version. This
project's dev environment is Windows + Python 3.13, so this system
cannot run here without WSL or a Linux/macOS machine — not a version
pin or toolchain issue like kenlm's, a hard platform gap in fairseq2n
itself. Confirmed via the actual pip resolver error and fairseq2n's
published wheel list, not assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vayas.audit.base import ASRSystem

DEFAULT_MODEL_CARD = "omniASR_LLM_300M_v2"
_LANG_CODE_MAP = {"hi": "hin_Deva"}


class OmnilingualASRSystem(ASRSystem):
    name = "meta-omnilingual-asr"

    def __init__(self, model_card: str = DEFAULT_MODEL_CARD) -> None:
        self._model_card = model_card
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self._pipeline = ASRInferencePipeline(model_card=self._model_card)

    def transcribe(self, audio_path: Path, lang: str) -> str:
        self._ensure_loaded()
        omni_lang = _LANG_CODE_MAP.get(lang, lang)
        transcriptions = self._pipeline.transcribe([str(audio_path)], lang=[omni_lang], batch_size=1)
        return transcriptions[0].strip()

    def unload(self) -> None:
        if self._pipeline is None:
            return
        del self._pipeline
        self._pipeline = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
