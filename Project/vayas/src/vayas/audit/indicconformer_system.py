"""AI4Bharat IndicConformer (600M, multilingual, incl. Hindi), zero-shot.

Verified interface: `ai4bharat/indic-conformer-600m-multilingual` on the
Hugging Face Hub, loaded via AutoModel(trust_remote_code=True); the
model's forward call takes (waveform, lang_code, decoding) directly rather
than the usual processor+model split. Hindi language code is "hi".
Requires onnxruntime/onnx per the model card.

GATED REPO (confirmed via `GET /api/models/ai4bharat/indic-conformer-600m-multilingual`
-> gated: "auto", same as indicwav2vec_system.py): requires a Hugging
Face account, accepting this repo's terms at
https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual, and
an access token exported as `HF_TOKEN`.

CONFIRMED WORKING END-TO-END (2026-08-16): real run on the project's dev
GPU against the same real Common Voice Hindi clip as mms_system.py
(common_voice_hi_26127974.mp3) produced an EXACT match to the gold
transcript: इंडिया टुडे ग्रुप ने प्रधानमंत्री को श्वेत पत्र सौंपा.
Downloads 404 files on first load (trust_remote_code pulls in per-
language ONNX components, not just the main torch checkpoint). The
ONNX-backed components run on CPU (CPUExecutionProvider), not the GPU,
even with `onnxruntime-gpu` installed — confirmed by trying it:
onnxruntime-gpu additionally needs a system-wide CUDA Toolkit + cuDNN 9
install (missing `cublasLt64_13.dll`), which is separate from and not
satisfied by torch's self-contained pip-installed CUDA runtime. Left on
plain CPU `onnxruntime` deliberately — correctness is unaffected (same
exact-match transcript either way), only throughput across a full batch;
revisit if IndicConformer's batch runtime becomes a bottleneck.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from vayas.audit.base import ASRSystem, default_device, load_audio_mono_16k

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformerSystem(ASRSystem):
    name = "indicconformer-600m-multilingual"

    def __init__(self, decoding: Literal["ctc", "rnnt"] = "ctc", device: Optional[str] = None) -> None:
        self._decoding = decoding
        self._device = device or default_device()
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(self._device)

    def transcribe(self, audio_path: Path, lang: str) -> str:
        self._ensure_loaded()

        wav = load_audio_mono_16k(audio_path).to(self._device)

        result = self._model(wav, lang, self._decoding)
        return str(result).strip()

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
