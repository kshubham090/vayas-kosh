"""AI4Bharat IndicWav2Vec (Hindi), zero-shot, via transformers.

Verified interface: `ai4bharat/indicwav2vec-hindi` on the Hugging Face
Hub, AutoModelForCTC + AutoProcessor. Model card notes it does not support
inference with a language model (no LM rescoring) — plain CTC decode only.

GATED REPO (confirmed via `GET /api/models/ai4bharat/indicwav2vec-hindi`
-> gated: "auto", and by an actual real run hitting a 401
GatedRepoError): requires a Hugging Face account, accepting this
specific repo's terms at https://huggingface.co/ai4bharat/indicwav2vec-hindi,
and an access token exported as `HF_TOKEN` — `transformers`/`huggingface_hub`
pick that env var up automatically, no code-level token handling needed
here. facebook/mms-1b-all and openai/whisper-large-v3 are NOT gated, for
comparison — this is specific to AI4Bharat's two repos in this harness.

CURRENTLY BLOCKED (2026-08-16), confirmed by a real run after gating was
resolved: this checkpoint's processor is Wav2Vec2ProcessorWithLM (ships
an n-gram LM for rescoring, contrary to the model card summary this
wrapper was originally written from, which said no LM support — the
model card was wrong/outdated). That requires `pyctcdecode` (installed
successfully) which in turn requires the `kenlm` Python bindings, whose
C extension fails to compile against Python 3.13's changed C API
(`_PyGC_FINALIZED`, `_PyDict_SetItem_KnownHash` missing;
`_PyLong_AsByteArray` signature changed) — a genuine upstream kenlm/
CPython-3.13 incompatibility, not a missing-toolchain issue on this
machine (MSVC build tools are present and were invoked correctly).
Unblocking this needs either Python <=3.12, a prebuilt kenlm wheel from
elsewhere (e.g. conda-forge), or an upstream kenlm fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vayas.audit.base import ASRSystem, default_device, load_audio_mono_16k

MODEL_ID = "ai4bharat/indicwav2vec-hindi"


class IndicWav2VecSystem(ASRSystem):
    name = "indicwav2vec-hindi"

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device or default_device()
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCTC, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForCTC.from_pretrained(MODEL_ID).to(self._device)

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
        if lang != "hi":
            raise ValueError(f"{self.name} is Hindi-only, got lang={lang!r}.")

        import torch

        self._ensure_loaded()

        wav = load_audio_mono_16k(audio_path)

        inputs = self._processor(wav.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(inputs.input_values.to(self._device)).logits
        ids = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(ids)[0].strip()
