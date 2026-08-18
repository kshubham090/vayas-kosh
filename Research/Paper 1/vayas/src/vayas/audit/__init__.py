"""Zero-shot ASR audit harness (Phase 4).

One wrapper per system (Whisper large-v3, MMS, Meta Omnilingual ASR,
IndicWhisper, IndicWav2Vec, IndicConformer, Sarvam) behind the common
ASRSystem.transcribe(audio_path, lang) interface, covering both
local-weights and API-backed systems. No training loops belong in this
module, ever — that is an automatic scope violation.

Wrapper model loading is lazy: importing this package or constructing a
wrapper never downloads anything. Actually calling .transcribe() does —
see each module's docstring for the specific model/API it hits and any
verification caveats.
"""

from vayas.audit.base import ASRSystem
from vayas.audit.indicconformer_system import IndicConformerSystem
from vayas.audit.indicwav2vec_system import IndicWav2VecSystem
from vayas.audit.indicwhisper_system import IndicWhisperSystem
from vayas.audit.mms_system import MMSSystem
from vayas.audit.omnilingual_system import OmnilingualASRSystem
from vayas.audit.runner import RunFailure, Utterance, run_batch
from vayas.audit.sarvam_system import SarvamSystem
from vayas.audit.whisper_system import WhisperSystem

__all__ = [
    "ASRSystem",
    "WhisperSystem",
    "MMSSystem",
    "OmnilingualASRSystem",
    "IndicWhisperSystem",
    "IndicWav2VecSystem",
    "IndicConformerSystem",
    "SarvamSystem",
    "Utterance",
    "RunFailure",
    "run_batch",
]
