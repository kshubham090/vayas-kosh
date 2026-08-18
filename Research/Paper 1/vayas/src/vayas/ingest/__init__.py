"""Audio validation and metadata-schema enforcement (Phase 2).

See audio.py for sample rate / channel / duration / silence validation,
and schema.py for data/metadata/speakers.csv's pydantic schema (matching
docs/protocol.md's confirmed IndicVoices-R Hindi source). Built and tested
against synthetic/dummy audio and records before any real data flows
through it.
"""

from vayas.ingest.audio import AudioInfo, AudioValidationError, read_audio_info, validate_audio, validate_audio_file
from vayas.ingest.schema import (
    AgeBand,
    AgeGroupRaw,
    SpeakerRecord,
    UtteranceRecord,
    acoustic_condition_for,
    age_band_for,
)

__all__ = [
    "AudioInfo",
    "AudioValidationError",
    "read_audio_info",
    "validate_audio",
    "validate_audio_file",
    "AgeBand",
    "AgeGroupRaw",
    "SpeakerRecord",
    "UtteranceRecord",
    "age_band_for",
    "acoustic_condition_for",
]
