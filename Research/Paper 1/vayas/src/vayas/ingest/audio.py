"""Audio validation (Phase 2): sample rate, channel count, minimum
duration, silence detection.

Built and tested against synthetic/dummy audio (see tests/test_ingest_audio.py)
per the handoff's explicit instruction not to wait on real recordings.
Uses `soundfile` (libsndfile), which covers WAV/FLAC — IndicVoices-R ships
WAV. Common Voice's MP3 is not decodable via this backend, but Common
Voice is not an ingestion dependency (docs/protocol.md sec.2.2: retained
only as supporting evidence, not used for real ingestion).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_SAMPLE_RATE_HZ = 16_000
MAX_CHANNELS = 2  # mono or stereo only
MIN_DURATION_SEC = 0.5
MAX_SILENCE_RATIO = 0.95
SILENCE_AMPLITUDE_THRESHOLD = 0.01


class AudioValidationError(ValueError):
    """Raised with a specific, actionable reason an audio file was rejected."""


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    sample_rate: int
    channels: int
    duration_sec: float
    silence_ratio: float


def read_audio_info(path: Path) -> AudioInfo:
    try:
        data, sample_rate = sf.read(path, always_2d=True)
    except Exception as e:
        raise AudioValidationError(f"{path}: could not be read as audio ({e}).") from e

    n_frames, n_channels = data.shape
    duration_sec = n_frames / sample_rate if sample_rate else 0.0
    mono = data.mean(axis=1)
    silence_ratio = float(np.mean(np.abs(mono) < SILENCE_AMPLITUDE_THRESHOLD)) if n_frames else 1.0

    return AudioInfo(
        path=path,
        sample_rate=sample_rate,
        channels=n_channels,
        duration_sec=duration_sec,
        silence_ratio=silence_ratio,
    )


def validate_audio(info: AudioInfo) -> None:
    """Raise AudioValidationError with a clear, specific reason, or return None if valid."""
    if info.sample_rate < MIN_SAMPLE_RATE_HZ:
        raise AudioValidationError(
            f"{info.path}: sample rate {info.sample_rate}Hz is below the {MIN_SAMPLE_RATE_HZ}Hz minimum."
        )
    if info.channels > MAX_CHANNELS:
        raise AudioValidationError(
            f"{info.path}: {info.channels} channels exceeds the {MAX_CHANNELS}-channel (mono/stereo) limit."
        )
    if info.duration_sec < MIN_DURATION_SEC:
        raise AudioValidationError(
            f"{info.path}: duration {info.duration_sec:.3f}s is below the {MIN_DURATION_SEC}s minimum."
        )
    if info.silence_ratio > MAX_SILENCE_RATIO:
        raise AudioValidationError(
            f"{info.path}: {info.silence_ratio:.1%} of samples are near-silent "
            f"(threshold {MAX_SILENCE_RATIO:.0%}) — likely a dead/empty recording."
        )


def validate_audio_file(path: Path) -> AudioInfo:
    """Read + validate in one call. Returns AudioInfo if valid, raises AudioValidationError otherwise."""
    info = read_audio_info(path)
    validate_audio(info)
    return info
