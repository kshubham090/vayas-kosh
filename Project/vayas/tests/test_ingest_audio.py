"""Phase 2 audio validation, tested against synthetic/dummy WAV files only
(handoff: don't wait on real recordings for working ingestion code)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vayas.ingest.audio import (
    MAX_CHANNELS,
    MIN_DURATION_SEC,
    MIN_SAMPLE_RATE_HZ,
    AudioValidationError,
    validate_audio_file,
)


def _write_wav(path: Path, *, sample_rate: int, duration_sec: float, channels: int, silent: bool) -> Path:
    n_frames = max(1, int(sample_rate * duration_sec))
    if silent:
        signal = np.zeros(n_frames, dtype=np.float32)
    else:
        t = np.arange(n_frames) / sample_rate
        signal = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    if channels > 1:
        signal = np.tile(signal[:, None], (1, channels))
    sf.write(path, signal, sample_rate)
    return path


def test_valid_audio_passes(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "ok.wav", sample_rate=16_000, duration_sec=2.0, channels=1, silent=False)
    info = validate_audio_file(path)
    assert info.sample_rate == 16_000
    assert info.channels == 1
    assert info.duration_sec == pytest.approx(2.0, abs=0.01)


def test_valid_stereo_audio_passes(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "stereo.wav", sample_rate=44_100, duration_sec=1.0, channels=2, silent=False)
    info = validate_audio_file(path)
    assert info.channels == 2


def test_low_sample_rate_rejected(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "low_sr.wav", sample_rate=8_000, duration_sec=2.0, channels=1, silent=False)
    assert 8_000 < MIN_SAMPLE_RATE_HZ
    with pytest.raises(AudioValidationError, match="sample rate"):
        validate_audio_file(path)


def test_too_many_channels_rejected(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "surround.wav", sample_rate=16_000, duration_sec=1.0, channels=6, silent=False)
    assert 6 > MAX_CHANNELS
    with pytest.raises(AudioValidationError, match="channels"):
        validate_audio_file(path)


def test_too_short_duration_rejected(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "short.wav", sample_rate=16_000, duration_sec=0.1, channels=1, silent=False)
    assert 0.1 < MIN_DURATION_SEC
    with pytest.raises(AudioValidationError, match="duration"):
        validate_audio_file(path)


def test_silent_audio_rejected(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "silent.wav", sample_rate=16_000, duration_sec=2.0, channels=1, silent=True)
    with pytest.raises(AudioValidationError, match="silent"):
        validate_audio_file(path)


def test_unreadable_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is not a wav file")
    with pytest.raises(AudioValidationError, match="could not be read"):
        validate_audio_file(path)
