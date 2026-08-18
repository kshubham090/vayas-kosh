"""Common zero-shot ASR system interface (Phase 4).

Every wrapper in this module implements ASRSystem.transcribe() and nothing
else — no .fit()/.train()/optimizer anywhere in src/vayas/audit is a hard
rule (handoff sec.4: "no script in this repo may contain a training loop").
Model loading is lazy (deferred to first .transcribe() call, not
__init__), so importing this module or constructing a wrapper never
triggers a download — only actually running an audit does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def default_device() -> str:
    """cuda if available, else cpu — checked lazily so importing this
    module doesn't require torch to already be importable at package
    import time (it's still a hard dependency, just not eagerly hit)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_mono_16k(audio_path: Path):
    """Load audio as a mono float32 torch tensor at 16kHz.

    Uses soundfile, not torchaudio.load() — torchaudio's decoder now
    requires torchcodec, which in turn requires a system FFmpeg install
    (not present in this project's dev environment); soundfile's bundled
    libsndfile (>=1.1) decodes MP3 directly with no extra system
    dependency, confirmed against a real Common Voice Hindi clip. Only
    resampling (a pure tensor op, no codec involved) uses torchaudio.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.mean(data, axis=1, keepdims=True).T)  # (1, n_frames)
    if sr != 16_000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16_000)(wav)
    return wav


class ASRSystem(ABC):
    """One system under audit. name identifies it in
    data/transcripts/hypotheses/{name}/{utt_id}.txt output paths."""

    name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, lang: str) -> str:
        """Run zero-shot inference on one audio file. Must not fine-tune,
        adapt, or otherwise update any model weights."""
        raise NotImplementedError

    def unload(self) -> None:
        """Release any loaded model/GPU memory. Default no-op (e.g. API-
        backed systems have nothing to unload). Local-weights wrappers
        override this — with only 4GB VRAM on this project's dev GPU
        (RTX 3050 laptop), run_batch() must free each system's memory
        before loading the next, or a multi-system run OOMs partway
        through instead of failing fast on the first system that
        genuinely doesn't fit."""
        pass
