"""Batch runner (Phase 4): run every configured ASRSystem over every
utterance, writing hypotheses to
data/transcripts/hypotheses/{system.name}/{utt_id}.txt.

Resumable (skips utterances that already have output) and fault-tolerant
per utterance (one failure is logged and skipped, not fatal to the whole
run) — a multi-hour, multi-system batch job over hundreds of speakers
should not be restarted from scratch over one bad file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

from vayas.audit.base import ASRSystem

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "transcripts" / "hypotheses"


class Utterance(NamedTuple):
    utt_id: str
    audio_path: Path
    lang: str


@dataclass(frozen=True)
class RunFailure:
    system_name: str
    utt_id: str
    error: str


def run_batch(
    systems: Iterable[ASRSystem],
    utterances: Iterable[Utterance],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[RunFailure]:
    failures: list[RunFailure] = []
    utterances = list(utterances)

    for system in systems:
        system_dir = output_dir / system.name
        system_dir.mkdir(parents=True, exist_ok=True)

        try:
            for utt in utterances:
                out_path = system_dir / f"{utt.utt_id}.txt"
                if out_path.exists():
                    continue
                try:
                    hypothesis = system.transcribe(utt.audio_path, utt.lang)
                except Exception as e:  # noqa: BLE001 - one bad utterance must not abort the batch
                    failures.append(RunFailure(system_name=system.name, utt_id=utt.utt_id, error=str(e)))
                    continue
                out_path.write_text(hypothesis, encoding="utf-8")
        finally:
            # Free this system's model/GPU memory before the next system
            # loads its own — required on this project's 4GB-VRAM dev GPU,
            # where holding two local models loaded at once risks OOM.
            system.unload()

    return failures
