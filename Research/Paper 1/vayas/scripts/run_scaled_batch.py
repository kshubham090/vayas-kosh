"""Scaled Phase 4 batch run, per the plan agreed 2026-08-18:

- MMS-1B + IndicConformer (fast systems): full manifest -- all 50 elderly
  speakers + a duration-matched control sample (~21h audio).
- Whisper large-v3 + IndicWhisper (slow systems, ~3-4x audio duration on
  this dev GPU per the pilot benchmark): capped at 3 utterances per
  elderly speaker, no control -- enough for a real per-speaker signal
  without a multi-day runtime.

Writes hypotheses to data/transcripts/hypotheses/{system}/{utt_id}.txt
(the real, non-pilot output directory) and prints per-system timing.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vayas.audit import (
    IndicConformerSystem,
    IndicWhisperSystem,
    MMSSystem,
    Utterance,
    WhisperSystem,
    run_batch,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "raw" / "batch_audio" / "manifest.csv"
AUDIO_DIR = ROOT / "data" / "raw" / "batch_audio"
OUTPUT_DIR = ROOT / "data" / "transcripts" / "hypotheses"
SLOW_CAP_PER_SPEAKER = 3


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_utterances(rows: list[dict]) -> list[Utterance]:
    return [Utterance(utt_id=r["utt_id"], audio_path=AUDIO_DIR / f"{r['utt_id']}.wav", lang="hi") for r in rows]


def cap_per_speaker(rows: list[dict], n: int) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker_id"]].append(r)
    capped = []
    for speaker_rows in by_speaker.values():
        capped.extend(speaker_rows[:n])
    return capped


def run_and_time(systems, utterances, label: str) -> None:
    print(f"\n=== {label}: {len(systems)} system(s) x {len(utterances)} clips ===", flush=True)
    for system in systems:
        t0 = time.time()
        failures = run_batch([system], utterances, output_dir=OUTPUT_DIR)
        elapsed = time.time() - t0
        print(
            f"{system.name}: {elapsed:.0f}s ({elapsed/3600:.2f}h) for {len(utterances)} clips, "
            f"{len(failures)} failures",
            flush=True,
        )
        for f in failures[:5]:
            print(f"  FAILED {f.utt_id}: {f.error}", flush=True)


def main() -> None:
    rows = load_manifest()
    elderly_rows = [r for r in rows if r["age_band"] == "60+"]
    control_rows = [r for r in rows if r["age_band"] == "control"]
    print(f"Manifest: {len(elderly_rows)} elderly, {len(control_rows)} control", flush=True)

    full_utterances = to_utterances(rows)
    slow_rows = cap_per_speaker(elderly_rows, SLOW_CAP_PER_SPEAKER)
    slow_utterances = to_utterances(slow_rows)
    print(f"Slow-system cap: {len(slow_utterances)} clips ({SLOW_CAP_PER_SPEAKER}/elderly speaker)", flush=True)

    overall_start = time.time()

    run_and_time([MMSSystem(), IndicConformerSystem()], full_utterances, "Fast systems, full manifest")
    run_and_time([WhisperSystem(), IndicWhisperSystem()], slow_utterances, "Slow systems, capped sample")

    print(f"\nTOTAL: {(time.time() - overall_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
