"""Per-clip WER/CER/divergence/substitution-insertion-deletion via jiwer.

jiwer's default pipeline (whitespace tokenization for words, raw
character sequence for CER) is used as-is -- no Hindi-specific
normalization (e.g. danda `.` handling) is applied. This is a documented
choice, not an oversight: adding normalization changes the numbers, so
it belongs in `docs/protocol.md` as an explicit decision before it's
silently baked into the metric, not decided implicitly in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer


@dataclass(frozen=True)
class ClipMetrics:
    utt_id: str
    speaker_id: str
    age_band: str
    system: str
    duration: float
    wer: float
    cer: float
    divergence: float  # wer - cer, per brief.pdf sec.3
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int
    hyp_words: int
    speaking_rate: float  # ref_words / duration, words/sec


def compute_clip_metrics(
    utt_id: str, speaker_id: str, age_band: str, system: str,
    duration: float, reference: str, hypothesis: str,
) -> ClipMetrics | None:
    """Returns None (not a zeroed record) when the gold reference is
    empty -- WER/CER against an empty reference is undefined, and a
    silent 0.0 or 1.0 would misrepresent a real result. Caller must
    track and report skipped counts, not just drop them.
    """
    if not reference.strip():
        return None

    word_out = jiwer.process_words(reference, hypothesis)
    char_out = jiwer.process_characters(reference, hypothesis)
    ref_words = word_out.hits + word_out.substitutions + word_out.deletions
    hyp_words = word_out.hits + word_out.substitutions + word_out.insertions
    speaking_rate = (ref_words / duration) if duration > 0 else float("nan")

    return ClipMetrics(
        utt_id=utt_id,
        speaker_id=speaker_id,
        age_band=age_band,
        system=system,
        duration=duration,
        wer=word_out.wer,
        cer=char_out.cer,
        divergence=word_out.wer - char_out.cer,
        substitutions=word_out.substitutions,
        deletions=word_out.deletions,
        insertions=word_out.insertions,
        hits=word_out.hits,
        ref_words=ref_words,
        hyp_words=hyp_words,
        speaking_rate=speaking_rate,
    )
