"""Metadata schema (Phase 2), matching docs/protocol.md's confirmed source
of record: IndicVoices-R Hindi.

Two record types, because IndicVoices-R's metadata is per-utterance
(26,318 rows) across far fewer speakers (368) — demographics (age, gender,
region) are stable per speaker, but acoustic/task attributes (snr,
duration, task_name) vary per clip:

- SpeakerRecord: one row per speaker_id -> data/metadata/speakers.csv.
- UtteranceRecord: one row per clip, references a speaker_id. Not yet
  written to a committed CSV (no clip-level table exists in the repo tree
  yet) — used internally once real audio ingestion needs to associate
  individual clips with their speaker's demographics.

No consent_id field — this project uses existing public data (see
docs/protocol.md's "Supersedes" note); there is no new-recruitment consent
gate to track. Fields below are only what IndicVoices-R's metadata
actually provides (docs/protocol.md sec.2.1) — nothing invented for
fields the source doesn't have (e.g. exact age, or recording device,
which crowdsourced-app corpora don't expose).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

# The 4 raw IndicVoices-R age_group labels, confirmed via the parquet
# ClassLabel metadata (docs/protocol.md sec.2.1, sec.5.1).
AgeGroupRaw = Literal["18-30", "30-45", "45-60", "60+"]

# docs/protocol.md sec.3.1: source data only supports a control-vs-60+ split,
# not the brief's 60s/70s/80s scheme.
AgeBand = Literal["control", "60+"]

_ELDERLY_AGE_GROUP: AgeGroupRaw = "60+"


def age_band_for(age_group_raw: AgeGroupRaw) -> AgeBand:
    return "60+" if age_group_raw == _ELDERLY_AGE_GROUP else "control"


def acoustic_condition_for(snr: Optional[float]) -> Optional[Literal["quiet", "ambient"]]:
    """Bucket snr into quiet/ambient.

    Threshold is provisional (20dB SNR is a common quiet/noisy speech
    cutoff) and not yet validated against this corpus's actual SNR
    distribution — revisit once real utterance-level data is ingested,
    per docs/protocol.md sec.3.3's open item on deriving this stratum.
    """
    if snr is None:
        return None
    return "quiet" if snr >= 20.0 else "ambient"


class SpeakerRecord(BaseModel):
    """One row of data/metadata/speakers.csv — per-speaker demographics,
    stable across all of a speaker's utterances."""

    model_config = {"frozen": True, "extra": "forbid"}

    speaker_id: str = Field(min_length=1)
    source_dataset: Literal["indicvoices_r_hindi"]
    lang: Literal["hi"] = "hi"

    age_group_raw: AgeGroupRaw
    gender: str = Field(min_length=1)

    # docs/protocol.md sec.4: confirmed spontaneous for this source — not a
    # per-row guess, a fixed fact about IndicVoices-R.
    speech_type: Literal["spontaneous"] = "spontaneous"

    region_state: Optional[str] = None
    region_district: Optional[str] = None

    # Not available from this source — crowdsourced/app-recorded corpora
    # don't expose per-speaker recording device or a single recording date
    # (a speaker's utterances may span multiple sessions). Left as None
    # rather than fabricated.
    recording_date: Optional[str] = None
    device: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def age_band(self) -> AgeBand:
        return age_band_for(self.age_group_raw)


class UtteranceRecord(BaseModel):
    """One row per clip — acoustic/task attributes that vary per utterance,
    not per speaker. References speaker_id but is not itself a row of
    speakers.csv."""

    model_config = {"frozen": True, "extra": "forbid"}

    speaker_id: str = Field(min_length=1)
    source_dataset: Literal["indicvoices_r_hindi"]

    task_name: Optional[str] = None
    snr: Optional[float] = None
    c50: Optional[float] = None
    speaking_rate: Optional[float] = None
    duration_sec: float = Field(gt=0)

    @computed_field  # type: ignore[misc]
    @property
    def acoustic_condition(self) -> Optional[Literal["quiet", "ambient"]]:
        return acoustic_condition_for(self.snr)
