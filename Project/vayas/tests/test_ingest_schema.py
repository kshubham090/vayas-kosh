"""Phase 2 metadata schema enforcement, tested against dummy records only."""

import pytest
from pydantic import ValidationError

from vayas.ingest.schema import SpeakerRecord, UtteranceRecord, age_band_for


def _speaker(**overrides) -> dict:
    base = dict(
        speaker_id="dummy-speaker-001",
        source_dataset="indicvoices_r_hindi",
        lang="hi",
        age_group_raw="60+",
        gender="Female",
        region_state="Madhya Pradesh",
        region_district="Bhopal",
    )
    base.update(overrides)
    return base


def _utterance(**overrides) -> dict:
    base = dict(
        speaker_id="dummy-speaker-001",
        source_dataset="indicvoices_r_hindi",
        task_name="KYP - Cooking",
        snr=25.0,
        c50=None,
        speaking_rate=None,
        duration_sec=8.03,
    )
    base.update(overrides)
    return base


def test_valid_speaker_derives_elderly_age_band() -> None:
    rec = SpeakerRecord(**_speaker())
    assert rec.age_band == "60+"
    assert rec.speech_type == "spontaneous"


def test_control_age_group_derives_control_band() -> None:
    rec = SpeakerRecord(**_speaker(age_group_raw="18-30"))
    assert rec.age_band == "control"


def test_age_band_for_matches_model_derivation() -> None:
    assert age_band_for("60+") == "60+"
    for band in ("18-30", "30-45", "45-60"):
        assert age_band_for(band) == "control"


def test_invalid_age_group_rejected() -> None:
    with pytest.raises(ValidationError):
        SpeakerRecord(**_speaker(age_group_raw="90+"))


def test_empty_speaker_id_rejected() -> None:
    with pytest.raises(ValidationError):
        SpeakerRecord(**_speaker(speaker_id=""))


def test_speaker_record_rejects_utterance_only_fields() -> None:
    with pytest.raises(ValidationError):
        SpeakerRecord(**_speaker(duration_sec=8.0))


def test_common_voice_source_rejected_not_a_dependency() -> None:
    # docs/protocol.md sec.2.2: Common Voice is checked-but-unused, not an
    # ingestion source — the schema only accepts what's actually used.
    with pytest.raises(ValidationError):
        SpeakerRecord(**_speaker(source_dataset="common_voice_hi"))


def test_valid_utterance_derives_quiet_condition() -> None:
    rec = UtteranceRecord(**_utterance())
    assert rec.acoustic_condition == "quiet"


def test_low_snr_utterance_derives_ambient_condition() -> None:
    rec = UtteranceRecord(**_utterance(snr=5.0))
    assert rec.acoustic_condition == "ambient"


def test_missing_snr_utterance_leaves_acoustic_condition_unset() -> None:
    rec = UtteranceRecord(**_utterance(snr=None))
    assert rec.acoustic_condition is None


def test_zero_duration_utterance_rejected() -> None:
    with pytest.raises(ValidationError):
        UtteranceRecord(**_utterance(duration_sec=0))
