"""Phase 0a: source-data feasibility check, IndicVoices-R fallback (handoff sec.0a step 3).

Counts distinct speakers and utterances per age_group in
`SPRINGLab/IndicVoices-R_Hindi` on Hugging Face, without downloading the
~46GB of embedded audio: the parquet export is columnar, so a
column-selective read over HTTP range requests (via fsspec + pyarrow)
transfers only the speaker_id/age_group columns.

No account/token required for this HF repo's parquet export — unlike
`ai4bharat/indicvoices_r`, which gates behind agreeing to share contact
info, this mirror's parquet files are publicly readable.

Real counts from this script are the only thing that may be written into
docs/protocol.md's stratification-feasibility section or
data/metadata/speakers.csv — never invented or estimated numbers
(handoff sec.4).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import fsspec
import pyarrow.parquet as pq

REPO = "SPRINGLab/IndicVoices-R_Hindi"
N_SHARDS = 10
SHARD_URL_TEMPLATE = (
    f"https://huggingface.co/datasets/{REPO}/resolve/refs%2Fconvert%2Fparquet/default/train/{{:04d}}.parquet"
)

# Confirmed via the parquet schema's embedded HuggingFace ClassLabel metadata
# for the age_group column — this is the full label set, in code order.
AGE_GROUP_LABELS = ["18-30", "30-45", "45-60", "60+"]
ELDERLY_LABEL = "60+"
STOPPING_RULE_MIN_N = 8


@dataclass(frozen=True)
class AgeGroupCount:
    age_group: str
    n_speakers: int
    n_utterances: int


def count_speakers_by_age_group() -> list[AgeGroupCount]:
    fs = fsspec.filesystem("https")
    speakers_by_group: dict[str, set[str]] = defaultdict(set)
    utterances_by_group: dict[str, int] = defaultdict(int)

    for i in range(N_SHARDS):
        url = SHARD_URL_TEMPLATE.format(i)
        with fs.open(url, "rb") as f:
            table = pq.read_table(f, columns=["speaker_id", "age_group"])
        speaker_ids = table.column("speaker_id").to_pylist()
        age_codes = table.column("age_group").to_pylist()
        for sid, code in zip(speaker_ids, age_codes):
            label = AGE_GROUP_LABELS[code] if code is not None and 0 <= code < len(AGE_GROUP_LABELS) else "unspecified"
            speakers_by_group[label].add(sid)
            utterances_by_group[label] += 1

    labels_seen = list(AGE_GROUP_LABELS) + (["unspecified"] if "unspecified" in speakers_by_group else [])
    return [
        AgeGroupCount(
            age_group=label,
            n_speakers=len(speakers_by_group.get(label, set())),
            n_utterances=utterances_by_group.get(label, 0),
        )
        for label in labels_seen
    ]


def main() -> None:
    counts = count_speakers_by_age_group()

    print(f"Source: {REPO}, {N_SHARDS} parquet shards, columns=[speaker_id, age_group]\n")
    print(f"{'age_group':<12}{'n_speakers':<12}{'n_utterances':<14}")
    for c in counts:
        print(f"{c.age_group:<12}{c.n_speakers:<12}{c.n_utterances:<14}")

    elderly = next((c for c in counts if c.age_group == ELDERLY_LABEL), None)
    if elderly is None:
        print(f"\nSTOPPING-RULE FLAG: no '{ELDERLY_LABEL}' band found at all.")
    elif elderly.n_speakers < STOPPING_RULE_MIN_N:
        print(
            f"\nSTOPPING-RULE FLAG (docs/protocol.md sec.5): '{ELDERLY_LABEL}' band has "
            f"{elderly.n_speakers} speakers, below n>={STOPPING_RULE_MIN_N}. Report as a "
            "limitation, do not drop silently."
        )
    else:
        print(f"\n'{ELDERLY_LABEL}' band clears n>={STOPPING_RULE_MIN_N} ({elderly.n_speakers} speakers).")


if __name__ == "__main__":
    main()
