# VAYAS — Age-Stratified Evaluation of Speech Recognition (Hindi)

Paper 1 of the Vyaskosh series. A portable protocol for auditing ASR systems on elderly speech,
instantiated on Hindi using existing public datasets.

- Project brief: [`docs/brief.pdf`](docs/brief.pdf)
- Execution spec: `../../Research/Paper 1/VAYAS_paper1_claude_code_handoff.md`
- Sampling protocol (Phase 1 output, gate doc): [`docs/protocol.md`](docs/protocol.md)

## Status

**Phase 1 (data provenance & sampling protocol) is done and reviewed.** Source of record:
IndicVoices-R Hindi — 50 elderly speakers (`age_group=60+`), 318-speaker matched control pool,
confirmed genuine spontaneous speech. Common Voice Hindi checked and retained as supporting
evidence (near-zero elderly representation). Single control-vs-60+ age band, per source data
limits — see `docs/protocol.md` for full detail.

**Phase 2 (ingestion)**: `data/metadata/speakers.csv` populated with all 368 real speakers.

**Phase 4 (audit harness)**: 4 of 7 originally-scoped systems confirmed working end-to-end on
real GPU inference (Whisper large-v3, MMS, IndicConformer, IndicWhisper) — IndicConformer and
IndicWhisper produce exact/near-exact matches against gold. Sarvam is in scope but deferred
(billed API, needs cost go-ahead). IndicWav2Vec and Omnilingual ASR are excluded — both hit real
platform gaps in this dev environment (kenlm doesn't build on Python 3.13; fairseq2n ships no
Windows wheels), not judged unusable in principle. See `docs/protocol.md` §1.

## Setup

```
uv sync
```

Run tests:

```
uv run pytest
```
