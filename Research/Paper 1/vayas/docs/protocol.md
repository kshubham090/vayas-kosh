# VAYAS Protocol — Data Provenance & Sampling (Hindi, Public-Data Path)

**Status: REVIEWED AND CONFIRMED by Shubham, 2026-08-16.** Source of record is **IndicVoices-R
Hindi** (§2.1) — 50 elderly speakers at `age_group=60+`, 318-speaker same-corpus control pool,
confirmed genuine spontaneous speech (§4), single control-vs-60+ age band (§3.1, the 60s/70s/80s
distinction from `brief.pdf` is not available from any checked public source). Common Voice
Hindi (§2.2) is retained and will be cited in the paper as supporting evidence for why
IndicVoices-R was necessary — its near-zero elderly representation (0 in 70s/80+, 2 in 60s) is
itself a data point for the public-data feasibility narrative (handoff §0a), not dropped from
the write-up. **Phase 1 is done.** Phase 2 (ingestion: `data/metadata/speakers.csv`, all 368 real
speakers) and Phase 4 (audit harness) are both underway — see §1's systems-under-audit note for
the confirmed five-system set (four verified, Sarvam deferred) and the two real platform-blocked
exclusions.

**Supersedes:** the recruitment/consent protocol implied by `brief.pdf` §4.2–§4.3 and the
"confirm real access to ≈50 elderly speakers per language" open gate in `brief.pdf` §4 (footer).
This project uses existing public datasets with age metadata; no new recording, consent, or
ethics gate applies to data collection. See `VAYAS_paper1_claude_code_handoff.md` §0 for the
full rationale (CHANGED 2026-08-13, CHANGED 2026-08-16).

---

## 1. Scope

- **Language:** Hindi only. The two-family (Indo-Aryan + Dravidian) design in `brief.pdf`
  Positioning §1 is dropped; Tamil and the LT-EDI Tamil shared-task dataset are out of scope
  entirely.
- **Systems under audit (zero-shot only, nothing fine-tuned) — confirmed 2026-08-16, after real
  end-to-end runs on the project's dev GPU against a real Hindi clip, compared against gold:**
  Whisper (large-v3), MMS, IndicWhisper, IndicConformer — all four confirmed working, with
  IndicConformer and IndicWhisper (both Hindi-specialized) producing exact or near-exact matches
  and Whisper large-v3 (general multilingual) showing more deviation, a plausible and
  differentiated result. **Sarvam** is in scope but not yet run — it's a billed API and running
  it needs an explicit go-ahead on cost, deferred for now, not dropped.
  **Excluded from the audit, confirmed via real attempts, not assumed:**
  - **IndicWav2Vec** — its gated HF repo access was resolved, but its processor
    (`Wav2Vec2ProcessorWithLM`) requires `pyctcdecode` → `kenlm`, whose C extension does not
    build against Python 3.13 (a genuine upstream kenlm/CPython incompatibility, confirmed via
    the actual compiler errors, not a missing-toolchain issue).
  - **Meta Omnilingual ASR** — `pip install omnilingual-asr` fails to resolve on Windows: its
    `fairseq2n` native backend ships no Windows wheels at all, for any Python version (confirmed
    via the pip resolver error and fairseq2n's published wheel list).
  Both are real platform gaps in this dev environment (Windows, Python 3.13), not judged
  unusable in principle — revisit if either upstream issue is fixed, or if this moves to
  Linux/WSL. See `src/vayas/audit/indicwav2vec_system.py` and `omnilingual_system.py` for the
  full technical detail. This drops the audit set from `brief.pdf` §2E's original seven Hindi
  systems to five (four confirmed + Sarvam deferred); the two exclusions and why should be
  reported honestly in the paper's Method section, not silently omitted. USM (Zhang et al. 2023)
  was never in scope — listed in the brief's reading list but not confirmed Hindi-capable at
  brief authoring time, and not revisited here.
- **Non-negotiables carried forward without exception** (handoff §0, brief.pdf §4):
  1. This protocol is frozen before it drives any downstream ingestion/audit code.
  2. Every choice justified in language-general terms — nothing Hindi-specific baked into
     "portable" code paths (`src/vayas/protocol`, `src/vayas/taxonomy`, etc.).
  3. Matched non-elderly control cohort (<60), same source/conditions, mandatory — no
     elderly-only dataset. Follows Koenecke, Nam, Lake et al. (2020, PNAS) matching design.
  4. Zero-shot audit only. No script in this repo may contain a training loop.
  5. Every metric reported stratified, with CIs and effect sizes — no bare point estimates,
     ever, anywhere in outputs (`src/vayas/stats`).
  6. Everything releasable: dataset, eval code, this protocol, the replication template.

### Scope note — positioning consequence (do not let this get glossed over later)

Dropping Tamil removes the evidence the brief's Positioning section (§1) rested on — "two
distinct families, multiple scripts, a steep resource gradient" — which is what made this audit
structurally closer to Gender Shades (multi-group) and Koenecke et al. (matched-cohort). A
single language can still demonstrate a portable *protocol*, but not a protocol that transfers
across a family/script boundary. This changes what claim the paper is entitled to make (narrower
"single-language proof-of-concept, explicit call for replication" vs. a depth-over-breadth
reframe trading breadth for statistical power). Handoff §0b requires Phase 9 (paper scaffold) to
carry an explicit note to revisit `brief.pdf` §1's Positioning/Objective wording once this is
decided — it must not be a silent find-and-replace of "Hindi and Tamil" → "Hindi".

---

## 2. Source data

Both candidate sources have now been pulled and counted (§5). The results reverse the brief's
implicit assumption that Common Voice would carry the elderly signal — **it doesn't; IndicVoices-R
does.** §2.1/§2.2 document both; §5 has the counts and the resulting recommendation.

### 2.1 IndicVoices-R Hindi — recommended source for the elderly + matched-control cohort

| Field | Value |
|---|---|
| Source | `SPRINGLab/IndicVoices-R_Hindi` on Hugging Face (Hindi subset of AI4Bharat's IndicVoices-R) |
| Corpus version | Hugging Face `refs/convert/parquet` export, `default/train`, 10 shards (`0000.parquet`–`0009.parquet`), 26,318 rows total, 368 unique speakers. Pulled and column-selectively read (speaker_id, age_group only — no audio) on 2026-08-16. |
| Access method | Public Hugging Face dataset repo — no gating, no account/token required for the parquet export used here (unlike the `ai4bharat/indicvoices_r` mirror, which does require agreeing to share contact info). |
| Access steps | Run `scripts/fetch_indicvoices_r_hindi.py` — column-selective read via `pyarrow.parquet` + `fsspec`'s HTTP filesystem against the HF-hosted parquet shards, requesting only `speaker_id`/`age_group` columns, avoiding the ~46GB of embedded audio. |
| Age resolution | `age_group` is a 4-class `ClassLabel`: `18-30`, `30-45`, `45-60`, `60+`. **This is coarser than Common Voice's decade brackets — there is no way to distinguish 60s/70s/80s within this source.** |
| Caveat (carries forward from handoff §0a step 3) | IndicVoices-R applies speech-enhancement preprocessing (denoising/dereverberation) that may suppress the acoustic degradation signal this paper measures. Now that this is the recommended primary elderly source rather than a hypothetical fallback, this caveat is live, not speculative — it must appear in every downstream document that cites results derived from it: dataset card, paper Method/Discussion/Limitations, replication template. |

### 2.2 Common Voice Hindi — checked, not usable for the elderly cohort

| Field | Value |
|---|---|
| Source | Mozilla Common Voice, Hindi (`hi`) validated split |
| Corpus version | `cv-corpus-26.0-2026-06-12` (Common Voice Scripted Speech 26.0 — Hindi). MDC dataset ID `cmqiod71900zgnr07uiyw57br`, archive checksum `2458dae795e8a7d6bb6704e539378e7beba0801756fc18432902ebb8241a3c8f`, `sizeBytes` 571685583 — verified via `GET /datasets/{id}` and matched against the downloaded archive's exact byte size before extraction. |
| Access method | Mozilla Data Collective (MDC) — as of the October 2025 platform migration, download requires a free authenticated account; anonymous/open download is no longer available. MDC's API additionally requires dataset terms to be accepted once via the web UI before `POST /datasets/{id}/download` will succeed (returns 403 otherwise) — this is not automatable. |
| Access steps | 1) Create a free Mozilla Data Collective account. 2) Open the [Hindi dataset page](https://mozilladatacollective.com/datasets/cmqiod71900zgnr07uiyw57br) and accept terms. 3) Generate an API key in profile settings, export as `MDC_TOKEN`. 4) Run `scripts/fetch_common_voice_hindi.py`. |
| Date pulled | 2026-08-16 |
| Status | Real counts (§5) show 0 speakers in 70s/80+, 2 in 60s — fails the n≥8 stopping rule outright. **Not used as the elderly-cohort source.** Confirmed role (2026-08-16): cited in the paper's data section as supporting evidence for why IndicVoices-R was necessary — its near-absence of elderly speakers in the most obvious public Hindi ASR corpus is itself part of the feasibility narrative (handoff §0a). Not a dependency of any Phase 2+ ingestion/audit code. |

### 2.3 Excluded

LT-EDI-2025 Tamil elderly/vulnerable-speaker shared task — Tamil is out of scope (§1).

**Reproducibility requirement:** given the same corpus version/shard set pulled via the same
access path, a stranger following this document must arrive at the same sample, for both sources.
Do not relax the version-pinning above.

---

## 3. Filtering and stratification logic

### 3.1 Age bands

**Superseded by §5's findings.** The brief's finer scheme (`brief.pdf` §3: control <60, 60–69,
70–79, 80+) assumed a source with decade-level resolution and real elderly representation.
Neither candidate source supports that in practice:

- Common Voice resolves to decade brackets (`sixties`/`seventies`/`eighties`) in principle, but
  has ~0 elderly speakers to bracket (§2.2, §5) — the finer resolution is moot without population.
- IndicVoices-R has real elderly speakers but only a single collapsed `60+` bucket (§2.1) — no
  60s/70s/80s distinction is possible from this source's public metadata at all.

**Recommended band scheme, pending Shubham's sign-off (§5):**

| Band | Source | Role |
|---|---|---|
| Control (<60) | IndicVoices-R Hindi, `age_group` in `{18-30, 30-45, 45-60}` | matched non-elderly cohort, same corpus as elderly band |
| Elderly (60+) | IndicVoices-R Hindi, `age_group = 60+` | single collapsed elderly band — no finer split available |

This is a larger resolution loss than the handoff anticipated (§0 flagged possible collapse to
60s/70s/80s; actual outcome is collapse to a single 60+ band). `brief.pdf` §1's Positioning
section already needs a rewrite over the Hindi-only pivot (§0b) — this is a second, independent
reason that section (and any Method-section age-banding language) needs revisiting before Phase 9,
not a variant of the same issue.

### 3.2 Control-cohort sampling

**Must be same-source as the elderly cohort**, not merely same-language — this is what makes the
Koenecke et al. matching design (§1 non-negotiable #3) actually hold: same corpus, same platform,
same recording/elicitation conditions, so the only systematic difference between cohorts is age.
Because the elderly cohort now comes from IndicVoices-R (§3.1), **the control cohort must also be
drawn from IndicVoices-R**, not Common Voice — mixing sources would reintroduce exactly the
cross-study confound (different platforms, different recording setups) that sourcing both cohorts
from one corpus was supposed to eliminate. IndicVoices-R's under-60 pool (318 speakers across its
three younger `age_group` buckets) is more than sufficient for this. Sampling within the control
band should still balance on gender and, where metadata allows, state/district, to avoid
confounding age with an unrelated demographic skew. Common Voice's own control pool (145 speakers,
§2.2) is not used for matching under this recommendation, since it has no matching elderly cohort
to pair with.

### 3.3 Stratification axes (per `brief.pdf` §3)

Every metric in `src/vayas/metrics` and `src/vayas/stats` is reported stratified across: age band
(§3.1), speech type, acoustic condition, and gender. Under the IndicVoices-R recommendation
(§2.1, §3.1–§3.2), acoustic condition can be derived more directly than Common Voice ever
supported — IndicVoices-R ships per-utterance `snr` and `c50` (clarity) numeric fields, which can
be binned into quiet/ambient strata directly, rather than relying on missing/absent metadata.
Speech type is confirmed spontaneous (§4) — `data/metadata/speakers.csv`'s `speech_type` column
should be populated `spontaneous` for all IndicVoices-R-sourced rows on that basis, not a guess.
The brief's "language
and language family" axis collapses to a constant (Hindi) under the single-language scope (§1)
and is retained as a stratum only for forward-compatibility with any future replication that adds
a second language.

---

## 4. Transcription-conventions deviation (state plainly, do not gloss over)

`brief.pdf` §4.6 specifies explicit transcription conventions for age-typical disfluency,
hesitation, and self-repair, assuming **spontaneous** speech recorded for this study.

**Under the Common Voice source (§2.2, not used for the elderly cohort — kept here for
completeness):** Common Voice Hindi is prompted, read speech, already transcribed by the platform
at collection time. Read speech carries measurably less disfluency than spontaneous speech, so
any degradation found would be a lower bound relative to spontaneous, real-world use. This
deviation no longer drives the elderly-cohort analysis under §3.1's recommendation, but still
applies to any future use of Common Voice as a secondary/robustness source.

**Under the IndicVoices-R source (§2.1, recommended primary) — CONFIRMED, resolved 2026-08-16:**
inspected a random sample of `text`/`verbatim`/`normalized`/`task_name` rows directly (via the
same column-selective parquet read as §5.1, sampling `text`+`verbatim`+`normalized` in addition
to the count columns). Two independent signals both point the same way:

1. **`task_name` values are topic-elicitation prompts, not read-aloud scripts** — e.g.
   "KYP - Cooking", "KYP - Gardening", "KYP - Drawing", "DOI - Entertainment", "Daily Life",
   alongside some command-style tasks ("Alexa Commands", "Ola/Uber Prompts"). This is a speaker
   talking about a topic, not reading fixed text.
2. **`verbatim` genuinely differs from `normalized`/`text` in ways only spontaneous speech
   produces**: colloquial spelling variants are preserved in `verbatim` and cleaned up
   elsewhere (e.g. verbatim `बौहौत ज्यादा` → normalized `बहुत ज़्यादा`), and at least one
   sampled utterance has a visible false start/disfluency carried into the verbatim transcript
   verbatim (`...हर एक हा हर चीज़ आइटम...` — a stutter/restart on "हर" that the cleaned
   `text`/`normalized` fields also happen to retain here, showing it wasn't scrubbed).

**Conclusion:** IndicVoices-R Hindi is genuine spontaneous, topic-elicited speech with disfluency
preserved in `verbatim` — this is closer to `brief.pdf` §4.6's original spontaneous-speech design
than Common Voice's read-speech-only corpus could ever have been, and is a **positive**, not a
deviation, for the source-switch recommendation in §5.1. `src/vayas/metrics`/`src/vayas/taxonomy`
should treat `verbatim` (not `text` or `normalized`) as the transcript to measure ASR hypotheses
against, since it's the one that actually preserves what `brief.pdf` §4.6 cares about. This
resolves the last open item blocking §5.1's sign-off list.

Consequence for Phase 3 (`src/vayas/transcribe/`): there is no new human transcription to perform
against either source — both ship existing transcripts. IAA tooling in that module remains useful
only for an independent quality spot-check on a sampled subset, which is optional, not a blocking
Phase 3 task for either data source.

---

## 5. Stopping / inclusion rule (handoff §0a step 4)

Decision gate, applied per age band once real counts exist:

- **n ≥ 8 speakers in a band** → band is included; stratified stats (CI, effect size) are
  reported for it per the non-negotiables in §1.
- **n < 8 speakers in a band** → band is **not dropped silently**. It is retained in every table
  with its true n, and the shortfall is reported as an explicit, honest limitation. If most or
  all elderly bands fall below this threshold, the paper's framing shifts from "here's the
  benchmark" to "here's what happens when you try to build this benchmark from the best
  available public data — and it's still not enough" (handoff §0a step 4) — itself a valid
  finding, not a failure to hide.

### 5.0 Common Voice Hindi — real counts

`cv-corpus-26.0-2026-06-12`, validated split only (produced by
`scripts/fetch_common_voice_hindi.py`, exact — not estimated):

| Bracket | n speakers | n validated clips |
|---|---|---|
| control (<60) | 145 | 7617 |
| 60s | 2 | 54 |
| 70s | **0** | 0 |
| 80+ | **0** | 0 |
| unspecified (age blank) | 222 | 3460 |

**Decision (handoff §0a step 4 applied):** every elderly band fails the n≥8 threshold — 60s is
80% short (2 of 8), and 70s/80+ have zero speakers each, not merely "thin." This is not a
partial-power situation to caveat; **the Common Voice Hindi validated split alone cannot support
this study's elderly bands at all.** Two things compound this, not just the raw shortfall:

1. **222 of 369 total speakers (60%) left age blank.** Common Voice's age field is
   self-reported and optional; it is plausible (unknowable from this data) that some elderly
   speakers exist in the "unspecified" pool. This cannot be used to rescue the sample — there is
   no way to distinguish an elderly non-reporter from a non-elderly one — but it means "zero
   elderly speakers" is really "zero *confirmed* elderly speakers out of a large unlabeled
   remainder," which is a meaningfully different (and worth stating precisely) claim than "no
   elderly speakers exist in this corpus."
2. This is the **validated split only**, per this protocol's own scope (§2). The `other.tsv`
   split (unvalidated/unreviewed clips) was not counted here and could not be used for the same
   study without a validation step this protocol doesn't currently define — noting it as a
   possible follow-up, not a fix already applied.

**This alone (handoff §0a step 4 applied to Common Voice only) would trigger the reframing to an
honest negative finding.** But handoff §0a step 3 requires checking the IndicVoices-R fallback
before concluding infeasibility — done below, and it changes the outcome.

### 5.1 IndicVoices-R Hindi — real counts

Produced by `scripts/fetch_indicvoices_r_hindi.py`, a column-selective read of
`speaker_id`/`age_group` across all 10 parquet shards of `SPRINGLab/IndicVoices-R_Hindi` (§2.1)
— exact, not estimated; every row was read, not sampled:

| `age_group` | n speakers | n utterances |
|---|---|---|
| 18-30 | 128 | 8,778 |
| 30-45 | 117 | 8,503 |
| 45-60 | 73 | 5,138 |
| **60+** | **50** | **3,899** |
| **Total** | **368** | **26,318** |

**Decision:** the 60+ band clears the n≥8 stopping rule with wide margin (50 vs. the n≥8 floor),
and the combined under-60 pool (318 speakers) is more than sufficient for a matched control
cohort drawn from the same corpus (§3.2). **Recommendation: IndicVoices-R Hindi becomes the
source of record for both the elderly and control cohorts; Common Voice Hindi is retained as a
documented, checked-but-unused source (§2.2), not the primary dataset.**

This recommendation carries three real costs, all already threaded through §2.1, §3.1, §3.3, §4
above — restated together here since they're the actual terms of the trade-off Shubham is
signing off on, not just individually scattered caveats:

1. **Age resolution collapses from a potential 60s/70s/80s scheme to a single 60+ band** (§3.1).
   This is a bigger loss than handoff §0 anticipated (it expected possible collapse to decade
   brackets, not to one combined elderly band) and is a second, independent reason `brief.pdf`
   §1's Positioning section needs the rewrite §0b already flagged — not a variant of that same
   issue, an additional one.
2. **The speech-enhancement/denoising caveat is now live, not hypothetical** (§2.1) — it applies
   to the dataset actually driving the paper's central comparison, not a fallback that might not
   be used.
3. **Speech-type classification (read vs. spontaneous) is unconfirmed** for IndicVoices-R (§4) —
   this needs to be resolved, not assumed favorably, before it's written into any release
   artifact.

**Confirmed by Shubham, 2026-08-16:** source switch to IndicVoices-R accepted as recommended
(§2.1, §3.1, §3.2); Common Voice retained as supporting evidence, not dropped (§2.2). Phase 1 is
closed — see the status line at the top of this document.

---

## 6. Metrics and statistical reporting (reference — see `src/vayas/metrics`, `src/vayas/stats`)

Full definitions live in code; per `brief.pdf` §3, all of: WER, CER, WER−CER divergence, task
success rate, entity accuracy, substitution/insertion/deletion ratios, speaking-rate × WER
correlation, and per-speaker (not just corpus-aggregate) distributions — every one stratified
per §3.3, every one reported as `(estimate, ci_low, ci_high, n)`, never a bare point estimate.

**Wording note on CER:** `brief.pdf` §3 frames CER as "primary for the Dravidian language;
morphology-robust" — that specific justification (Dravidian morphology) no longer applies with
Tamil out of scope (§1). CER remains relevant for Hindi on the same morphology-robustness
grounds (Hindi is morphologically rich relative to English, per Thennal D K, James, Gopinath &
Ashraf K, 2025, NAACL Findings — the non-negotiable citation grounding the WER–CER decomposition
per `brief.pdf` §2D), so the metric is retained; only its stated justification changes from
"the Dravidian language" to "Hindi's own morphology."

---

## 7. Key citations grounding this protocol (full list: `docs/brief.pdf` §2)

- Koenecke, Nam, Lake, Rickford, Jurafsky & Goel (2020), PNAS 117(14):7684–7689 — control-cohort
  matching design, adopted directly in §3.2.
- Pellegrini et al. (2012), Springer LNCS — closest non-English age-banding precedent for the
  finer 60/70/80+ scheme `brief.pdf` originally specified; retained here as the citation for what
  the *ideal* band structure would be, even though §3.1's actual recommendation collapses to a
  single 60+ band under the source data's real constraints.
- Thennal D K, James, Gopinath & Ashraf K (2025), NAACL Findings, 4941–4950 — grounds the
  WER–CER decomposition (§6).
- Buolamwini & Gebru (2018), FAccT (Gender Shades) — structural audit blueprint: build a
  benchmark, train nothing, audit existing systems.

---

## 8. Definition of done

Per handoff §2 Phase 1: this document is complete and reviewed by Shubham, reflecting **real**
counts from §5 (not projected ones), before any Phase 2 ingestion code is pointed at real data.
Real counts exist for both candidate sources (§5.0, §5.1), §4's speech-type question is resolved
(confirmed spontaneous), and Shubham has explicitly confirmed the resulting recommendation:
IndicVoices-R Hindi as source of record, single control-vs-60+ age scheme, Common Voice retained
as supporting evidence. **This document is done. Phase 2 may begin.**
