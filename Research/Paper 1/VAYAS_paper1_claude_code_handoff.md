# VAYAS (Vyaskosh Paper 1) — Claude Code Handoff

**Owner:** Shubham Gupta, Amity University, shubham.kumar59@s.amity.edu
**Source of truth:** `VAYAS_project_brief_v1.0.pdf` (attached in repo as `docs/brief.pdf`) — this handoff operationalizes it. If anything here conflicts with the brief, the brief wins.

**What this document is:** a task-by-task execution spec for Claude Code to build the VAYAS research pipeline — protocol instruments, data pipeline, ASR audit harness, metrics/stats, error taxonomy, and release artifacts. It does NOT write the paper prose (that comes last, off real results). It also does not do recruitment/ethics approval — that's Shubham's track, running in parallel.

**Non-negotiables carried over from the brief (do not violate these while building anything):**
1. Protocol is written and frozen *before* any recording happens.
2. Every language/system choice must be justified in language-general terms — nothing Hindi-specific baked into "portable" code paths.
3. Matched non-elderly control cohort, same prompts/conditions, is mandatory — no elderly-only dataset.
4. Zero-shot audit only. Nothing gets fine-tuned or trained. If a script trains something, it's out of scope for this paper.
5. Every metric gets reported stratified + with CIs/effect sizes. No bare point estimates, ever, anywhere in outputs.
6. Everything is releasable: dataset, eval code, protocol doc, replication template. Build with that as a constraint, not an afterthought.

---

## 0. Working assumptions (confirm/override before Phase 1)

**CHANGED (2026-08-13): This project uses existing public datasets, not new primary recruitment.** No new recording, no new consent/ethics gate for data collection.

**CHANGED (2026-08-16): Single-language design — Hindi only.** The two-family (Indo-Aryan + Dravidian) design from the brief is dropped. Tamil, and the Tamil-specific LT-EDI shared-task dataset, are out of scope entirely. See §0b for why this needs a positioning rewrite, not just a data-pipeline change.

| Item | Default assumption | Where it's used |
|---|---|---|
| Language | Hindi only | prompt/domain filtering, ASR system selection |
| Data source | Public datasets with existing age metadata — Mozilla Common Voice (Hindi) | replaces primary recruitment entirely |
| Scale target | Whatever the source data actually contains once counted — see Phase 0a below. No longer a recruitment target; now a **reporting requirement**: if real counts fall short of a defensible per-band n, that shortfall gets reported honestly in the paper, not padded | stratification feasibility, paper framing |
| Age bands | <60 (control), 60–69, 70–79, 80+ where source data resolution allows; Common Voice only gives decade brackets (sixties/seventies/eighties), so bands may need to collapse to 60s/70s/80+ | stratification code everywhere |
| Domain for prompts | N/A — no new elicitation, this only applied to the abandoned recruitment path | — |
| Systems under audit | Whisper (large-v3), MMS, Meta Omnilingual ASR, IndicWhisper, IndicWav2Vec, IndicConformer, Sarvam Saarika (all systems that support Hindi — drop any Tamil-only or Dravidian-focused entries) | audit harness |
| Repo language | Python 3.11, uv/venv for env management | everything |

### 0a. New Phase 0 — Source data feasibility check (do this before anything else)
This replaces the old "recruitment feasibility" gate. Concretely:
1. Script to download/access Common Voice Hindi validated split (via Mozilla Data Collective — requires free account signup, not open anonymous download as of Oct 2025 platform migration).
2. Count speakers per age bracket (sixties/seventies/eighties), on **validated clips only**. Report exact numbers — do not estimate.
3. Same for any candidate fallback source (IndicVoices-R age-group metadata — flag its speech-enhancement preprocessing as a caveat if used, since denoising/dereverb may suppress the acoustic degradation signal the paper measures).
4. Decision gate: if elderly-band counts are too thin to support per-band stratified stats (rule of thumb: need n≥8 per band minimum to report any CI/effect size meaningfully), the paper's framing shifts from "here's the benchmark" to "here's what happens when you try to build this benchmark from the best available public data — and it's still not enough," which is itself a valid, honest finding.

### 0b. Positioning consequence — needs a real rewrite, not just a find-and-replace
The brief's Positioning section (§1) explicitly built the "methodology is the contribution, India is the existence proof" claim on a two-family design: *"Indian languages are chosen as a deliberately difficult instantiation: two distinct families, multiple scripts, a steep resource gradient."* That specific evidence disappears with Hindi-only — a single language can describe a portable protocol but can't demonstrate it transferring across a family/script boundary, which is what made the audit closer to Gender Shades' (multi-group) and Koenecke's (matched-cohort) structural precedent.

This doesn't kill the paper — it changes what claim the paper is entitled to make. Options to resolve later, not now:
- **Narrower, still-honest claim:** "portable protocol, single-language proof-of-concept, explicit call for replication" — frame Hindi as instantiation #1, with the released protocol/replication template as the actual portability mechanism (someone else runs it in Yoruba, not you).
- **Depth-over-breadth reframe:** single language allows more speakers, finer age bands, more systems audited — trade breadth for statistical power, which is defensible if said explicitly.
Do NOT let this get glossed over silently in the paper draft — Phase 9 (paper scaffold) should carry an explicit note to revisit the Positioning/Objective section wording once this is decided, not just swap "Hindi and Tamil" → "Hindi" in the brief's prose.

---

## 1. Repo structure to scaffold first

```
vayas/
├── docs/
│   ├── brief.pdf
│   ├── protocol.md              # ← Phase 1 output, this is THE gate doc
│   ├── error_taxonomy.md
│   ├── dataset_card.md           # HF-style datasheet
│   └── replication_template.md
├── instruments/
│   ├── consent_form_{hi,ta}.md
│   ├── elicitation_prompts_{hi,ta}.json
│   └── recruitment_tracker.csv
├── data/
│   ├── raw/                      # gitignored, audio never committed
│   ├── metadata/
│   │   └── speakers.csv          # id, age_band, lang, gender, region, consent_id
│   └── transcripts/
│       ├── gold/
│       └── iaa/                  # inter-annotator overlap subsets
├── src/vayas/
│   ├── protocol/                 # prompt generation, consent doc rendering
│   ├── ingest/                   # audio validation, metadata schema enforcement
│   ├── transcribe/                # transcription conventions, IAA (kappa/alpha)
│   ├── audit/                    # ASR system wrappers, zero-shot inference
│   ├── metrics/                  # WER, CER, entity acc, task success, sub/ins/del
│   ├── stats/                    # stratified breakdowns, bootstrap CIs, effect sizes
│   ├── taxonomy/                 # error-tagging tool
│   └── release/                  # dataset card + model card generators
├── notebooks/                    # exploratory only, nothing canonical lives here
├── tests/
├── pyproject.toml
└── README.md
```

Claude Code: scaffold this exactly, empty stubs with docstrings, before writing any real logic. Commit as the first commit.

---

## 2. Phase breakdown

**NOTE: Phase 1 below is rewritten for the public-data path (§0a). The original recruitment/consent version is no longer in scope — no new human-subjects recording is happening.**

### Phase 1 — Data Provenance & Sampling Protocol (replaces old recruitment protocol)
**Goal:** produce `docs/protocol.md` — but now it documents *how existing public data was selected, filtered, and stratified*, not how new speakers were recruited. Still needs to be reproducible by a stranger: given the same public dataset, they should get the same sample.

Tasks:
- Document exact source(s) per language (e.g. Common Voice v25 Hindi validated split), exact download/access method, exact date pulled (public datasets get updated — pin a version).
- Document exact filtering logic: which age labels count as which band, how "control <60" is sampled from the same corpus (same source, same recording conditions — this is actually a *cleaner* control-matching case than new recruitment, since it's genuinely the same platform/mic conditions).
- Write the stopping/inclusion rule from §0a step 4 explicitly, including what happens if bands are underpowered (report as a limitation, don't drop the band silently).
- Write transcription-conventions note: Common Voice transcripts are prompted/read speech transcribed by the platform already — document how this interacts with the brief's "age-typical disfluency" conventions (read speech has less disfluency than spontaneous; this is a real deviation from the brief's spontaneous-speech design and should be stated plainly in protocol.md, not glossed over).
- (LT-EDI Tamil is out of scope — no longer applicable.)

**Definition of done:** `docs/protocol.md` complete + reviewed by Shubham, reflecting real counts from §0a (not projected ones), before Phase 2 audit code is pointed at real data.

### Phase 2 — Data ingestion & metadata
- `data/metadata/speakers.csv` schema: speaker_id, lang, age, age_band, gender, region, consent_id, recording_date, speech_type (read/spontaneous), acoustic_condition (quiet/ambient), device.
- `src/vayas/ingest/`: audio validation (sample rate, mono/stereo, min duration, silence detection), automatic metadata schema enforcement, rejects malformed uploads with clear errors.
- Build this against synthetic/dummy audio first — don't wait on real recordings to have working ingestion code.

### Phase 3 — Transcription pipeline
- `src/vayas/transcribe/`: tooling to support human transcription against the Phase 1 conventions (not ASR — this is gold-standard human transcript creation).
- Inter-annotator agreement: subset of utterances double-transcribed, compute agreement (word-level, e.g. Cohen's kappa or Krippendorff's alpha on segmented tokens) — code this now against dummy transcript pairs so it's ready.

### Phase 4 — ASR audit harness (zero-shot only)
- `src/vayas/audit/`: one wrapper per system (Whisper, MMS, Omnilingual, IndicWhisper, IndicWav2Vec, IndicConformer, Sarvam Saarika) behind a common interface: `transcribe(audio_path, lang) -> hypothesis_text`.
- Batch runner that takes `data/metadata/speakers.csv` + audio, runs every system on every utterance, writes hypotheses to `data/transcripts/hypotheses/{system}/{utt_id}.txt`.
- No fine-tuning code should exist anywhere in this module — a code review gate: if any script has a training loop, that's a scope violation.
- Handle API vs. local-weights systems differently (Sarvam Saarika likely API; IndicWhisper/Wav2Vec/Conformer local) — abstract this behind the same interface.

### Phase 5 — Metrics
Per brief §3, all of:
- WER, CER (use a maintained library — jiwer or similar; verify CER handles Devanagari script tokenization correctly).
- WER−CER divergence.
- Task success rate: did the utterance resolve to correct welfare scheme (needs a small intent/entity gold-label set per utterance).
- Entity accuracy: scheme names, numbers, dates, place names — needs an entity extraction/matching step, likely rule-based given small scale.
- Substitution/insertion/deletion ratios (jiwer gives this natively).
- Speaking rate × WER correlation.
- Per-speaker distribution (not just corpus aggregate — worst-case speaker is the point).

All metrics computed stratified across: age band, language/family, speech type, acoustic condition, gender.

### Phase 6 — Statistics
- `src/vayas/stats/`: bootstrap confidence intervals for every stratified metric, effect sizes (e.g. Cohen's d or rank-biserial depending on distribution) for elderly-vs-control comparisons.
- No function in this module should return a bare mean — every output type is `(estimate, ci_low, ci_high, n)`.

### Phase 7 — Error taxonomy
- `src/vayas/taxonomy/`: tagging tool (even a simple CLI or notebook widget) to categorize failure instances into a portable taxonomy (e.g. acoustic/pitch-related, disfluency-related, OOV/lexical, morphological, code-switching, environmental) — built against the transcripts once real audit hypotheses exist.
- Taxonomy categories defined language-generally even though only Hindi is instantiated here — this is what keeps the taxonomy itself portable/reusable per §0b, even though the language coverage isn't.

### Phase 8 — Release artifacts
- `docs/dataset_card.md` (Datasheets for Datasets format).
- Model/audit cards per system audited (Model Cards format) — documenting what was audited, not building new models.
- `docs/replication_template.md`: fill-in-the-blank doc for someone replicating this in a third language.
- Ensure `src/vayas/audit` and `src/vayas/metrics` are runnable standalone as the "evaluation code" release artifact — someone should be able to clone, point at their own audio+metadata, and get stratified metrics out.

### Phase 9 — Paper scaffold (last, only once real numbers exist)
- LaTeX repo structure mirroring brief's section headers (Objective, Reading, Metrics, Method, Results, Discussion).
- Figures pipeline: scripts that regenerate every paper figure/table directly from `src/vayas/stats` outputs — no hand-copied numbers into LaTeX, ever, to avoid drift between code and paper.

---

## 3. Libraries (starting point, Claude Code can substitute with justification)
- Audio/ASR: `torch`, `transformers`, `openai-whisper` or HF Whisper, model-specific SDKs for MMS/Omnilingual/Sarvam.
- Metrics: `jiwer` (verify Indic CER correctness), custom entity matcher.
- Stats: `scipy`, `numpy`, bootstrap via `scikit-learn` or manual.
- IAA: `nltk.agreement` or `krippendorff` package.
- Data validation: `pydantic` for metadata schema enforcement.

## 4. What Claude Code should NOT do
- Don't invent recruitment numbers or synthetic "real" speaker data and treat it as real — dummy data must be clearly marked and never flow into `docs/dataset_card.md` or any release artifact.
- Don't write paper prose/claims ahead of Phase 9 — no results section drafted before real metrics exist.
- Don't fine-tune or train any ASR component under any framing.
- Don't skip the stratified+CI requirement "for now" — retrofitting this later is expensive and error-prone.

## 5. Immediate next action for Claude Code
Scaffold the repo (§1), then execute Phase 1 (protocol doc + instruments) end to end, using the assumptions in §0 unless Shubham has overridden them. Stop and flag before Phase 2 if any Phase 1 output wasn't explicitly reviewed.
