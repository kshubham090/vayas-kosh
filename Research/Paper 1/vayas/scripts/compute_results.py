"""Phase 5/6: compute WER/CER metrics against gold transcripts and the
age-stratified statistics (CI + effect size) the brief requires as a
non-negotiable (sec.4, "every metric reported stratified, with CIs and
effect sizes -- no bare point estimates, ever").

Reads every manifest_*.csv under data/transcripts/hypotheses/ (one per
batch run: slow_batch, fast_batch, slow_control, wave3) to recover each
clip's gold verbatim/speaker_id/age_band/duration, keyed by utt_id --
utt_id is unique within the source corpus, so clips repeated across
manifests (a known, documented overlap between wave1 and wave2, see
vayas-batch-slow-wave3's own exclusion logic) just resolve to the same
gold record.

For each of the 4 confirmed systems, reads every hypothesis .txt file
actually produced, computes per-clip WER/CER/divergence/S-I-D, then
aggregates to per-speaker means and runs the stratified stats.

Writes:
  data/results/clip_metrics.csv   -- one row per (system, clip)
  data/results/summary.csv        -- one row per (system, age_band):
                                      estimate, ci_low, ci_high, n_speakers
  data/results/effect_sizes.csv   -- one row per system: Cohen's d,
                                      rank-biserial r, Mann-Whitney p,
                                      for elderly vs control, per metric
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vayas.metrics import compute_clip_metrics
from vayas.stats import bootstrap_mean_ci, cohens_d, rank_biserial, spearman_correlation

HYPOTHESES_DIR = ROOT / "data" / "transcripts" / "hypotheses"
RESULTS_DIR = ROOT / "data" / "results"
SYSTEMS = ["mms-1b-all", "indicconformer-600m-multilingual", "whisper-large-v3", "indicwhisper-hindi"]
MIN_SPEAKERS_PER_BAND = 8  # decision gate, protocol.md sec.4


def load_gold() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    manifest_paths = sorted(HYPOTHESES_DIR.glob("manifest_*.csv"))
    if not manifest_paths:
        raise SystemExit(f"No manifest_*.csv found under {HYPOTHESES_DIR}")
    for path in manifest_paths:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                gold[row["utt_id"]] = row
    print(f"Loaded gold metadata for {len(gold)} unique clips from {len(manifest_paths)} manifests", flush=True)
    return gold


def compute_all_clip_metrics(gold: dict[str, dict]) -> list:
    all_metrics = []
    skipped_empty_ref = 0
    skipped_no_gold = 0

    for system in SYSTEMS:
        system_dir = HYPOTHESES_DIR / system
        if not system_dir.exists():
            print(f"  WARNING: no hypotheses dir for {system}, skipping", flush=True)
            continue
        txt_files = list(system_dir.glob("*.txt"))
        print(f"  {system}: {len(txt_files)} hypothesis files", flush=True)

        for txt_path in txt_files:
            utt_id = txt_path.stem
            row = gold.get(utt_id)
            if row is None:
                skipped_no_gold += 1
                continue
            hypothesis = txt_path.read_text(encoding="utf-8")
            metrics = compute_clip_metrics(
                utt_id=utt_id,
                speaker_id=row["speaker_id"],
                age_band=row["age_band"],
                system=system,
                duration=float(row["duration"]),
                reference=row["verbatim"],
                hypothesis=hypothesis,
            )
            if metrics is None:
                skipped_empty_ref += 1
                continue
            all_metrics.append(metrics)

    print(f"Computed metrics for {len(all_metrics)} clips "
          f"(skipped {skipped_empty_ref} empty-gold, {skipped_no_gold} no-gold-match)", flush=True)
    return all_metrics


def write_clip_metrics(all_metrics: list) -> Path:
    out_path = RESULTS_DIR / "clip_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "utt_id", "speaker_id", "age_band", "duration", "wer", "cer",
                    "divergence", "substitutions", "deletions", "insertions", "hits",
                    "ref_words", "hyp_words", "speaking_rate"])
        for m in all_metrics:
            w.writerow([m.system, m.utt_id, m.speaker_id, m.age_band, m.duration, m.wer, m.cer,
                        m.divergence, m.substitutions, m.deletions, m.insertions, m.hits,
                        m.ref_words, m.hyp_words, m.speaking_rate])
    print(f"Wrote {len(all_metrics)} rows to {out_path}", flush=True)
    return out_path


def per_speaker_means(all_metrics: list, system: str, age_band: str, metric: str) -> list[float]:
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for m in all_metrics:
        if m.system == system and m.age_band == age_band:
            by_speaker[m.speaker_id].append(getattr(m, metric))
    return [sum(vals) / len(vals) for vals in by_speaker.values()]


def write_summary(all_metrics: list) -> None:
    out_path = RESULTS_DIR / "summary.csv"
    rows = []
    age_bands = sorted({m.age_band for m in all_metrics})
    for system in SYSTEMS:
        for age_band in age_bands:
            for metric in ("wer", "cer", "divergence"):
                vals = per_speaker_means(all_metrics, system, age_band, metric)
                est = bootstrap_mean_ci(vals)
                flag = "LOW_N" if est.n < MIN_SPEAKERS_PER_BAND and est.n > 0 else ""
                rows.append([system, age_band, metric, est.estimate, est.ci_low, est.ci_high, est.n, flag])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "age_band", "metric", "estimate", "ci_low", "ci_high", "n_speakers", "flag"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}", flush=True)

    print("\n=== Summary (per-speaker mean, 95% bootstrap CI) ===", flush=True)
    for r in rows:
        system, age_band, metric, est, lo, hi, n, flag = r
        print(f"  {system:35s} {age_band:8s} {metric:11s} {est:.4f} [{lo:.4f}, {hi:.4f}] n={n} {flag}", flush=True)


def write_effect_sizes(all_metrics: list) -> None:
    out_path = RESULTS_DIR / "effect_sizes.csv"
    rows = []
    for system in SYSTEMS:
        for metric in ("wer", "cer", "divergence"):
            elderly = per_speaker_means(all_metrics, system, "60+", metric)
            control = per_speaker_means(all_metrics, system, "control", metric)
            d = cohens_d(elderly, control)
            r, p = rank_biserial(elderly, control)
            rows.append([system, metric, len(elderly), len(control), d, r, p])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "metric", "n_elderly_speakers", "n_control_speakers",
                     "cohens_d", "rank_biserial_r", "mannwhitney_p"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)

    print("\n=== Effect sizes: elderly vs control (positive = elderly higher error) ===", flush=True)
    for r in rows:
        system, metric, n_e, n_c, d, rb, p = r
        print(f"  {system:35s} {metric:11s} n_e={n_e} n_c={n_c}  Cohen's d={d:.3f}  "
              f"rank-biserial={rb:.3f}  p={p:.4f}", flush=True)


def write_speaking_rate_correlation(all_metrics: list) -> None:
    print("\n=== Speaking-rate x WER (Spearman, per system, across all clips) ===", flush=True)
    for system in SYSTEMS:
        rates = [m.speaking_rate for m in all_metrics if m.system == system]
        wers = [m.wer for m in all_metrics if m.system == system]
        rho, p = spearman_correlation(rates, wers)
        print(f"  {system:35s} rho={rho:.3f}  p={p:.4f}  n={len(rates)}", flush=True)


DEGENERATE_WER_THRESHOLD = 1.0  # WER > 1.0 is only possible via runaway
# insertion (hypothesis has far more words than reference) -- a
# mathematical signature of decoder repetition-loop hallucination, not
# normal transcription difficulty. Found by inspecting indicwhisper's
# worst clips: real repeated-token degeneration ("ठीक है ना" x9 -> "ना"
# x30 -> "शा" x100), not a metrics bug.


def write_degenerate_rate(all_metrics: list) -> None:
    out_path = RESULTS_DIR / "degenerate_output_rate.csv"
    rows = []
    for system in SYSTEMS:
        clips = [m for m in all_metrics if m.system == system]
        n = len(clips)
        degenerate = [m for m in clips if m.wer > DEGENERATE_WER_THRESHOLD]
        n_degen = len(degenerate)
        avg_dur_degen = sum(m.duration for m in degenerate) / n_degen if n_degen else float("nan")
        normal = [m for m in clips if m.wer <= DEGENERATE_WER_THRESHOLD]
        avg_dur_normal = sum(m.duration for m in normal) / len(normal) if normal else float("nan")
        rows.append([system, n, n_degen, n_degen / n if n else float("nan"), avg_dur_degen, avg_dur_normal])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "n_clips", "n_degenerate", "degenerate_rate", "avg_duration_degenerate", "avg_duration_normal"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)

    print("\n=== Degenerate-output rate (WER > 1.0, repetition-loop signature) ===", flush=True)
    for system, n, n_degen, rate, dd, dn in rows:
        print(f"  {system:35s} {n_degen:5d}/{n:5d} = {100*rate:.2f}%   "
              f"avg dur: degenerate={dd:.2f}s normal={dn:.2f}s", flush=True)


def write_trimmed_summary(all_metrics: list) -> None:
    """Same as write_summary but excluding degenerate clips, so the
    system ranking can be checked for sensitivity to the repetition-loop
    pathology rather than assumed unaffected.
    """
    out_path = RESULTS_DIR / "summary_trimmed.csv"
    trimmed = [m for m in all_metrics if m.wer <= DEGENERATE_WER_THRESHOLD]
    rows = []
    age_bands = sorted({m.age_band for m in trimmed})
    for system in SYSTEMS:
        for age_band in age_bands:
            for metric in ("wer", "cer"):
                vals = per_speaker_means(trimmed, system, age_band, metric)
                est = bootstrap_mean_ci(vals)
                rows.append([system, age_band, metric, est.estimate, est.ci_low, est.ci_high, est.n])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "age_band", "metric", "estimate", "ci_low", "ci_high", "n_speakers"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)

    print("\n=== Trimmed summary (degenerate clips excluded, per-speaker mean) ===", flush=True)
    for r in rows:
        system, age_band, metric, est, lo, hi, n = r
        print(f"  {system:35s} {age_band:8s} {metric:5s} {est:.4f} [{lo:.4f}, {hi:.4f}] n={n}", flush=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gold = load_gold()
    all_metrics = compute_all_clip_metrics(gold)
    write_clip_metrics(all_metrics)
    write_summary(all_metrics)
    write_effect_sizes(all_metrics)
    write_speaking_rate_correlation(all_metrics)
    write_degenerate_rate(all_metrics)
    write_trimmed_summary(all_metrics)

    print("\nNOTE: task success rate and entity accuracy (brief.pdf sec.3) are NOT "
          "computed here -- deferred, see src/vayas/metrics/__init__.py for why.", flush=True)


if __name__ == "__main__":
    main()
