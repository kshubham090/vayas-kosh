"""Re-score every clip's WER/CER against the `normalized` reference
instead of `verbatim`, and compare the age-effect pattern to the
verbatim-based results in data/results/. This is the direct test of the
gold-reference confound flagged in the paper's Limitations: verbatim
preserves disfluencies no audited system is trained to produce, so any
elderly-control difference in disfluency rate could produce a WER/CER
gap attributable to reference style, not acoustic recognition.

Requires data/transcripts/hypotheses/normalized_references.csv, produced
by fetch_normalized_references.py.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vayas.metrics import compute_clip_metrics
from vayas.stats import bootstrap_mean_ci, cohens_d, rank_biserial

HYPOTHESES_DIR = ROOT / "data" / "transcripts" / "hypotheses"
RESULTS_DIR = ROOT / "data" / "results"
SYSTEMS = ["mms-1b-all", "indicconformer-600m-multilingual", "whisper-large-v3", "indicwhisper-hindi"]


def load_gold_and_normalized() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for path in sorted(HYPOTHESES_DIR.glob("manifest_*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                gold[row["utt_id"]] = row

    norm_path = HYPOTHESES_DIR / "normalized_references.csv"
    with open(norm_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["utt_id"] in gold:
                gold[row["utt_id"]]["normalized"] = row["normalized"]

    n_missing = sum(1 for r in gold.values() if "normalized" not in r)
    print(f"Loaded gold for {len(gold)} clips, {n_missing} missing a normalized reference", flush=True)
    return gold


def compute_all_clip_metrics(gold: dict[str, dict]) -> list:
    all_metrics = []
    skipped_empty_ref = 0
    skipped_no_normalized = 0

    for system in SYSTEMS:
        system_dir = HYPOTHESES_DIR / system
        if not system_dir.exists():
            continue
        txt_files = list(system_dir.glob("*.txt"))
        print(f"  {system}: {len(txt_files)} hypothesis files", flush=True)

        for txt_path in txt_files:
            utt_id = txt_path.stem
            row = gold.get(utt_id)
            if row is None or "normalized" not in row:
                skipped_no_normalized += 1
                continue
            hypothesis = txt_path.read_text(encoding="utf-8")
            metrics = compute_clip_metrics(
                utt_id=utt_id,
                speaker_id=row["speaker_id"],
                age_band=row["age_band"],
                system=system,
                duration=float(row["duration"]),
                reference=row["normalized"],
                hypothesis=hypothesis,
            )
            if metrics is None:
                skipped_empty_ref += 1
                continue
            all_metrics.append(metrics)

    print(f"Computed metrics for {len(all_metrics)} clips "
          f"(skipped {skipped_empty_ref} empty-ref, {skipped_no_normalized} no-normalized-match)", flush=True)
    return all_metrics


def per_speaker_means(all_metrics: list, system: str, age_band: str, metric: str) -> list[float]:
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for m in all_metrics:
        if m.system == system and m.age_band == age_band:
            by_speaker[m.speaker_id].append(getattr(m, metric))
    return [sum(vals) / len(vals) for vals in by_speaker.values()]


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        p = pvals[idx]
        val = min(prev, p * n / i)
        adjusted[idx] = val
        prev = val
    return adjusted


def main() -> None:
    gold = load_gold_and_normalized()
    all_metrics = compute_all_clip_metrics(gold)

    out_path = RESULTS_DIR / "clip_metrics_normalized.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "utt_id", "speaker_id", "age_band", "wer", "cer"])
        for m in all_metrics:
            w.writerow([m.system, m.utt_id, m.speaker_id, m.age_band, m.wer, m.cer])
    print(f"Wrote {len(all_metrics)} rows to {out_path}", flush=True)

    print("\n=== Summary (normalized reference, per-speaker mean, 95% bootstrap CI) ===", flush=True)
    summary_rows = []
    for system in SYSTEMS:
        for age_band in ("60+", "control"):
            for metric in ("wer", "cer"):
                vals = per_speaker_means(all_metrics, system, age_band, metric)
                est = bootstrap_mean_ci(vals)
                summary_rows.append([system, age_band, metric, est.estimate, est.ci_low, est.ci_high, est.n])
                print(f"  {system:35s} {age_band:8s} {metric:5s} {est.estimate:.4f} [{est.ci_low:.4f}, {est.ci_high:.4f}] n={est.n}", flush=True)

    with open(RESULTS_DIR / "summary_normalized.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "age_band", "metric", "estimate", "ci_low", "ci_high", "n_speakers"])
        w.writerows(summary_rows)

    print("\n=== Effect sizes (normalized reference) + comparison to verbatim ===", flush=True)
    verbatim_effects = {}
    with open(RESULTS_DIR / "effect_sizes.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            verbatim_effects[(row["system"], row["metric"])] = float(row["cohens_d"])

    effect_rows = []
    tests = []
    for system in SYSTEMS:
        for metric in ("wer", "cer"):
            elderly = per_speaker_means(all_metrics, system, "60+", metric)
            control = per_speaker_means(all_metrics, system, "control", metric)
            d = cohens_d(elderly, control)
            rb, p = rank_biserial(elderly, control)
            tests.append((system, metric, d, rb, p))

    pvals = [t[4] for t in tests]
    adj = benjamini_hochberg(pvals)
    for (system, metric, d, rb, p), p_bh in zip(tests, adj):
        v_d = verbatim_effects.get((system, metric), float("nan"))
        same_sign = "same sign" if (d * v_d > 0) else "SIGN FLIP"
        effect_rows.append([system, metric, d, rb, p, p_bh, v_d, same_sign])
        print(f"  {system:35s} {metric:5s} normalized_d={d:+.3f} verbatim_d={v_d:+.3f} [{same_sign}] "
              f"p={p:.4f} p_BH={p_bh:.4f}", flush=True)

    with open(RESULTS_DIR / "effect_sizes_normalized.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "metric", "cohens_d_normalized", "rank_biserial_r", "p", "p_BH", "cohens_d_verbatim", "direction"])
        w.writerows(effect_rows)
    print(f"\nWrote {len(effect_rows)} rows to {RESULTS_DIR / 'effect_sizes_normalized.csv'}", flush=True)


if __name__ == "__main__":
    main()
