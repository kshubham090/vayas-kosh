"""Follow-up analysis after external review flagged real gaps in the
first results pass: per-speaker clip-count imbalance, BH-corrected
significance, degenerate rate by cohort, trimmed-data significance
retest, and a minimum-detectable-effect note. Reuses clip_metrics.csv
from compute_results.py rather than recomputing WER/CER.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from scipy import stats as scipy_stats
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
SYSTEMS = ["mms-1b-all", "indicconformer-600m-multilingual", "whisper-large-v3", "indicwhisper-hindi"]


def load_clips():
    with open(RESULTS / "clip_metrics.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def per_speaker_means(rows, system, age_band, metric):
    by_speaker = defaultdict(list)
    for r in rows:
        if r["system"] == system and r["age_band"] == age_band:
            by_speaker[r["speaker_id"]].append(float(r[metric]))
    return {sid: sum(v) / len(v) for sid, v in by_speaker.items()}


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def rank_biserial(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float((2 * u) / (len(a) * len(b)) - 1), float(p)


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


def main():
    rows = load_clips()

    print("=" * 70)
    print("1. PER-SPEAKER CLIP-COUNT IMBALANCE (elderly vs control)")
    print("=" * 70)
    for system in SYSTEMS:
        for band in ["60+", "control"]:
            counts = defaultdict(int)
            for r in rows:
                if r["system"] == system and r["age_band"] == band:
                    counts[r["speaker_id"]] += 1
            n_speakers = len(counts)
            n_clips = sum(counts.values())
            avg = n_clips / n_speakers if n_speakers else float("nan")
            print(f"  {system:35s} {band:8s} {n_clips:5d} clips / {n_speakers:4d} speakers = {avg:6.2f} clips/speaker")

    print()
    print("=" * 70)
    print("2. BENJAMINI-HOCHBERG CORRECTED SIGNIFICANCE (8 tests: 4 systems x WER/CER)")
    print("=" * 70)
    tests = []
    for system in SYSTEMS:
        for metric in ("wer", "cer"):
            elderly = list(per_speaker_means(rows, system, "60+", metric).values())
            control = list(per_speaker_means(rows, system, "control", metric).values())
            d = cohens_d(elderly, control)
            rb, p = rank_biserial(elderly, control)
            tests.append((system, metric, d, rb, p))
    pvals = [t[4] for t in tests]
    adj = benjamini_hochberg(pvals)
    for (system, metric, d, rb, p), p_adj in zip(tests, adj):
        flag = "SIG" if p_adj < 0.05 else "ns"
        print(f"  {system:35s} {metric:5s} d={d:+.3f} rb={rb:+.3f} p={p:.4f} p_BH={p_adj:.4f} [{flag}]")

    print()
    print("=" * 70)
    print("3. DEGENERATE-OUTPUT RATE BY COHORT (WER > 1.0)")
    print("=" * 70)
    for system in SYSTEMS:
        for band in ["60+", "control"]:
            clips = [r for r in rows if r["system"] == system and r["age_band"] == band]
            n = len(clips)
            n_degen = sum(1 for r in clips if float(r["wer"]) > 1.0)
            rate = n_degen / n if n else float("nan")
            print(f"  {system:35s} {band:8s} {n_degen:4d}/{n:5d} = {100*rate:.2f}%")

    print()
    print("=" * 70)
    print("4. TRIMMED-DATA SIGNIFICANCE RETEST (degenerate clips excluded)")
    print("=" * 70)
    trimmed = [r for r in rows if float(r["wer"]) <= 1.0]
    for system in SYSTEMS:
        for metric in ("wer", "cer"):
            elderly = list(per_speaker_means(trimmed, system, "60+", metric).values())
            control = list(per_speaker_means(trimmed, system, "control", metric).values())
            d = cohens_d(elderly, control)
            rb, p = rank_biserial(elderly, control)
            print(f"  {system:35s} {metric:5s} d={d:+.3f} rb={rb:+.3f} p={p:.4f}")

    print()
    print("=" * 70)
    print("5. GAP-CLOSURE: IndicWhisper WER, untrimmed vs trimmed")
    print("=" * 70)
    for label, dataset in [("untrimmed", rows), ("trimmed", trimmed)]:
        e = per_speaker_means(dataset, "indicwhisper-hindi", "60+", "wer")
        c = per_speaker_means(dataset, "indicwhisper-hindi", "control", "wer")
        gap = (sum(e.values()) / len(e) - sum(c.values()) / len(c)) * 100
        print(f"  {label}: elderly-control gap = {gap:.2f} pp")

    print()
    print("=" * 70)
    print("6. MINIMUM DETECTABLE EFFECT (n=50 vs n~313, alpha=.05, power=.80)")
    print("=" * 70)
    from scipy.stats import norm
    n1, n2 = 50, 313
    alpha, power = 0.05, 0.80
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    mde = (z_alpha + z_beta) * np.sqrt(1 / n1 + 1 / n2)
    print(f"  MDE (Cohen's d) at n1={n1}, n2={n2}: d = {mde:.3f}")

    print()
    print("=" * 70)
    print("7. SPEAKING-RATE x WER, PARTIAL CORRELATION CONTROLLING FOR REF LENGTH")
    print("=" * 70)
    for system in SYSTEMS:
        clips = [r for r in rows if r["system"] == system]
        rate = np.array([float(r["speaking_rate"]) for r in clips])
        wer = np.array([float(r["wer"]) for r in clips])
        ref_words = np.array([float(r["ref_words"]) for r in clips])
        mask = (rate > 0) & (rate < 10) & (ref_words > 0)
        rate, wer, ref_words = rate[mask], wer[mask], ref_words[mask]
        r_rate_wer = scipy_stats.spearmanr(rate, wer)
        r_rate_ref = scipy_stats.spearmanr(rate, ref_words)
        r_ref_wer = scipy_stats.spearmanr(ref_words, wer)
        r_rw, r_rr, r_fw = r_rate_wer.statistic, r_rate_ref.statistic, r_ref_wer.statistic
        partial = (r_rw - r_rr * r_fw) / np.sqrt((1 - r_rr**2) * (1 - r_fw**2))
        print(f"  {system:35s} raw rho={r_rw:+.3f}  partial rho (ctrl ref_words)={partial:+.3f}")


if __name__ == "__main__":
    main()
