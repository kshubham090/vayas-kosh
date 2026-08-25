"""Bootstrap confidence intervals and effect sizes, resampled at the
SPEAKER level, not the clip level.

Multiple clips from the same speaker are not independent observations --
treating them as such (clip-level resampling) would understate variance
and overstate confidence. Callers must aggregate per-clip metrics to one
value per speaker (e.g. mean WER per speaker) before passing them here.
This is what makes the per-speaker distributions in brief.pdf sec.3 a
statistical requirement, not just a reporting nicety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class Estimate:
    estimate: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_mean_ci(
    per_speaker_values: list[float], n_boot: int = 10_000, ci: float = 0.95, seed: int = 0,
) -> Estimate:
    arr = np.asarray(per_speaker_values, dtype=float)
    n = len(arr)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return Estimate(float(arr[0]), float(arr[0]), float(arr[0]), 1)
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return Estimate(float(arr.mean()), float(lo), float(hi), n)


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Positive d means group_a > group_b."""
    a, b = np.asarray(group_a, dtype=float), np.asarray(group_b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def rank_biserial(group_a: list[float], group_b: list[float]) -> tuple[float, float]:
    """Nonparametric effect size via Mann-Whitney U -- more robust than
    Cohen's d for the small, likely-skewed per-speaker samples here.
    Returns (rank_biserial_correlation, mannwhitney_p_value). Positive
    correlation means group_a tends larger than group_b.
    """
    a, b = np.asarray(group_a, dtype=float), np.asarray(group_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    # scipy's mannwhitneyu(a, b) returns U for sample a, which grows
    # toward len(a)*len(b) as a's values tend LARGER than b's -- so
    # 2*U/(na*nb) - 1 gives +1 when a is uniformly larger, -1 when
    # uniformly smaller, matching cohens_d's "positive = a > b" sign
    # convention. (The more common textbook formula, 1 - 2U/(na*nb), is
    # the OPPOSITE sign of this -- verified against scipy's actual U
    # definition after a real sign mismatch against cohens_d surfaced it.)
    u_stat, p_value = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    r = (2 * u_stat) / (len(a) * len(b)) - 1
    return float(r), float(p_value)


def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """(rho, p_value) for e.g. speaking-rate x WER, per brief.pdf sec.3."""
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)
