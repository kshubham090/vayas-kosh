"""Stratified statistics (Phase 6): bootstrap confidence intervals and
effect sizes (Cohen's d / rank-biserial) for every metric, every stratum.

No function in this module may return a bare point estimate — every
output is (estimate, ci_low, ci_high, n).
"""

from vayas.stats.bootstrap import (
    Estimate,
    bootstrap_mean_ci,
    cohens_d,
    rank_biserial,
    spearman_correlation,
)

__all__ = ["Estimate", "bootstrap_mean_ci", "cohens_d", "rank_biserial", "spearman_correlation"]
