"""Pairwise comparison of mobility differences between targets, across replicates.

The statistics half of the legacy ``compare_targetmz_differences_with_boxplot``,
without plotting and without the import-time ``statannot`` requirement. Fixed:

* Tukey p-values are matched to pairs by group name. The legacy code zipped
  them against its own pair list, but statsmodels orders pairs by *sorted*
  group, so whenever targets were not already ascending the stars went on the
  wrong comparisons.
* ``correction="none"`` works; the legacy code raised for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from .config import StatsConfig
from .results import StatsResult

PAIR_COLUMNS = ["group_a", "group_b", "n_a", "n_b", "statistic", "raw_p_value",
                "adjusted_p_value", "reject"]


def compare_targets(differences: pd.DataFrame, config: StatsConfig = StatsConfig(),
                    window_pair: tuple[int, int] = (0, 1)) -> StatsResult:
    """Pairwise t-tests of ``config.metric`` between targets.

    Only targets measured in at least ``config.min_samples`` distinct samples
    enter the test, as in the legacy code.
    """
    if config.metric not in differences.columns:
        raise ValueError(f"metric {config.metric!r} is not a column of the differences table")
    d = differences
    if {"window_a", "window_b"} <= set(d.columns):
        d = d[(d.window_a == window_pair[0]) & (d.window_b == window_pair[1])]
    counts = d.groupby("target_mz")["sample_id"].nunique()
    valid = counts[counts >= config.min_samples].index
    data = d[d.target_mz.isin(valid)].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No target m/z has data in at least {config.min_samples} samples "
                         "for pairwise analysis.")

    groups = list(data["target_mz"].unique())        # order of appearance, as before
    values = {g: data.loc[data.target_mz == g, config.metric].to_numpy(dtype=float)
              for g in groups}
    rows = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            stat, p = ttest_ind(values[a], values[b], nan_policy="omit",
                                equal_var=config.equal_var)
            rows.append({"group_a": a, "group_b": b, "n_a": len(values[a]),
                         "n_b": len(values[b]), "statistic": float(stat), "raw_p_value": float(p)})
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS[:6])
    adjusted, reject = _correct(pairs, data, config)
    pairs["adjusted_p_value"] = adjusted
    pairs["reject"] = reject
    return StatsResult(config.metric, config.correction, config.alpha, data, pairs)


def _correct(pairs: pd.DataFrame, data: pd.DataFrame, config: StatsConfig):
    raw = pairs["raw_p_value"].to_numpy(dtype=float)
    adjusted = np.full(len(raw), np.nan)
    reject = np.zeros(len(raw), dtype=bool)
    if len(raw) == 0:
        return adjusted, reject

    if config.correction == "tukey":
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        clean = data[["target_mz", config.metric]].dropna()
        result = pairwise_tukeyhsd(clean[config.metric], clean["target_mz"], alpha=config.alpha)
        names = result.groupsunique
        lookup = {}
        for (i, j), p, rej in zip(zip(*result._multicomp.pairindices), result.pvalues,
                                  result.reject):
            lookup[frozenset((names[i], names[j]))] = (float(p), bool(rej))
        for k, (a, b) in enumerate(zip(pairs.group_a, pairs.group_b)):
            adjusted[k], reject[k] = lookup[frozenset((a, b))]
        return adjusted, reject

    ok = ~np.isnan(raw)       # zero-variance groups give NaN; leave them unadjusted
    if config.correction == "none":
        adjusted[ok] = raw[ok]
        reject[ok] = raw[ok] < config.alpha
    elif ok.any():
        from statsmodels.stats.multitest import multipletests

        rej, adj, _, _ = multipletests(raw[ok], alpha=config.alpha, method=config.correction)
        adjusted[ok], reject[ok] = adj, rej
    return adjusted, reject


def significance_stars(p: float) -> str:
    """The legacy plot's thresholds: **** < 1e-4 < *** < 1e-3 < ** < 1e-2 < * < 0.05."""
    if p is None or np.isnan(p):
        return ""
    for threshold, stars in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if p <= threshold:
            return stars
    return "ns"
