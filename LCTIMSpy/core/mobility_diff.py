"""Mobility difference between RT windows: the core isomer measurement.

The science is the legacy ``extract_and_calculate_mobility_difference``:
intensity-weighted mean mobility and peak (argmax) mobility per window, and the
absolute differences between windows. What changed:

* any number of windows (>= 2); differences are reported for every pair,
* windows are identified by index with explicit ``rt_start``/``rt_end`` rather
  than by an ``"start-end"`` string that depended on float formatting,
* dropped targets are returned with a reason instead of printed,
* optional per-sample RT offsets, so drifting replicates need no re-windowing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from .config import QueryConfig, RtWindow
from .mobil import load_points, mobilograms_from_points
from .results import MobilityDiffResult, MobilogramSeries

PER_WINDOW_COLUMNS = ["sample_id", "target_mz", "window", "rt_start", "rt_end", "avg_mobility",
                      "peak_mobility", "max_intensity", "max_intensity_raw", "total_intensity",
                      "n_bins", "included"]
DIFF_COLUMNS = ["sample_id", "target_mz", "window_a", "window_b", "avg_mobility_a",
                "avg_mobility_b", "mobility_difference", "peak_mobility_a", "peak_mobility_b",
                "peak_mobility_difference"]
EXCLUDED_COLUMNS = ["sample_id", "target_mz", "reason"]


def window_stats(series: MobilogramSeries) -> dict:
    total = float(series.intensity.sum())
    return {
        "avg_mobility": (float(np.average(series.mobility, weights=series.intensity))
                         if total != 0 else float("nan")),
        "peak_mobility": float(series.mobility[int(np.argmax(series.intensity))]),
        "max_intensity": float(series.intensity.max()),
        "max_intensity_raw": series.max_intensity_raw,
        "total_intensity": total,
        "n_bins": len(series.mobility),
    }


def _exclusion_reason(by_window: dict[int, MobilogramSeries], n_windows: int,
                      config: QueryConfig) -> str | None:
    """Same checks, in the same order, as the legacy function."""
    missing = [i for i in range(n_windows) if i not in by_window]
    if missing:
        return f"no data in window {', '.join(map(str, missing))}"
    if config.intensity_threshold is not None:
        maxima = [by_window[i].max_intensity_raw if config.threshold_on == "raw"
                  else float(by_window[i].intensity.max()) for i in range(n_windows)]
        if min(maxima) < config.intensity_threshold:
            shown = ", ".join(f"{m:g}" for m in maxima)
            return (f"max intensity below {config.intensity_threshold:g} "
                    f"({config.threshold_on}: {shown})")
    if any(by_window[i].intensity.sum() == 0 for i in range(n_windows)):
        return "zero total intensity in a window"
    return None


def mobility_difference_from_points(
    points: pd.DataFrame,
    sample_id,
    targets: Sequence[float],
    windows: Sequence[RtWindow],
    config: QueryConfig = QueryConfig(),
) -> tuple[list[dict], list[dict], list[dict]]:
    per_window, diffs, excluded = [], [], []
    for target in targets:
        series = mobilograms_from_points(points, target, windows, config)
        by_window = {s.window: s for s in series}
        reason = _exclusion_reason(by_window, len(windows), config)
        stats = {i: window_stats(s) for i, s in by_window.items()}
        for i, s in by_window.items():
            per_window.append({"sample_id": sample_id, "target_mz": float(target), "window": i,
                               "rt_start": s.rt_start, "rt_end": s.rt_end, **stats[i],
                               "included": reason is None})
        if reason is not None:
            excluded.append({"sample_id": sample_id, "target_mz": float(target), "reason": reason})
            continue
        for a, b in combinations(range(len(windows)), 2):
            sa, sb = stats[a], stats[b]
            diffs.append({
                "sample_id": sample_id, "target_mz": float(target), "window_a": a, "window_b": b,
                "avg_mobility_a": sa["avg_mobility"], "avg_mobility_b": sb["avg_mobility"],
                "mobility_difference": abs(sb["avg_mobility"] - sa["avg_mobility"]),
                "peak_mobility_a": sa["peak_mobility"], "peak_mobility_b": sb["peak_mobility"],
                "peak_mobility_difference": abs(sb["peak_mobility"] - sa["peak_mobility"]),
            })
    return per_window, diffs, excluded


def mobility_difference(
    cache,
    entries: Mapping[str, object],
    targets: Sequence[float],
    windows: Sequence[RtWindow],
    config: QueryConfig = QueryConfig(),
    rt_offsets: Mapping[str, float] | None = None,
) -> MobilityDiffResult:
    """Mobility differences for every sample in ``entries`` (sample_id -> CacheEntry)."""
    if len(windows) < 2:
        raise ValueError("at least two RT windows are needed to compute a mobility difference")
    rt_offsets = rt_offsets or {}
    per_window, diffs, excluded = [], [], []
    for sample_id, entry in entries.items():
        shift = rt_offsets.get(sample_id, 0.0)
        shifted = [RtWindow(w.start + shift, w.end + shift) for w in windows]
        points = load_points(cache, entry, targets, config)
        pw, d, ex = mobility_difference_from_points(points, sample_id, targets, shifted, config)
        per_window += pw
        diffs += d
        excluded += ex
    return MobilityDiffResult(
        per_window=pd.DataFrame(per_window, columns=PER_WINDOW_COLUMNS),
        differences=pd.DataFrame(diffs, columns=DIFF_COLUMNS),
        excluded=pd.DataFrame(excluded, columns=EXCLUDED_COLUMNS),
    )
