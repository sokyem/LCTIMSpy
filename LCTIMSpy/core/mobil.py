"""Mobilograms: intensity per mobility bin inside an RT window, read from the cache.

Logic follows the legacy ``extract_mobilogram`` -- filter by m/z, keep points
inside each RT window (inclusive), sum per mobility bin, optionally Gaussian-
smooth, then subtract the minimum -- with two differences: the caller's data is
never mutated, and the maximum before baseline subtraction is kept so the
intensity threshold can be applied to it (see ``QueryConfig.threshold_on``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import QueryConfig, RtWindow
from .results import MobilogramSeries
from .smoothing import gaussian
from .transforms import baseline_correct


def mobilograms_from_points(
    points: pd.DataFrame,
    target: float,
    windows: Sequence[RtWindow],
    config: QueryConfig = QueryConfig(),
) -> list[MobilogramSeries]:
    """One series per window with signal; windows without any points are omitted."""
    h = config.tolerance.half_width(target)
    hit = points[np.abs(points["mz"].to_numpy() - target) <= h]
    rt = hit["retention_time"].to_numpy()
    out = []
    for i, w in enumerate(windows):
        in_window = hit[(rt >= w.start) & (rt <= w.end)]
        if in_window.empty:
            continue
        grouped = in_window.groupby("mobility", sort=True)["intensity"].sum()
        intensity = grouped.to_numpy(dtype=float)
        if config.mobilogram_sigma is not None:
            intensity = gaussian(intensity, config.mobilogram_sigma)
        raw_max = float(intensity.max())
        if config.baseline_correction:
            intensity = baseline_correct(intensity)
        out.append(MobilogramSeries(
            target_mz=float(target),
            window=i,
            rt_start=w.start,
            rt_end=w.end,
            mobility=grouped.index.to_numpy(dtype=float),
            intensity=intensity,
            max_intensity_raw=raw_max,
        ))
    return out


def load_points(cache, entry, targets: Sequence[float], config: QueryConfig) -> pd.DataFrame:
    """All cached points near any target, with retention time, in one read."""
    from .chrom import target_windows

    return cache.load_mz_window(entry, target_windows(targets, config.tolerance),
                                columns=("mz", "intensity", "mobility", "frame"))


def mobilograms(cache, entry, targets: Sequence[float], windows: Sequence[RtWindow],
                config: QueryConfig = QueryConfig()) -> list[MobilogramSeries]:
    points = load_points(cache, entry, targets, config)
    return [s for t in targets for s in mobilograms_from_points(points, t, windows, config)]
