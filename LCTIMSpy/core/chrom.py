"""Extracted-ion chromatograms and RT peak detection, read from the cache.

``chromatograms_from_points`` reproduces the legacy ``extract_chromatograms``
(points with ``|mz - target| <= tol``, summed per retention time). The cache
version adds what the legacy one could not do cheaply: one pushdown read for all
targets, and zero-filling frames with no signal so every series shares the MS1
RT axis (gaps otherwise draw as straight lines and hide peak boundaries).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from .config import MzTolerance
from .results import ChromatogramSeries
from .smoothing import savgol

# Widen pushdown reads by a hair so float rounding at the window edge can never
# drop a point that the exact |mz - target| <= tol filter would keep.
_PUSHDOWN_EPS = 1e-9


def target_windows(targets: Sequence[float], tol: MzTolerance) -> list[tuple[float, float]]:
    return [(t - tol.half_width(t) - _PUSHDOWN_EPS, t + tol.half_width(t) + _PUSHDOWN_EPS)
            for t in targets]


def chromatograms_from_points(
    points: pd.DataFrame,
    targets: Sequence[float],
    tol: MzTolerance,
    rt_axis: np.ndarray | None = None,
    labels: Sequence[str | None] | None = None,
) -> list[ChromatogramSeries]:
    """EICs from a frame of ``mz``, ``intensity``, ``retention_time`` points.

    With ``rt_axis``, each series is reindexed onto it with zeros; without it,
    only retention times that have signal appear (the legacy behaviour), and a
    target with no signal at all is omitted.
    """
    labels = list(labels) if labels is not None else [None] * len(targets)
    mz = points["mz"].to_numpy()
    out = []
    for target, label in zip(targets, labels):
        hit = points[np.abs(mz - target) <= tol.half_width(target)]
        summed = hit.groupby("retention_time", sort=True)["intensity"].sum()
        if rt_axis is not None:
            summed = summed.reindex(rt_axis, fill_value=0.0)
        elif summed.empty:
            continue
        out.append(ChromatogramSeries(
            target_mz=float(target),
            retention_time=summed.index.to_numpy(dtype=float),
            intensity=summed.to_numpy(dtype=float),
            label=label,
        ))
    return out


def chromatograms(cache, entry, targets: Sequence[float], tol: MzTolerance = MzTolerance(),
                  labels: Sequence[str | None] | None = None,
                  fill_missing: bool = True) -> list[ChromatogramSeries]:
    """EICs for ``targets`` from a cache entry, in one read."""
    points = cache.load_mz_window(entry, target_windows(targets, tol),
                                  columns=("mz", "intensity", "frame"))
    rt_axis = None
    if fill_missing:
        rt_axis = np.unique(cache.frame_index(entry)["retention_time"].to_numpy())
    return chromatograms_from_points(points, targets, tol, rt_axis, labels)


def summed_chromatogram(series: Sequence[ChromatogramSeries]) -> ChromatogramSeries:
    """Sum series that share an RT axis (as zero-filled cache series do)."""
    if not series:
        raise ValueError("no chromatograms to sum")
    rt = series[0].retention_time
    if any(len(s.retention_time) != len(rt) or not np.array_equal(s.retention_time, rt)
           for s in series):
        raise ValueError("chromatograms must share an RT axis; use fill_missing=True")
    return ChromatogramSeries(float("nan"), rt.copy(), np.sum([s.intensity for s in series], axis=0),
                              label="sum")


@dataclass(frozen=True)
class DetectedPeak:
    apex_rt: float
    start_rt: float       # left edge at half height
    end_rt: float         # right edge at half height
    height: float
    prominence: float


def find_rt_peaks(series: ChromatogramSeries, n: int = 2, min_prominence: float = 0.05,
                  smooth_points: int = 5) -> list[DetectedPeak]:
    """The ``n`` most prominent peaks with FWHM bounds, returned in RT order.

    ``min_prominence`` is a fraction of the series maximum. The isomer workflow
    is definitionally two peaks, so ``n=2`` seeds the RT windows.
    """
    rt, y = series.retention_time, series.intensity
    if len(y) < 3 or not np.any(y > 0):
        return []
    ys = np.clip(savgol(y, smooth_points), 0, None)
    idx, props = find_peaks(ys, prominence=min_prominence * ys.max())
    if len(idx) == 0:
        return []
    top = np.argsort(props["prominences"])[::-1][:n]
    idx, prominences = idx[top], props["prominences"][top]
    _, _, left, right = peak_widths(ys, idx, rel_height=0.5)
    positions = np.arange(len(rt))
    peaks = [
        DetectedPeak(
            apex_rt=float(rt[i]),
            start_rt=float(np.interp(lo, positions, rt)),
            end_rt=float(np.interp(hi, positions, rt)),
            height=float(ys[i]),
            prominence=float(p),
        )
        for i, lo, hi, p in zip(idx, left, right, prominences)
    ]
    return sorted(peaks, key=lambda pk: pk.apex_rt)
