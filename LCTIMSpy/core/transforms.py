"""Small array transforms used by queries and figures."""

from __future__ import annotations

import numpy as np


def normalize(values: np.ndarray, out_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Min-max scale into ``out_range``; NaN counts as 0, a flat series maps to the low end.

    Same semantics as the legacy ``normalize_series``.
    """
    x = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    lo, hi = out_range
    if x.size == 0:
        return x
    span = x.max() - x.min()
    if span == 0:
        return np.full(x.shape, lo, dtype=float)
    return (x - x.min()) / span * (hi - lo) + lo


def baseline_correct(values: np.ndarray) -> np.ndarray:
    """Subtract the minimum, as Bruker DataAnalysis does for mobilograms."""
    x = np.asarray(values, dtype=float)
    return x - x.min() if x.size else x.copy()


def lttb_downsample(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets: keep ``n_out`` points that preserve the visual shape.

    Used to send long chromatograms to the browser without losing peaks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n_out >= n or n_out < 3:
        return x.copy(), y.copy()
    keep = np.empty(n_out, dtype=np.int64)
    keep[0], keep[-1] = 0, n - 1
    edges = np.linspace(1, n - 1, n_out - 1).astype(np.int64)
    a = 0
    for i in range(n_out - 2):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        nxt_lo, nxt_hi = edges[i + 1], max(edges[i + 2] if i + 2 < len(edges) else n, edges[i + 1] + 1)
        avg_x, avg_y = x[nxt_lo:nxt_hi].mean(), y[nxt_lo:nxt_hi].mean()
        area = np.abs(
            (x[a] - avg_x) * (y[lo:hi] - y[a]) - (x[a] - x[lo:hi]) * (avg_y - y[a])
        )
        a = lo + int(np.argmax(area))
        keep[i + 1] = a
    return x[keep], y[keep]
