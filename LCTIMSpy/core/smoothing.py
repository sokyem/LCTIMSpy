"""Smoothing for chromatograms and mobilograms.

The legacy ``_apply_smoothing`` documented ``'gaussian'`` but only handled
``'savgol'`` and silently returned raw data for anything else. Here every
accepted method is implemented and anything else raises.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

SmoothMethod = Literal["none", "savgol", "gaussian"]


def savgol(values: np.ndarray, points: int = 5) -> np.ndarray:
    """Savitzky-Golay (polyorder 2), window logic unchanged from the legacy code.

    The window is forced odd and >= 3, shrunk to fit short series, and series
    with fewer than 3 points are returned unchanged.
    """
    values = np.asarray(values, dtype=float)
    win = max(int(points), 3)
    if win % 2 == 0:
        win -= 1
    n = len(values)
    if n < win:
        win = n if n % 2 == 1 else n - 1
        if win < 3:
            return values.copy()
    return savgol_filter(values, window_length=win, polyorder=2)


def gaussian(values: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    return gaussian_filter1d(values, sigma=sigma)


def smooth(values: np.ndarray, method: SmoothMethod = "none", points: int = 5,
           sigma: float = 1.0) -> np.ndarray:
    """Apply ``method``; ``points`` is the savgol window, ``sigma`` the gaussian width."""
    if method in (None, "none"):
        return np.asarray(values, dtype=float).copy()
    if method == "savgol":
        return savgol(values, points)
    if method == "gaussian":
        return gaussian(values, sigma)
    raise ValueError(f"unknown smoothing method {method!r}; use 'none', 'savgol' or 'gaussian'")
