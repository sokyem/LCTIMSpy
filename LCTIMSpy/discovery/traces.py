"""Rebuild a species' signal from the raw cache and analyse it in RT and mobility.

A *species* here is an accurate m/z and charge (from a feature or a PSM group).
Its signal is read straight from the extraction cache -- not from PSM sampling
-- summed over the first isotopes of its envelope for S/N:

* ``species_chromatogram``  intensity on the full MS1 RT axis (zero-filled)
* ``species_mobilogram``    intensity per mobility bin inside an RT window
* ``analyze_species``       every RT peak, pair resolutions, and the mobility
                            populations inside each RT peak

The mobility axis stays in the cache's coordinate (1/K0 for timsTOF).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .peaks import (
    MobilityAnalysis,
    MobilityConfig,
    RtPair,
    RtPeak,
    RtPeakConfig,
    detect_mobility_populations,
    detect_rt_peaks,
    rt_pairs,
)
from .synthetic import NEUTRON


@dataclass(frozen=True)
class TraceConfig:
    ppm: float = 10.0                 # m/z tolerance per isotope
    n_isotopes: int = 2               # isotopes summed (mono + n-1)
    mobility_bin: float | None = None # None: the cache's own bins (1/K0 is pre-binned)
    rt: RtPeakConfig = RtPeakConfig()
    mobility: MobilityConfig = MobilityConfig()

    def to_dict(self) -> dict:
        return asdict(self)


def isotope_windows(mono_mz: float, charge: int, n_isotopes: int, ppm: float
                    ) -> list[tuple[float, float]]:
    out = []
    for k in range(n_isotopes):
        mz = mono_mz + k * NEUTRON / max(int(charge), 1)
        tol = mz * ppm * 1e-6
        out.append((mz - tol, mz + tol))
    return out


def _points(cache, entry, mono_mz, charge, cfg: TraceConfig) -> pd.DataFrame:
    return cache.load_mz_window(entry, isotope_windows(mono_mz, charge, cfg.n_isotopes, cfg.ppm),
                                columns=("mz", "intensity", "mobility", "frame"))


def species_chromatogram(cache, entry, mono_mz: float, charge: int,
                         cfg: TraceConfig = TraceConfig(), points: pd.DataFrame | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    pts = points if points is not None else _points(cache, entry, mono_mz, charge, cfg)
    axis = np.unique(cache.frame_index(entry)["retention_time"].to_numpy(float))
    summed = pts.groupby("retention_time")["intensity"].sum()
    return axis, summed.reindex(axis, fill_value=0.0).to_numpy(float)


def species_mobilogram(cache, entry, mono_mz: float, charge: int, rt_start: float,
                       rt_end: float, cfg: TraceConfig = TraceConfig(),
                       points: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    pts = points if points is not None else _points(cache, entry, mono_mz, charge, cfg)
    sel = pts[(pts["retention_time"] >= rt_start) & (pts["retention_time"] <= rt_end)]
    sel = sel[np.isfinite(sel["mobility"])]
    if sel.empty:
        return np.array([]), np.array([])
    if cfg.mobility_bin:
        key = np.round(sel["mobility"].to_numpy() / cfg.mobility_bin) * cfg.mobility_bin
        grouped = sel.assign(_m=key).groupby("_m")["intensity"].sum()
    else:
        grouped = sel.groupby("mobility")["intensity"].sum()
    return grouped.index.to_numpy(float), grouped.to_numpy(float)


@dataclass
class SpeciesAnalysis:
    mono_mz: float
    charge: int
    rt_peaks: list[RtPeak]
    rt_pairs: list[RtPair]
    mobility: list[MobilityAnalysis | None]     # one per RT peak (None for shoulders)
    coordinate: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"mono_mz": self.mono_mz, "charge": self.charge, "coordinate": self.coordinate,
                "rt_peaks": [p.to_dict() for p in self.rt_peaks],
                "rt_pairs": [p.to_dict() for p in self.rt_pairs],
                "mobility": [m.to_dict() if m else None for m in self.mobility],
                "flags": self.flags}


def analyze_species(cache, entry, mono_mz: float, charge: int,
                    cfg: TraceConfig = TraceConfig(),
                    rt_range: tuple[float, float] | None = None) -> SpeciesAnalysis:
    """RT peaks of the species' trace, and mobility populations within each."""
    pts = _points(cache, entry, mono_mz, charge, cfg)
    rt, y = species_chromatogram(cache, entry, mono_mz, charge, cfg, points=pts)
    if rt_range is not None:
        keep = (rt >= rt_range[0]) & (rt <= rt_range[1])
        rt, y = rt[keep], y[keep]
    peaks = detect_rt_peaks(rt, y, cfg.rt)
    pairs = rt_pairs(peaks, rt, y, cfg.rt)
    coordinate = entry.mobility.get("coordinate") or "none"
    mobility: list[MobilityAnalysis | None] = []
    flags = []
    if not entry.mobility.get("present"):
        flags.append("no ion mobility in this dataset: RT-only analysis")
    for p in peaks:
        if p.kind != "peak" or not entry.mobility.get("present"):
            mobility.append(None)
            continue
        x, m = species_mobilogram(cache, entry, mono_mz, charge, p.start_rt, p.end_rt, cfg,
                                  points=pts)
        mobility.append(detect_mobility_populations(x, m, cfg.mobility, coordinate))
    return SpeciesAnalysis(float(mono_mz), int(charge), peaks, pairs, mobility, coordinate, flags)
