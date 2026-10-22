"""Everything the candidate viewer draws, read back from the raw cache.

Measured and calculated quantities are kept apart in the payload: raw traces
are measured; averagine isotope positions/intensities and fitted population
centres are calculated, and labelled so.
"""

from __future__ import annotations

import numpy as np

from ..core.transforms import lttb_downsample
from .synthetic import NEUTRON, PROTON, averagine_pattern
from .traces import TraceConfig, isotope_windows, species_chromatogram, species_mobilogram

RT_PAD = 1.0                  # minutes shown either side of the species
MAP_RT_BINS = 160
MAP_MOBILITY_BINS = 120


def _species(c: dict) -> list[tuple[str, dict]]:
    out = [("A", c["species_a"])]
    if c.get("species_b"):
        out.append(("B", c["species_b"]))
    return out


def candidate_detail(cache, entry, c: dict, cfg: TraceConfig = TraceConfig()) -> dict:
    mono, z = float(c["mono_mz"]), int(c["charge"])
    species = _species(c)
    pts = cache.load_mz_window(entry, isotope_windows(mono, z, cfg.n_isotopes, cfg.ppm),
                               columns=("mz", "intensity", "mobility", "frame"))
    rt_lo = min(s["rt_start"] for _, s in species) - RT_PAD
    rt_hi = max(s["rt_end"] for _, s in species) + RT_PAD

    rt, y = species_chromatogram(cache, entry, mono, z, cfg, points=pts)
    keep = (rt >= rt_lo) & (rt <= rt_hi)
    rt, y = lttb_downsample(rt[keep], y[keep], 1500)

    mobilograms = []
    for name, s in species:
        x, m = species_mobilogram(cache, entry, mono, z, s["rt_start"], s["rt_end"], cfg,
                                  points=pts)
        mobilograms.append({"species": name, "x": x.tolist(), "y": m.tolist(),
                            "centre": s.get("mobility"), "fwhm": s.get("mobility_fwhm")})

    heat = None
    sel = pts[(pts["retention_time"] >= rt_lo) & (pts["retention_time"] <= rt_hi)
              & np.isfinite(pts["mobility"])]
    if len(sel):
        mob = sel["mobility"].to_numpy(float)
        centres = [s["mobility"] for _, s in species if s.get("mobility") is not None]
        m_lo = min(min(centres) - 0.05, np.percentile(mob, 1)) if centres else mob.min()
        m_hi = max(max(centres) + 0.05, np.percentile(mob, 99)) if centres else mob.max()
        inside = (mob >= m_lo) & (mob <= m_hi)
        h, xe, ye = np.histogram2d(sel["retention_time"].to_numpy(float)[inside], mob[inside],
                                   bins=[MAP_RT_BINS, MAP_MOBILITY_BINS], range=[[rt_lo, rt_hi],
                                                                                 [m_lo, m_hi]],
                                   weights=sel["intensity"].to_numpy(float)[inside])
        heat = {"rt": ((xe[:-1] + xe[1:]) / 2).tolist(),
                "mobility": ((ye[:-1] + ye[1:]) / 2).tolist(),
                "z": h.T.tolist()}

    ms1 = []
    for name, s in species:
        lo, hi = mono - 0.5, mono + 4 * NEUTRON / z + 0.5
        win = cache.load_mz_window(entry, [(lo, hi)], columns=("mz", "intensity", "mobility",
                                                                "frame"))
        half_rt = max((s["rt_end"] - s["rt_start"]) / 4, 0.01)
        ok = np.abs(win["retention_time"] - s["rt_apex"]) <= half_rt
        if s.get("mobility") is not None and np.isfinite(win["mobility"]).any():
            half_m = max((s.get("mobility_fwhm") or 0) / 2, 0.005)
            ok &= np.abs(win["mobility"] - s["mobility"]) <= half_m
        w = win[ok]
        bins = np.round(w["mz"].to_numpy() / 0.005) * 0.005
        prof = w.assign(_b=bins).groupby("_b")["intensity"].sum()
        mass = (mono - PROTON) * z
        pattern = averagine_pattern(mass, 5)
        ms1.append({"species": name, "mz": prof.index.tolist(), "intensity": prof.tolist(),
                    "averagine_mz": [mono + k * NEUTRON / z for k in range(5)],
                    "averagine_relative": pattern.tolist()})

    return {
        "candidate_id": c["candidate_id"], "mono_mz": mono, "charge": z,
        "coordinate": entry.mobility.get("coordinate"),
        "chromatogram": {"rt": rt.tolist(), "intensity": y.tolist(),
                         "isotopes_summed": cfg.n_isotopes, "ppm": cfg.ppm},
        "species": [{"name": n, **s} for n, s in species],
        "mobilograms": mobilograms, "map": heat, "ms1": ms1,
        "mirror": (c.get("msms") or {}).get("mirror"),
        "labels": {
            "measured": ["chromatogram", "mobilograms", "map", "ms1 profile", "MS/MS spectra",
                         "RT apex/boundaries", "areas"],
            "calculated": ["mono m/z from the isotope envelope", "neutral mass",
                           "population centres and FWHM (fitted)", "averagine isotope pattern",
                           "relative mobility difference", "Rs", "spectral similarity"],
        },
    }
