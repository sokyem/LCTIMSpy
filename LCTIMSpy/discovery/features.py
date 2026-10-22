"""Untargeted MS1 feature detection over m/z, retention time and ion mobility.

Two stages, both configurable (``FeatureFinderConfig``), none instrument-specific:

1. **Traces** -- coherent signal of one ion: points are assigned to m/z
   *lanes* (maxima of the intensity-weighted m/z density on a ppm axis); each
   lane is split at RT gaps and rasterised into a small RT x mobility image;
   watershed segmentation of the smoothed image separates neighbouring peaks
   -- so two species at the same m/z but different RT or mobility become two
   traces. Each trace records apex, boundaries, area, height, S/N and its m/z
   spread.
2. **Isotope envelopes** -- traces at Δm/z = 1.00336/z with matching RT and
   mobility apexes are linked into one feature per ion and charge, scored
   against an averagine pattern. The monoisotopic peak is corrected when a
   preceding isotope is present. Traces that join no envelope are kept as
   features with unknown charge (no neutral mass is calculated for them).

Output columns follow spec §3; quantities are labelled measured vs calculated
in ``FEATURE_COLUMNS_DOC``. Coordinates stay in the cache's mobility
coordinate (1/K0 for timsTOF); CCS is not derived here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from ..core.progress import CancelToken, ProgressReporter, StageTimer, null_reporter
from .synthetic import NEUTRON, PROTON, averagine_pattern


@dataclass(frozen=True)
class FeatureFinderConfig:
    slab_width: float = 10.0            # Da per processing slab (bounds memory)
    lane_ppm: float = 8.0               # half-width of an m/z lane
    density_bin_ppm: float = 1.0
    density_sigma_ppm: float = 2.0      # ~ the instrument's m/z jitter
    max_rt_gap_frames: int = 4          # split a lane where no signal for this many frames
    mobility_bin: float = 0.002         # image resolution in the mobility coordinate
    smooth_sigma: tuple[float, float] = (1.0, 1.0)   # (frames, mobility bins)
    region_fraction: float = 0.10       # trace extent: pixels above this fraction of apex
    min_snr: float = 5.0
    min_frames: int = 3
    min_points: int = 5
    max_charge: int = 4
    max_isotopes: int = 5
    isotope_ppm: float = 10.0
    isotope_rt_frames: float = 2.0      # max apex RT difference within an envelope
    isotope_mobility_rel: float = 0.01  # max relative apex mobility difference
    min_isotope_cosine: float = 0.6

    def to_dict(self) -> dict:
        return asdict(self)


TRACE_COLUMNS = [
    "trace_id", "mz", "mz_ppm_sd", "rt_apex", "rt_start", "rt_end", "frames",
    "mobility_apex", "mobility_low", "mobility_high", "height", "area", "n_points", "snr",
]

FEATURE_COLUMNS_DOC = {
    "feature_id": "identifier",
    "mz": "measured: intensity-weighted m/z of the monoisotopic trace",
    "charge": "calculated from isotope spacing; <NA> when no envelope was found",
    "neutral_mass": "calculated: (mz - proton) * charge; NaN without a charge",
    "rt_apex / rt_start / rt_end": "measured, minutes",
    "mobility_apex / mobility_low / mobility_high": "measured, in the cache's mobility coordinate",
    "area": "measured: summed intensity of all isotope traces (integrated)",
    "height": "measured: highest binned intensity of any isotope trace",
    "isotope_intensities": "measured: area of each isotope, mono first",
    "isotope_cosine": "calculated: similarity of the pattern to averagine",
    "mz_ppm_sd": "measured: m/z spread of the monoisotopic trace (mass precision)",
    "isotope_spacing_error_ppm": "calculated: mean deviation of isotope m/z from mono + k*1.00336/z, ppm",
    "snr": "measured: apex over the local noise level",
}


# ---- stage 1: traces ----------------------------------------------------------------


def _lanes(mz: np.ndarray, intensity: np.ndarray, cfg: FeatureFinderConfig) -> np.ndarray:
    """Lane index per point (-1 = no lane), from maxima of the m/z density."""
    u = np.log(mz) * 1e6                                   # 1 unit = 1 ppm
    lo, hi = u.min() - 3 * cfg.lane_ppm, u.max() + 3 * cfg.lane_ppm
    nbins = int(math.ceil((hi - lo) / cfg.density_bin_ppm)) + 1
    idx = ((u - lo) / cfg.density_bin_ppm).astype(np.int64)
    density = np.bincount(idx, weights=intensity, minlength=nbins).astype(float)
    smooth = ndimage.gaussian_filter1d(density, cfg.density_sigma_ppm / cfg.density_bin_ppm)
    window = int(round(cfg.lane_ppm / cfg.density_bin_ppm))
    peaks = np.flatnonzero((smooth == ndimage.maximum_filter1d(smooth, 2 * window + 1))
                           & (smooth > 0))
    if len(peaks) == 0:
        return np.full(len(mz), -1)
    centres = lo + peaks * cfg.density_bin_ppm
    pos = np.searchsorted(centres, u)
    left = np.clip(pos - 1, 0, len(centres) - 1)
    right = np.clip(pos, 0, len(centres) - 1)
    nearest = np.where(np.abs(u - centres[left]) <= np.abs(u - centres[right]), left, right)
    return np.where(np.abs(u - centres[nearest]) <= cfg.lane_ppm, nearest, -1)


def _ascent_labels(smooth: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label every pixel with the local maximum it reaches by steepest ascent.

    Each pixel points to its brightest 3x3 neighbour (itself if none is
    brighter); pointer jumping follows the chain to a maximum. Pixels whose
    maximum is not in ``keep`` (noise) get label 0. Exact for separated peaks,
    and -- unlike a priority-flood watershed -- no tie-breaking order can hand
    one peak's flank to its neighbour. Returns (labels, apex value per label).
    """
    h, w = smooth.shape
    idx = np.arange(h * w).reshape(h, w)
    pad = np.pad(smooth, 1, constant_values=-np.inf)
    pad_idx = np.pad(idx, 1, constant_values=-1)
    best, best_idx = smooth.copy(), idx.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            v = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            j = pad_idx[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            better = v > best
            best = np.where(better, v, best)
            best_idx = np.where(better, j, best_idx)
    ptr = best_idx.ravel()
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    maxima = np.flatnonzero((ptr == np.arange(h * w)) & keep.ravel())
    label_of = np.zeros(h * w, dtype=np.int64)
    label_of[maxima] = np.arange(1, len(maxima) + 1)
    return label_of[ptr].reshape(h, w), smooth.ravel()[maxima]


def _segment_traces(fpos, mob, mz, inten, rt_of_pos, noise, cfg, has_mobility):
    """Watershed-separated peaks in one lane segment's RT x mobility image."""
    f0 = fpos.min()
    fi = fpos - f0
    if has_mobility:
        m0 = np.nanmin(mob)
        mi = np.floor((mob - m0) / cfg.mobility_bin).astype(np.int64)
    else:
        m0, mi = 0.0, np.zeros(len(mob), dtype=np.int64)
    shape = (int(fi.max()) + 1, int(mi.max()) + 1)
    raw = np.zeros(shape)
    np.add.at(raw, (fi, mi), inten)
    sigma = cfg.smooth_sigma if has_mobility else (cfg.smooth_sigma[0], 0.0)
    smooth = ndimage.gaussian_filter(raw, sigma=sigma, mode="constant")
    # Kept maxima must clear the S/N threshold in the raw apex neighbourhood.
    local = ndimage.maximum_filter(raw, size=3, mode="constant")
    labels, peak_vals = _ascent_labels(smooth, keep=(smooth > 0) & (local >= cfg.min_snr * noise))
    n = len(peak_vals)
    if n == 0:
        return []
    # Each trace extends over its basin, down to a fraction of its own apex.
    apex_of_label = np.concatenate(([np.inf], peak_vals))
    labels = np.where(smooth >= cfg.region_fraction * apex_of_label[labels], labels, 0)

    point_label = labels[fi, mi]
    out = []
    for lab in range(1, n + 1):
        sel = point_label == lab
        if sel.sum() < cfg.min_points:
            continue
        frames_here = np.unique(fi[sel])
        if len(frames_here) < cfg.min_frames:
            continue
        w = inten[sel]
        region = labels == lab
        apex_pix = np.unravel_index(np.argmax(np.where(region, smooth, -1)), shape)
        mzw = float(np.average(mz[sel], weights=w))
        ppm_sd = float(np.sqrt(np.average((mz[sel] - mzw) ** 2, weights=w)) / mzw * 1e6)
        rows_f = np.flatnonzero(region.any(axis=1))
        if has_mobility:
            m_sel = mob[sel]
            mob_apex = float(np.average(m_sel, weights=w)) if len(m_sel) else np.nan
            mob_low, mob_high = float(m_sel.min()), float(m_sel.max())
            # Apex mobility from the image apex column (sharper than the mean).
            col = apex_pix[1]
            in_col = sel & (mi == col)
            if in_col.any():
                mob_apex = float(np.average(mob[in_col], weights=inten[in_col]))
        else:
            mob_apex = mob_low = mob_high = np.nan
        height = float(raw[region].max())
        out.append({
            "mz": mzw, "mz_ppm_sd": ppm_sd,
            "rt_apex": float(rt_of_pos[f0 + apex_pix[0]]),
            "rt_start": float(rt_of_pos[f0 + rows_f.min()]),
            "rt_end": float(rt_of_pos[f0 + rows_f.max()]),
            "frames": int(len(frames_here)),
            "mobility_apex": mob_apex, "mobility_low": mob_low, "mobility_high": mob_high,
            "height": height, "area": float(w.sum()), "n_points": int(sel.sum()),
            "snr": float(height / noise) if noise > 0 else np.inf,
        })
    return out


def find_traces(points: pd.DataFrame, rt_of_pos: np.ndarray, frame_pos: np.ndarray,
                cfg: FeatureFinderConfig, noise: float | None = None) -> pd.DataFrame:
    """Traces in a frame of points (mz, intensity, mobility, frame-position)."""
    if points.empty:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    mz = points["mz"].to_numpy(float)
    inten = points["intensity"].to_numpy(float)
    mob = points["mobility"].to_numpy(float)
    has_mobility = bool(np.isfinite(mob).any())
    if noise is None:
        noise = float(np.median(inten))
    lanes = _lanes(mz, inten, cfg)
    order = np.lexsort((frame_pos, lanes))
    lanes_s, fpos_s = lanes[order], frame_pos[order]
    rows = []
    starts = np.flatnonzero(np.diff(lanes_s, prepend=-2) != 0)
    ends = np.append(starts[1:], len(lanes_s))
    for a, b in zip(starts, ends):
        if lanes_s[a] < 0 or b - a < cfg.min_points:
            continue
        idx = order[a:b]
        f = fpos_s[a:b]
        cuts = np.flatnonzero(np.diff(f) > cfg.max_rt_gap_frames) + 1
        for seg in np.split(np.arange(len(idx)), cuts):
            if len(seg) < cfg.min_points:
                continue
            sel = idx[seg]
            m = mob[sel] if has_mobility else np.full(len(sel), np.nan)
            if has_mobility and not np.isfinite(m).all():
                ok = np.isfinite(m)
                sel, m = sel[ok], m[ok]
                if len(sel) < cfg.min_points:
                    continue
            rows += _segment_traces(frame_pos[sel], m, mz[sel], inten[sel], rt_of_pos, noise,
                                    cfg, has_mobility)
    df = pd.DataFrame(rows, columns=TRACE_COLUMNS[1:])
    df.insert(0, "trace_id", np.arange(len(df)))
    return df


# ---- stage 2: isotope envelopes -------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def assemble_features(traces: pd.DataFrame, cfg: FeatureFinderConfig,
                      frame_minutes: float) -> pd.DataFrame:
    """Link traces into isotope envelopes; one feature per ion and charge."""
    if traces.empty:
        return pd.DataFrame()
    t = traces.sort_values("mz").reset_index(drop=True)
    mz = t["mz"].to_numpy(float)
    rt = t["rt_apex"].to_numpy(float)
    mob = t["mobility_apex"].to_numpy(float)
    area = t["area"].to_numpy(float)
    rt_tol = cfg.isotope_rt_frames * frame_minutes
    used = np.zeros(len(t), dtype=bool)

    def partner(i: int, target_mz: float) -> int | None:
        tol = target_mz * cfg.isotope_ppm * 1e-6
        lo, hi = np.searchsorted(mz, [target_mz - tol, target_mz + tol])
        best, best_d = None, np.inf
        for j in range(lo, hi):
            if j == i or abs(rt[j] - rt[i]) > rt_tol:
                continue
            if np.isfinite(mob[i]) and np.isfinite(mob[j]) and \
                    abs(mob[j] - mob[i]) > cfg.isotope_mobility_rel * mob[i]:
                continue
            d = abs(mz[j] - target_mz)
            if d < best_d:
                best, best_d = j, d
        return best

    def envelope(mono: int, z: int) -> list[int]:
        chain, cur = [mono], mono
        for k in range(1, cfg.max_isotopes):
            nxt = partner(cur, mz[mono] + k * NEUTRON / z)
            if nxt is None or used[nxt]:
                break
            chain.append(nxt)
            cur = nxt
        return chain

    features = []
    for i in np.argsort(-area):                          # most intense first
        if used[i]:
            continue
        best = None
        for z in range(1, cfg.max_charge + 1):
            mono = i
            # Walk back: a preceding isotope means i is not the monoisotopic peak.
            for _ in range(cfg.max_isotopes - 1):
                prev = partner(mono, mz[mono] - NEUTRON / z)
                if prev is None or used[prev]:
                    break
                mono = prev
            chain = envelope(mono, z)
            if len(chain) < 2:
                continue
            mass = (mz[mono] - PROTON) * z
            observed = area[chain]
            expected = averagine_pattern(mass, len(chain))
            cos = _cosine(observed, expected)
            key = (len(chain), cos)
            if cos >= cfg.min_isotope_cosine and (best is None or key > best[0]):
                best = (key, z, mono, chain, cos)
        if best is None:
            continue
        _, z, mono, chain, cos = best
        used[chain] = True
        rows = t.iloc[chain]
        # Mass accuracy of the envelope: each isotope's m/z against where the
        # monoisotopic m/z and the charge place it, in ppm of m/z.
        iso_mz = rows["mz"].to_numpy(float)
        expected = mz[mono] + np.arange(len(chain)) * NEUTRON / z
        spacing_err = float(np.mean((iso_mz[1:] - expected[1:]) / expected[1:]) * 1e6)
        w = rows["area"].to_numpy()
        mobs = rows["mobility_apex"].to_numpy()
        features.append({
            "mz": float(mz[mono]), "charge": z, "neutral_mass": float((mz[mono] - PROTON) * z),
            "rt_apex": float(rows.loc[rows["area"].idxmax(), "rt_apex"]),
            "rt_start": float(rows["rt_start"].min()), "rt_end": float(rows["rt_end"].max()),
            "mobility_apex": (float(np.average(mobs, weights=w))
                              if np.isfinite(mobs).all() else np.nan),
            "mobility_low": float(rows["mobility_low"].min()),
            "mobility_high": float(rows["mobility_high"].max()),
            "area": float(w.sum()), "height": float(rows["height"].max()),
            "n_isotopes": len(chain), "isotope_intensities": [float(x) for x in w],
            "isotope_cosine": cos, "isotope_spacing_error_ppm": spacing_err,
            "mz_ppm_sd": float(t.iloc[mono]["mz_ppm_sd"]), "snr": float(rows["snr"].max()),
            "trace_ids": [int(x) for x in rows["trace_id"]],
        })
    # Traces in no envelope: kept as features of unknown charge, never dropped.
    for i in np.flatnonzero(~used):
        r = t.iloc[i]
        features.append({
            "mz": float(r["mz"]), "charge": pd.NA, "neutral_mass": np.nan,
            "rt_apex": float(r["rt_apex"]), "rt_start": float(r["rt_start"]),
            "rt_end": float(r["rt_end"]), "mobility_apex": float(r["mobility_apex"]),
            "mobility_low": float(r["mobility_low"]), "mobility_high": float(r["mobility_high"]),
            "area": float(r["area"]), "height": float(r["height"]), "n_isotopes": 1,
            "isotope_intensities": [float(r["area"])], "isotope_cosine": np.nan,
            "isotope_spacing_error_ppm": np.nan, "mz_ppm_sd": float(r["mz_ppm_sd"]),
            "snr": float(r["snr"]), "trace_ids": [int(r["trace_id"])],
        })
    out = pd.DataFrame(features).sort_values("area", ascending=False).reset_index(drop=True)
    out.insert(0, "feature_id", np.arange(len(out)))
    out["charge"] = out["charge"].astype("Int64")
    return out


# ---- driver ---------------------------------------------------------------------------


def find_features(cache, entry, cfg: FeatureFinderConfig = FeatureFinderConfig(),
                  mz_range: tuple[float, float] | None = None,
                  reporter: ProgressReporter = null_reporter,
                  cancel: CancelToken | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(features, traces)`` for a cache entry, processed in m/z slabs."""
    cancel = cancel or CancelToken()
    frames = cache.frame_index(entry)
    frame_ids = frames["frame"].to_numpy(np.int64)
    rt_of_pos = frames["retention_time"].to_numpy(float)
    frame_minutes = float(np.median(np.diff(rt_of_pos))) if len(rt_of_pos) > 1 else 0.01
    if mz_range is None:
        mz_range = (entry.stats.get("mz_min") or 100.0, (entry.stats.get("mz_max") or 1700.0)
                    + 1e-6)
    lo, hi = mz_range
    edges = np.arange(lo, hi + cfg.slab_width, cfg.slab_width)
    timer = StageTimer("features", len(edges) - 1, reporter)
    traces = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        cancel.raise_if_cancelled()
        margin = 3 * cfg.lane_ppm * 1e-6 * b
        pts = cache.load_mz_window(entry, [(a - margin, b + margin)],
                                   columns=("mz", "intensity", "mobility", "frame"),
                                   with_retention_time=False)
        if len(pts):
            pos = np.searchsorted(frame_ids, pts["frame"].to_numpy(np.int64))
            tr = find_traces(pts, rt_of_pos, pos, cfg)
            traces.append(tr[(tr["mz"] >= a) & (tr["mz"] < b)])
        timer.update(k + 1)
    traces = [t for t in traces if len(t)]
    all_traces = (pd.concat(traces, ignore_index=True) if traces
                  else pd.DataFrame(columns=TRACE_COLUMNS))
    numeric = [c for c in TRACE_COLUMNS if c != "trace_id"]
    all_traces[numeric] = all_traces[numeric].astype(float)
    all_traces["trace_id"] = np.arange(len(all_traces))
    features = assemble_features(all_traces, cfg, frame_minutes)
    features.attrs.update({"config": cfg.to_dict(), "columns": FEATURE_COLUMNS_DOC,
                           "mobility": dict(entry.mobility)})
    return features, all_traces
