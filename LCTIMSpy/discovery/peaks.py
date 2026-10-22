"""Peak analysis in retention time and ion mobility (spec §6-7).

**Retention time.** ``detect_rt_peaks`` finds every peak in a chromatogram with
its apex, boundaries, FWHM, integrated area, height, S/N and asymmetry, plus
*shoulders* (a peak on a flank with no local maximum, found from curvature).
``rt_pairs`` describes each neighbouring pair by resolution
Rs = 1.18 (t2 - t1) / (w1/2 + w2/2) -- widths from each peak's outer side, since
fusion inflates the shared side -- and valley depth, and classifies it as
baseline-resolved, partially overlapping, shoulder or coeluting. Apex
separation alone is never the criterion.

**Ion mobility.** ``detect_mobility_populations`` decides whether a mobilogram
holds one population or two by comparing three models -- one Gaussian, two
Gaussians, one *skewed* peak -- on BIC. Two populations are reported only if
the two-Gaussian model wins clearly over *both* single-peak models (so an
asymmetric peak is not split), the components are resolved, the minor one is
not negligible, and their separation exceeds a multiple of the instrument's
mobility precision. That precision is a parameter to be established from
replicate measurements on the instrument; the default is a placeholder, and
results say so.

Relative mobility difference = 100 * |IM2 - IM1| / reference, with the
coordinate and the reference stated alongside every value.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import skewnorm

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))      # 2.3548


# ---- retention time ------------------------------------------------------------------


@dataclass(frozen=True)
class RtPeakConfig:
    smooth_points: int = 7                # Savitzky-Golay window (points); odd
    min_snr: float = 5.0
    min_rel_prominence: float = 0.03      # of the chromatogram maximum
    boundary_fraction: float = 0.05       # outer boundary: signal falls to this x apex
    min_fwhm: float = 0.005               # minutes
    max_fwhm: float = 2.0
    baseline_resolution: float = 1.5      # Rs at or above: baseline-resolved
    partial_resolution: float = 0.8       # Rs at or above: partially overlapping
    shoulder_min_rel_height: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RtPeak:
    apex_rt: float
    start_rt: float
    end_rt: float
    fwhm: float
    height: float                          # measured (raw intensity at the apex)
    area: float                            # measured: integrated above the baseline
    snr: float
    asymmetry: float                       # tailing factor at 10% height (b/a); 1 = symmetric
    kind: str = "peak"                     # "peak" | "shoulder"
    flags: list[str] = field(default_factory=list)
    half_width_left: float = float("nan")   # apex to half height, leading side (min)
    half_width_right: float = float("nan")  # apex to half height, trailing side (min)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RtPair:
    first: int
    second: int
    delta_rt: float
    resolution: float                      # approximate when the peaks are fused (Rs < ~1)
    valley_ratio: float                    # valley height / smaller apex height
    separation: str                        # baseline-resolved | partially overlapping | shoulder | coeluting

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_noise(y: np.ndarray) -> float:
    """Robust noise level from point-to-point differences (insensitive to peaks)."""
    d = np.diff(y)
    d = d[d != 0]
    if len(d) == 0:
        return max(1e-12, float(np.max(y)) * 1e-3)
    return max(float(np.median(np.abs(d))) / 0.6745 / math.sqrt(2.0),
               float(np.max(y)) * 1e-4, 1e-12)


def _smooth(y: np.ndarray, points: int) -> np.ndarray:
    w = max(5, points | 1)
    if len(y) < w:
        return y.astype(float)
    return savgol_filter(y.astype(float), w, 3)


def _interp_x(x: np.ndarray, pos: float) -> float:
    return float(np.interp(pos, np.arange(len(x)), x))


def _asymmetry(y: np.ndarray, apex: int) -> float:
    """Tailing factor b/a at 10% of the apex (USP convention)."""
    level = 0.1 * y[apex]
    left = apex
    while left > 0 and y[left] > level:
        left -= 1
    right = apex
    while right < len(y) - 1 and y[right] > level:
        right += 1
    a, b = apex - left, right - apex
    return float(b / a) if a > 0 else float("nan")


def detect_rt_peaks(rt: np.ndarray, y: np.ndarray, cfg: RtPeakConfig = RtPeakConfig()
                    ) -> list[RtPeak]:
    """All peaks and shoulders in a chromatogram, in RT order."""
    rt = np.asarray(rt, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 5 or not np.any(y > 0):
        return []
    ys = _smooth(y, cfg.smooth_points)
    noise = estimate_noise(y)
    baseline = float(np.percentile(ys, 5))
    min_prom = max(cfg.min_snr * noise, cfg.min_rel_prominence * ys.max())
    idx, props = find_peaks(ys, prominence=min_prom)
    if len(idx) == 0:
        return []
    half = peak_widths(ys, idx, rel_height=0.5)
    peaks = []
    for k, i in enumerate(idx):
        # Inner boundaries: valleys toward neighbouring apexes. Outer: where the
        # smoothed signal falls to boundary_fraction of the apex height.
        level = baseline + cfg.boundary_fraction * (ys[i] - baseline)
        lo = int(idx[k - 1]) + int(np.argmin(ys[idx[k - 1]:i + 1])) if k > 0 else None
        hi = int(i) + int(np.argmin(ys[i:idx[k + 1] + 1])) if k + 1 < len(idx) else None
        if lo is None:
            lo = i
            while lo > 0 and ys[lo] > level:
                lo -= 1
        if hi is None:
            hi = i
            while hi < len(ys) - 1 and ys[hi] > level:
                hi += 1
        seg = slice(lo, hi + 1)
        area = float(np.trapezoid(np.clip(y[seg] - baseline, 0, None), rt[seg]))
        fwhm = _interp_x(rt, half[3][k]) - _interp_x(rt, half[2][k])
        flags = []
        if lo == 0 or hi == len(ys) - 1:
            flags.append("truncated at the chromatogram edge")
        if fwhm < cfg.min_fwhm:
            flags.append(f"narrower than {cfg.min_fwhm} min (spike?)")
        if fwhm > cfg.max_fwhm:
            flags.append(f"wider than {cfg.max_fwhm} min")
        height = float(y[i])
        peaks.append(RtPeak(
            apex_rt=float(rt[i]), start_rt=float(rt[lo]), end_rt=float(rt[hi]), fwhm=fwhm,
            height=height, area=area, snr=float((ys[i] - baseline) / noise),
            asymmetry=_asymmetry(ys, int(i)), flags=flags,
            half_width_left=float(rt[i]) - _interp_x(rt, half[2][k]),
            half_width_right=_interp_x(rt, half[3][k]) - float(rt[i])))
    peaks += _shoulders(rt, y, ys, idx, baseline, noise, cfg)
    return sorted(peaks, key=lambda p: p.apex_rt)


def _shoulders(rt, y, ys, apexes, baseline, noise, cfg) -> list[RtPeak]:
    """Peaks hidden on a flank: a local minimum of the second derivative there."""
    d2 = np.gradient(np.gradient(ys))
    cand, _ = find_peaks(-d2, prominence=np.max(-d2) * 0.05 if np.max(-d2) > 0 else None)
    out = []
    top = ys.max()
    for c in cand:
        if np.min(np.abs(apexes - c)) <= 3:                # curvature minimum of a real apex
            continue
        nearest = int(apexes[np.argmin(np.abs(apexes - c))])
        # On a flank the slope keeps its sign between the shoulder and the apex
        # (skipping the points next to the apex, where it passes through zero).
        if c > nearest:
            slope = np.gradient(ys)[nearest + 3:c + 1]
        else:
            slope = np.gradient(ys)[c:nearest - 2]
        if len(slope) == 0 or not (np.all(slope >= 0) or np.all(slope <= 0)):
            continue
        if ys[c] - baseline < max(cfg.shoulder_min_rel_height * (top - baseline),
                                  cfg.min_snr * noise):
            continue
        out.append(RtPeak(
            apex_rt=float(rt[c]), start_rt=float(rt[max(0, c - 2)]),
            end_rt=float(rt[min(len(rt) - 1, c + 2)]), fwhm=float("nan"), height=float(y[c]),
            area=float("nan"), snr=float((ys[c] - baseline) / noise), asymmetry=float("nan"),
            kind="shoulder",
            flags=["shoulder: no local maximum; apex and area are not separable"]))
    return out


def rt_pairs(peaks: list[RtPeak], rt: np.ndarray, y: np.ndarray,
             cfg: RtPeakConfig = RtPeakConfig()) -> list[RtPair]:
    """Resolution and separation class for each neighbouring pair of peaks."""
    rt = np.asarray(rt, dtype=float)
    ys = _smooth(np.asarray(y, dtype=float), cfg.smooth_points)
    out = []
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        delta = b.apex_rt - a.apex_rt
        if a.kind == "shoulder" or b.kind == "shoulder":
            out.append(RtPair(i, i + 1, delta, float("nan"), float("nan"), "shoulder"))
            continue
        # Fused peaks inflate the half-height width on the shared side, so each
        # peak's width is taken from its outer side (doubled).
        w_a, w_b = 2 * a.half_width_left, 2 * b.half_width_right
        res = 1.18 * delta / (w_a + w_b) if w_a + w_b > 0 else float("nan")
        lo, hi = np.searchsorted(rt, [a.apex_rt, b.apex_rt])
        valley = float(ys[lo:hi + 1].min()) if hi > lo else float("nan")
        ratio = valley / min(ys[lo], ys[min(hi, len(ys) - 1)])
        if res >= cfg.baseline_resolution:
            sep = "baseline-resolved"
        elif res >= cfg.partial_resolution:
            sep = "partially overlapping"
        else:
            sep = "coeluting"
        out.append(RtPair(i, i + 1, delta, float(res), float(ratio), sep))
    return out


# ---- ion mobility ---------------------------------------------------------------------


@dataclass(frozen=True)
class MobilityConfig:
    min_points: int = 7
    delta_bic: float = 10.0               # two populations must win by at least this
    min_minor_fraction: float = 0.05      # smaller population's share of the signal
    min_resolution: float = 0.5           # 1.18 * Δμ / (FWHM1 + FWHM2)
    # Instrument mobility precision (relative SD of repeated 1/K0 measurements of
    # one species). PLACEHOLDER: establish from replicate data on the instrument.
    precision_rel: float = 0.005
    precision_multiple: float = 3.0       # Δμ must exceed this many precisions
    reference: str = "mean"               # "mean" | "lower" | "first"
    broad_fwhm_rel: float = 0.03          # a single population wider than this is flagged
    precision_is_placeholder: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MobilityPopulation:
    centre: float
    fwhm: float
    fraction: float                       # of the fitted signal
    resolving_power: float                # centre / FWHM

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MobilityAnalysis:
    n_populations: int                    # 0 (insufficient data), 1 or 2
    populations: list[MobilityPopulation]
    coordinate: str
    relative_difference_pct: float | None = None
    reference: str | None = None
    resolution: float | None = None
    bic: dict = field(default_factory=dict)          # model -> BIC
    verdict: str = ""
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["populations"] = [p.to_dict() for p in self.populations]
        return d


def _gauss(x, a, mu, s):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def _m1(x, a, mu, s, c):
    return _gauss(x, a, mu, s) + c


def _m2(x, a1, mu1, s1, a2, mu2, s2, c):
    return _gauss(x, a1, mu1, s1) + _gauss(x, a2, mu2, s2) + c


def _mskew(x, a, mu, s, alpha, c):
    return a * skewnorm.pdf(x, alpha, loc=mu, scale=s) + c


def _fit(model, x, y, w, p0, bounds):
    try:
        p, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, sigma=w, absolute_sigma=False,
                         maxfev=20000)
    except (RuntimeError, ValueError):
        return None, np.inf
    chi2 = float(np.sum(((y - model(x, *p)) / w) ** 2))
    return p, chi2


def detect_mobility_populations(x: np.ndarray, y: np.ndarray,
                                cfg: MobilityConfig = MobilityConfig(),
                                coordinate: str = "1/K0") -> MobilityAnalysis:
    """One or two mobility populations in a mobilogram; see module docstring."""
    x = np.asarray(x, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 0, None)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < cfg.min_points or y.max() <= 0:
        return MobilityAnalysis(0, [], coordinate, verdict="insufficient data",
                                flags=[f"fewer than {cfg.min_points} mobility bins with signal"])
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    span = x.max() - x.min()
    step = float(np.median(np.diff(x))) if n > 1 else span
    w = np.sqrt(np.maximum(y, y.max() * 1e-3))              # ~ counting statistics
    total_w = np.sum(y)
    mu0 = float(np.sum(x * y) / total_w)
    s0 = float(math.sqrt(max(np.sum(y * (x - mu0) ** 2) / total_w, step ** 2)))
    a0, c0 = float(y.max()), float(np.percentile(y, 5))
    lo_s, hi_s = step / 2, span
    b1 = ([0, x.min(), lo_s, 0], [np.inf, x.max(), hi_s, a0])
    p1, chi1 = _fit(_m1, x, y, w, [a0, mu0, s0, c0], b1)

    # Two-component start: the two most prominent smoothed maxima, else mu0 ± s0.
    ys = _smooth(y, 5)
    idx, props = find_peaks(ys, prominence=a0 * 0.02)
    if len(idx) >= 2:
        top = idx[np.argsort(props["prominences"])[-2:]]
        m_a, m_b = sorted(x[top])
    else:
        m_a, m_b = mu0 - s0, mu0 + s0
    b2 = ([0, x.min(), lo_s, 0, x.min(), lo_s, 0],
          [np.inf, x.max(), hi_s, np.inf, x.max(), hi_s, a0])
    p2, chi2 = _fit(_m2, x, y, w, [a0, m_a, s0 / 2, a0 / 2, m_b, s0 / 2, c0], b2)

    bs = ([0, x.min() - span, lo_s, -20, 0], [np.inf, x.max() + span, 2 * span, 20, a0])
    best_skew = (None, np.inf)
    for alpha0 in (-4.0, 4.0):
        p, chi = _fit(_mskew, x, y, w, [a0 * s0 * 2.5, mu0, s0 * 1.5, alpha0, c0], bs)
        if chi < best_skew[1]:
            best_skew = (p, chi)
    ps, chis = best_skew

    # Quasi-likelihood correction: real intensity scatter is not Poisson (it is
    # partly multiplicative), which inflates chi2 so that any extra parameter
    # "wins" on BIC. The overdispersion c_hat is estimated from the most general
    # model and divides every chi2 before the models are compared (QBIC).
    general = chi2 if np.isfinite(chi2) else chi1
    k_general = 7 if np.isfinite(chi2) else 4
    c_hat = max(1.0, general / max(n - k_general, 1)) if np.isfinite(general) else 1.0
    bic = {"one gaussian": chi1 / c_hat + 4 * math.log(n),
           "two gaussians": chi2 / c_hat + 7 * math.log(n),
           "one skewed": chis / c_hat + 5 * math.log(n)}
    single_bic = min(bic["one gaussian"], bic["one skewed"])
    flags: list[str] = []
    if cfg.precision_is_placeholder:
        flags.append(f"mobility precision {cfg.precision_rel:.1%} is a placeholder; establish it "
                     "from replicate measurements on this instrument")

    def single(verdict: str) -> MobilityAnalysis:
        if p1 is None:
            return MobilityAnalysis(0, [], coordinate, bic=bic, verdict="fit failed", flags=flags)
        a, mu, s, _ = p1
        fwhm = FWHM_PER_SIGMA * s
        if fwhm / mu > cfg.broad_fwhm_rel:
            flags.append(f"broad distribution (FWHM {fwhm / mu:.1%} of centre): may hide "
                         "unresolved populations or reflect poor feature quality")
        if bic["one skewed"] + cfg.delta_bic < bic["one gaussian"]:
            flags.append("asymmetric distribution (skewed model fits better than a Gaussian)")
        return MobilityAnalysis(1, [MobilityPopulation(float(mu), float(fwhm), 1.0,
                                                       float(mu / fwhm))],
                                coordinate, bic=bic, verdict=verdict, flags=flags)

    if p2 is None or bic["two gaussians"] + cfg.delta_bic > single_bic:
        why = ("one population" if p2 is None or bic["two gaussians"] >= single_bic else
               "one population (two-population model not better by the required margin)")
        return single(why)
    a1, mu1, s1, a2, mu2, s2, _ = p2
    if mu2 < mu1:
        a1, mu1, s1, a2, mu2, s2 = a2, mu2, s2, a1, mu1, s1
    f1, f2 = FWHM_PER_SIGMA * s1, FWHM_PER_SIGMA * s2
    area1, area2 = a1 * s1, a2 * s2
    frac1 = area1 / (area1 + area2)
    resolution = 1.18 * (mu2 - mu1) / (f1 + f2)
    ref = {"mean": (mu1 + mu2) / 2, "lower": min(mu1, mu2), "first": mu1}[cfg.reference]
    rel = 100.0 * abs(mu2 - mu1) / ref
    if min(frac1, 1 - frac1) < cfg.min_minor_fraction:
        return single("one dominant population (second component below the minimum fraction)")
    if resolution < cfg.min_resolution:
        flags.append(f"two-population model preferred but resolution {resolution:.2f} < "
                     f"{cfg.min_resolution}: treated as one unresolved distribution")
        return single("unresolved")
    if abs(mu2 - mu1) / ref < cfg.precision_multiple * cfg.precision_rel:
        flags.append(f"separation {rel:.2f}% is within {cfg.precision_multiple:g}x the mobility "
                     "precision: not distinguishable from measurement scatter")
        return single("separation below instrument precision")
    pops = [MobilityPopulation(float(mu1), float(f1), float(frac1), float(mu1 / f1)),
            MobilityPopulation(float(mu2), float(f2), float(1 - frac1), float(mu2 / f2))]
    return MobilityAnalysis(2, pops, coordinate, relative_difference_pct=float(rel),
                            reference=f"{cfg.reference} of the two centres",
                            resolution=float(resolution), bic=bic, verdict="two populations",
                            flags=flags)
