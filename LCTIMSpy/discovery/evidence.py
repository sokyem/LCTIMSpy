"""Data-only evidence that two observed species are isomers -- no identifications needed.

Isomers share an elemental composition, so, independently of any database search:

* isotope envelopes  each species' measured M0..Mn distribution, compared with the
                     other species (and with averagine). A different distribution
                     points to another composition (isobaric, not isomeric) or to a
                     co-eluting interference.
* monoisotopic m/z   measured separately for each species; should agree within tolerance.
* charge states      the same neutral mass at other charges. The RT split (or the
                     mobility split) reproduced at another charge is an independent
                     observation of the pair. Each species' charge-state distribution is
                     reported; conformers can change it, so a difference is not exclusionary.

MS/MS and mobility evidence come from msms.py and peaks.py; ``summarize_support``
combines every line after MS/MS validation. None of this establishes structure: the
output is a putative candidate with its evidence laid out, never an identification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..core.progress import CancelToken, ProgressEvent, ProgressReporter, null_reporter
from .psms import PROTON
from .synthetic import averagine_pattern
from .traces import TraceConfig, analyze_species, isotope_windows


@dataclass(frozen=True)
class EvidenceConfig:
    n_isotopes: int = 4
    ppm: float = 10.0                     # per-isotope m/z window
    min_isotope_cosine: float = 0.95      # A vs B envelopes agree at or above this
    min_averagine_cosine: float = 0.90    # below: envelope distorted (interference?)
    mass_agreement_ppm: float = 5.0       # A vs B measured monoisotopic m/z
    charge_offsets: tuple[int, ...] = (-1, 1, 2)
    max_charge: int = 6
    rt_match_min: float = 0.1             # other-charge apex within this of the species apex
    min_alt_relative_area: float = 0.02   # other-charge area / species area to count as observed
    distribution_diff: float = 0.2        # |main-charge fraction A - B| reported as different

    def to_dict(self) -> dict:
        return asdict(self)


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ---- isotope envelopes --------------------------------------------------------------


def species_envelope(cache, entry, mono_mz: float, charge: int, sp: dict,
                     cfg: EvidenceConfig = EvidenceConfig()) -> dict:
    """Measured isotope intensities in the species' core (RT apex region x mobility population)."""
    wins = isotope_windows(mono_mz, charge, cfg.n_isotopes, cfg.ppm)
    pts = cache.load_mz_window(entry, wins, columns=("mz", "intensity", "mobility", "frame"))
    half_rt = max((sp["rt_end"] - sp["rt_start"]) / 4, 0.01)
    ok = np.abs(pts["retention_time"].to_numpy(float) - sp["rt_apex"]) <= half_rt
    centre = _f(sp.get("mobility"))
    mob = pts["mobility"].to_numpy(float)
    if np.isfinite(centre) and np.isfinite(mob).any():
        fwhm = _f(sp.get("mobility_fwhm"))
        half_m = max(fwhm / 2 if np.isfinite(fwhm) else 0.0, 0.005)
        ok &= np.abs(mob - centre) <= half_m
    mz = pts["mz"].to_numpy(float)[ok]
    inten = pts["intensity"].to_numpy(float)[ok]
    env = np.zeros(cfg.n_isotopes)
    mono_measured = float("nan")
    for k, (lo, hi) in enumerate(wins):
        sel = (mz >= lo) & (mz <= hi)
        env[k] = inten[sel].sum()
        if k == 0 and env[0] > 0:
            mono_measured = float(np.average(mz[sel], weights=inten[sel]))
    top = env.max()
    return {"intensities": env.tolist(), "relative": (env / top).tolist() if top > 0 else None,
            "mono_mz_measured": mono_measured, "n_points": int(ok.sum())}


def compare_envelopes(env_a: dict, env_b: dict, neutral_mass: float, mono_mz: float,
                      cfg: EvidenceConfig = EvidenceConfig()) -> tuple[dict, list[str]]:
    flags: list[str] = []
    expected = averagine_pattern(neutral_mass, cfg.n_isotopes)
    out = {"a": env_a, "b": env_b, "averagine_relative": expected.tolist(),
           "cosine": None, "max_deviation": None, "averagine_cosine_a": None,
           "averagine_cosine_b": None, "mono_mz_diff_ppm": None,
           "verdict": "insufficient signal", "mass_verdict": "insufficient signal"}
    ra, rb = env_a["relative"], env_b["relative"]
    usable = [r is not None and sum(v > 0 for v in r) >= 2 for r in (ra, rb)]
    if all(usable):
        a, b = np.asarray(ra), np.asarray(rb)
        cos = _cosine(a, b)
        # Deviation only where averagine expects a measurable isotope.
        mask = expected >= 0.1
        out.update(cosine=cos, max_deviation=float(np.max(np.abs(a - b)[mask])),
                   averagine_cosine_a=_cosine(a, expected), averagine_cosine_b=_cosine(b, expected))
        out["verdict"] = "agree" if cos >= cfg.min_isotope_cosine else "differ"
        if out["verdict"] == "differ":
            flags.append(f"isotope distributions of A and B differ (cosine {cos:.3f}): a different "
                         "elemental composition (isobaric, not isomeric) or an interference")
        for side in ("a", "b"):
            c = out[f"averagine_cosine_{side}"]
            if c < cfg.min_averagine_cosine:
                flags.append(f"{side.upper()}: isotope envelope deviates from averagine "
                             f"(cosine {c:.2f}): possible co-eluting interference or non-peptide")
    else:
        missing = [s for s, u in zip("AB", usable) if not u]
        flags.append(f"isotope envelope too weak to compare for species {' and '.join(missing)}")
    ma, mb = env_a["mono_mz_measured"], env_b["mono_mz_measured"]
    if np.isfinite(ma) and np.isfinite(mb):
        diff = (ma - mb) / mono_mz * 1e6
        out["mono_mz_diff_ppm"] = float(diff)
        out["mass_verdict"] = "agree" if abs(diff) <= cfg.mass_agreement_ppm else "differ"
        if out["mass_verdict"] == "differ":
            flags.append(f"measured monoisotopic m/z of A and B differ by {diff:.1f} ppm "
                         f"(tolerance {cfg.mass_agreement_ppm:g} ppm)")
    return out, flags


# ---- charge states ------------------------------------------------------------------


def _match_peak(analysis, sp: dict, cfg: EvidenceConfig) -> int | None:
    best, gap = None, cfg.rt_match_min
    for i, p in enumerate(analysis.rt_peaks):
        d = abs(p.apex_rt - sp["rt_apex"])
        if p.kind == "peak" and d <= gap:
            best, gap = i, d
    return best


def charge_states(cache, entry, row: dict, cfg: EvidenceConfig = EvidenceConfig(),
                  trace: TraceConfig = TraceConfig()) -> tuple[dict, list[str]]:
    """The candidate's neutral mass at other charges, attributed to species A and B."""
    z = int(row["charge"])
    mass = float(row["neutral_mass"])
    a, b = row["species_a"], row["species_b"]
    shared = a["rt_peak"] == b["rt_peak"]
    states = []
    for dz in cfg.charge_offsets:
        zz = z + dz
        if zz < 1 or zz > cfg.max_charge:
            continue
        mono = mass / zz + PROTON
        an = analyze_species(cache, entry, mono, zz, trace)
        state = {"charge": zz, "mono_mz": mono, "a": None, "b": None, "reproduced": False}
        hit = {}
        for side, sp in (("a", a), ("b", b)):
            i = _match_peak(an, sp, cfg)
            if i is None:
                continue
            p, m = an.rt_peaks[i], an.mobility[i]
            area = _f(sp.get("rt_area"))
            rel = p.area / area if np.isfinite(area) and area > 0 else float("nan")
            if np.isfinite(rel) and rel < cfg.min_alt_relative_area:
                continue
            hit[side] = i
            state[side] = {"rt_apex": p.apex_rt, "area": p.area, "relative_area": rel,
                           "n_populations": m.n_populations if m is not None else 0,
                           "mobility": [q.centre for q in m.populations] if m is not None else []}
        if shared:
            alt = state["a"]
            state["reproduced"] = bool(alt and alt["n_populations"] >= 2)
        else:
            state["reproduced"] = "a" in hit and "b" in hit and hit["a"] != hit["b"]
        if state["a"] or state["b"]:
            states.append(state)

    flags = []
    reproduced = [s["charge"] for s in states if s["reproduced"]]
    observed = [s["charge"] for s in states]
    split = "mobility split" if shared else "RT split"
    if reproduced:
        verdict = "reproduced"
        flags.append(f"{split} reproduced at charge " + ", ".join(f"{c}+" for c in reproduced))
    elif observed:
        verdict = "not reproduced"
        flags.append(f"also observed at charge {', '.join(f'{c}+' for c in observed)}, "
                     f"but the {split} is not reproduced there")
    else:
        verdict = "single charge state"

    distribution = None
    if not shared and states:
        frac = {}
        for side, sp in (("a", a), ("b", b)):
            main = _f(sp.get("rt_area"))
            alts = sum(s[side]["area"] for s in states if s[side])
            frac[side] = main / (main + alts) if np.isfinite(main) and main + alts > 0 else None
        distribution = {"main_charge_fraction_a": frac["a"], "main_charge_fraction_b": frac["b"]}
        if frac["a"] is not None and frac["b"] is not None \
                and abs(frac["a"] - frac["b"]) > cfg.distribution_diff:
            flags.append(f"charge-state distributions differ ({z}+ is {frac['a']:.0%} of A's "
                         f"signal, {frac['b']:.0%} of B's): conformers or different structures; "
                         "not exclusionary")
    return {"charge": z, "states": states, "verdict": verdict, "shared_rt_window": shared,
            "distribution": distribution}, flags


# ---- per candidate ------------------------------------------------------------------


def data_evidence(cache, entry, row: dict, cfg: EvidenceConfig = EvidenceConfig(),
                  trace: TraceConfig = TraceConfig()) -> tuple[dict | None, list[str]]:
    if row.get("species_b") is None:
        return None, []
    mono, z = float(row["mono_mz"]), int(row["charge"])
    env_a = species_envelope(cache, entry, mono, z, row["species_a"], cfg)
    env_b = species_envelope(cache, entry, mono, z, row["species_b"], cfg)
    iso, f1 = compare_envelopes(env_a, env_b, float(row["neutral_mass"]), mono, cfg)
    charges, f2 = charge_states(cache, entry, row, cfg, trace)
    return {"isotopes": iso, "charge_states": charges}, f1 + f2


def add_data_evidence(cache, entry, candidates: pd.DataFrame,
                      cfg: EvidenceConfig = EvidenceConfig(), trace: TraceConfig = TraceConfig(),
                      reporter: ProgressReporter = null_reporter,
                      cancel: CancelToken | None = None) -> pd.DataFrame:
    out = candidates.copy()
    evidence, flags = [], []
    rows = out.to_dict(orient="records")
    for i, row in enumerate(rows):
        if cancel is not None:
            cancel.raise_if_cancelled()
        reporter(ProgressEvent("evidence", i, len(rows), "Comparing isotopes and charge states"))
        ev, new = data_evidence(cache, entry, row, cfg, trace)
        evidence.append(ev)
        flags.append(list(row["flags"]) + [f for f in new if f not in row["flags"]])
    out["data_evidence"] = evidence
    out["flags"] = flags
    return out


MSMS_LINE = {
    "supporting": ("similar fragmentation, sequence-explained", True),
    "similar, sequence not confirmed": ("similar fragmentation", True),
    "dissimilar": ("different fragmentation (does not exclude isomerization)", None),
    "ambiguous": ("precursor association ambiguous", None),
    "missing": ("MS/MS missing for a species", None),
}


def summarize_support(row: dict) -> dict | None:
    """Every data-only line of evidence for a pair, and an overall level.

    ``strong``:   isotope envelopes and monoisotopic m/z agree, AND the split is
                  reproduced at another charge or the MS/MS is similar
    ``moderate``: isotope envelopes and monoisotopic m/z agree
    ``weak``:     an envelope or mass disagreement, or too little signal to compare
    """
    ev = row.get("data_evidence")
    if not ev or row.get("species_b") is None:
        return None
    iso, ch = ev["isotopes"], ev["charge_states"]
    lines = [
        {"name": "RT / mobility separation", "result": row["classification"], "supports": None},
        {"name": "Isotope distribution", "supports": {"agree": True, "differ": False}.get(iso["verdict"]),
         "result": iso["verdict"] + (f" (cosine {iso['cosine']:.3f})" if iso["cosine"] is not None else "")},
        {"name": "Monoisotopic m/z", "supports": {"agree": True, "differ": False}.get(iso["mass_verdict"]),
         "result": iso["mass_verdict"] + (f" ({iso['mono_mz_diff_ppm']:+.1f} ppm)"
                                          if iso["mono_mz_diff_ppm"] is not None else "")},
    ]
    reproduced = [s["charge"] for s in ch["states"] if s["reproduced"]]
    lines.append({"name": "Other charge states",
                  "supports": True if reproduced else None,
                  "result": ("split reproduced at " + ", ".join(f"{c}+" for c in reproduced))
                  if reproduced else ch["verdict"]})
    msms_text, msms_ok = MSMS_LINE.get(row.get("msms_status") or "", (row.get("msms_status")
                                                                      or "not assessed", None))
    lines.append({"name": "MS/MS", "result": msms_text, "supports": msms_ok})
    agree = iso["verdict"] == "agree" and iso["mass_verdict"] == "agree"
    corroborated = bool(reproduced) or msms_ok is True
    level = "strong" if agree and corroborated else "moderate" if agree else "weak"
    return {"level": level, "lines": lines}
