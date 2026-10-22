"""MS/MS validation of isomer candidates (spec §9, §10, §12).

For each candidate, MS/MS spectra (DDA-PASEF precursors, or mzML MS2) are
assigned to each observed species by precursor m/z, charge, RT and -- with
mobility -- the precursor's 1/K0 inside the species' population. Assignments
are graded:

    unambiguous   one species, clean isolation
    ambiguous     the isolation mobility range spans both populations of the
                  pair, another feature was co-isolated, or the precursor has
                  no mobility / no charge

The species' spectra are compared (cosine and normalized spectral angle on
square-root intensities after removing the precursor region; matched
fragments; intensity correlation), against the within-species reproducibility
when available, and annotated with the candidate sequence's b/y ions.

Interpretation follows the spec: similar fragmentation *supports* a shared
sequence but does not prove identical structures; different fragmentation
does *not* exclude isomerization. Only candidates whose two species both have
unambiguous, good-quality, mutually similar spectra explained by the same
sequence -- with strong feature quality -- become "High-confidence putative".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .fragments import UnknownModification, fragment_ladder
from .synthetic import NEUTRON


@dataclass(frozen=True)
class MsmsConfig:
    precursor_ppm: float = 15.0
    precursor_isotopes: int = 2           # a "largest peak" m/z may be isotope 1 or 2
    rt_margin: float = 0.02               # minutes
    mobility_window_rel: float = 0.01     # precursor 1/K0 within centre ± max(FWHM/2, this)
    fragment_ppm: float = 20.0
    fragment_da_floor: float = 0.005
    top_n_peaks: int = 150
    min_peaks: int = 10
    coisolation_min_rel: float = 0.10     # co-isolated feature height relative to the target
    # Feature RT bounds stop at a fraction of the apex; an ion still elutes in
    # its tails, so the co-isolation window extends each side by this x width.
    coisolation_rt_extension: float = 0.5
    similarity_threshold: float = 0.70    # cosine on sqrt intensities
    # Between-species similarity may fall at most this far below the
    # within-species (replicate) similarity to count as "similar".
    max_similarity_drop: float = 0.10
    min_annotated_fraction: float = 0.30  # of fragment intensity explained by b/y
    min_annotated_ions: int = 4
    high_conf_min_snr: float = 20.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---- spectra -------------------------------------------------------------------------------


def preprocess(mz: np.ndarray, intensity: np.ndarray, cfg: MsmsConfig,
               precursor_mz: float | None = None, isolation_width: float | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Drop the precursor region, keep the top-N peaks, sqrt-scale, unit L2 norm."""
    mz = np.asarray(mz, dtype=float)
    it = np.asarray(intensity, dtype=float)
    if precursor_mz is not None and isolation_width:
        keep = np.abs(mz - precursor_mz) > isolation_width / 2
        mz, it = mz[keep], it[keep]
    if len(mz) > cfg.top_n_peaks:
        top = np.argsort(it)[-cfg.top_n_peaks:]
        mz, it = mz[top], it[top]
    order = np.argsort(mz)
    mz, it = mz[order], np.sqrt(np.clip(it[order], 0, None))
    norm = np.linalg.norm(it)
    return mz, it / norm if norm > 0 else it


def match_peaks(mz1, mz2, it1, it2, cfg: MsmsConfig) -> list[tuple[int, int]]:
    """One-to-one fragment matches within tolerance, most intense pairs first."""
    pairs = []
    j0 = 0
    for i, m in enumerate(mz1):
        tol = max(m * cfg.fragment_ppm * 1e-6, cfg.fragment_da_floor)
        while j0 < len(mz2) and mz2[j0] < m - tol:
            j0 += 1
        j = j0
        while j < len(mz2) and mz2[j] <= m + tol:
            pairs.append((it1[i] * it2[j], i, j))
            j += 1
    used1, used2, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used1 and j not in used2:
            used1.add(i)
            used2.add(j)
            out.append((i, j))
    return sorted(out)


def similarity(s1, s2, cfg: MsmsConfig) -> dict:
    """Cosine, normalized spectral angle, matches and intensity correlation."""
    (mz1, it1), (mz2, it2) = s1, s2
    if len(mz1) == 0 or len(mz2) == 0:
        return {"cosine": float("nan"), "spectral_angle": float("nan"), "n_matched": 0,
                "pearson": float("nan"), "matches": []}
    m = match_peaks(mz1, mz2, it1, it2, cfg)
    dot = float(sum(it1[i] * it2[j] for i, j in m))
    cos = min(1.0, dot)                                       # both are unit vectors
    angle = 1 - 2 * math.acos(max(-1.0, cos)) / math.pi
    if len(m) >= 3:
        a = np.log([it1[i] for i, _ in m])
        b = np.log([it2[j] for _, j in m])
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    else:
        r = float("nan")
    return {"cosine": cos, "spectral_angle": float(angle), "n_matched": len(m), "pearson": r,
            "matches": m}


def annotate(spec, peptide: str, mods: str, precursor_charge: int, cfg: MsmsConfig) -> dict:
    """Share of (sqrt) fragment intensity explained by the sequence's b/y ions."""
    mz, it = spec
    if len(mz) == 0:
        return {"fraction": 0.0, "n_ions": 0, "ions": []}
    try:
        ladders = [fragment_ladder(peptide, mods, 1)]
        if precursor_charge and precursor_charge >= 2:
            ladders.append(fragment_ladder(peptide, mods, 2))
    except UnknownModification as exc:
        return {"fraction": float("nan"), "n_ions": 0, "ions": [],
                "note": f"cannot annotate: {exc}"}
    theo = np.concatenate([lad[0] for lad in ladders])
    labels = sum((lad[1] for lad in ladders), [])
    order = np.argsort(theo)
    theo, labels = theo[order], [labels[k] for k in order]
    explained, ions = 0.0, []
    for k, m in enumerate(mz):
        tol = max(m * cfg.fragment_ppm * 1e-6, cfg.fragment_da_floor)
        pos = np.searchsorted(theo, m)
        for q in (pos - 1, pos):
            if 0 <= q < len(theo) and abs(theo[q] - m) <= tol:
                explained += it[k] ** 2
                ions.append(labels[q])
                break
    return {"fraction": float(explained / np.sum(it ** 2)), "n_ions": len(set(ions)),
            "ions": sorted(set(ions))}


# ---- precursor assignment -------------------------------------------------------------------


def _isolation_contains(pre, mz: float) -> bool:
    w = pre.isolation_width if np.isfinite(pre.isolation_width) else 2.0
    return abs(mz - pre.isolation_mz) <= w / 2


def assign_precursors(precursors: pd.DataFrame, species: dict, mono_mz: float, charge: int,
                      cfg: MsmsConfig) -> pd.DataFrame:
    """Precursors that belong to one observed species, with ambiguity reasons."""
    if precursors.empty:
        return precursors.assign(isotope=[], reasons=[])
    p = precursors
    mono = p["mz_monoisotopic"].to_numpy(float)
    largest = p["mz_largest"].to_numpy(float)
    z = pd.to_numeric(p["charge"], errors="coerce").to_numpy(float)
    rt = p["rt"].to_numpy(float)
    mob = p["mobility"].to_numpy(float)
    tol = mono_mz * cfg.precursor_ppm * 1e-6
    iso = np.full(len(p), -1)
    for k in range(cfg.precursor_isotopes + 1):
        target = mono_mz + k * NEUTRON / charge
        hit_l = np.abs(largest - target) <= tol
        iso[(iso < 0) & hit_l] = k
    iso[np.abs(mono - mono_mz) <= tol] = 0
    ok = iso >= 0
    ok &= np.isnan(z) | (z == charge)
    ok &= (rt >= species["rt_start"] - cfg.rt_margin) & (rt <= species["rt_end"] + cfg.rt_margin)
    centre = species["mobility"]
    if np.isfinite(centre):
        fwhm = species["mobility_fwhm"]
        half = max(fwhm / 2 if np.isfinite(fwhm) else 0.0, cfg.mobility_window_rel * centre)
        ok &= ~np.isfinite(mob) | (np.abs(mob - centre) <= half)
    out = p[ok].copy()
    out["isotope"] = iso[ok]
    reasons = []
    for row in out.itertuples():
        r = []
        if pd.isna(row.charge):
            r.append("precursor charge undetermined")
        if np.isfinite(centre) and not np.isfinite(row.mobility):
            r.append("precursor has no mobility: cannot tell which population was fragmented")
        reasons.append(r)
    out["reasons"] = reasons
    return out


def _coisolated(pre, features: pd.DataFrame | None, mono_mz: float, charge: int,
                cfg: MsmsConfig) -> list[str]:
    """Other ions inside this precursor's isolation window (m/z, RT and mobility).

    Features at the target's own m/z and charge are the target -- or its isomer
    partner, which the "spans both populations" check covers -- and are skipped.
    Heights are compared feature to feature, so the threshold is like for like.
    """
    if features is None or features.empty:
        return []
    f = features[features["charge"].notna()]
    ext = cfg.coisolation_rt_extension * (f["rt_end"] - f["rt_start"]) + cfg.rt_margin
    in_rt = (f["rt_start"] - ext <= pre.rt) & (pre.rt <= f["rt_end"] + ext)
    f = f[in_rt]
    tol = mono_mz * cfg.precursor_ppm * 1e-6
    is_target = (np.abs(f["mz"] - mono_mz) <= tol) & (f["charge"] == charge)
    target_height = float(f.loc[is_target, "height"].max()) if is_target.any() else float("nan")
    out = []
    for row in f[~is_target].itertuples():
        if np.isfinite(pre.isolation_mobility_low) and np.isfinite(row.mobility_apex):
            if not (pre.isolation_mobility_low <= row.mobility_apex <= pre.isolation_mobility_high):
                continue
        z = int(row.charge)
        if not any(_isolation_contains(pre, row.mz + k * NEUTRON / z)
                   for k in range(int(row.n_isotopes))):
            continue
        if np.isfinite(target_height):
            rel = row.height / target_height
            if rel >= cfg.coisolation_min_rel:
                out.append(f"co-isolated with feature {int(row.feature_id)} "
                           f"(m/z {row.mz:.4f}, {z}+, {rel:.0%} of target height)")
        else:
            out.append(f"co-isolated with feature {int(row.feature_id)} (m/z {row.mz:.4f}, "
                       f"{z}+); target feature not detected, relative intensity unknown")
    return out


# ---- per-candidate validation -----------------------------------------------------------------


def _species_msms(cache, entry, precursors, row, side, other, features, cfg):
    sp = row[side]
    assigned = assign_precursors(precursors, sp, row["mono_mz"], row["charge"], cfg)
    spectra, info = {}, []
    raw = cache.load_msms(entry, assigned["precursor_id"].tolist()) if len(assigned) else {}
    for pre in assigned.itertuples():
        reasons = list(pre.reasons)
        # The partner can only have been co-isolated if it was eluting when this
        # precursor was selected (an RT-separated partner of equal mobility was not).
        partner_eluting = other is not None and (
            other["rt_start"] - cfg.rt_margin <= pre.rt <= other["rt_end"] + cfg.rt_margin)
        if partner_eluting and np.isfinite(other["mobility"]) and \
                np.isfinite(pre.isolation_mobility_low) and \
                pre.isolation_mobility_low <= other["mobility"] <= pre.isolation_mobility_high:
            reasons.append("isolation mobility range spans both populations of the pair")
        reasons += _coisolated(pre, features, row["mono_mz"], row["charge"], cfg)
        mz, it = raw.get(int(pre.precursor_id), (np.array([]), np.array([])))
        n_peaks = len(mz)
        if n_peaks < cfg.min_peaks:
            reasons.append(f"poor MS/MS: {n_peaks} peaks")
        spectra[int(pre.precursor_id)] = preprocess(mz, it, cfg, pre.isolation_mz,
                                                    pre.isolation_width)
        info.append({"precursor_id": int(pre.precursor_id), "rt": float(pre.rt),
                     "mobility": float(pre.mobility), "intensity": float(pre.intensity),
                     "n_peaks": n_peaks, "ambiguous": bool(reasons), "reasons": reasons})
    clean = [i for i in info if not i["ambiguous"]]
    pool = clean or info
    best = max(pool, key=lambda i: i["intensity"]) if pool else None
    within = []
    ids = [i["precursor_id"] for i in pool]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            within.append(similarity(spectra[ids[a]], spectra[ids[b]], cfg)["cosine"])
    return {"n_assigned": len(info), "n_unambiguous": len(clean), "spectra": info,
            "best_precursor": best["precursor_id"] if best else None,
            "best_ambiguous": bool(best["ambiguous"]) if best else None,
            "within_similarity_median": float(np.median(within)) if within else float("nan"),
            }, (spectra[best["precursor_id"]] if best else None)


def _differential(sa, sb, matches, k=5) -> list[dict]:
    """The fragments whose relative intensity differs most between the two spectra."""
    rows = []
    matched_a = {i for i, _ in matches}
    matched_b = {j for _, j in matches}
    for i, j in matches:
        rows.append({"mz": float(sa[0][i]), "a": float(sa[1][i]), "b": float(sb[1][j]),
                     "log_ratio": float(np.log((sa[1][i] + 1e-9) / (sb[1][j] + 1e-9)))})
    for i in set(range(len(sa[0]))) - matched_a:
        rows.append({"mz": float(sa[0][i]), "a": float(sa[1][i]), "b": 0.0, "log_ratio": np.inf})
    for j in set(range(len(sb[0]))) - matched_b:
        rows.append({"mz": float(sb[0][j]), "a": 0.0, "b": float(sb[1][j]), "log_ratio": -np.inf})
    rows.sort(key=lambda r: -(max(r["a"], r["b"]) * min(abs(r["log_ratio"]), 5.0)))
    return rows[:k]


def validate_candidate(cache, entry, row: dict, precursors: pd.DataFrame,
                       features: pd.DataFrame | None, cfg: MsmsConfig) -> dict:
    a, b = row["species_a"], row["species_b"]
    msms_a, spec_a = _species_msms(cache, entry, precursors, row, "species_a", b, features, cfg)
    result = {"a": msms_a, "b": None, "similarity": None, "annotation": {}, "mirror": None,
              "differential_fragments": []}
    spec_b = None
    if b is not None:
        result["b"], spec_b = _species_msms(cache, entry, precursors, row, "species_b", a,
                                            features, cfg)
    if spec_a is not None and spec_b is not None:
        sim = similarity(spec_a, spec_b, cfg)
        result["similarity"] = {k: v for k, v in sim.items() if k != "matches"}
        result["mirror"] = {"a_mz": spec_a[0].tolist(), "a_intensity": spec_a[1].tolist(),
                            "b_mz": spec_b[0].tolist(), "b_intensity": spec_b[1].tolist(),
                            "matches": [list(m) for m in sim["matches"]]}
        result["differential_fragments"] = _differential(spec_a, spec_b, sim["matches"])
    keys = row.get("mod_keys") or ([row["mod_key"]] if row.get("mod_key") else [])
    keys = keys or sorted(set(a.get("mod_keys", [])) | set((b or {}).get("mod_keys", [])))
    for key in keys:
        peptide, _, mods = key.partition("|")
        result["annotation"][key] = {
            "a": annotate(spec_a, peptide, mods, row["charge"], cfg) if spec_a is not None else None,
            "b": annotate(spec_b, peptide, mods, row["charge"], cfg) if spec_b is not None else None,
        }
    return result


def _supports(ann: dict | None, cfg: MsmsConfig) -> bool:
    return bool(ann) and np.isfinite(ann["fraction"]) and \
        ann["fraction"] >= cfg.min_annotated_fraction and ann["n_ions"] >= cfg.min_annotated_ions


def grade(row: dict, msms: dict, cfg: MsmsConfig) -> tuple[str, str, list[str]]:
    """(category, msms_status, flags) after MS/MS validation."""
    category = row["category"]
    flags = [f for f in row["flags"] if f != "high-confidence requires MS/MS validation (phase 4)"]
    if row["status"] == "excluded" or row["species_b"] is None:
        return category, "not assessed", flags
    ma, mb = msms["a"], msms["b"]
    if not ma["n_assigned"] or not mb["n_assigned"]:
        missing = [s for s, m in (("A", ma), ("B", mb)) if not m["n_assigned"]]
        flags.append(f"no MS/MS assigned to species {' and '.join(missing)}")
        return category, "missing", flags
    sim = msms["similarity"] or {}
    cos = sim.get("cosine", float("nan"))
    ambiguous = ma["best_ambiguous"] or mb["best_ambiguous"]
    if ambiguous:
        reasons = {r for m in (ma, mb) for s in m["spectra"]
                   if s["precursor_id"] == m["best_precursor"] for r in s["reasons"]}
        flags.append("MS/MS association ambiguous: " + "; ".join(sorted(reasons)))
    within = [m["within_similarity_median"] for m in (ma, mb)
              if np.isfinite(m["within_similarity_median"])]
    baseline = min(within) if within else float("nan")
    similar = (np.isfinite(cos) and cos >= cfg.similarity_threshold
               and (not np.isfinite(baseline) or cos >= baseline - cfg.max_similarity_drop))
    if np.isfinite(cos) and not similar:
        detail = (f"cosine {cos:.2f} vs {baseline:.2f} between replicate spectra"
                  if np.isfinite(baseline) else f"cosine {cos:.2f}")
        flags.append(f"MS/MS patterns differ ({detail}); this does not exclude isomerization")
    supported = [k for k, v in msms["annotation"].items()
                 if _supports(v["a"], cfg) and _supports(v["b"], cfg)]
    status = ("ambiguous" if ambiguous else "supporting" if similar and supported
              else "similar, sequence not confirmed" if similar else "dissimilar")
    if category == "Unresolved candidate":
        return category, status, flags
    quality_ok = (row["species_a"]["snr"] >= cfg.high_conf_min_snr
                  and row["species_b"]["snr"] >= cfg.high_conf_min_snr
                  and not any(f.startswith("low-abundance") or "broad" in f for f in flags))
    one_sided = row["sequence_status"] == "only one species has a sequence assignment"
    if status == "supporting" and category == "Sequence-supported candidate" and quality_ok:
        category = "High-confidence putative candidate"
    elif status == "supporting" and one_sided:
        category = "Sequence-supported candidate"
        flags.append("shared sequence supported by MS/MS similarity and fragment annotation, "
                     "not by a database PSM for both species")
    if category != "High-confidence putative candidate" and status == "supporting" \
            and not quality_ok:
        flags.append(f"not high-confidence: feature quality below threshold "
                     f"(S/N >= {cfg.high_conf_min_snr:g}, no low-abundance/broad flags)")
    return category, status, flags


def validate_msms(cache, entry, candidates: pd.DataFrame, features: pd.DataFrame | None = None,
                  cfg: MsmsConfig = MsmsConfig()) -> pd.DataFrame:
    """Add MS/MS evidence to every candidate and update its evidence category."""
    out = candidates.copy()
    precursors = cache.load_precursors(entry)
    details, cats, statuses, flag_lists, cosines = [], [], [], [], []
    for row in out.to_dict(orient="records"):
        if precursors.empty:
            msms = {"a": None, "b": None, "similarity": None, "annotation": {}, "mirror": None,
                    "differential_fragments": []}
            cat, status = row["category"], "no MS/MS in this run"
            flags = row["flags"] + ["no MS/MS spectra in this run"]
        else:
            msms = validate_candidate(cache, entry, row, precursors, features, cfg)
            cat, status, flags = grade(row, msms, cfg)
        details.append(msms)
        cats.append(cat)
        statuses.append(status)
        flag_lists.append(flags)
        cosines.append((msms.get("similarity") or {}).get("cosine", float("nan")))
    out["category"] = cats
    out["msms_status"] = statuses
    out["msms_cosine"] = cosines
    out["flags"] = flag_lists
    out["msms"] = details
    out.attrs.update(candidates.attrs)
    out.attrs["msms_config"] = cfg.to_dict()
    return out
