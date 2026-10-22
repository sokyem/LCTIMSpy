"""Joint RT-IM isomer candidate discovery and evidence classification (spec §8, §11, §12).

Every accurate-mass group (mass-first) or identified peptide (sequence-first)
is re-measured from the raw cache. Each RT peak -- or each mobility population
inside an RT peak -- becomes an *observed species*; every pair of observed
species of one group is a candidate, classified by what separates them:

    A  RT-separated        resolved in RT (Rs >= rt_min_resolution) and
                           dRT >= rt_min_delta; mobility not distinct
    B  IM-separated        distinct mobility populations, whatever the RT
    C  RT- and IM-separated both of the above
    D  unresolved          matching mass/sequence evidence that RT and IM do not
                           resolve (shoulder, coelution, resolved but below the dRT
                           screen, PSM sampling disagreeing with the trace, ...)

Neither threshold is required for the other: a mobility-distinct pair is kept
whatever its RT. Every candidate records the measured differences, each
criterion with its threshold and outcome, warnings, and -- if excluded --
the reasons. Excluded candidates are kept, marked ``status = "excluded"``.

Evidence categories describe the strength of evidence, never a mechanism:

    High-confidence putative   requires MS/MS validation (phase 4); not assigned here
    Sequence-supported         both species supported by PSMs of one and the same
                               sequence and modification state
    Mass-matched               compatible accurate masses, insufficient sequence evidence
    Unresolved                 class D
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .grouping import GroupingConfig, artifact_flags, group_by_mass, is_exclusionary
from .psms import PROTON
from .sequence_first import SequenceFirstConfig, _rt_clusters, filter_psms
from .traces import SpeciesAnalysis, TraceConfig, analyze_species

CLASSES = {"A": "A: RT-separated", "B": "B: IM-separated",
           "C": "C: RT- and IM-separated", "D": "D: unresolved"}


@dataclass(frozen=True)
class CandidateConfig:
    # Screening thresholds. Defaults follow the published PepSIFinder workflow
    # (1.0 min; 2% mobility variation for Aplysia, 4% for rat); neither is
    # universal -- set them from the chromatographic method and the
    # instrument's measured mobility precision.
    rt_min_delta: float = 1.0
    rt_min_resolution: float = 0.8
    mobility_min_rel_pct: float = 2.0
    min_relative_area: float = 0.01
    match_ppm: float = 10.0                 # PSM precursor m/z vs species m/z
    psm_mobility_window_rel: float = 0.01   # PSM supports a population within centre ± this
    analyze_single_feature_groups: bool = False
    split_pattern_min: int = 5              # recurring dRT among this many other candidates
    split_pattern_rt: float = 0.05
    contaminant_pattern: str = r"CON_|contam|cRAP|KRT\d|keratin|TRYP_|trypsin"
    trace: TraceConfig = TraceConfig()
    grouping: GroupingConfig = GroupingConfig()
    sequence: SequenceFirstConfig = SequenceFirstConfig()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObservedSpecies:
    rt_peak: int
    population: int                   # index within the RT peak's mobility populations
    n_populations: int
    kind: str                         # "peak" | "shoulder"
    rt_apex: float
    rt_start: float
    rt_end: float
    rt_area: float                    # measured: integrated RT peak area
    signal: float                     # rt_area x population fraction (relative LC-MS signal)
    height: float
    mobility: float                   # population centre, NaN without mobility
    mobility_fwhm: float
    snr: float
    psm_count: int = 0
    mod_keys: list[str] = field(default_factory=list)
    feature_ids: list[int] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def observe_species(analysis: SpeciesAnalysis) -> list[ObservedSpecies]:
    out = []
    for i, (p, m) in enumerate(zip(analysis.rt_peaks, analysis.mobility)):
        pops = m.populations if m is not None and m.n_populations else []
        mflags = list(m.flags) if m is not None else []
        if m is not None and m.verdict in ("unresolved", "separation below instrument precision"):
            mflags.append(f"mobility: {m.verdict}")
        if not pops:
            out.append(ObservedSpecies(i, 0, 0, p.kind, p.apex_rt, p.start_rt, p.end_rt,
                                       p.area, p.area, p.height, np.nan, np.nan, p.snr,
                                       flags=list(p.flags) + mflags))
            continue
        for k, pop in enumerate(pops):
            area = p.area * pop.fraction if np.isfinite(p.area) else np.nan
            out.append(ObservedSpecies(i, k, len(pops), p.kind, p.apex_rt, p.start_rt, p.end_rt,
                                       p.area, area, p.height, pop.centre, pop.fwhm, p.snr,
                                       flags=list(p.flags) + mflags))
    return out


def _support(species: list[ObservedSpecies], psms: pd.DataFrame | None, cfg: CandidateConfig
             ) -> None:
    """Count the PSMs inside each species (RT bounds; mobility around the population)."""
    if psms is None or psms.empty:
        return
    rt = psms["rt"].to_numpy(float)
    mob = psms["mobility"].to_numpy(float)
    for s in species:
        ok = (rt >= s.rt_start) & (rt <= s.rt_end)
        if np.isfinite(s.mobility):
            half = max(s.mobility_fwhm / 2 if np.isfinite(s.mobility_fwhm) else 0,
                       cfg.psm_mobility_window_rel * s.mobility)
            ok &= ~np.isfinite(mob) | (np.abs(mob - s.mobility) <= half)
        hits = psms[ok]
        s.psm_count = len(hits)
        s.mod_keys = sorted(hits["mod_key"].unique())


def _attach_features(species: list[ObservedSpecies], features: pd.DataFrame | None) -> None:
    if features is None or features.empty:
        return
    for s in species:
        ok = (features["rt_apex"] >= s.rt_start) & (features["rt_apex"] <= s.rt_end)
        if np.isfinite(s.mobility):
            ok &= (features["mobility_low"] <= s.mobility) & (features["mobility_high"] >= s.mobility)
        s.feature_ids = [int(x) for x in features.loc[ok, "feature_id"]]


def _resolution(a: ObservedSpecies, b: ObservedSpecies, analysis: SpeciesAnalysis) -> float:
    pa, pb = analysis.rt_peaks[a.rt_peak], analysis.rt_peaks[b.rt_peak]
    first, second = (pa, pb) if pa.apex_rt <= pb.apex_rt else (pb, pa)
    w = 2 * first.half_width_left + 2 * second.half_width_right
    return float(1.18 * (second.apex_rt - first.apex_rt) / w) if w > 0 else float("nan")


def classify_pair(a: ObservedSpecies, b: ObservedSpecies, analysis: SpeciesAnalysis,
                  cfg: CandidateConfig) -> dict:
    """Measured differences, criteria and class for one pair of observed species."""
    if b.rt_apex < a.rt_apex or (b.rt_apex == a.rt_apex and b.mobility < a.mobility):
        a, b = b, a
    same_peak = a.rt_peak == b.rt_peak
    delta_rt = b.rt_apex - a.rt_apex
    resolution = float("nan") if same_peak else _resolution(a, b, analysis)
    shoulder = "shoulder" in (a.kind, b.kind)
    rt_resolved = (not same_peak and not shoulder and np.isfinite(resolution)
                   and resolution >= cfg.rt_min_resolution)
    rt_ok = rt_resolved and delta_rt >= cfg.rt_min_delta
    if np.isfinite(a.mobility) and np.isfinite(b.mobility):
        delta_mob = b.mobility - a.mobility
        ref = (a.mobility + b.mobility) / 2
        rel_pct = 100 * abs(delta_mob) / ref
        prec = cfg.trace.mobility
        im_ok = (rel_pct >= cfg.mobility_min_rel_pct
                 and abs(delta_mob) / ref >= prec.precision_multiple * prec.precision_rel)
    else:
        delta_mob = rel_pct = float("nan")
        im_ok = False
    cls = "C" if rt_ok and im_ok else "A" if rt_ok else "B" if im_ok else "D"
    if same_peak:
        rt_sep = "same RT peak"
    elif shoulder:
        rt_sep = "shoulder"
    elif not np.isfinite(resolution):
        rt_sep = "unknown"
    elif resolution >= cfg.trace.rt.baseline_resolution:
        rt_sep = "baseline-resolved"
    elif resolution >= cfg.trace.rt.partial_resolution:
        rt_sep = "partially overlapping"
    else:
        rt_sep = "coeluting"
    criteria = {
        "rt_resolution": {"value": resolution, "threshold": cfg.rt_min_resolution,
                          "met": bool(rt_resolved)},
        "delta_rt": {"value": delta_rt, "threshold": cfg.rt_min_delta,
                     "met": bool(delta_rt >= cfg.rt_min_delta)},
        "mobility_rel_pct": {"value": rel_pct, "threshold": cfg.mobility_min_rel_pct,
                             "met": bool(im_ok)},
    }
    notes = []
    if cls == "D":
        if rt_resolved and delta_rt < cfg.rt_min_delta:
            notes.append(f"resolved in RT but dRT {delta_rt:.2f} min is below the "
                         f"{cfg.rt_min_delta:g}-min screening threshold")
        if shoulder:
            notes.append("one species is a shoulder without its own apex")
        if not rt_resolved and not same_peak and not shoulder:
            notes.append(f"RT peaks not resolved (Rs {resolution:.2f})")
    return {"a": a, "b": b, "delta_rt": float(delta_rt), "rt_resolution": resolution,
            "rt_separation": rt_sep, "delta_mobility": float(delta_mob),
            "mobility_rel_diff_pct": float(rel_pct), "classification": CLASSES[cls],
            "criteria": criteria, "notes": notes}


def _evidence(pair: dict) -> tuple[str, str, list[str]]:
    """(category, sequence status, exclusion reasons) from the PSM support of a pair."""
    a, b = pair["a"], pair["b"]
    keys_a = set(a.mod_keys)
    keys_b = set(b.mod_keys) if b is not None else set()
    reasons = []
    if keys_a and keys_b and keys_a.isdisjoint(keys_b):
        reasons.append("isobaric, not isomeric: the species are identified as different "
                       f"sequences or modification states ({sorted(keys_a)} vs {sorted(keys_b)})")
    if pair["classification"] == CLASSES["D"]:
        category = "Unresolved candidate"
    elif keys_a and keys_b and keys_a == keys_b and len(keys_a) == 1:
        category = "Sequence-supported candidate"
    else:
        category = "Mass-matched candidate"
    if keys_a and keys_b and keys_a == keys_b and len(keys_a) == 1:
        status = "both species identified as the same sequence"
    elif keys_a or keys_b:
        status = "only one species has a sequence assignment" if not (keys_a and keys_b) \
            else "ambiguous sequence assignments"
    else:
        status = "no sequence assignment (unidentified mass-matched species)"
    return category, status, reasons


def _record(route, group_id, z, mono_mz, pair, cfg, extra_flags=(), key=None) -> dict:
    a, b = pair["a"], pair["b"]
    category, seq_status, reasons = _evidence(pair)
    flags = list(extra_flags) + [f"A: {f}" for f in a.flags]
    if b is not None:
        flags += [f"B: {f}" for f in b.flags]
        total = a.signal + b.signal
        frac = a.signal / total if total > 0 else float("nan")
        if np.isfinite(frac) and min(frac, 1 - frac) < cfg.min_relative_area:
            flags.append(f"low-abundance partner ({min(frac, 1 - frac):.1%} of the signal)")
    else:
        frac = float("nan")
    flags += pair["notes"]
    if not reasons and category != "Unresolved candidate":
        flags.append("high-confidence requires MS/MS validation (phase 4)")
    return {
        "route": route, "group": group_id, "charge": int(z), "mono_mz": float(mono_mz),
        "neutral_mass": float((mono_mz - PROTON) * z), "mod_key": key,
        "classification": pair["classification"], "category": category,
        "sequence_status": seq_status,
        "delta_rt": pair["delta_rt"], "rt_resolution": pair["rt_resolution"],
        "rt_separation": pair["rt_separation"], "delta_mobility": pair["delta_mobility"],
        "mobility_rel_diff_pct": pair["mobility_rel_diff_pct"],
        "mobility_reference": "mean of the two population centres",
        "relative_signal_fraction_a": frac,
        "species_a": a.to_dict(), "species_b": b.to_dict() if b is not None else None,
        "criteria": pair["criteria"], "flags": flags,
        "status": "excluded" if reasons else "candidate", "exclusion_reasons": reasons,
    }


def _pairs_for(route, group_id, z, mono_mz, analysis, species, cfg, extra_flags=(), key=None,
               psm_clusters: int | None = None):
    rows = []
    for a, b in combinations(species, 2):
        rows.append(_record(route, group_id, z, mono_mz, classify_pair(a, b, analysis, cfg), cfg,
                            extra_flags, key))
    if len(species) == 1:
        s = species[0]
        why = [f for f in s.flags if f.startswith("mobility:") or "broad" in f]
        if psm_clusters and psm_clusters >= 2:
            why.append(f"PSMs fall into {psm_clusters} RT clusters but the trace shows one "
                       "peak: two elution events cannot be confirmed")
        if why:
            pair = {"a": s, "b": None, "delta_rt": float("nan"), "rt_resolution": float("nan"),
                    "rt_separation": "single species", "delta_mobility": float("nan"),
                    "mobility_rel_diff_pct": float("nan"), "classification": CLASSES["D"],
                    "criteria": {}, "notes": why}
            rows.append(_record(route, group_id, z, mono_mz, pair, cfg, extra_flags, key))
    return rows


# ---- routes -----------------------------------------------------------------------------


def discover_mass_first(cache, entry, features: pd.DataFrame, psms: pd.DataFrame | None = None,
                        cfg: CandidateConfig = CandidateConfig()) -> list[dict]:
    """Candidates from accurate-mass groups of detected features."""
    f = features[features["charge"].notna()].copy()
    if f.empty:
        return []
    f["mass_group"] = group_by_mass(f, cfg.grouping)
    f["artifact_flags"] = artifact_flags(f, cfg.grouping)
    rows = []
    for (g, z), sub in f.groupby(["mass_group", "charge"]):
        if len(sub) < 2 and not cfg.analyze_single_feature_groups:
            continue
        mono = float(np.average(sub["mz"], weights=sub["area"]))
        analysis = analyze_species(cache, entry, mono, int(z), cfg.trace)
        species = observe_species(analysis)
        _attach_features(species, sub)
        if psms is not None:
            near = psms[(np.abs(psms["precursor_mz"] - mono) <= mono * cfg.match_ppm * 1e-6)
                        & (psms["charge"] == z)]
            _support(species, near, cfg)
        artifact = {int(r.feature_id): r.artifact_flags for r in sub.itertuples()}
        for row in _pairs_for("mass-first", int(g), z, mono, analysis, species, cfg):
            for side in ("species_a", "species_b"):
                sp = row[side]
                if sp is None:
                    continue
                for fid in sp["feature_ids"]:
                    for flag in artifact.get(fid, []):
                        label = f"{side[-1].upper()}: {flag}"
                        if is_exclusionary(flag):
                            row["exclusion_reasons"].append(label)
                            row["status"] = "excluded"
                        elif label not in row["flags"]:
                            row["flags"].append(label)
            rows.append(row)
    return rows


def discover_sequence_first(cache, entry, psms: pd.DataFrame,
                            cfg: CandidateConfig = CandidateConfig()
                            ) -> tuple[list[dict], pd.DataFrame]:
    """Candidates from identified peptides; returns (candidates, excluded PSMs)."""
    kept, excluded = filter_psms(psms, cfg.sequence)
    rows = []
    for (key, z), g in kept.groupby(["mod_key", "charge"]):
        calc = g["calculated_neutral_mass"].median()
        mono = float(calc / z + PROTON) if np.isfinite(calc) else float(g["precursor_mz"].median())
        analysis = analyze_species(cache, entry, mono, int(z), cfg.trace)
        species = observe_species(analysis)
        near = kept[(np.abs(kept["precursor_mz"] - mono) <= mono * cfg.match_ppm * 1e-6)
                    & (kept["charge"] == z)]
        _support(species, near, cfg)
        clusters = int(_rt_clusters(g["rt"].to_numpy(float), cfg.sequence.psm_rt_gap).max()) + 1
        flags = []
        if re.search(cfg.contaminant_pattern, str(g["protein"].iloc[0]), re.IGNORECASE):
            flags.append(f"contaminant protein ({g['protein'].iloc[0]})")
        rows += _pairs_for("sequence-first", key, z, mono, analysis, species, cfg, flags, key,
                           psm_clusters=clusters)
    return rows, excluded


def _same_candidate(r1: dict, r2: dict, cfg: CandidateConfig) -> bool:
    if r1["charge"] != r2["charge"]:
        return False
    if abs(r1["neutral_mass"] - r2["neutral_mass"]) / r1["neutral_mass"] * 1e6 > cfg.match_ppm:
        return False
    for side in ("species_a", "species_b"):
        s1, s2 = r1[side], r2[side]
        if (s1 is None) != (s2 is None):
            return False
        if s1 is None:
            continue
        if abs(s1["rt_apex"] - s2["rt_apex"]) > 0.05:
            return False
        if np.isfinite(s1["mobility"]) and np.isfinite(s2["mobility"]) and \
                abs(s1["mobility"] - s2["mobility"]) > 0.003:
            return False
    return True


def _post_checks(rows: list[dict], cfg: CandidateConfig) -> None:
    """Checks that need the whole candidate list: peak splitting, charge consistency."""
    paired = [r for r in rows if r["species_b"] is not None]
    for r in paired:
        if r["classification"] not in (CLASSES["A"], CLASSES["C"]):
            continue
        similar = [o for o in paired if o is not r and o["classification"] in
                   (CLASSES["A"], CLASSES["C"])
                   and abs(o["delta_rt"] - r["delta_rt"]) <= cfg.split_pattern_rt
                   and abs(o["neutral_mass"] - r["neutral_mass"]) > 1.0]
        if len(similar) >= cfg.split_pattern_min:
            r["flags"].append(f"possible chromatographic peak splitting: {len(similar)} other "
                              f"candidates show the same dRT ({r['delta_rt']:.2f} min)")
        for o in paired:
            if o is r or o["charge"] == r["charge"]:
                continue
            if abs(o["neutral_mass"] - r["neutral_mass"]) / r["neutral_mass"] * 1e6 <= cfg.match_ppm \
                    and abs(o["delta_rt"] - r["delta_rt"]) <= 0.1:
                r["flags"].append(f"consistent at charge {o['charge']}+ "
                                  f"({o['classification']})")


def discover(cache, entry, features: pd.DataFrame | None = None,
             psms: pd.DataFrame | None = None, cfg: CandidateConfig = CandidateConfig()
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both routes, merged. Returns (candidates, excluded PSMs with reasons)."""
    rows: list[dict] = []
    excluded = pd.DataFrame()
    if psms is not None and len(psms):
        seq_rows, excluded = discover_sequence_first(cache, entry, psms, cfg)
        # Peptides of equal mass are analysed separately but observe the same
        # species pair; keep one candidate and list every sequence key.
        for r in seq_rows:
            twin = next((s for s in rows if _same_candidate(s, r, cfg)), None)
            if twin is None:
                r["mod_keys"] = [r["mod_key"]]
                rows.append(r)
            elif r["mod_key"] not in twin["mod_keys"]:
                twin["mod_keys"].append(r["mod_key"])
    if features is not None and len(features):
        kept = filter_psms(psms, cfg.sequence)[0] if psms is not None and len(psms) else None
        for r in discover_mass_first(cache, entry, features, kept, cfg):
            twin = next((s for s in rows if s["route"] in ("sequence-first", "both")
                         and _same_candidate(s, r, cfg)), None)
            if twin is None:
                rows.append(r)
                continue
            twin["route"] = "both"
            twin["group"] = f"{twin['group']} / mass group {r['group']}"
            for flag in r["flags"]:
                if flag not in twin["flags"]:
                    twin["flags"].append(flag)
            for reason in r["exclusion_reasons"]:
                if reason not in twin["exclusion_reasons"]:
                    twin["exclusion_reasons"].append(reason)
                    twin["status"] = "excluded"
    for r in rows:
        r.setdefault("mod_keys", [r["mod_key"]] if r["mod_key"] else [])
    _post_checks(rows, cfg)
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(0, "candidate_id", np.arange(len(df)))
    df.attrs["config"] = cfg.to_dict()
    return df, excluded
