"""Sequence-first discovery: start from search-engine identifications.

Steps (each traceable -- nothing is dropped without a recorded reason):

1. ``filter_psms``          decoys, q-value, minimum length (the paper used >= 5
                            residues: shorter peptides are not identified reliably)
2. ``pepsifinder_candidates`` the published PepSIFinder rule, reproduced exactly
                            for continuity with Okyem et al. (2025)
3. ``psm_groups``           per (run, sequence + modification state, charge):
                            PSM-level RT clusters, i.e. separate elution events
                            as sampled by MS/MS
4. ``annotate_features``    attach PSMs to detected MS1 features (m/z, charge,
                            RT and mobility containment), which joins this route
                            to the mass-first one

PSM-level RT and mobility spread is a *sampling* of the signal, not the signal:
one broad or tailing peak can spread PSMs over more than a minute. Phase 2
replaces it with peak detection on the raw traces; this module keeps the
PSM-level view for comparison and for PepSIFinder compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceFirstConfig:
    min_length: int = 5
    max_q_value: float | None = 0.01       # applied only where the engine reports q-values
    exclude_decoys: bool = True
    psm_rt_gap: float = 0.5                 # minutes between PSMs that splits an RT cluster
    # Feature annotation tolerances
    match_ppm: float = 10.0
    rt_margin: float = 0.05                 # minutes added to feature RT bounds
    mobility_margin_rel: float = 0.01       # relative margin on feature mobility bounds

    def to_dict(self) -> dict:
        return asdict(self)


def filter_psms(psms: pd.DataFrame, cfg: SequenceFirstConfig = SequenceFirstConfig()
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(kept, excluded)``; ``excluded`` carries a ``reason`` per PSM."""
    reasons = pd.Series("", index=psms.index, dtype=object)

    def mark(mask, why):
        fresh = mask & (reasons == "")
        reasons[fresh] = why

    if cfg.exclude_decoys:
        mark(psms["is_decoy"].astype(bool), "decoy")
    if cfg.max_q_value is not None:
        q = pd.to_numeric(psms["q_value"], errors="coerce")
        mark(q.notna() & (q > cfg.max_q_value), f"q-value > {cfg.max_q_value}")
    mark(psms["peptide"].str.len() < cfg.min_length, f"shorter than {cfg.min_length} residues")
    mark(psms["charge"].isna(), "no charge state")
    mark(psms["rt"].isna(), "no retention time")
    excluded = psms[reasons != ""].assign(reason=reasons[reasons != ""])
    return psms[reasons == ""].copy(), excluded


def pepsifinder_candidates(psms: pd.DataFrame, rt_tol: float = 1.0,
                           ims_diff: float = 0.04) -> pd.DataFrame:
    """The published PepSIFinder rule, reproduced exactly.

    Per peptide (sequence + modifications): RT spread = max - min PSM RT over
    *all* charge states; mobility variation = (max - min) / mean of PSM 1/K0
    per (peptide, charge). A peptide is a candidate when *both* exceed their
    thresholds (strictly). One row per peptide, as in the original.

    Kept for reproducibility of published results. Its known limits -- PSM
    sampling instead of peaks, RT pooled across charges, both criteria
    required -- are what the multidimensional engine addresses.
    """
    df = psms.copy()
    frac = df.groupby(["mod_key", "charge"])["mobility"].agg(
        lambda x: (x.max() - x.min()) / x.mean()).rename("frac_variation")
    df = df.join(frac, on=["mod_key", "charge"])
    rt = df.groupby("mod_key")["rt"].agg(["min", "max"]).rename(
        columns={"min": "first_RT", "max": "last_RT"})
    rt["max_RT_diff"] = rt["last_RT"] - rt["first_RT"]
    df = df.join(rt, on="mod_key")
    hit = df[(df["frac_variation"] > ims_diff) & (df["max_RT_diff"] > rt_tol)]
    hit = hit.drop_duplicates(subset=["mod_key"]).reset_index(drop=True)
    return hit[["mod_key", "peptide", "modified_peptide", "charge", "first_RT", "last_RT",
                "max_RT_diff", "mobility", "frac_variation", "precursor_mz",
                "observed_neutral_mass", "spectrum_ref", "protein"]]


def _rt_clusters(rt: np.ndarray, gap: float) -> np.ndarray:
    order = np.argsort(rt)
    labels = np.empty(len(rt), dtype=np.int64)
    labels[order] = np.cumsum(np.concatenate(([0], np.diff(rt[order]) > gap)))
    return labels


def psm_groups(psms: pd.DataFrame, cfg: SequenceFirstConfig = SequenceFirstConfig()
               ) -> pd.DataFrame:
    """One row per (run, mod_key, charge, RT cluster) with PSM-level RT/mobility stats."""
    rows = []
    for (run, key, z), g in psms.groupby(["run", "mod_key", "charge"], sort=False):
        labels = _rt_clusters(g["rt"].to_numpy(float), cfg.psm_rt_gap)
        n_clusters = int(labels.max()) + 1
        for c in range(n_clusters):
            sub = g[labels == c]
            mob = sub["mobility"].dropna()
            rows.append({
                "run": run, "mod_key": key, "peptide": sub["peptide"].iloc[0], "charge": int(z),
                "cluster": c, "n_clusters": n_clusters, "n_psms": len(sub),
                "rt_first": float(sub["rt"].min()), "rt_last": float(sub["rt"].max()),
                "rt_median": float(sub["rt"].median()),
                "mobility_median": float(mob.median()) if len(mob) else np.nan,
                "mobility_min": float(mob.min()) if len(mob) else np.nan,
                "mobility_max": float(mob.max()) if len(mob) else np.nan,
                "precursor_mz": float(sub["precursor_mz"].median()),
                "observed_neutral_mass": float(sub["observed_neutral_mass"].median()),
                "calculated_neutral_mass": float(sub["calculated_neutral_mass"].median()),
                "best_score": float(sub["score"].max()) if sub["score"].notna().any() else np.nan,
                "protein": sub["protein"].iloc[0],
            })
    return pd.DataFrame(rows)


def annotate_features(features: pd.DataFrame, psms: pd.DataFrame,
                      cfg: SequenceFirstConfig = SequenceFirstConfig()) -> pd.DataFrame:
    """Attach PSMs to the features they fall inside; never transfer identities.

    A PSM supports a feature when its precursor m/z matches the feature's
    monoisotopic m/z (``match_ppm``), its charge equals the feature charge, its
    RT lies within the feature's RT bounds (plus ``rt_margin``) and -- when both
    have mobility -- its 1/K0 lies within the feature's mobility bounds (plus
    ``mobility_margin_rel``). Adds ``psm_count``, ``sequences`` (sorted
    mod_keys), ``best_mod_key`` (most PSMs, then best score) and
    ``sequence_ambiguous`` (more than one mod_key). Features with no PSM keep
    an empty identity.
    """
    out = features.copy()
    counts, seqs, best, ambiguous = [], [], [], []
    p_mz = psms["precursor_mz"].to_numpy(float)
    p_z = psms["charge"].astype("Float64").to_numpy(float, na_value=np.nan)
    p_rt = psms["rt"].to_numpy(float)
    p_mob = psms["mobility"].to_numpy(float)
    order = np.argsort(p_mz)
    sorted_mz = p_mz[order]
    for f in out.itertuples():
        z = f.charge
        if pd.isna(z):
            counts.append(0), seqs.append([]), best.append(None), ambiguous.append(False)
            continue
        tol = f.mz * cfg.match_ppm * 1e-6
        lo, hi = np.searchsorted(sorted_mz, [f.mz - tol, f.mz + tol])
        idx = order[lo:hi]
        ok = (p_z[idx] == float(z)) & (p_rt[idx] >= f.rt_start - cfg.rt_margin) \
            & (p_rt[idx] <= f.rt_end + cfg.rt_margin)
        if np.isfinite(f.mobility_low) and np.isfinite(f.mobility_high):
            m = p_mob[idx]
            margin = cfg.mobility_margin_rel * f.mobility_apex
            ok &= ~np.isfinite(m) | ((m >= f.mobility_low - margin) & (m <= f.mobility_high + margin))
        hits = psms.iloc[idx[ok]]
        counts.append(len(hits))
        keys = sorted(hits["mod_key"].unique())
        seqs.append(keys)
        ambiguous.append(len(keys) > 1)
        if keys:
            tally = hits.groupby("mod_key").agg(n=("mod_key", "size"), s=("score", "max"))
            best.append(tally.sort_values(["n", "s"], ascending=False).index[0])
        else:
            best.append(None)
    out["psm_count"] = counts
    out["sequences"] = seqs
    out["best_mod_key"] = best
    out["sequence_ambiguous"] = ambiguous
    return out
