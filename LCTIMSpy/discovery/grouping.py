"""Accurate-mass grouping of features and artifact relationships between them (spec §5, §11).

``group_by_mass`` clusters features by neutral monoisotopic mass within a ppm
tolerance (default 5 ppm). A cluster never spans more than the tolerance from
its first member, so a ladder of masses cannot chain unrelated species
together, and only features with a charge (hence a mass) take part.

``artifact_flags`` relates features that coelute (same RT apex) by mass
differences that betray a chemical or assignment artifact rather than a
distinct species. Every relationship is a recorded flag naming the related
feature -- nothing is removed here:

    isotope / monoisotopic error   +1.00336 Da (k = 1, 2), same charge, RT and mobility
    Na+ / K+ / NH4+ adduct          +21.98194 / +37.95588 / +17.02655 Da, same RT
    in-source loss                  -18.01056 (H2O) / -17.02655 (NH3) Da, same RT
    charge-state family             same mass, other charge, same RT (supporting evidence)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .synthetic import NEUTRON

ADDUCTS = {"Na+": 21.981944, "K+": 37.955882, "NH4+": 17.026549}
NEUTRAL_LOSSES = {"H2O": 18.010565, "NH3": 17.026549}


@dataclass(frozen=True)
class GroupingConfig:
    mass_ppm: float = 5.0
    artifact_ppm: float = 10.0
    coelution_rt: float = 0.05             # minutes between apexes to call two features coeluting
    isotope_mobility_rel: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


def group_by_mass(features: pd.DataFrame, cfg: GroupingConfig = GroupingConfig()) -> pd.Series:
    """Accurate-mass group id per feature (``<NA>`` for features without a charge)."""
    groups = pd.Series(pd.NA, index=features.index, dtype="Int64")
    has_mass = features["neutral_mass"].notna()
    masses = features.loc[has_mass, "neutral_mass"].sort_values()
    gid, anchor = -1, None
    for idx, m in masses.items():
        if anchor is None or (m - anchor) / anchor * 1e6 > cfg.mass_ppm:
            gid += 1
            anchor = m
        groups[idx] = gid
    return groups


def _near(values: np.ndarray, target: float, ppm: float) -> np.ndarray:
    return np.flatnonzero(np.abs(values - target) <= abs(target) * ppm * 1e-6)


def artifact_flags(features: pd.DataFrame, cfg: GroupingConfig = GroupingConfig()) -> pd.Series:
    """A list of artifact flags per feature (empty when none apply)."""
    flags = pd.Series([[] for _ in range(len(features))], index=features.index, dtype=object)
    f = features[features["neutral_mass"].notna()]
    if f.empty:
        return flags
    mass = f["neutral_mass"].to_numpy(float)
    rt = f["rt_apex"].to_numpy(float)
    mob = f["mobility_apex"].to_numpy(float)
    z = f["charge"].astype(int).to_numpy()
    fid = f["feature_id"].to_numpy()
    idx = f.index.to_numpy()

    def coeluting(i, j):
        return abs(rt[i] - rt[j]) <= cfg.coelution_rt

    for i in range(len(f)):
        out = []
        for k in (1, 2):
            for j in _near(mass, mass[i] - k * NEUTRON, cfg.artifact_ppm):
                same_mob = (not np.isfinite(mob[i]) or not np.isfinite(mob[j])
                            or abs(mob[i] - mob[j]) <= cfg.isotope_mobility_rel * mob[j])
                if z[j] == z[i] and coeluting(i, j) and same_mob:
                    out.append(f"possible isotope (+{k}) of feature {fid[j]}: "
                               "monoisotopic peak may be misassigned")
        for name, delta in ADDUCTS.items():
            for j in _near(mass, mass[i] - delta, cfg.artifact_ppm):
                if coeluting(i, j):
                    out.append(f"possible {name} adduct of feature {fid[j]}")
        for name, delta in NEUTRAL_LOSSES.items():
            for j in _near(mass, mass[i] + delta, cfg.artifact_ppm):
                if coeluting(i, j):
                    out.append(f"possible in-source {name} loss from feature {fid[j]}")
        for j in _near(mass, mass[i], cfg.mass_ppm):
            if j != i and z[j] != z[i] and coeluting(i, j):
                out.append(f"same species observed at charge {z[j]}+ (feature {fid[j]})")
        flags[idx[i]] = out
    return flags


def is_exclusionary(flag: str) -> bool:
    """Flags that make a feature an artifact of another, not a distinct species."""
    return flag.startswith(("possible isotope", "possible Na+", "possible K+", "possible NH4+",
                            "possible in-source"))
