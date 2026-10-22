"""Peptide-spectrum matches from database-search engines, normalized to one table.

Supported: PEAKS (PSM CSV), MSFragger/FragPipe (psm.tsv), MaxQuant
(evidence.txt / msms.txt), mzIdentML 1.1/1.2, and mzTab 1.0.

Normalized columns (units fixed, whatever the engine used):

    engine, run, spectrum_ref        provenance
    peptide                          stripped sequence (uppercase, no flanks/mods)
    modified_peptide                 as the engine wrote it
    mods                             canonical "pos:delta" list ("3:+14.02;12:-0.98");
                                     pos 0 = N-terminus, len+1 = C-terminus
    mod_key                          peptide + "|" + mods: same key = same
                                     sequence *and* modification state
    charge
    precursor_mz                     observed (measured)
    observed_neutral_mass            z * (precursor_mz - proton)      [calculated]
    calculated_neutral_mass          from the sequence, by the engine [calculated]
    mass_error_ppm
    rt                               minutes (measured)
    mobility                         1/K0, V·s/cm² (measured), may be missing
    mobility_low, mobility_high      a reported 1/K0 range (PEAKS), may be missing
    ccs                              Å², only if the engine reported it [derived]
    score, score_name, q_value, is_decoy, protein

Sequences are never transferred between PSMs: a PSM's identity is only what
its engine assigned to that spectrum.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.errors import LCTIMSError

PROTON = 1.007276466812

PSM_COLUMNS = [
    "engine", "run", "spectrum_ref", "peptide", "modified_peptide", "mods", "mod_key", "charge",
    "precursor_mz", "observed_neutral_mass", "calculated_neutral_mass", "mass_error_ppm",
    "rt", "mobility", "mobility_low", "mobility_high", "ccs",
    "score", "score_name", "q_value", "is_decoy", "protein",
]

ENGINES = ("peaks", "msfragger", "maxquant", "mzidentml", "mztab")

# Monoisotopic mass deltas for modification names and UNIMOD ids that engines
# write without a mass. Unknown names are kept as names in the key.
MOD_MASSES = {
    "oxidation": 15.994915, "ox": 15.994915,
    "acetyl": 42.010565, "ac": 42.010565,
    "phospho": 79.966331, "ph": 79.966331,
    "deamidation": 0.984016, "deamidated": 0.984016, "de": 0.984016,
    "amidated": -0.984016, "amidation": -0.984016, "am": -0.984016,
    "methyl": 14.015650, "me": 14.015650,
    "dimethyl": 28.031300,
    "carbamidomethyl": 57.021464, "cam": 57.021464,
    "gln->pyro-glu": -17.026549, "glu->pyro-glu": -18.010565, "pyro-glu": -17.026549,
    "sulfo": 79.956815,
}
UNIMOD_MASSES = {1: 42.010565, 2: -0.984016, 4: 57.021464, 7: 0.984016, 21: 79.966331,
                 27: -18.010565, 28: -17.026549, 34: 14.015650, 35: 15.994915, 36: 28.031300,
                 40: 79.956815}


class PsmFormatError(LCTIMSError):
    """A PSM file is not in a recognized format, or lacks required columns."""


# ---- modification canonicalization --------------------------------------------


def _fmt_delta(delta: float | None, name: str | None = None) -> str:
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return (name or "?").strip()
    return f"{delta:+.2f}"


def make_mod_key(peptide: str, mods: list[tuple[int, str]]) -> tuple[str, str]:
    """Canonical ``(mods, mod_key)`` from ``[(position, delta_or_name)]``."""
    mods_str = ";".join(f"{p}:{d}" for p, d in sorted(mods, key=lambda m: (m[0], m[1])))
    return mods_str, f"{peptide}|{mods_str}"


_FLANKS = re.compile(r"^[A-Z\-_]\.(.+)\.[A-Z\-_]$")


def _strip_flanks(seq: str) -> str:
    m = _FLANKS.match(seq)
    return m.group(1) if m else seq


def parse_peaks_peptide(seq: str) -> tuple[str, list[tuple[int, str]]]:
    """PEAKS notation: ``(+42.01)PEPM(+15.99)TIDE`` -> ("PEPMTIDE", [(0,'+42.01'), (4,'+15.99')])."""
    seq = _strip_flanks(seq.strip())
    peptide, mods, pos = [], [], 0
    for token in re.finditer(r"([A-Z])|\(([+-]\d+(?:\.\d+)?)\)", seq):
        if token.group(1):
            peptide.append(token.group(1))
            pos += 1
        else:
            mods.append((pos, _fmt_delta(float(token.group(2)))))
    return "".join(peptide), mods


def parse_msfragger_mods(peptide: str, assigned: str) -> list[tuple[int, str]]:
    """``"5M(15.9949), N-term(42.0106)"`` -> [(5,'+15.99'), (0,'+42.01')]."""
    mods = []
    for part in str(assigned or "").split(","):
        part = part.strip()
        m = re.match(r"(N-term|C-term|\d+)[A-Za-z]?\((-?\d+(?:\.\d+)?)\)", part)
        if not m:
            continue
        where = m.group(1)
        pos = 0 if where == "N-term" else len(peptide) + 1 if where == "C-term" else int(where)
        mods.append((pos, _fmt_delta(float(m.group(2)))))
    return mods


def parse_maxquant_modified(seq: str) -> tuple[str, list[tuple[int, str]]]:
    """``_(Acetyl (Protein N-term))PEPM(Oxidation (M))IDE_`` or ``_PEPM(ox)IDE_``."""
    s = seq.strip().strip("_")
    peptide, mods, pos, i = [], [], 0, 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth, j = 1, i + 1
            while j < len(s) and depth:
                depth += {"(": 1, ")": -1}.get(s[j], 0)
                j += 1
            name = s[i + 1:j - 1]
            base = re.sub(r"\s*\(.*\)$", "", name).strip().lower()
            delta = MOD_MASSES.get(base)
            mods.append((pos, _fmt_delta(delta, base)))
            i = j
        elif c.isalpha():
            peptide.append(c.upper())
            pos += 1
            i += 1
        else:
            i += 1
    return "".join(peptide), mods


# ---- per-engine readers -------------------------------------------------------------


def _read_table(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        head = f.readline()
    sep = "\t" if head.count("\t") > head.count(",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False, encoding="utf-8-sig")


def _require(df: pd.DataFrame, cols: list[str], engine: str, path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise PsmFormatError(
            f"{path.name}: not a {engine} PSM file, missing {missing}. "
            f"Columns found: {list(df.columns)[:25]}")


def _col(df: pd.DataFrame, *names, default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_peaks(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    _require(df, ["Peptide", "RT", "z", "m/z"], "PEAKS", path)
    parsed = df["Peptide"].astype(str).map(parse_peaks_peptide)
    rng = _col(df, "1/k0 Range", "1/K0 Range", default=None).astype("string")
    parts = rng.str.split("-", n=1, expand=True) if rng.notna().any() else None
    low = _num(parts[0]) if parts is not None else np.nan
    high = _num(parts[1]) if parts is not None and parts.shape[1] > 1 else np.nan
    mobility = _num(_col(df, "1/k0", "1/K0"))
    if mobility.isna().all() and parts is not None:
        mobility = (low + high) / 2           # PEAKS reports a range; its midpoint, as PepSIFinder
    return pd.DataFrame({
        "engine": "peaks",
        "run": _col(df, "Source File", "Fraction", default="").astype(str),
        "spectrum_ref": _col(df, "Scan", "Id", default="").astype(str),
        "peptide": parsed.map(lambda t: t[0]),
        "modified_peptide": df["Peptide"].astype(str),
        "_mods": parsed.map(lambda t: t[1]),
        "charge": _num(df["z"]),
        "precursor_mz": _num(df["m/z"]),
        "calculated_neutral_mass": _num(_col(df, "Mass")),
        "mass_error_ppm": _num(_col(df, "ppm")),
        "rt": _num(df["RT"]),
        "mobility": mobility,
        "mobility_low": low,
        "mobility_high": high,
        "ccs": _num(_col(df, "CCS")),
        "score": _num(_col(df, "-10lgP")),
        "score_name": "-10lgP",
        "q_value": np.nan,
        "is_decoy": False,
        "protein": _col(df, "Accession", "Protein Accession", default="").astype(str),
    })


def read_msfragger(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    _require(df, ["Spectrum", "Peptide", "Charge", "Retention"], "MSFragger psm.tsv", path)
    peptide = df["Peptide"].astype(str).str.upper()
    assigned = _col(df, "Assigned Modifications", default="").fillna("").astype(str)
    mods = [parse_msfragger_mods(p, a) for p, a in zip(peptide, assigned)]
    mz = _num(_col(df, "Observed M/Z", "Calibrated Observed M/Z"))
    proteins = _col(df, "Protein", default="").astype(str)
    return pd.DataFrame({
        "engine": "msfragger",
        "run": _col(df, "Spectrum File", default="").astype(str),
        "spectrum_ref": df["Spectrum"].astype(str),
        "peptide": peptide,
        "modified_peptide": _col(df, "Modified Peptide", default="").fillna("").astype(str),
        "_mods": mods,
        "charge": _num(df["Charge"]),
        "precursor_mz": mz,
        "calculated_neutral_mass": _num(_col(df, "Calculated Peptide Mass")),
        "mass_error_ppm": np.nan,
        "rt": _num(df["Retention"]) / 60.0,                 # psm.tsv reports seconds
        "mobility": _num(_col(df, "Ion Mobility")),
        "mobility_low": np.nan,
        "mobility_high": np.nan,
        "ccs": np.nan,
        "score": _num(_col(df, "Hyperscore", "Expectation")),
        "score_name": "Hyperscore" if "Hyperscore" in df.columns else "Expectation",
        "q_value": np.nan,
        "is_decoy": proteins.str.contains(r"rev_|DECOY", regex=True),
        "protein": proteins,
    })


def read_maxquant(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    _require(df, ["Sequence", "Charge", "Retention time", "Raw file"], "MaxQuant", path)
    modseq = _col(df, "Modified sequence", default="").fillna("").astype(str)
    parsed = [parse_maxquant_modified(m) if m else (s, []) for m, s in
              zip(modseq, df["Sequence"].astype(str))]
    return pd.DataFrame({
        "engine": "maxquant",
        "run": df["Raw file"].astype(str),
        "spectrum_ref": _col(df, "MS/MS scan number", "Scan number", "MS/MS Scan Number",
                             default="").astype(str),
        "peptide": [p for p, _ in parsed],
        "modified_peptide": modseq,
        "_mods": [m for _, m in parsed],
        "charge": _num(df["Charge"]),
        "precursor_mz": _num(_col(df, "m/z")),
        "calculated_neutral_mass": _num(_col(df, "Mass")),
        "mass_error_ppm": _num(_col(df, "Mass error [ppm]", "Mass Error [ppm]")),
        "rt": _num(df["Retention time"]),                   # minutes
        "mobility": _num(_col(df, "1/K0", "Ion mobility index")),
        "mobility_low": np.nan,
        "mobility_high": np.nan,
        "ccs": _num(_col(df, "CCS")),
        "score": _num(_col(df, "Score")),
        "score_name": "Score",
        "q_value": np.nan,
        "is_decoy": _col(df, "Reverse", default="").fillna("").astype(str).eq("+"),
        "protein": _col(df, "Leading razor protein", "Proteins", default="").astype(str),
    })


def _unit_minutes(value, unit: str | None) -> float:
    v = float(value)
    u = (unit or "").lower()
    return v / 60.0 if "second" in u or u in ("s", "sec", "uo:0000010") else v


def read_mzidentml(path: Path) -> pd.DataFrame:
    from pyteomics import mzid

    rows = []
    with mzid.read(str(path), retrieve_refs=True, read_schema=False) as reader:
        for result in reader:
            rt = rt_unit = mobility = None
            for key, value in result.items():
                k = key.lower()
                if k in ("scan start time", "retention time"):
                    rt = value
                    # pyteomics attaches the cvParam's unit name as ``unit_info``.
                    rt_unit = str(getattr(value, "unit_info", "") or "")
                elif "inverse reduced ion mobility" in k:
                    mobility = float(value)
            for item in result.get("SpectrumIdentificationItem", []):
                if int(item.get("rank", 1)) != 1:
                    continue
                pep = str(item.get("PeptideSequence", "")).upper()
                mods = []
                for m in item.get("Modification", []) or []:
                    delta = m.get("monoisotopicMassDelta")
                    mods.append((int(m.get("location", 0)),
                                 _fmt_delta(float(delta) if delta is not None else None,
                                            m.get("name"))))
                evidence = item.get("PeptideEvidenceRef", []) or []
                q = score = None
                score_name = None
                for key, value in item.items():
                    k = key.lower()
                    if q is None and ("q-value" in k or "qvalue" in k or "q value" in k):
                        q = float(value)
                    elif score is None and ("score" in k or "expect" in k) and \
                            isinstance(value, (int, float)):
                        score, score_name = float(value), key
                rows.append({
                    "engine": "mzidentml",
                    "run": str(result.get("location", result.get("spectraData_ref", ""))),
                    "spectrum_ref": str(result.get("spectrumID", "")),
                    "peptide": pep,
                    "modified_peptide": pep,
                    "_mods": mods,
                    "charge": item.get("chargeState"),
                    "precursor_mz": item.get("experimentalMassToCharge"),
                    "calculated_neutral_mass": (
                        (float(item["calculatedMassToCharge"]) - PROTON) * int(item["chargeState"])
                        if item.get("calculatedMassToCharge") is not None else np.nan),
                    "mass_error_ppm": np.nan,
                    "rt": _unit_minutes(rt, rt_unit) if rt is not None else np.nan,
                    "mobility": mobility if mobility is not None else np.nan,
                    "mobility_low": np.nan, "mobility_high": np.nan, "ccs": np.nan,
                    "score": score if score is not None else np.nan,
                    "score_name": score_name or "",
                    "q_value": q if q is not None else np.nan,
                    "is_decoy": any(bool(e.get("isDecoy")) for e in evidence),
                    "protein": ";".join(str(e.get("accession", "")) for e in evidence),
                })
    if not rows:
        raise PsmFormatError(f"{path.name}: no rank-1 spectrum identifications found")
    return pd.DataFrame(rows)


def read_mztab(path: Path) -> pd.DataFrame:
    from pyteomics import mztab

    tab = mztab.MzTab(str(path))
    df = tab.spectrum_match_table
    if df is None or len(df) == 0:
        raise PsmFormatError(f"{path.name}: mzTab file has no PSM section")
    df = df.reset_index(drop=True)
    _require(df, ["sequence", "charge", "exp_mass_to_charge", "retention_time"], "mzTab", path)

    def mods_of(seq, text):
        out = []
        for part in str(text or "").split(","):
            m = re.match(r"\s*(\d+)-(?:UNIMOD:(\d+)|\[[^\]]*\]|(.+))", part)
            if not m or part.strip() in ("", "null"):
                continue
            pos = int(m.group(1))
            if m.group(2):
                uid = int(m.group(2))
                out.append((pos, _fmt_delta(UNIMOD_MASSES.get(uid), f"UNIMOD:{uid}")))
            else:
                out.append((pos, _fmt_delta(None, (m.group(3) or "?").strip())))
        return out

    def first_rt(value):
        s = str(value).split("|")[0]
        try:
            return float(s) / 60.0                      # mzTab 1.0: seconds
        except ValueError:
            return np.nan

    mobility_cols = [c for c in df.columns if "ion_mobility" in c.lower()]
    score_col = next((c for c in df.columns if c.startswith("search_engine_score")), None)
    seq = df["sequence"].astype(str).str.upper()
    return pd.DataFrame({
        "engine": "mztab",
        "run": df.get("spectra_ref", pd.Series("", index=df.index)).astype(str)
        .str.split(":").str[0],
        "spectrum_ref": df.get("spectra_ref", pd.Series("", index=df.index)).astype(str),
        "peptide": seq,
        "modified_peptide": seq,
        "_mods": [mods_of(s, m) for s, m in zip(seq, df.get("modifications", [None] * len(df)))],
        "charge": _num(df["charge"]),
        "precursor_mz": _num(df["exp_mass_to_charge"]),
        "calculated_neutral_mass": (_num(_col(df, "calc_mass_to_charge")) - PROTON)
        * _num(df["charge"]),
        "mass_error_ppm": np.nan,
        "rt": df["retention_time"].map(first_rt),
        "mobility": _num(df[mobility_cols[0]]) if mobility_cols else np.nan,
        "mobility_low": np.nan, "mobility_high": np.nan, "ccs": np.nan,
        "score": _num(df[score_col]) if score_col else np.nan,
        "score_name": score_col or "",
        "q_value": np.nan,
        "is_decoy": _col(df, "opt_global_cv_MS:1002217_decoy_peptide", default=0)
        .fillna(0).astype(str).isin(["1", "1.0", "true", "True"]),
        "protein": _col(df, "accession", default="").astype(str),
    })


READERS = {"peaks": read_peaks, "msfragger": read_msfragger, "maxquant": read_maxquant,
           "mzidentml": read_mzidentml, "mztab": read_mztab}


def detect_format(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".mzid",) or p.name.lower().endswith(".mzid.gz"):
        return "mzidentml"
    if suffix == ".mztab":
        return "mztab"
    with open(p, encoding="utf-8-sig", errors="replace") as f:
        head = f.readline()
    cols = {c.strip().strip('"') for c in re.split(r"[\t,]", head)}
    if head.startswith("MTD"):
        return "mztab"
    if {"Spectrum", "Retention"} <= cols and ("Hyperscore" in cols or "Expectation" in cols):
        return "msfragger"
    if {"Raw file", "Sequence"} <= cols:
        return "maxquant"
    if {"Peptide", "RT", "z"} <= cols:
        return "peaks"
    raise PsmFormatError(f"{p.name}: cannot tell which search engine wrote it "
                         f"(first columns: {sorted(cols)[:15]})")


def read_psms(path: str | Path, engine: str | None = None) -> pd.DataFrame:
    """Read a PSM export into the normalized table (see module docstring)."""
    path = Path(path)
    engine = engine or detect_format(path)
    if engine not in READERS:
        raise PsmFormatError(f"unknown engine {engine!r}; choose one of {ENGINES}")
    df = READERS[engine](path)
    df["charge"] = pd.to_numeric(df["charge"], errors="coerce").astype("Int64")
    df["precursor_mz"] = pd.to_numeric(df["precursor_mz"], errors="coerce")
    df["observed_neutral_mass"] = (df["precursor_mz"] - PROTON) * df["charge"].astype(float)
    calc = pd.to_numeric(df["calculated_neutral_mass"], errors="coerce")
    df["calculated_neutral_mass"] = calc
    ppm = pd.to_numeric(df["mass_error_ppm"], errors="coerce")
    derived = (df["observed_neutral_mass"] - calc) / calc * 1e6
    df["mass_error_ppm"] = ppm.where(ppm.notna(), derived)
    keys = [make_mod_key(p, m) for p, m in zip(df["peptide"], df["_mods"])]
    df["mods"] = [k[0] for k in keys]
    df["mod_key"] = [k[1] for k in keys]
    df = df.drop(columns=["_mods"])
    for col in ("rt", "mobility", "mobility_low", "mobility_high", "ccs", "score", "q_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_decoy"] = df["is_decoy"].fillna(False).astype(bool)
    df.attrs.update({"engine": engine, "source": str(path),
                     "units": {"rt": "min", "mobility": "1/K0 (V·s/cm²)", "ccs": "Å²",
                               "masses": "Da (neutral, monoisotopic)"}})
    return df[PSM_COLUMNS]
