"""Sequence-first route: filtering, PepSIFinder compatibility, PSM groups, annotation."""

import numpy as np
import pandas as pd
import pytest

from LCTIMSpy.discovery.psms import PROTON, make_mod_key
from LCTIMSpy.discovery.sequence_first import (
    SequenceFirstConfig,
    annotate_features,
    filter_psms,
    pepsifinder_candidates,
    psm_groups,
)


def psm(peptide, z, rt, mob, mz=None, mods=(), run="run1", decoy=False, q=np.nan, score=50.0):
    mods_str, key = make_mod_key(peptide, list(mods))
    mz = mz if mz is not None else 500.0 + len(peptide)
    return {"engine": "peaks", "run": run, "spectrum_ref": "", "peptide": peptide,
            "modified_peptide": peptide, "mods": mods_str, "mod_key": key, "charge": z,
            "precursor_mz": mz, "observed_neutral_mass": (mz - PROTON) * z,
            "calculated_neutral_mass": (mz - PROTON) * z, "mass_error_ppm": 0.0, "rt": rt,
            "mobility": mob, "mobility_low": np.nan, "mobility_high": np.nan, "ccs": np.nan,
            "score": score, "score_name": "-10lgP", "q_value": q, "is_decoy": decoy,
            "protein": "P1"}


def frame(rows):
    df = pd.DataFrame(rows)
    df["charge"] = df["charge"].astype("Int64")
    return df


def published_get_candidates(df, RT_tol, ims_diff):
    """PepSIFinder.isomer_finder.get_candidates (github.com/sokyem/PepSIFinder),
    verbatim logic on its own column names, as the oracle."""
    df = df.copy()
    frac_df = df.groupby(["Peptide", "z"])["1/k0"].agg(
        lambda x: (x.max() - x.min()) / x.mean()).reset_index(name="frac_variation")
    df = df.merge(frac_df, on=["Peptide", "z"], how="left")
    rt = df.groupby("Peptide")["RT"].agg(["min", "max"]).reset_index()
    rt["max_RT_diff"] = rt["max"] - rt["min"]
    df = df.merge(rt, on="Peptide", how="left")
    c = df[(df["frac_variation"] > ims_diff) & (df["max_RT_diff"] > RT_tol)]
    return set(c.drop_duplicates(subset=["Peptide"])["Peptide"])


def test_pepsifinder_mode_reproduces_the_published_rule():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(40):
        pep = f"PEPTIDE{chr(65 + i % 20)}{'K' * (i // 20)}"
        for _ in range(rng.integers(2, 6)):
            rows.append(psm(pep, int(rng.integers(2, 4)), float(rng.uniform(10, 14)),
                            float(rng.uniform(0.85, 0.98))))
    df = frame(rows)
    ours = set(pepsifinder_candidates(df, rt_tol=1.0, ims_diff=0.04)["peptide"])
    theirs = published_get_candidates(
        df.rename(columns={"mod_key": "Peptide", "charge": "z", "rt": "RT", "mobility": "1/k0"}),
        RT_tol=1.0, ims_diff=0.04)
    assert {k.split("|")[0] for k in theirs} == ours
    assert 0 < len(ours) < 40                                 # the fixture exercises both sides


def test_pepsifinder_needs_both_criteria():
    df = frame([psm("RTONLYPEP", 2, 10.0, 0.900), psm("RTONLYPEP", 2, 12.0, 0.901),
                psm("IMONLYPEP", 2, 10.0, 0.90), psm("IMONLYPEP", 2, 10.1, 0.96),
                psm("BOTHPEPTD", 2, 10.0, 0.90), psm("BOTHPEPTD", 2, 12.0, 0.96)])
    assert set(pepsifinder_candidates(df, 1.0, 0.04)["peptide"]) == {"BOTHPEPTD"}


def test_filtering_records_a_reason_for_every_exclusion():
    df = frame([psm("GOODPEPTIDE", 2, 10, 0.9), psm("DECOYPEPT", 2, 10, 0.9, decoy=True),
                psm("POORQPEPT", 2, 10, 0.9, q=0.2), psm("SHRT", 2, 10, 0.9)])
    kept, excluded = filter_psms(df)
    assert list(kept["peptide"]) == ["GOODPEPTIDE"]
    assert dict(zip(excluded["peptide"], excluded["reason"])) == {
        "DECOYPEPT": "decoy", "POORQPEPT": "q-value > 0.01", "SHRT": "shorter than 5 residues"}


def test_psm_groups_split_elution_events_and_keep_states_apart():
    rows = [psm("GDYFDVEW", 2, rt, 0.91) for rt in (40.0, 40.1, 40.2)]
    rows += [psm("GDYFDVEW", 2, rt, 0.93) for rt in (41.6, 41.7)]
    rows += [psm("GDYFDVEW", 2, 41.0, 0.92, mods=[(2, "+14.02")])]          # methyl-Asp
    groups = psm_groups(frame(rows))
    plain = groups[groups["mod_key"] == "GDYFDVEW|"]
    assert list(plain["n_psms"]) == [3, 2] and set(plain["n_clusters"]) == {2}
    assert plain.iloc[1]["rt_first"] == pytest.approx(41.6)
    assert len(groups[groups["mod_key"] == "GDYFDVEW|2:+14.02"]) == 1


def test_annotation_respects_rt_mobility_and_never_transfers_identity():
    features = pd.DataFrame({
        "feature_id": [0, 1, 2], "mz": [600.30, 600.30, 800.4], "charge": [2, 2, pd.NA],
        "rt_start": [9.9, 11.9, 5.0], "rt_end": [10.1, 12.1, 5.2],
        "mobility_apex": [0.90, 0.93, 0.95], "mobility_low": [0.89, 0.92, 0.94],
        "mobility_high": [0.91, 0.94, 0.96]})
    features["charge"] = features["charge"].astype("Int64")
    psms = frame([psm("ISOMERPEPK", 2, 10.0, 0.90, mz=600.302),
                  psm("ISOMERPEPK", 2, 10.05, 0.95, mz=600.302),             # wrong mobility
                  psm("OTHERPEPTK", 2, 10.0, 0.90, mz=600.301)])
    out = annotate_features(features, psms, SequenceFirstConfig())
    f0, f1, f2 = out.iloc[0], out.iloc[1], out.iloc[2]
    assert f0["psm_count"] == 2 and f0["sequence_ambiguous"]
    assert set(f0["sequences"]) == {"ISOMERPEPK|", "OTHERPEPTK|"}
    assert f1["psm_count"] == 0 and pd.isna(f1["best_mod_key"])              # no transfer
    assert f2["psm_count"] == 0                                                # unknown charge
