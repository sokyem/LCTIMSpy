"""Phase 3: joint RT-IM candidate discovery on a synthetic run with ground truth.

Synthetic species and PSMs test the classification *logic*. They are not
evidence about real isomers (spec §15).
"""

import numpy as np
import pandas as pd
import pytest

from LCTIMSpy.core.io.cache import Cache
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.discovery.candidates import CLASSES, discover
from LCTIMSpy.discovery.features import find_features
from LCTIMSpy.discovery.grouping import artifact_flags, group_by_mass
from LCTIMSpy.discovery.psms import PROTON, PSM_COLUMNS, make_mod_key
from LCTIMSpy.discovery.sequence_first import pepsifinder_candidates
from LCTIMSpy.discovery.synthetic import Scenario, Species, SyntheticFrameSource, write_synthetic_d

S = dict(rt_sigma=0.06)
SPECIES = [
    Species("P1", 1200.600, 2, rt=2.0, mobility=0.900, sequence="PEPTIDEAAK", **S),
    Species("P1'", 1200.600, 2, rt=3.5, mobility=0.902, sequence="PEPTIDEAAK", **S),
    Species("P2", 1500.700, 2, rt=4.5, mobility=1.000, sequence="PEPTIDEBBK", **S),
    Species("P2'", 1500.700, 2, rt=4.5, mobility=1.040, sequence="PEPTIDEBBK", **S),
    Species("P3", 1800.800, 2, rt=5.5, mobility=0.950, sequence="PEPTIDECCK", **S),
    Species("P3'", 1800.800, 2, rt=7.0, mobility=0.990, sequence="PEPTIDECCK", **S),
    Species("P3z3", 1800.800, 3, rt=5.5, mobility=0.800, sequence="PEPTIDECCK", **S),
    Species("P3'z3", 1800.800, 3, rt=7.0, mobility=0.830, sequence="PEPTIDECCK", **S),
    Species("P4", 1350.650, 2, rt=8.0, mobility=0.920, sequence="PEPTIDEDDK", **S),
    Species("P4'", 1350.650, 2, rt=8.6, mobility=0.921, sequence="PEPTIDEDDK", **S),
    Species("P5", 1650.800, 2, rt=9.5, mobility=0.970, sequence="PEPTIDEEEK", **S),
    Species("P5+Na", 1650.800 + 21.981944, 2, rt=9.5, mobility=0.990, **S),
    Species("P6a", 1450.700, 2, rt=10.0, mobility=0.930, sequence="SEQAAAAAAK", **S),
    Species("P6b", 1450.700, 2, rt=11.5, mobility=0.950, sequence="SEQBBBBBBK", **S),
    Species("P8", 1250.620, 2, rt=12.0, mobility=0.910, sequence="PEPTIDEFFK", **S),
    Species("P8'", 1250.620, 2, rt=13.5, mobility=0.910, **S),                # unidentified
]


def synthetic_psms(species, n=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sp in species:
        if not sp.sequence:
            continue
        _, key = make_mod_key(sp.sequence, [])
        for _ in range(n):
            mz = sp.mono_mz * (1 + rng.normal(0, 2e-6))
            rows.append({
                "engine": "synthetic", "run": "s", "spectrum_ref": "", "peptide": sp.sequence,
                "modified_peptide": sp.sequence, "mods": "", "mod_key": key, "charge": sp.charge,
                "precursor_mz": mz, "observed_neutral_mass": (mz - PROTON) * sp.charge,
                "calculated_neutral_mass": sp.neutral_mass, "mass_error_ppm": 0.0,
                "rt": sp.rt + rng.normal(0, 0.02), "mobility": sp.mobility + rng.normal(0, 0.001),
                "mobility_low": np.nan, "mobility_high": np.nan, "ccs": np.nan, "score": 80.0,
                "score_name": "synthetic", "q_value": 0.001, "is_decoy": False, "protein": "P_X"})
    df = pd.DataFrame(rows, columns=PSM_COLUMNS)
    df["charge"] = df["charge"].astype("Int64")
    return df


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidates")
    scenario = Scenario(SPECIES, n_frames=870, seconds_per_frame=1.0)
    cache = Cache(root / "cache")
    entry = extract_to_cache(write_synthetic_d(root, "s.d", scenario), cache=cache,
                             frame_source=SyntheticFrameSource(scenario))
    features, _ = find_features(cache, entry)
    psms = synthetic_psms(SPECIES)
    candidates, excluded = discover(cache, entry, features, psms)
    return candidates, features, psms


def rows_for(candidates, mass, charge=2):
    hit = candidates[(np.abs(candidates["neutral_mass"] - mass) / mass * 1e6 < 10)
                     & (candidates["charge"] == charge)]
    return hit[hit["species_b"].notna()]


def test_rt_separated_isomer_is_class_a_and_sequence_supported(result):
    c = rows_for(result[0], 1200.600)
    assert len(c) == 1
    r = c.iloc[0]
    assert r["classification"] == CLASSES["A"]
    assert r["category"] == "Sequence-supported candidate"
    assert r["delta_rt"] == pytest.approx(1.5, abs=0.03)
    assert r["route"] == "both"                                # found by both routes, merged
    assert r["criteria"]["delta_rt"]["met"] and not r["criteria"]["mobility_rel_pct"]["met"]
    assert "high-confidence requires MS/MS validation (phase 4)" in r["flags"]


def test_mobility_only_isomer_is_class_b_despite_zero_drt(result):
    r = rows_for(result[0], 1500.700).iloc[0]
    assert r["classification"] == CLASSES["B"] and r["rt_separation"] == "same RT peak"
    assert r["mobility_rel_diff_pct"] == pytest.approx(100 * 0.04 / 1.02, abs=0.2)


def test_joint_isomer_is_class_c_and_consistent_across_charges(result):
    z2 = rows_for(result[0], 1800.800, 2).iloc[0]
    z3 = rows_for(result[0], 1800.800, 3).iloc[0]
    assert z2["classification"] == z3["classification"] == CLASSES["C"]
    assert any(f.startswith("consistent at charge 3+") for f in z2["flags"])


def test_resolved_but_below_screening_threshold_is_unresolved_not_dropped(result):
    r = rows_for(result[0], 1350.650).iloc[0]
    assert r["classification"] == CLASSES["D"] and r["category"] == "Unresolved candidate"
    assert r["rt_separation"] == "baseline-resolved"
    assert any("below the 1-min screening threshold" in f for f in r["flags"])


def test_single_species_makes_no_candidate(result):
    assert rows_for(result[0], 1650.800).empty


def test_different_sequences_of_equal_mass_are_excluded_with_reason(result):
    c = rows_for(result[0], 1450.700)
    assert len(c) == 1                                    # one candidate, both sequences listed
    r = c.iloc[0]
    assert r["status"] == "excluded"
    assert "isobaric, not isomeric" in r["exclusion_reasons"][0]
    assert sorted(r["mod_keys"]) == ["SEQAAAAAAK|", "SEQBBBBBBK|"]


def test_pure_gaussian_mobility_is_not_called_asymmetric(result):
    for r in result[0].itertuples():
        assert not any("asymmetric" in f for f in r.flags), (r.neutral_mass, r.flags)


def test_identity_is_not_transferred_to_an_unidentified_partner(result):
    r = rows_for(result[0], 1250.620).iloc[0]
    assert r["classification"] == CLASSES["A"]
    assert r["category"] == "Mass-matched candidate"
    assert r["sequence_status"] == "only one species has a sequence assignment"
    assert {r["species_a"]["psm_count"] > 0, r["species_b"]["psm_count"] > 0} == {True, False}


def test_adduct_is_flagged_on_the_feature(result):
    _, features, _ = result
    f = features[features["charge"].notna()].copy()
    f["mass_group"] = group_by_mass(f)
    flags = artifact_flags(f)
    adduct = f[np.abs(f["neutral_mass"] - (1650.8 + 21.981944)) < 0.01]
    assert len(adduct) == 1
    assert any("Na+ adduct" in x for x in flags[adduct.index[0]])


def test_every_candidate_is_traceable(result):
    for r in result[0].itertuples():
        assert r.status in ("candidate", "excluded")
        assert isinstance(r.flags, list) and isinstance(r.exclusion_reasons, list)
        if r.species_b is not None:
            for crit in r.criteria.values():
                assert {"value", "threshold", "met"} <= set(crit)


def test_engine_finds_what_the_published_rule_misses(result):
    candidates, _, psms = result
    published = set(pepsifinder_candidates(psms, rt_tol=1.0, ims_diff=0.04)["peptide"])
    assert published == {"PEPTIDECCK"}                          # only the RT+IM pair
    found = {k.split("|")[0] for k in candidates.loc[
        (candidates["status"] == "candidate") & candidates["classification"].isin(
            [CLASSES["A"], CLASSES["B"], CLASSES["C"]]), "mod_key"].dropna()}
    assert {"PEPTIDEAAK", "PEPTIDEBBK", "PEPTIDECCK", "PEPTIDEFFK"} <= found
