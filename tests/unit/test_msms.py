"""Phase 4: MS/MS validation on a synthetic DDA-PASEF run, plus the spectral primitives.

Synthetic spectra test the logic (association, ambiguity, similarity, grading);
they are not evidence about real isomers (spec §15).
"""

import numpy as np
import pandas as pd
import pytest

from LCTIMSpy.core.fragments import calculate_fragment_mz
from LCTIMSpy.core.io.cache import Cache
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.discovery.candidates import CLASSES, discover
from LCTIMSpy.discovery.features import find_features
from LCTIMSpy.discovery.fragments import fragment_ladder
from LCTIMSpy.discovery.msms import (
    MsmsConfig,
    annotate,
    match_peaks,
    preprocess,
    similarity,
    validate_msms,
)
from LCTIMSpy.discovery.psms import PROTON, PSM_COLUMNS, make_mod_key
from LCTIMSpy.discovery.synthetic import Scenario, Species, SyntheticFrameSource, write_synthetic_d

S = dict(rt_sigma=0.08)
SPECIES = [
    Species("H1", 1200.600, 2, rt=1.5, mobility=0.900, sequence="PEPTIDEAAK", **S),
    Species("H1'", 1200.600, 2, rt=3.0, mobility=0.901, sequence="PEPTIDEAAK", **S),
    Species("H2", 1300.650, 2, rt=4.5, mobility=0.920, sequence="PEPTIDEGGK", **S),
    Species("H2'", 1300.650, 2, rt=6.0, mobility=0.921, sequence="PEPTIDEGGK", msms_variant=1.2, **S),
    Species("H3", 1500.700, 2, rt=7.5, mobility=1.000, sequence="PEPTIDECCK", **S),
    Species("H3'", 1500.700, 2, rt=7.5, mobility=1.040, sequence="PEPTIDECCK", **S),
    Species("H4", 1600.750, 2, rt=9.0, mobility=1.000, sequence="PEPTIDEDDK", mobility_sigma=0.004, **S),
    Species("H4'", 1600.750, 2, rt=9.0, mobility=1.025, sequence="PEPTIDEDDK", mobility_sigma=0.004, **S),
    Species("H5", 1700.800, 2, rt=10.5, mobility=0.950, sequence="PEPTIDEEEK", **S),
    Species("H5'", 1700.800, 2, rt=12.0, mobility=0.951, sequence="PEPTIDEEEK", **S),   # no PSMs
    Species("H6", 1400.700, 2, rt=13.5, mobility=0.960, sequence="PEPTIDEFFK", **S),
    Species("H6'", 1400.700, 2, rt=15.0, mobility=0.961, sequence="PEPTIDEFFK", **S),
    Species("Y", 1401.500, 2, rt=13.5, mobility=0.962, sequence="SEQYYYYYYK", **S),  # co-isolated
]
NO_PSMS = {"H5'"}


def synthetic_psms(species, n=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sp in species:
        if not sp.sequence or sp.name in NO_PSMS:
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
def validated(tmp_path_factory):
    root = tmp_path_factory.mktemp("msms")
    scenario = Scenario(SPECIES, n_frames=1000, seconds_per_frame=1.0, msms_every=3,
                        isolation_scans=30)
    cache = Cache(root / "cache")
    entry = extract_to_cache(write_synthetic_d(root, "s.d", scenario), cache=cache,
                             frame_source=SyntheticFrameSource(scenario))
    features, _ = find_features(cache, entry)
    candidates, _ = discover(cache, entry, features, synthetic_psms(SPECIES))
    return validate_msms(cache, entry, candidates, features)


def row_for(c, mass, charge=2):
    hit = c[(np.abs(c["neutral_mass"] - mass) / mass * 1e6 < 10) & (c["charge"] == charge)
            & c["species_b"].notna()]
    assert len(hit) == 1, hit[["neutral_mass", "classification", "status"]]
    return hit.iloc[0]


def test_identical_fragmentation_gives_high_confidence(validated):
    r = row_for(validated, 1200.600)
    assert r["classification"] == CLASSES["A"]
    assert r["msms_status"] == "supporting" and r["msms_cosine"] > 0.9
    assert r["category"] == "High-confidence putative candidate"
    ann = r["msms"]["annotation"]["PEPTIDEAAK|"]
    assert ann["a"]["fraction"] > 0.5 and ann["b"]["fraction"] > 0.5
    assert r["msms"]["mirror"]["matches"]
    assert not any("phase 4" in f for f in r["flags"])


def test_different_fragmentation_does_not_exclude_isomerization(validated):
    r = row_for(validated, 1300.650)
    assert r["status"] == "candidate"
    assert r["msms_status"] == "dissimilar"
    assert r["category"] == "Sequence-supported candidate"
    assert any("does not exclude isomerization" in f for f in r["flags"])
    assert len(r["msms"]["differential_fragments"]) == 5


def test_mobility_pair_with_clean_isolation_is_unambiguous(validated):
    r = row_for(validated, 1500.700)
    assert r["classification"] == CLASSES["B"]
    assert r["msms"]["a"]["n_unambiguous"] > 0 and r["msms"]["b"]["n_unambiguous"] > 0
    assert r["msms_status"] == "supporting"


def test_isolation_spanning_both_populations_is_ambiguous(validated):
    r = row_for(validated, 1600.750)
    assert r["classification"] == CLASSES["B"]
    assert r["msms_status"] == "ambiguous"
    assert any("spans both populations" in f for f in r["flags"])
    assert r["category"] != "High-confidence putative candidate"


def test_unidentified_partner_supported_by_msms_not_by_psm(validated):
    r = row_for(validated, 1700.800)
    assert r["category"] == "Sequence-supported candidate"
    assert r["sequence_status"] == "only one species has a sequence assignment"
    assert any("not by a database PSM" in f for f in r["flags"])


def test_coisolation_is_reported(validated):
    r = row_for(validated, 1400.700)
    assert r["msms_status"] == "ambiguous"
    assert any("co-isolated with feature" in f for f in r["flags"])


def test_candidates_without_msms_keep_their_category(tmp_path):
    from tests.unit.test_candidates import SPECIES as NO_MSMS_SPECIES
    from tests.unit.test_candidates import synthetic_psms as psms_for

    scenario = Scenario(NO_MSMS_SPECIES[:2], n_frames=300, seconds_per_frame=1.0)
    cache = Cache(tmp_path / "cache")
    entry = extract_to_cache(write_synthetic_d(tmp_path, "s.d", scenario), cache=cache,
                             frame_source=SyntheticFrameSource(scenario))
    features, _ = find_features(cache, entry)
    candidates, _ = discover(cache, entry, features, psms_for(NO_MSMS_SPECIES[:2]))
    out = validate_msms(cache, entry, candidates, features)
    assert (out["msms_status"] == "no MS/MS in this run").all()
    assert list(out["category"]) == list(candidates["category"])


# ---- primitives ---------------------------------------------------------------------------


def test_fragment_ladder_matches_the_core_calculator():
    mz, labels = fragment_ladder("PEPTIDEK")
    core = calculate_fragment_mz("PEPTIDEK", (1,))
    np.testing.assert_allclose(mz[:7], core["b1+"], atol=1e-6)
    np.testing.assert_allclose(mz[7:], sorted(core["y1+"]), atol=1e-6)
    assert labels[0] == "b1" and labels[-1] == "y7"


def test_modifications_shift_the_right_ions():
    plain, _ = fragment_ladder("GDYFDVEW")
    methyl, _ = fragment_ladder("GDYFDVEW", "2:+14.02")
    n = 7
    assert methyl[0] == pytest.approx(plain[0])                  # b1 (G) unchanged
    assert methyl[1] == pytest.approx(plain[1] + 14.02)          # b2 carries the methyl-Asp
    assert methyl[n + 5] == pytest.approx(plain[n + 5])          # y6 does not contain residue 2
    assert methyl[n + 6] == pytest.approx(plain[n + 6] + 14.02)  # y7 does


def test_similarity_and_matching():
    cfg = MsmsConfig()
    mz = np.array([200.1, 300.2, 450.3, 600.4, 750.5])
    it = np.array([10.0, 50.0, 100.0, 30.0, 5.0])
    a = preprocess(mz, it, cfg)
    assert similarity(a, a, cfg)["cosine"] == pytest.approx(1.0)
    b = preprocess(mz + 3.0, it, cfg)
    assert similarity(a, b, cfg)["cosine"] == 0.0
    shifted = preprocess(mz * (1 + 5e-6), it, cfg)                   # within 20 ppm
    assert similarity(a, shifted, cfg)["n_matched"] == 5
    assert match_peaks(np.array([100.0, 100.001]), np.array([100.0005]),
                       np.array([1.0, 2.0]), np.array([1.0]), cfg) == [(1, 0)]   # one-to-one


def test_annotation_explains_a_pure_ladder():
    cfg = MsmsConfig()
    mz, _ = fragment_ladder("PEPTIDEAAK")
    spec = preprocess(mz, np.ones(len(mz)), cfg)
    ann = annotate(spec, "PEPTIDEAAK", "", 2, cfg)
    assert ann["fraction"] == pytest.approx(1.0) and ann["n_ions"] == len(mz)
    bad = annotate(spec, "PEPTIDEAAK", "3:oxidation", 2, cfg)
    assert np.isnan(bad["fraction"]) and "cannot annotate" in bad["note"]
