"""Data-only isomer evidence: no PSMs, only the raw MS1/MS2 of a synthetic DDA-PASEF run.

Synthetic species test the evidence *logic*; they are not evidence about real
isomers (spec §15).
"""

import json

import numpy as np
import pytest

from LCTIMSpy.core.io.cache import Cache
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.discovery.candidates import CLASSES
from LCTIMSpy.discovery.evidence import EvidenceConfig, compare_envelopes, summarize_support
from LCTIMSpy.discovery.pipeline import run_discovery
from LCTIMSpy.discovery.report import to_csv
from LCTIMSpy.discovery.synthetic import Scenario, Species, SyntheticFrameSource, write_synthetic_d

S = dict(rt_sigma=0.08)
DIFFERENT_PATTERN = (1.0, 0.25, 0.55, 0.10)     # not averagine: another composition
SPECIES = [
    # RT-separated pair, seen again at 3+
    Species("D1", 1200.600, 2, rt=1.5, mobility=0.900, sequence="PEPTIDEAAK", **S),
    Species("D1'", 1200.600, 2, rt=3.0, mobility=0.901, sequence="PEPTIDEAAK", **S),
    Species("D1z3", 1200.600, 3, rt=1.5, mobility=0.750, sequence="PEPTIDEAAK", abundance=5e4, **S),
    Species("D1'z3", 1200.600, 3, rt=3.0, mobility=0.751, sequence="PEPTIDEAAK", abundance=5e4, **S),
    # mobility-separated pair (same RT), split seen again at 3+
    Species("D2", 1500.700, 2, rt=5.0, mobility=1.000, sequence="PEPTIDECCK", **S),
    Species("D2'", 1500.700, 2, rt=5.0, mobility=1.040, sequence="PEPTIDECCK", **S),
    Species("D2z3", 1500.700, 3, rt=5.0, mobility=0.820, sequence="PEPTIDECCK", abundance=6e4, **S),
    Species("D2'z3", 1500.700, 3, rt=5.0, mobility=0.855, sequence="PEPTIDECCK", abundance=6e4, **S),
    # RT-separated pair at one charge only; MS/MS is the corroborating line
    Species("D3", 1300.650, 2, rt=7.0, mobility=0.920, sequence="PEPTIDEGGK", **S),
    Species("D3'", 1300.650, 2, rt=8.5, mobility=0.921, sequence="PEPTIDEGGK", **S),
    # same mass, different isotope distribution: isobaric, not isomeric
    Species("X", 1400.700, 2, rt=10.0, mobility=0.960, sequence="PEPTIDEFFK", **S),
    Species("X'", 1400.700, 2, rt=11.5, mobility=0.962, sequence="PEPTIDEWWK",
            isotope_pattern=DIFFERENT_PATTERN, **S),
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidence")
    scenario = Scenario(SPECIES, n_frames=780, seconds_per_frame=1.0, msms_every=3,
                        isolation_scans=30)
    cache = Cache(root / "cache")
    entry = extract_to_cache(write_synthetic_d(root, "s.d", scenario), cache=cache,
                             frame_source=SyntheticFrameSource(scenario))
    meta = run_discovery(cache, entry, root / "out")           # no PSM file at all
    records = json.loads((root / "out" / "candidates.json").read_text(encoding="utf-8"))
    return meta, records


def pair(records, mass, charge=2):
    hit = [r for r in records if abs(r["neutral_mass"] - mass) / mass * 1e6 < 10
           and r["charge"] == charge and r["species_b"] is not None]
    assert len(hit) == 1, [(r["classification"], r["species_a"]["rt_apex"]) for r in hit]
    return hit[0]


def line(r, name):
    return next(x for x in r["data_support"]["lines"] if x["name"] == name)


def test_rt_split_is_supported_by_isotopes_mass_and_a_second_charge(run):
    r = pair(run[1], 1200.600)
    assert r["route"] == "mass-first" and r["classification"] == CLASSES["A"]
    iso = r["data_evidence"]["isotopes"]
    assert iso["verdict"] == "agree" and iso["cosine"] > 0.98
    assert iso["mass_verdict"] == "agree" and abs(iso["mono_mz_diff_ppm"]) < 5
    ch = r["data_evidence"]["charge_states"]
    assert ch["verdict"] == "reproduced"
    assert [s["charge"] for s in ch["states"] if s["reproduced"]] == [3]
    assert r["data_support"]["level"] == "strong"
    assert "RT split reproduced at charge 3+" in r["flags"]


def test_mobility_split_reproduced_at_another_charge(run):
    r = pair(run[1], 1500.700)
    assert r["classification"] == CLASSES["B"]
    ch = r["data_evidence"]["charge_states"]
    assert ch["shared_rt_window"] and ch["verdict"] == "reproduced"
    z3 = next(s for s in ch["states"] if s["charge"] == 3)
    assert z3["a"]["n_populations"] == 2
    assert ch["distribution"] is None          # one RT window: charge shares are not attributable
    assert r["data_support"]["level"] == "strong"


def test_single_charge_pair_is_corroborated_by_msms_without_any_identification(run):
    r = pair(run[1], 1300.650)
    assert r["data_evidence"]["charge_states"]["verdict"] == "single charge state"
    assert r["msms_status"] == "similar, sequence not confirmed"
    assert line(r, "MS/MS")["supports"] is True
    assert r["data_support"]["level"] == "strong"
    assert r["category"] == "Mass-matched candidate"    # no identification: never upgraded


def test_different_isotope_distribution_is_weak_and_explained(run):
    r = pair(run[1], 1400.700)
    iso = r["data_evidence"]["isotopes"]
    assert iso["verdict"] == "differ" and iso["cosine"] < 0.95
    assert line(r, "Isotope distribution")["supports"] is False
    assert r["data_support"]["level"] == "weak"
    assert any(f.startswith("isotope distributions of A and B differ") for f in r["flags"])
    assert any(f.startswith("B: isotope envelope deviates from averagine") for f in r["flags"])


def test_counts_and_report_carry_the_data_support(run):
    meta, records = run
    assert meta["counts"]["by_data_support"]["strong"] >= 3
    assert meta["counts"]["by_data_support"]["weak"] >= 1
    assert "evidence_s" in meta["timings"]
    csv = to_csv(records)
    assert "data_support" in csv.splitlines()[0] and "3+(split)" in csv


def test_envelope_comparison_needs_two_isotopes():
    weak = {"intensities": [10, 0, 0, 0], "relative": [1, 0, 0, 0], "mono_mz_measured": np.nan,
            "n_points": 1}
    good = {"intensities": [10, 7, 3, 1], "relative": [1, .7, .3, .1], "mono_mz_measured": 600.0,
            "n_points": 40}
    out, flags = compare_envelopes(weak, good, 1198.0, 600.0, EvidenceConfig())
    assert out["verdict"] == "insufficient signal" and out["mass_verdict"] == "insufficient signal"
    assert flags == ["isotope envelope too weak to compare for species A"]


def test_summary_is_none_for_single_species():
    assert summarize_support({"species_b": None, "data_evidence": None}) is None
