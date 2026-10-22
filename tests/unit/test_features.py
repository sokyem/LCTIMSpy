"""Untargeted MS1 feature detection on synthetic runs with known ground truth."""

import numpy as np
import pytest

from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.discovery.features import FeatureFinderConfig, assemble_features, find_features
from LCTIMSpy.discovery.synthetic import (
    NEUTRON,
    Scenario,
    Species,
    SyntheticFrameSource,
    averagine_pattern,
    write_synthetic_d,
)

SPECIES = [
    Species("A", 1200.600, 2, rt=2.00, mobility=0.900, rt_sigma=0.06),
    Species("A-iso", 1200.600, 2, rt=3.00, mobility=0.930, rt_sigma=0.06),     # RT + IM
    Species("B", 1500.700, 2, rt=4.00, mobility=1.000, rt_sigma=0.06),
    Species("B-iso", 1500.700, 2, rt=4.00, mobility=1.040, rt_sigma=0.06),     # IM only
    Species("C3", 2100.000, 3, rt=5.00, mobility=0.850, rt_sigma=0.06, abundance=6e4),
    Species("C2", 2100.000, 2, rt=5.00, mobility=1.050, rt_sigma=0.06, abundance=4e4),
]


@pytest.fixture(scope="module")
def detected(tmp_path_factory):
    from LCTIMSpy.core.io.cache import Cache

    root = tmp_path_factory.mktemp("features")
    scenario = Scenario(SPECIES, n_frames=330, seconds_per_frame=1.0)
    d = write_synthetic_d(root, "synthetic.d", scenario)
    cache = Cache(root / "cache")
    entry = extract_to_cache(d, cache=cache, frame_source=SyntheticFrameSource(scenario))
    features, traces = find_features(cache, entry, FeatureFinderConfig())
    return features, traces


def match(features, sp, rt_tol=0.03, mob_tol=0.004, ppm=5.0):
    f = features[features["charge"].notna()]
    hit = f[(f["charge"] == sp.charge)
            & (np.abs(f["neutral_mass"] - sp.neutral_mass) / sp.neutral_mass * 1e6 <= ppm)
            & (np.abs(f["rt_apex"] - sp.rt) <= rt_tol)
            & (np.abs(f["mobility_apex"] - sp.mobility) <= mob_tol)]
    return hit


@pytest.mark.parametrize("sp", SPECIES, ids=[s.name for s in SPECIES])
def test_every_species_is_found_once_with_correct_charge_and_mass(detected, sp):
    features, _ = detected
    hit = match(features, sp)
    assert len(hit) == 1, f"{sp.name}: {len(hit)} matching features"
    row = hit.iloc[0]
    assert row["n_isotopes"] >= 3
    assert row["isotope_cosine"] > 0.9
    assert row["rt_start"] < sp.rt < row["rt_end"]
    assert row["mobility_low"] < sp.mobility < row["mobility_high"]
    assert abs(row["isotope_spacing_error_ppm"]) < 5          # synthetic jitter is 2 ppm


def test_mobility_only_isomers_stay_two_features(detected):
    features, _ = detected
    b, b_iso = match(features, SPECIES[2]), match(features, SPECIES[3])
    assert len(b) == 1 and len(b_iso) == 1
    assert b.iloc[0]["feature_id"] != b_iso.iloc[0]["feature_id"]


def test_noise_makes_no_confident_charged_features(detected):
    features, _ = detected
    charged = features[features["charge"].notna()]
    explained = np.zeros(len(charged), dtype=bool)
    for sp in SPECIES:
        explained |= charged["feature_id"].isin(match(features, sp)["feature_id"]).to_numpy()
    spurious = charged[~explained]
    smallest = min(sp.abundance for sp in SPECIES)
    assert (spurious["height"] < 0.05 * smallest).all(), spurious[["mz", "charge", "height"]]


def test_monoisotopic_peak_is_recovered_when_a_later_isotope_is_most_intense():
    """For heavy peptides the 2nd isotope is most intense; the mono must still win."""
    import pandas as pd

    mass, z = 3500.0, 2
    pattern = averagine_pattern(mass, 5)
    assert pattern.argmax() >= 1
    mono_mz = mass / z + 1.007276466812
    traces = pd.DataFrame({
        "trace_id": range(5), "mz": [mono_mz + k * NEUTRON / z for k in range(5)],
        "mz_ppm_sd": 1.0, "rt_apex": 10.0, "rt_start": 9.9, "rt_end": 10.1, "frames": 10,
        "mobility_apex": 1.1, "mobility_low": 1.09, "mobility_high": 1.11,
        "height": pattern * 1e5, "area": pattern * 1e6, "n_points": 50, "snr": 100.0})
    f = assemble_features(traces, FeatureFinderConfig(), frame_minutes=0.02)
    top = f.iloc[0]
    assert top["charge"] == z and top["n_isotopes"] == 5
    assert top["mz"] == pytest.approx(mono_mz)


def test_unmatched_traces_are_kept_with_unknown_charge():
    import pandas as pd

    traces = pd.DataFrame({
        "trace_id": [0], "mz": [600.0], "mz_ppm_sd": 1.0, "rt_apex": 5.0, "rt_start": 4.9,
        "rt_end": 5.1, "frames": 8, "mobility_apex": 0.9, "mobility_low": 0.89,
        "mobility_high": 0.91, "height": 1e4, "area": 1e5, "n_points": 30, "snr": 50.0})
    f = assemble_features(traces, FeatureFinderConfig(), frame_minutes=0.02)
    assert len(f) == 1 and pd.isna(f.iloc[0]["charge"]) and np.isnan(f.iloc[0]["neutral_mass"])
