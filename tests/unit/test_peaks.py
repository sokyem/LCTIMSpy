"""RT peak detection and mobility-population detection on shapes with known answers."""

import math

import numpy as np
import pytest
from scipy.stats import skewnorm

from LCTIMSpy.discovery.peaks import (
    MobilityConfig,
    detect_mobility_populations,
    detect_rt_peaks,
    rt_pairs,
)

RT = np.arange(0, 10, 0.01)                    # 10 min at 0.6 s per point
SIGMA = 0.05                                    # FWHM 0.118 min


def gauss(x, mu, s, a=1e5):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def two(delta, h2=1.0):
    return gauss(RT, 5.0, SIGMA) + gauss(RT, 5.0 + delta, SIGMA, 1e5 * h2)


def delta_for(rs):
    return rs * 2 * 2.3548 * SIGMA / 1.18


# ---- retention time ----------------------------------------------------------------------


def test_single_peak_area_width_and_symmetry():
    peaks = detect_rt_peaks(RT, gauss(RT, 4.0, SIGMA))
    assert len(peaks) == 1
    p = peaks[0]
    assert p.apex_rt == pytest.approx(4.0, abs=0.01)
    assert p.area == pytest.approx(1e5 * SIGMA * math.sqrt(2 * math.pi), rel=0.03)
    assert p.fwhm == pytest.approx(2.3548 * SIGMA, rel=0.05)
    assert p.asymmetry == pytest.approx(1.0, abs=0.15)
    assert p.start_rt < 4.0 - 2 * SIGMA and p.end_rt > 4.0 + 2 * SIGMA and p.flags == []


# Rs read from a fused profile is approximate: the tolerance widens with overlap.
@pytest.mark.parametrize("rs,expected,rel", [(2.0, "baseline-resolved", 0.10),
                                             (1.0, "partially overlapping", 0.15),
                                             (0.6, "coeluting", 0.30)])
def test_pair_classification_by_resolution(rs, expected, rel):
    y = two(delta_for(rs))
    peaks = detect_rt_peaks(RT, y)
    assert [p.kind for p in peaks] == ["peak", "peak"]
    pair = rt_pairs(peaks, RT, y)[0]
    assert pair.separation == expected
    assert pair.resolution == pytest.approx(rs, rel=rel)


def test_resolved_pair_areas_are_split_correctly():
    peaks = detect_rt_peaks(RT, two(delta_for(2.5), h2=0.4))
    expected = 1e5 * SIGMA * math.sqrt(2 * math.pi)
    assert peaks[0].area == pytest.approx(expected, rel=0.05)
    assert peaks[1].area == pytest.approx(0.4 * expected, rel=0.05)


def test_hidden_shoulder_is_reported():
    y = gauss(RT, 5.0, SIGMA) + gauss(RT, 5.0 + 2.2 * SIGMA, SIGMA, 3.5e4)
    peaks = detect_rt_peaks(RT, y)
    kinds = [p.kind for p in peaks]
    assert kinds.count("peak") == 1 and kinds.count("shoulder") == 1
    shoulder = next(p for p in peaks if p.kind == "shoulder")
    assert shoulder.apex_rt > 5.0 and "no local maximum" in shoulder.flags[0]
    assert rt_pairs(peaks, RT, y)[0].separation == "shoulder"


def test_tailing_peak_stays_one_peak():
    tau = 0.15                                  # exponential tail (EMG)
    kernel = np.exp(-np.arange(0, 1.5, 0.01) / tau)
    y = np.convolve(gauss(RT, 4.0, SIGMA), kernel / kernel.sum())[:len(RT)]
    peaks = detect_rt_peaks(RT, y)
    assert [p.kind for p in peaks] == ["peak"]
    assert peaks[0].asymmetry > 1.3


def test_noise_does_not_split_a_peak():
    rng = np.random.default_rng(1)
    y = gauss(RT, 6.0, SIGMA) + rng.normal(0, 1500, len(RT)).clip(0) + 500
    peaks = detect_rt_peaks(RT, y)
    assert len([p for p in peaks if p.kind == "peak"]) == 1


def test_peak_at_the_edge_is_flagged():
    peaks = detect_rt_peaks(RT, gauss(RT, 0.02, SIGMA))
    assert any("edge" in f for f in peaks[0].flags)


def test_empty_and_flat_inputs():
    assert detect_rt_peaks(RT, np.zeros(len(RT))) == []
    assert detect_rt_peaks(RT[:3], np.ones(3)) == []


# ---- ion mobility --------------------------------------------------------------------------

MOB = np.arange(0.95, 1.10, 0.0005)


def noisy(y, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(y + rng.normal(0, 0.01 * y.max(), len(y)), 0, None)


def test_two_separated_populations():
    y = noisy(gauss(MOB, 1.00, 0.006) + gauss(MOB, 1.04, 0.006, 6e4))
    r = detect_mobility_populations(MOB, y)
    assert r.n_populations == 2 and r.verdict == "two populations"
    assert [p.centre for p in r.populations] == pytest.approx([1.00, 1.04], abs=0.001)
    assert r.relative_difference_pct == pytest.approx(100 * 0.04 / 1.02, abs=0.1)
    assert r.reference == "mean of the two centres" and r.coordinate == "1/K0"
    assert r.populations[0].fraction == pytest.approx(1 / 1.6, abs=0.03)
    assert r.resolution > 1.0


def test_one_gaussian_is_one_population():
    r = detect_mobility_populations(MOB, noisy(gauss(MOB, 1.02, 0.006)))
    assert r.n_populations == 1
    assert r.populations[0].centre == pytest.approx(1.02, abs=0.0005)
    assert r.populations[0].resolving_power == pytest.approx(1.02 / (2.3548 * 0.006), rel=0.05)


def test_asymmetric_single_peak_is_not_split():
    y = noisy(1e3 * skewnorm.pdf(MOB, 6, loc=1.0, scale=0.015))
    r = detect_mobility_populations(MOB, y)
    assert r.n_populations == 1, r.bic
    assert any("asymmetric" in f for f in r.flags)


def test_broad_distribution_is_flagged_not_split():
    r = detect_mobility_populations(MOB, noisy(gauss(MOB, 1.02, 0.02)))
    assert r.n_populations == 1 and any("broad" in f for f in r.flags)


def test_close_pair_is_not_claimed_as_two():
    r = detect_mobility_populations(MOB, noisy(gauss(MOB, 1.00, 0.006) + gauss(MOB, 1.006, 0.006)))
    assert r.n_populations == 1


def test_negligible_minor_component_is_one_population():
    y = noisy(gauss(MOB, 1.00, 0.006) + gauss(MOB, 1.05, 0.006, 2e3))
    r = detect_mobility_populations(MOB, y)
    assert r.n_populations == 1 and "dominant" in r.verdict


def test_separation_below_instrument_precision_is_not_an_isomer():
    x = np.arange(0.98, 1.03, 0.0002)
    y = noisy(gauss(x, 1.000, 0.0012) + gauss(x, 1.010, 0.0012))          # well resolved, 1%
    r = detect_mobility_populations(x, y, MobilityConfig(precision_rel=0.005))
    assert r.n_populations == 1 and r.verdict == "separation below instrument precision"
    loose = detect_mobility_populations(x, y, MobilityConfig(precision_rel=0.001))
    assert loose.n_populations == 2


def test_reference_choice_changes_only_the_percentage():
    y = noisy(gauss(MOB, 1.00, 0.006) + gauss(MOB, 1.04, 0.006))
    mean = detect_mobility_populations(MOB, y, MobilityConfig(reference="mean"))
    lower = detect_mobility_populations(MOB, y, MobilityConfig(reference="lower"))
    assert lower.relative_difference_pct == pytest.approx(4.0, abs=0.1)
    assert mean.relative_difference_pct < lower.relative_difference_pct


def test_insufficient_data_and_placeholder_flag():
    r = detect_mobility_populations(MOB[:4], np.ones(4))
    assert r.n_populations == 0 and r.verdict == "insufficient data"
    ok = detect_mobility_populations(MOB, noisy(gauss(MOB, 1.02, 0.006)))
    assert any("placeholder" in f for f in ok.flags)
