"""Species re-measured from the raw cache: RT peaks and mobility populations together."""

import pytest

from LCTIMSpy.core.io.cache import Cache
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.discovery.synthetic import (
    PROTON,
    Scenario,
    Species,
    SyntheticFrameSource,
    write_synthetic_d,
)
from LCTIMSpy.discovery.traces import analyze_species, species_mobilogram

SPECIES = [
    Species("A", 1200.600, 2, rt=2.00, mobility=0.900, rt_sigma=0.06),
    Species("A-iso", 1200.600, 2, rt=3.00, mobility=0.930, rt_sigma=0.06),    # RT + IM
    Species("B", 1500.700, 2, rt=4.00, mobility=1.000, rt_sigma=0.06),
    Species("B-iso", 1500.700, 2, rt=4.00, mobility=1.040, rt_sigma=0.06),    # IM only
    Species("C", 2100.000, 3, rt=5.00, mobility=0.850, rt_sigma=0.06),
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("traces")
    scenario = Scenario(SPECIES, n_frames=330, seconds_per_frame=1.0)
    cache = Cache(root / "cache")
    entry = extract_to_cache(write_synthetic_d(root, "s.d", scenario), cache=cache,
                             frame_source=SyntheticFrameSource(scenario))
    return cache, entry


def mono(mass, z):
    return mass / z + PROTON


def test_rt_separated_isomers_give_two_resolved_peaks_each_one_population(run):
    cache, entry = run
    a = analyze_species(cache, entry, mono(1200.6, 2), 2)
    peaks = [p for p in a.rt_peaks if p.kind == "peak"]
    assert [round(p.apex_rt, 1) for p in peaks] == [2.0, 3.0]
    assert a.rt_pairs[0].separation == "baseline-resolved"
    centres = [m.populations[0].centre for m in a.mobility]
    assert [m.n_populations for m in a.mobility] == [1, 1]
    assert centres == pytest.approx([0.900, 0.930], abs=0.002)
    assert a.coordinate == "1/K0"


def test_mobility_only_isomers_split_inside_one_rt_peak(run):
    cache, entry = run
    b = analyze_species(cache, entry, mono(1500.7, 2), 2)
    peaks = [p for p in b.rt_peaks if p.kind == "peak"]
    assert len(peaks) == 1 and peaks[0].apex_rt == pytest.approx(4.0, abs=0.03)
    m = b.mobility[0]
    assert m.n_populations == 2, (m.verdict, m.flags)
    assert [p.centre for p in m.populations] == pytest.approx([1.000, 1.040], abs=0.002)
    assert m.relative_difference_pct == pytest.approx(100 * 0.04 / 1.02, abs=0.2)


def test_single_species_is_one_peak_one_population(run):
    cache, entry = run
    c = analyze_species(cache, entry, mono(2100.0, 3), 3)
    assert len([p for p in c.rt_peaks if p.kind == "peak"]) == 1
    assert c.mobility[0].n_populations == 1
    assert c.mobility[0].populations[0].centre == pytest.approx(0.850, abs=0.002)


def test_mobilogram_window_and_serialization(run):
    cache, entry = run
    x, y = species_mobilogram(cache, entry, mono(1500.7, 2), 2, 3.9, 4.1)
    assert len(x) > 20 and y.max() > 0
    d = analyze_species(cache, entry, mono(1500.7, 2), 2).to_dict()
    assert d["mobility"][0]["n_populations"] == 2 and d["coordinate"] == "1/K0"
