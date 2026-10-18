"""Phase 2: analysis primitives over the cache, checked against the legacy code.

The legacy functions (``LCTIMSpy.lctims``, ``LCTIMSpy.mobility_analysis``) take
an in-memory DataFrame of points. Every regression test below loads the full
cache into such a frame, runs the legacy function on it, and requires the new
cache-backed code to produce the same numbers. That is the "science-preserving"
check the plan asks for, run on synthetic data until real data is available.
"""

import importlib.util

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from LCTIMSpy import cli  # noqa: E402
from LCTIMSpy import lctims as legacy  # noqa: E402
from LCTIMSpy.core import smoothing, transforms  # noqa: E402
from LCTIMSpy.core.chrom import (  # noqa: E402
    chromatograms,
    chromatograms_from_points,
    find_rt_peaks,
    summed_chromatogram,
)
from LCTIMSpy.core.config import MzTolerance, QueryConfig, RtWindow, StatsConfig  # noqa: E402
from LCTIMSpy.core.errors import LCTIMSError  # noqa: E402
from LCTIMSpy.core.export import write_figure, write_table  # noqa: E402
from LCTIMSpy.core.figures import (  # noqa: E402
    chromatogram_figure,
    mobility_difference_figure,
    mobilogram_figure,
)
from LCTIMSpy.core.fragments import calculate_fragment_mz, target_list  # noqa: E402
from LCTIMSpy.core.io.cache import Cache  # noqa: E402
from LCTIMSpy.core.io.extract import extract_to_cache  # noqa: E402
from LCTIMSpy.core.mobil import mobilograms, mobilograms_from_points  # noqa: E402
from LCTIMSpy.core.mobility_diff import mobility_difference  # noqa: E402
from LCTIMSpy.core.stats import compare_targets, significance_stars  # noqa: E402
from tests.conftest import (  # noqa: E402
    ISOMERS,
    N_FRAMES,
    SECONDS_PER_FRAME,
    TARGETS,
    WINDOWS,
    FakeFrameSource,
    is_ms1,
    make_fake_d,
)

TOL = 0.02
RT_WINDOWS = [RtWindow(*w) for w in WINDOWS]


@pytest.fixture(scope="module")
def replicates(tmp_path_factory):
    """Three replicate datasets, extracted into one cache."""
    root = tmp_path_factory.mktemp("replicates")
    cache = Cache(root / "cache")
    paths, entries = {}, {}
    for i in range(3):
        d = make_fake_d(root / "data", f"rep{i}.d")
        entry = extract_to_cache(d, cache=cache, frame_source=FakeFrameSource(seed=i))
        paths[entry.dataset.name] = d
        entries[entry.dataset.name] = entry
    return cache, entries, paths


@pytest.fixture(scope="module")
def one(replicates):
    cache, entries, _ = replicates
    sid, entry = next(iter(entries.items()))
    return cache, sid, entry


def all_points(cache, entry) -> pd.DataFrame:
    """Every cached point, shaped like the legacy extractor's DataFrame."""
    return cache.load_mz_window(entry, [(0.0, 1e9)])


# ---- regression against the legacy implementation ---------------------------

def test_chromatograms_match_legacy(one):
    cache, _, entry = one
    points = all_points(cache, entry)
    old = legacy.extract_chromatograms(points, list(TARGETS), mz_tolerance=TOL)
    new = chromatograms_from_points(points, TARGETS, MzTolerance(TOL))
    from_cache = chromatograms(cache, entry, TARGETS, MzTolerance(TOL), fill_missing=False)
    assert len(new) == len(from_cache) == len(TARGETS)
    for series, cached in zip(new, from_cache):
        o = old[old.target_mz == series.target_mz]
        np.testing.assert_allclose(series.retention_time, o.retention_time)
        np.testing.assert_allclose(series.intensity, o.intensity)
        np.testing.assert_allclose(cached.intensity, series.intensity)


@pytest.mark.parametrize("sigma", [None, 1.0])
@pytest.mark.parametrize("baseline", [True, False])
def test_mobilograms_match_legacy(one, sigma, baseline):
    cache, _, entry = one
    points = all_points(cache, entry)
    config = QueryConfig(tolerance=MzTolerance(TOL), mobilogram_sigma=sigma,
                         baseline_correction=baseline)
    for target in TARGETS:
        old = legacy.extract_mobilogram(points.copy(), target, WINDOWS, mz_tolerance=TOL,
                                        apply_smoothing=sigma is not None, sigma=sigma or 1,
                                        baseline_correction=baseline)
        new = mobilograms_from_points(points, target, RT_WINDOWS, config)
        assert len(new) == 2
        for s in new:
            o = old[old.rt_range == f"{s.rt_start}-{s.rt_end}"]
            np.testing.assert_allclose(s.mobility, o.mobility)
            np.testing.assert_allclose(s.intensity, o.intensity)


def test_mobility_difference_matches_legacy(replicates):
    from LCTIMSpy.mobility_analysis import extract_and_calculate_mobility_difference

    cache, entries, _ = replicates
    config = QueryConfig(tolerance=MzTolerance(TOL), intensity_threshold=100)
    new = mobility_difference(cache, entries, TARGETS, RT_WINDOWS, config).to_legacy_frame()
    assert len(new) == len(TARGETS) * len(entries)
    for sid, entry in entries.items():
        old = extract_and_calculate_mobility_difference(
            all_points(cache, entry), list(TARGETS), WINDOWS, mz_tolerance=TOL,
            intensity_threshold=100)
        got = new[new.sample_id == sid].drop(columns="sample_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(got, old, check_dtype=False, rtol=1e-6)


def test_mobility_difference_recovers_the_isomer_shift(replicates):
    cache, entries, _ = replicates
    result = mobility_difference(cache, entries, TARGETS, RT_WINDOWS,
                                 QueryConfig(tolerance=MzTolerance(TOL)))
    shift = ISOMERS[1][1] - ISOMERS[0][1]
    by_target = result.differences.groupby("target_mz")["mobility_difference"].mean()
    assert by_target[TARGETS[0]] == pytest.approx(shift, abs=0.001)
    assert by_target[TARGETS[1]] == pytest.approx(shift + 0.004, abs=0.001)


def test_compare_targets_matches_legacy_p_values(replicates, tmp_path):
    from LCTIMSpy.mobility_analysis import compare_targetmz_differences_with_boxplot

    cache, entries, _ = replicates
    dfs = {sid: all_points(cache, e) for sid, e in entries.items()}
    _, old_ttest, _ = compare_targetmz_differences_with_boxplot(
        dfs, list(TARGETS), WINDOWS, mz_tolerance=TOL, save_figure=True,
        file_name=str(tmp_path / "legacy"))
    diffs = mobility_difference(cache, entries, TARGETS, RT_WINDOWS,
                                QueryConfig(tolerance=MzTolerance(TOL), intensity_threshold=None))
    new = compare_targets(diffs.differences, StatsConfig())
    np.testing.assert_allclose(new.pairs.raw_p_value, old_ttest.raw_p_value)
    np.testing.assert_allclose(new.pairs.adjusted_p_value, old_ttest.adjusted_p_value)
    assert new.pairs.reject.tolist() == list(old_ttest.reject)


# ---- behaviour the legacy code did not have ----------------------------------

def test_mobilograms_do_not_mutate_input(one):
    cache, _, entry = one
    points = all_points(cache, entry)
    before = points.copy()
    mobilograms_from_points(points, TARGETS[0], RT_WINDOWS)
    pd.testing.assert_frame_equal(points, before)


def test_zero_filled_chromatograms_share_the_ms1_axis(one):
    cache, _, entry = one
    filled = chromatograms(cache, entry, TARGETS, MzTolerance(TOL))
    sparse = chromatograms(cache, entry, TARGETS, MzTolerance(TOL), fill_missing=False)
    n_ms1 = sum(is_ms1(i) for i in range(1, N_FRAMES + 1))
    for f, s in zip(filled, sparse):
        assert len(f.retention_time) == n_ms1
        assert f.intensity.sum() == pytest.approx(s.intensity.sum())
    assert summed_chromatogram(filled).intensity.sum() == pytest.approx(
        sum(f.intensity.sum() for f in filled))


def test_find_rt_peaks_locates_both_isomers(one):
    cache, _, entry = one
    total = summed_chromatogram(chromatograms(cache, entry, TARGETS, MzTolerance(TOL)))
    peaks = find_rt_peaks(total, n=2)
    assert len(peaks) == 2
    for peak, (apex_frame, _) in zip(peaks, ISOMERS):
        apex_rt = apex_frame * SECONDS_PER_FRAME / 60
        assert peak.apex_rt == pytest.approx(apex_rt, abs=0.02)
        assert peak.start_rt < apex_rt < peak.end_rt


def test_n_windows_give_every_pair(replicates):
    cache, entries, _ = replicates
    windows = RT_WINDOWS + [RtWindow(0.35, 0.45)]
    result = mobility_difference(cache, entries, TARGETS, windows,
                                 QueryConfig(tolerance=MzTolerance(TOL)))
    pairs = set(zip(result.differences.window_a, result.differences.window_b))
    assert pairs == {(0, 1), (0, 2), (1, 2)}
    assert len(result.differences) == 3 * len(TARGETS) * len(entries)
    with pytest.raises(ValueError):
        mobility_difference(cache, entries, TARGETS, RT_WINDOWS[:1])


def test_exclusions_are_reported_with_reasons(replicates):
    cache, entries, _ = replicates
    empty_window = [RT_WINDOWS[0], RtWindow(5.0, 6.0)]        # after the run ends
    r = mobility_difference(cache, entries, TARGETS, empty_window,
                            QueryConfig(tolerance=MzTolerance(TOL)))
    assert r.differences.empty
    assert r.excluded.reason.str.contains("no data in window 1").all()

    r = mobility_difference(cache, entries, TARGETS, RT_WINDOWS,
                            QueryConfig(tolerance=MzTolerance(TOL), intensity_threshold=1e9))
    assert len(r.excluded) == len(TARGETS) * len(entries)
    assert r.excluded.reason.str.contains("below").all()
    assert not r.per_window.included.any()


def test_threshold_on_raw_versus_corrected(one):
    cache, sid, entry = one
    base = mobility_difference(cache, {sid: entry}, TARGETS[:1], RT_WINDOWS,
                               QueryConfig(tolerance=MzTolerance(TOL), intensity_threshold=None))
    pw = base.per_window
    corrected, raw = pw.max_intensity.min(), pw.max_intensity_raw.min()
    assert raw > corrected
    threshold = (corrected + raw) / 2
    on = {}
    for mode in ("corrected", "raw"):
        cfg = QueryConfig(tolerance=MzTolerance(TOL), intensity_threshold=threshold,
                          threshold_on=mode)
        on[mode] = mobility_difference(cache, {sid: entry}, TARGETS[:1], RT_WINDOWS, cfg)
    assert len(on["corrected"].excluded) == 1
    assert len(on["raw"].differences) == 1


def test_rt_offset_shifts_one_sample(replicates):
    cache, entries, _ = replicates
    moved = next(iter(entries))
    r = mobility_difference(cache, entries, TARGETS, RT_WINDOWS,
                            QueryConfig(tolerance=MzTolerance(TOL)), rt_offsets={moved: 10.0})
    assert set(r.excluded.sample_id) == {moved}
    assert moved not in set(r.differences.sample_id)


def test_cache_mobilograms_equal_points_version(one):
    cache, _, entry = one
    cfg = QueryConfig(tolerance=MzTolerance(TOL))
    from_cache = mobilograms(cache, entry, TARGETS, RT_WINDOWS, cfg)
    points = all_points(cache, entry)
    direct = [s for t in TARGETS for s in mobilograms_from_points(points, t, RT_WINDOWS, cfg)]
    assert len(from_cache) == len(direct) == 4
    for a, b in zip(from_cache, direct):
        np.testing.assert_allclose(a.intensity, b.intensity)


# ---- stats -------------------------------------------------------------------

def _groups_frame():
    """Targets in *descending* order of appearance: the case the legacy Tukey got wrong."""
    rng = np.random.default_rng(1)
    rows = []
    for target, mean in ((900.0, 0.050), (700.0, 0.040), (500.0, 0.041)):
        for s in range(4):
            rows.append({"sample_id": f"s{s}", "target_mz": target,
                         "mobility_difference": mean + rng.normal(0, 0.001)})
    return pd.DataFrame(rows)


def test_tukey_p_values_are_matched_to_the_right_pairs():
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    data = _groups_frame()
    result = compare_targets(data, StatsConfig(correction="tukey"))
    direct = pairwise_tukeyhsd(data.mobility_difference, data.target_mz)
    expected = {}
    for (i, j), p in zip(zip(*direct._multicomp.pairindices), direct.pvalues):
        expected[frozenset((direct.groupsunique[i], direct.groupsunique[j]))] = p
    for row in result.pairs.itertuples():
        assert row.adjusted_p_value == pytest.approx(expected[frozenset((row.group_a, row.group_b))])
    # 700 vs 500 is the only non-significant pair.
    ns = result.pairs[~result.pairs.reject]
    assert {frozenset(p) for p in zip(ns.group_a, ns.group_b)} == {frozenset((700.0, 500.0))}


def test_correction_none_and_min_samples():
    data = _groups_frame()
    none = compare_targets(data, StatsConfig(correction="none"))
    np.testing.assert_allclose(none.pairs.adjusted_p_value, none.pairs.raw_p_value)
    with pytest.raises(ValueError, match="at least 5 samples"):
        compare_targets(data, StatsConfig(min_samples=5))


def test_significance_stars():
    assert [significance_stars(p) for p in (1e-5, 5e-4, 5e-3, 0.03, 0.2)] == [
        "****", "***", "**", "*", "ns"]


# ---- smoothing, transforms, fragments --------------------------------------

def test_savgol_matches_legacy_and_gaussian_is_real():
    from LCTIMSpy.plot_chromatograms_mobilograms import _apply_smoothing

    rng = np.random.default_rng(0)
    for n in (2, 4, 5, 20):
        y = pd.Series(rng.uniform(0, 10, n))
        np.testing.assert_allclose(smoothing.smooth(y, "savgol", 5), _apply_smoothing(y, "savgol", 5))
    y = np.zeros(21)
    y[10] = 1.0
    g = smoothing.smooth(y, "gaussian", sigma=2)
    assert g.max() < 1.0 and g.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        smoothing.smooth(y, "boxcar")


def test_normalize_matches_legacy_and_lttb_keeps_shape():
    from LCTIMSpy.plot_chromatograms_mobilograms import normalize_series

    s = pd.Series([3.0, np.nan, 7.0, 5.0])
    np.testing.assert_allclose(transforms.normalize(s, (0, 100)), normalize_series(s, (0, 100)))
    assert (transforms.normalize([2.0, 2.0]) == 0).all()

    x = np.linspace(0, 10, 5000)
    y = np.exp(-((x - 6.3) ** 2) / 0.01)
    xs, ys = transforms.lttb_downsample(x, y, 200)
    assert len(xs) == 200 and xs[0] == x[0] and xs[-1] == x[-1]
    assert ys.max() == pytest.approx(y.max(), rel=0.02)
    assert np.all(np.diff(xs) > 0)


def test_target_list_labels():
    targets = target_list("PEPTIDE", (1,))
    assert len(targets) == 12
    ions = calculate_fragment_mz("PEPTIDE", (1,))
    assert targets[0] == ("b1 (1+)", ions["b1+"][0])
    assert targets[6] == ("y6 (1+)", ions["y1+"][0])
    assert targets[-1] == ("y1 (1+)", ions["y1+"][-1])
    assert target_list("PEPTIDE", (1, 2), ["y2+"])[0][0] == "y6 (2+)"


# ---- figures and export -----------------------------------------------------

def test_figures_build_without_showing(one, monkeypatch):
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda *a, **k: pytest.fail("show() called"))
    cache, sid, entry = one
    eics = chromatograms(cache, entry, TARGETS, MzTolerance(TOL), labels=["y1", "y2"])
    fig = chromatogram_figure(eics, RT_WINDOWS, title=sid)
    assert len(fig.data) == 2 and len(fig.layout.shapes) == 2
    mobs = mobilograms(cache, entry, TARGETS, RT_WINDOWS, QueryConfig(tolerance=MzTolerance(TOL)))
    assert len(mobilogram_figure(mobs).data) == 4

    data = _groups_frame()
    stats = compare_targets(data, StatsConfig())
    box = mobility_difference_figure(stats.data, stats=stats)
    assert len(box.data) == 3
    assert len(box.layout.shapes) == int(stats.pairs.reject.sum())


def test_export_writers(tmp_path):
    import plotly.graph_objects as go

    df = pd.DataFrame({"a": [1, 2]})
    assert pd.read_csv(write_table(df, tmp_path / "t.csv")).equals(df)
    html = write_figure(go.Figure(), tmp_path / "f.html")
    assert "plotly" in html.read_text(encoding="utf-8").lower()
    with pytest.raises(ValueError):
        write_figure(go.Figure(), tmp_path / "f.gif")
    if importlib.util.find_spec("kaleido") is None:
        with pytest.raises(LCTIMSError, match="kaleido"):
            write_figure(go.Figure(), tmp_path / "f.png")


# ---- CLI ------------------------------------------------------------------

def test_cli_analyze_end_to_end(replicates, tmp_path, capsys):
    cache, _, paths = replicates
    out = tmp_path / "out"
    argv = ["--cache-dir", str(cache.root), "analyze", *map(str, paths.values()),
            "--mz", *map(str, TARGETS), "--windows", "0.3:0.5", "0.7:0.9", "--out", str(out)]
    assert cli.main(argv) == 0
    for name in ("targets.csv", "mobility_per_window.csv", "mobility_difference.csv",
                 "excluded.csv", "mobility_difference_legacy_format.csv", "stats_pairs.csv",
                 "mobility_difference.html", "chromatograms_rep0.html", "mobilograms_rep2.html"):
        assert (out / name).is_file(), name
    legacy_fmt = pd.read_csv(out / "mobility_difference_legacy_format.csv")
    assert len(legacy_fmt) == len(TARGETS) * 3
    assert "significant" in capsys.readouterr().out


def test_cli_analyze_auto_windows(replicates, tmp_path, capsys):
    cache, _, paths = replicates
    argv = ["--cache-dir", str(cache.root), "analyze", *map(str, paths.values()),
            "--mz", *map(str, TARGETS), "--windows", "auto", "--out", str(tmp_path / "o")]
    assert cli.main(argv) == 0
    assert "auto windows" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "o" / "mobility_difference.csv")) == len(TARGETS) * 3


def test_cli_analyze_requires_extraction(tmp_path, capsys):
    d = make_fake_d(tmp_path, "new.d")
    argv = ["--cache-dir", str(tmp_path / "cache"), "analyze", str(d), "--mz", "500.25",
            "--windows", "0.3:0.5", "0.7:0.9", "--out", str(tmp_path / "o")]
    assert cli.main(argv) == 1
    assert "lctimspy extract" in capsys.readouterr().err
