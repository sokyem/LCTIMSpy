"""``lctimspy`` command-line interface.

    lctimspy doctor                 check the SDK, dependencies and cache
    lctimspy extract A.d [B.d ...]  extract into the cache (Ctrl-C cancels cleanly)
    lctimspy analyze A.d B.d C.d --peptide SEQ --windows 18.3:19.0 21.5:22.2
    lctimspy serve [--open]          run the local web server for the GUI
    lctimspy desktop                 open the GUI in a native window
    lctimspy cache list|clear|remove KEY|sweep
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import time
from pathlib import Path

from . import __version__

EXIT_CANCELLED = 130


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class BarReporter:
    """Single-line progress bar on stderr, redrawn at most 10 times a second."""

    def __init__(self, label: str, stream=sys.stderr):
        self.label = label
        self.stream = stream
        self._last = 0.0
        self._stage = None

    def __call__(self, event) -> None:
        now = time.monotonic()
        finished = event.done >= event.total
        if event.stage == self._stage and not finished and now - self._last < 0.1:
            return
        if self._stage is not None and event.stage != self._stage:
            self.stream.write("\n")
        self._stage, self._last = event.stage, now
        width = 24
        filled = int(width * event.fraction)
        bar = "#" * filled + "-" * (width - filled)
        rate = f"{event.rate:.1f}/s" if event.rate else ""
        line = (
            f"\r{self.label} {event.stage:<8} [{bar}] {event.fraction:6.1%} "
            f"{event.done}/{event.total} {rate:>9} ETA {_fmt_eta(event.eta_s)} {event.message}"
        )
        self.stream.write(line[:160].ljust(100))
        self.stream.flush()

    def close(self) -> None:
        if self._stage is not None:
            self.stream.write("\n")
            self.stream.flush()


# ---- doctor -----------------------------------------------------------------

def cmd_doctor(args) -> int:
    from .core.io.cache import Cache
    from .core.sdk import probe_sdk

    ok = True
    print(f"LCTIMSpy {__version__}")
    print(f"Python   {platform.python_version()} ({platform.system()} {platform.machine()})")

    print("\nDependencies")
    for dist, required in (("numpy", True), ("pandas", True), ("pyarrow", True),
                           ("platformdirs", True), ("scipy", True), ("plotly", False),
                           ("matplotlib", False)):
        try:
            version = importlib.metadata.version(dist)
            print(f"  ok       {dist} {version}")
        except importlib.metadata.PackageNotFoundError:
            tag = "MISSING " if required else "absent  "
            ok = ok and not required
            print(f"  {tag} {dist}" + ("" if required else "  (needed by legacy plot functions)"))

    print("\nBruker TDF-SDK (needed only for extraction)")
    status = probe_sdk()
    if status.available:
        print(f"  ok       pyTDFSDK {status.version or '?'}")
        print(f"           {status.path}")
    else:
        print(f"  unavailable: {status.error}")
        if status.remediation:
            print(f"  fix:         {status.remediation}")
        print("  Cached datasets can still be analysed on this machine.")

    cache = Cache(args.cache_dir)
    entries = cache.entries()
    print("\nCache")
    print(f"  dir      {cache.root}")
    print(f"  entries  {len(entries)}  ({_fmt_bytes(sum(e.bytes for e in entries))} of "
          f"{_fmt_bytes(cache.max_bytes)} limit)")
    return 0 if ok else 1


# ---- extract ----------------------------------------------------------------

def cmd_extract(args) -> int:
    from .core.config import ExtractionConfig
    from .core.errors import CancelledError, LCTIMSError
    from .core.io.cache import Cache
    from .core.io.extract import extract_to_cache
    from .core.io.paths import is_network_path

    config = ExtractionConfig(
        mobility_bin_width=args.mobility_bin_width, ms1_only=not args.include_msms
    )
    cache = Cache(args.cache_dir)
    failures = 0
    for path in args.paths:
        if is_network_path(path):
            print(f"warning: {path} is on a network drive; extraction will be slower.",
                  file=sys.stderr)
        reporter = BarReporter(Path(path).name)
        try:
            entry = extract_to_cache(path, config, cache=cache, force=args.force,
                                     reporter=reporter)
        except (KeyboardInterrupt, CancelledError):
            reporter.close()
            print("Cancelled. No partial cache was left behind.", file=sys.stderr)
            return EXIT_CANCELLED
        except LCTIMSError as exc:
            reporter.close()
            print(f"error: {path}: {exc}", file=sys.stderr)
            remediation = getattr(exc, "remediation", None)
            if remediation:
                print(f"fix:   {remediation}", file=sys.stderr)
            failures += 1
            continue
        reporter.close()
        stats = entry.stats
        print(f"{entry.dataset.name}: {entry.rows:,} points, {_fmt_bytes(entry.bytes)}, "
              f"key {entry.key}")
        if stats.get("frames_failed"):
            print(f"  warning: {stats['frames_failed']} frames failed; first error: "
                  f"{stats['first_error']}", file=sys.stderr)
    return 1 if failures else 0


# ---- analyze ----------------------------------------------------------------

def _resolve_entries(specs, cache, config):
    """Map each CLI dataset (a .d path or a cache key) to (sample_id, CacheEntry)."""
    import re

    from .core.errors import DatasetError
    from .core.io.paths import validate_d_folder

    out = {}
    for spec in specs:
        if re.fullmatch(r"[0-9a-f]{32}", spec):
            entry = cache.get(spec)
            if entry is None:
                raise DatasetError(f"No cache entry {spec}; see `lctimspy cache list`.")
        else:
            entry = cache.lookup(validate_d_folder(spec), config)
            if entry is None:
                raise DatasetError(f"{spec} has not been extracted with these settings; "
                                   f"run `lctimspy extract {spec}` first.")
        name, n = entry.dataset.name, 2
        while name in out:
            name, n = f"{entry.dataset.name}_{n}", n + 1
        out[name] = entry
    return out


def _targets(args):
    from .core.fragments import target_list

    if args.mz:
        return [(None, mz) for mz in args.mz]
    return target_list(args.peptide, tuple(args.charges), args.ions or None)


def cmd_analyze(args) -> int:
    import pandas as pd

    from .core.chrom import chromatograms, find_rt_peaks, summed_chromatogram
    from .core.config import ExtractionConfig, MzTolerance, QueryConfig, RtWindow, StatsConfig
    from .core.errors import LCTIMSError
    from .core.export import write_figure, write_table
    from .core.figures import chromatogram_figure, mobility_difference_figure, mobilogram_figure
    from .core.io.cache import Cache
    from .core.mobil import mobilograms
    from .core.mobility_diff import mobility_difference
    from .core.stats import compare_targets

    cache = Cache(args.cache_dir)
    try:
        entries = _resolve_entries(
            args.datasets, cache, ExtractionConfig(mobility_bin_width=args.mobility_bin_width))
    except LCTIMSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    targets = _targets(args)
    if not targets:
        print("error: no targets (check --ions against the series names, e.g. y1+ b2+)",
              file=sys.stderr)
        return 1
    labels = [label for label, _ in targets]
    mzs = [mz for _, mz in targets]
    query = QueryConfig(
        tolerance=MzTolerance(args.tol, args.tol_unit),
        mobilogram_sigma=args.sigma,
        baseline_correction=not args.no_baseline,
        intensity_threshold=None if args.threshold < 0 else args.threshold,
        threshold_on=args.threshold_on,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_table(pd.DataFrame({"label": labels, "target_mz": mzs}), out / "targets.csv")

    eics = {sid: chromatograms(cache, e, mzs, query.tolerance, labels) for sid, e in entries.items()}
    if args.windows == ["auto"]:
        first = next(iter(eics))
        peaks = find_rt_peaks(summed_chromatogram(eics[first]), n=2)
        if len(peaks) < 2:
            print(f"error: found {len(peaks)} RT peak(s) in {first}; pass --windows START:END ...",
                  file=sys.stderr)
            return 1
        windows = [RtWindow(round(p.start_rt, 3), round(p.end_rt, 3)) for p in peaks]
        print("auto windows (from " + first + "): "
              + "  ".join(f"{w.start:g}:{w.end:g}" for w in windows))
    else:
        try:
            windows = [RtWindow.parse(w) for w in args.windows]
        except ValueError as exc:
            print(f"error: --windows: {exc}", file=sys.stderr)
            return 1
        if len(windows) < 2:
            print("error: give at least two --windows", file=sys.stderr)
            return 1

    for sid, entry in entries.items():
        write_figure(chromatogram_figure(eics[sid], windows, title=f"EIC — {sid}"),
                     out / f"chromatograms_{sid}.html")
        mobs = mobilograms(cache, entry, mzs, windows, query)
        write_figure(mobilogram_figure(mobs, title=f"Mobilograms — {sid}"),
                     out / f"mobilograms_{sid}.html")

    result = mobility_difference(cache, entries, mzs, windows, query)
    label_of = dict(zip(mzs, labels))
    for df in (result.per_window, result.differences, result.excluded):
        df.insert(2, "label", df["target_mz"].map(label_of))
    write_table(result.per_window, out / "mobility_per_window.csv")
    write_table(result.differences, out / "mobility_difference.csv")
    write_table(result.excluded, out / "excluded.csv")
    if len(windows) == 2:
        write_table(result.to_legacy_frame(), out / "mobility_difference_legacy_format.csv")

    n_samples = len(entries)
    print(f"{len(mzs)} targets x {n_samples} samples x {len(windows)} windows: "
          f"{len(result.differences)} differences, {len(result.excluded)} excluded")
    stats_cfg = StatsConfig(metric=args.metric, correction=args.correction, alpha=args.alpha)
    if n_samples >= stats_cfg.min_samples:
        try:
            stats = compare_targets(result.differences, stats_cfg)
        except ValueError as exc:
            print(f"stats skipped: {exc}")
        else:
            write_table(stats.pairs, out / "stats_pairs.csv")
            write_figure(mobility_difference_figure(stats.data, stats.metric, stats,
                                                    title="Mobility difference by target"),
                         out / "mobility_difference.html")
            print(f"stats: {int(stats.pairs.reject.sum())} of {len(stats.pairs)} pairs "
                  f"significant ({stats.correction}, alpha {stats.alpha:g})")
    else:
        print(f"stats skipped: need at least {stats_cfg.min_samples} samples")
    print(f"results written to {out.resolve()}")
    return 0


# ---- serve ------------------------------------------------------------------

def cmd_serve(args) -> int:
    try:
        from .server.app import create_app
        from .server.runner import bind_loopback, make_server, url_for
        from .server.settings import ServerSettings
    except ImportError as exc:
        print(f"error: the server needs extra packages ({exc.name}); "
              'install with: pip install -e ".[server]"', file=sys.stderr)
        return 1

    settings = ServerSettings()
    if args.cache_dir:
        settings.cache_dir = Path(args.cache_dir)
    if args.no_token:
        settings.token = None
    try:
        sock = bind_loopback(args.port)
    except OSError as exc:
        print(f"error: cannot listen on 127.0.0.1:{args.port}: {exc}", file=sys.stderr)
        return 1
    server = make_server(create_app(settings))
    url = url_for(sock, settings.token)
    print(f"LCTIMSpy server running at {url}")
    print(f"API docs: {url_for(sock, None)}api/docs" + ("" if args.no_token else
          "  (open the URL above first to authenticate)"))
    print("Press Ctrl-C to stop.", flush=True)   # flush: launchers read the URL from a pipe
    if args.open:
        import webbrowser

        webbrowser.open(url)
    server.run(sockets=[sock])
    return 0


# ---- desktop ----------------------------------------------------------------

def cmd_desktop(args) -> int:
    try:
        import webview  # noqa: F401
    except ImportError:
        print('error: the desktop app needs pywebview; install with: pip install -e ".[desktop]"',
              file=sys.stderr)
        return 1
    import os

    if args.cache_dir:
        os.environ["LCTIMSPY_CACHE_DIR"] = args.cache_dir
    from .desktop.main import main as desktop_main

    return desktop_main(args.desktop_args)


# ---- cache ------------------------------------------------------------------

def cmd_cache(args) -> int:
    from .core.io.cache import Cache

    cache = Cache(args.cache_dir)
    if args.cache_cmd == "list":
        entries = cache.entries()
        if not entries:
            print(f"No cached datasets in {cache.root}")
            return 0
        for e in entries:
            used = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.last_used))
            print(f"{e.key}  {_fmt_bytes(e.bytes):>9}  {e.rows:>14,} pts  used {used}  "
                  f"{e.dataset.source_path}")
        print(f"\n{len(entries)} entries, {_fmt_bytes(sum(e.bytes for e in entries))} "
              f"in {cache.root}")
    elif args.cache_cmd == "remove":
        if not cache.remove(args.key):
            print(f"No removable entry {args.key}", file=sys.stderr)
            return 1
    elif args.cache_cmd == "clear":
        print(f"Removed {cache.clear()} entries from {cache.root}")
    elif args.cache_cmd == "sweep":
        removed = cache.sweep_partials()
        print(f"Removed {len(removed)} leftover temp folders")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lctimspy", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"LCTIMSpy {__version__}")
    parser.add_argument("--cache-dir", default=None,
                        help="cache location (default: per-user cache dir, or LCTIMSPY_CACHE_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the SDK, dependencies and cache").set_defaults(
        func=cmd_doctor)

    p = sub.add_parser("extract", help="extract .d folders into the cache")
    p.add_argument("paths", nargs="+", metavar="D_FOLDER")
    p.add_argument("--mobility-bin-width", type=float, default=0.0005)
    p.add_argument("--include-msms", action="store_true",
                   help="also extract MS/MS frames (default: MS1 only)")
    p.add_argument("--force", action="store_true", help="re-extract even if cached")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("analyze", help="mobility difference between RT windows, from the cache")
    p.add_argument("datasets", nargs="+", metavar="D_FOLDER_OR_KEY")
    tg = p.add_mutually_exclusive_group(required=True)
    tg.add_argument("--peptide", help='sequence; end with "(-0.98)" for C-terminal amidation')
    tg.add_argument("--mz", type=float, nargs="+", help="explicit target m/z values")
    p.add_argument("--charges", type=int, nargs="+", default=[1, 2])
    p.add_argument("--ions", nargs="+", help="series to keep, e.g. y1+ b1+ (default: all)")
    p.add_argument("--windows", nargs="+", required=True, metavar="START:END",
                   help='RT windows in minutes, or "auto" to detect the two main peaks')
    p.add_argument("--tol", type=float, default=0.02, help="m/z tolerance (default 0.02)")
    p.add_argument("--tol-unit", choices=["Da", "ppm"], default="Da")
    p.add_argument("--sigma", type=float, default=None, help="gaussian smoothing of mobilograms")
    p.add_argument("--no-baseline", action="store_true", help="skip baseline subtraction")
    p.add_argument("--threshold", type=float, default=100,
                   help="min max-intensity per window; negative disables (default 100)")
    p.add_argument("--threshold-on", choices=["corrected", "raw"], default="corrected")
    p.add_argument("--metric", choices=["mobility_difference", "peak_mobility_difference"],
                   default="mobility_difference")
    p.add_argument("--correction", choices=["bonferroni", "holm", "tukey", "none"],
                   default="bonferroni")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mobility-bin-width", type=float, default=0.0005,
                   help="must match the extraction (default 0.0005)")
    p.add_argument("--out", default="lctims_results", help="output folder")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("serve", help="run the local web server for the GUI")
    p.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: any free)")
    p.add_argument("--open", action="store_true", help="open the browser")
    p.add_argument("--no-token", action="store_true",
                   help="disable the per-run access token (not recommended)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("desktop", help="open the GUI in a native window",
                       description="Options: --browser, --smoke, --debug")
    p.add_argument("desktop_args", nargs=argparse.REMAINDER, metavar="...")
    p.set_defaults(func=cmd_desktop)

    p = sub.add_parser("cache", help="inspect or clean the cache")
    csub = p.add_subparsers(dest="cache_cmd", required=True)
    csub.add_parser("list")
    csub.add_parser("clear")
    csub.add_parser("sweep", help="delete temp folders left by killed extractions")
    csub.add_parser("remove").add_argument("key")
    p.set_defaults(func=cmd_cache)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
