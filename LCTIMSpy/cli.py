"""``lctimspy`` command-line interface.

    lctimspy doctor                 check the SDK, dependencies and cache
    lctimspy extract A.d [B.d ...]  extract into the cache (Ctrl-C cancels cleanly)
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
