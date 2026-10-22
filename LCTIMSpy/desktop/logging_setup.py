"""Logging for a windowed app, where there may be no console at all.

In a ``console=False`` PyInstaller build (and under pythonw) ``sys.stdout`` and
``sys.stderr`` are None, so the first stray ``print`` -- from uvicorn, pythonnet
or any library -- raises. Both are redirected to the log file, and
``faulthandler`` writes native crashes (access violations in WebView2 or the
Bruker library) to ``crash_log.txt`` instead of vanishing.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_dir: Path, debug: bool = False) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "lctimspy.log"
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3,
                                                   encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    if sys.stdout is None or sys.stderr is None:
        stream = open(log_file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives for the process
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream

    crash = open(log_dir / "crash_log.txt", "a", encoding="utf-8")  # noqa: SIM115 - lives for the process
    faulthandler.enable(file=crash, all_threads=True)
    return log_file
