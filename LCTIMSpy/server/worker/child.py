"""Worker process entry point: ``python -m LCTIMSpy.server.worker.child '<spec json>'``.

Under PyInstaller there is no ``python.exe``, so the frozen app re-invokes itself
as ``LCTIMSpy.exe --lctims-worker '<spec json>'`` and dispatches here before any
heavy import.

Spec kinds:
    {"kind": "extract", "path", "cache_dir", "config", "force"}
    {"kind": "copy", "src", "dest_root"}
Every spec may carry "cancel_file" and "parent_pid" (see watch_for_cancel).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from .protocol import encode

PROGRESS_INTERVAL_S = 0.2
CANCEL_POLL_S = 0.2
COPY_CHUNK = 8 * 1024 * 1024


def emit(kind: str, **data) -> None:
    sys.stdout.write(encode(kind, **data) + "\n")
    sys.stdout.flush()


class ThrottledReporter:
    """Forward progress at most every PROGRESS_INTERVAL_S, plus stage starts and ends."""

    def __init__(self):
        self._last = 0.0
        self._stage = None

    def __call__(self, event) -> None:
        now = time.monotonic()
        boundary = event.stage != self._stage or event.done >= event.total or event.done == 0
        if not boundary and now - self._last < PROGRESS_INTERVAL_S:
            return
        self._stage, self._last = event.stage, now
        emit("progress", stage=event.stage, done=event.done, total=event.total,
             rate=event.rate, eta_s=event.eta_s, message=event.message)


def _parent_alive_check(pid: int | None):
    """A zero-cost probe for "is the server still running?"."""
    if not pid:
        return lambda: True
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        SYNCHRONIZE, WAIT_OBJECT_0 = 0x00100000, 0
        # Holding the handle pins the process object, so pid reuse cannot fool us.
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return lambda: False
        return lambda: kernel32.WaitForSingleObject(handle, 0) != WAIT_OBJECT_0
    original = os.getppid()
    return lambda: os.getppid() == original


def watch_for_cancel(token, cancel_file: str | None, parent_pid: int | None) -> threading.Thread:
    """Cancel when the server creates ``cancel_file`` or the server process exits.

    Deliberately not a blocking read on stdin: on Windows, while a synchronous
    read is pending on a pipe, loading a DLL whose C runtime inspects the
    standard handles blocks -- ``import numpy`` hangs, and so could the Bruker
    library. Polling a file and a process handle never blocks.
    """
    parent_alive = _parent_alive_check(parent_pid)
    flag = Path(cancel_file) if cancel_file else None

    def run():
        while not token.cancelled:
            if (flag is not None and flag.exists()) or not parent_alive():
                token.cancel()
                return
            time.sleep(CANCEL_POLL_S)

    t = threading.Thread(target=run, name="cancel-watch", daemon=True)
    t.start()
    return t


def _frame_source(spec: dict):
    """Test hook: a fake frame source, honoured only when LCTIMSPY_TESTING=1."""
    target = spec.get("frame_source")
    if not target:
        return None
    if os.environ.get("LCTIMSPY_TESTING") != "1":
        raise RuntimeError("frame_source overrides are only allowed when LCTIMSPY_TESTING=1")
    import importlib

    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)(**spec.get("frame_source_kwargs", {}))


def run_extract(spec: dict, token) -> dict:
    from ...core.config import ExtractionConfig
    from ...core.io.cache import Cache
    from ...core.io.extract import extract_to_cache

    entry = extract_to_cache(
        spec["path"],
        ExtractionConfig.from_dict(spec.get("config") or {}),
        cache=Cache(spec.get("cache_dir")),
        force=bool(spec.get("force")),
        reporter=ThrottledReporter(),
        cancel=token,
        frame_source=_frame_source(spec),
        on_temp_dir=announce_temp_dir,
    )
    return {"key": entry.key, "rows": entry.rows, "bytes": entry.bytes, "stats": entry.stats}


def announce_temp_dir(path: Path) -> None:
    """Tell the server where partial output lives, so it can remove it if it must kill us."""
    emit("log", message=f"writing to {path}", temp_dir=str(path))


def run_copy(spec: dict, token) -> dict:
    """Copy a .d folder to local scratch; atomic, resumable only by restarting."""
    from ...core.io.paths import validate_d_folder
    from ...core.progress import StageTimer

    src = validate_d_folder(spec["src"])
    dest_root = Path(spec["dest_root"])
    dest_root.mkdir(parents=True, exist_ok=True)
    final = dest_root / src.name
    files = [p for p in src.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)

    if final.exists() and _same_tree(src, final, files):
        return {"path": str(final), "bytes": total, "skipped": True}

    tmp = dest_root / f"{src.name}.tmp-{uuid.uuid4().hex[:8]}"
    announce_temp_dir(tmp)
    timer = StageTimer("copy", total, ThrottledReporter())
    done = 0
    try:
        for f in files:
            target = tmp / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "rb") as fin, open(target, "wb") as fout:
                while chunk := fin.read(COPY_CHUNK):
                    token.raise_if_cancelled()
                    fout.write(chunk)
                    done += len(chunk)
                    timer.update(done, f.name)
            shutil.copystat(f, target)
        timer.update(total)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"path": str(final), "bytes": total, "skipped": False}


def _same_tree(src: Path, dest: Path, files: list[Path]) -> bool:
    for f in files:
        other = dest / f.relative_to(src)
        if not other.is_file():
            return False
        a, b = f.stat(), other.stat()
        if a.st_size != b.st_size or abs(a.st_mtime - b.st_mtime) > 2:
            return False
    return True


RUNNERS = {"extract": run_extract, "copy": run_copy}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except (AttributeError, ValueError):
        pass
    from ...core.errors import CancelledError, LCTIMSError
    from ...core.progress import CancelToken

    try:
        spec = json.loads(argv[0])
        runner = RUNNERS[spec["kind"]]
    except (IndexError, ValueError, KeyError) as exc:
        emit("error", error="BadSpec", message=f"invalid worker spec: {exc}", remediation=None)
        return 2

    token = CancelToken()
    watch_for_cancel(token, spec.get("cancel_file"), spec.get("parent_pid"))
    try:
        result = runner(spec, token)
    except (CancelledError, KeyboardInterrupt):
        emit("cancelled")
        return 0
    except LCTIMSError as exc:
        emit("error", error=type(exc).__name__, message=str(exc),
             remediation=getattr(exc, "remediation", None))
        return 1
    except Exception as exc:  # report, don't traceback into the void
        emit("error", error=type(exc).__name__, message=str(exc), remediation=None)
        return 1
    emit("result", **result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
