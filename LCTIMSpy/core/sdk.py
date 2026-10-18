"""Lazy loading and probing of the Bruker TDF-SDK.

The legacy module called ``init_tdf_sdk_api()`` at import, so ``import
LCTIMSpy`` failed outright without the SDK. Here the library is loaded on first
use, and ``probe_sdk()`` reports *why* it is unavailable instead of raising, so
``lctimspy doctor`` and the UI can explain the fix.

Set ``LCTIMSPY_TIMSDATA`` to the full path of ``timsdata.dll`` /
``libtimsdata.so`` to use a library other than the one bundled with pyTDFSDK.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import SdkUnavailableError

INSTALL_HINT = 'pip install -e ".[bruker]"   (from the LCTIMSpy checkout)'
_LIBRARY_NAMES = {"Windows": "timsdata.dll", "Linux": "libtimsdata.so"}


@dataclass(frozen=True)
class SdkStatus:
    available: bool
    version: str | None = None
    path: str | None = None
    error: str | None = None
    remediation: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_lock = threading.Lock()
_library = None


def _library_path() -> tuple[Path | None, SdkStatus | None]:
    """Locate the native library, or return a failed status explaining why not."""
    override = os.environ.get("LCTIMSPY_TIMSDATA")
    if override:
        return Path(override), None

    system = platform.system()
    if system not in _LIBRARY_NAMES:
        return None, SdkStatus(
            available=False,
            error=f"Bruker does not ship the TDF-SDK for {system}.",
            remediation="Extract on Windows or Linux; cached datasets can be analysed anywhere.",
        )

    spec = importlib.util.find_spec("pyTDFSDK")
    if spec is None or spec.origin is None:
        return None, SdkStatus(
            available=False,
            error="pyTDFSDK is not installed.",
            remediation=INSTALL_HINT,
        )
    # pyTDFSDK installs the library at <site-packages>/TDF-SDK/, next to the package.
    return Path(spec.origin).parent.parent / "TDF-SDK" / _LIBRARY_NAMES[system], None


def _version() -> str | None:
    try:
        return importlib.metadata.version("pyTDFSDK")
    except importlib.metadata.PackageNotFoundError:
        return None


def _load():
    """Load the library once per process. Raises SdkUnavailableError."""
    global _library
    with _lock:
        if _library is not None:
            return _library
        path, failure = _library_path()
        if failure is not None:
            raise SdkUnavailableError(failure.error, failure.remediation)
        if not path.is_file():
            raise SdkUnavailableError(
                f"TDF-SDK library not found at {path}.",
                "Reinstall pyTDFSDK, or set LCTIMSPY_TIMSDATA to the library's full path.",
            )
        try:
            from pyTDFSDK.init_tdf_sdk import init_tdf_sdk_api
        except ImportError as exc:
            raise SdkUnavailableError(f"pyTDFSDK failed to import: {exc}", INSTALL_HINT) from exc
        try:
            _library = init_tdf_sdk_api(str(path))
        except OSError as exc:
            raise SdkUnavailableError(
                f"Could not load {path}: {exc}",
                "Install the Microsoft Visual C++ Redistributable (x64) and check the "
                "library matches this Python's architecture.",
            ) from exc
        return _library


def load_sdk():
    """Return the initialized TDF-SDK ctypes library, loading it on first call."""
    return _load()


def probe_sdk() -> SdkStatus:
    """Report whether extraction is possible on this machine. Never raises."""
    path, failure = _library_path()
    if failure is not None:
        return failure
    try:
        _load()
    except SdkUnavailableError as exc:
        return SdkStatus(False, _version(), str(path), str(exc), exc.remediation)
    return SdkStatus(True, _version(), str(path))
