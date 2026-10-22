"""App directories, ``.d`` folder validation, and network-path detection."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from ..errors import DatasetError

APP_NAME = "LCTIMSpy"
REQUIRED_FILES = ("analysis.tdf", "analysis.tdf_bin")


def cache_root() -> Path:
    """Where extraction caches live. ``LCTIMSPY_CACHE_DIR`` overrides the default.

    Deliberately never next to the ``.d`` folder: those live on read-only shares
    and instrument drives.
    """
    override = os.environ.get("LCTIMSPY_CACHE_DIR")
    if override:
        return Path(override)
    import platformdirs

    return Path(platformdirs.user_cache_dir(APP_NAME, APP_NAME)) / "cache"


def validate_d_folder(path: str | os.PathLike) -> Path:
    """Return the resolved path of a timsTOF ``.d`` folder, or raise DatasetError."""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"{p} does not exist.")
    if not p.is_dir():
        raise DatasetError(f"{p} is not a folder; expected a Bruker .d folder.")
    if (p / "analysis.tsf").exists() and not (p / "analysis.tdf").exists():
        raise DatasetError(f"{p} is a TSF dataset (no ion mobility); only TDF is supported.")
    missing = [name for name in REQUIRED_FILES if not (p / name).is_file()]
    if missing:
        raise DatasetError(f"{p} is not a timsTOF .d folder: missing {', '.join(missing)}.")
    return p.resolve()


def validate_source(path: str | os.PathLike) -> Path:
    """A ``.d`` folder (Bruker) or an ``.mzML`` file; raises DatasetError otherwise."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".mzml") or name.endswith(".mzml.gz"):
        if not p.is_file():
            raise DatasetError(f"{p} does not exist or is not a file.")
        return p.resolve()
    return validate_d_folder(p)


def is_network_path(path: str | os.PathLike) -> bool:
    """Best-effort check for UNC paths, mapped network drives and network mounts."""
    s = str(path)
    if s.startswith(("\\\\", "//")):
        return True
    if platform.system() == "Windows":
        drive = os.path.splitdrive(os.path.abspath(s))[0]
        if not drive:
            return False
        import ctypes

        DRIVE_REMOTE = 4
        return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == DRIVE_REMOTE
    if platform.system() == "Linux":
        return _linux_mount_type(os.path.realpath(s)) in {
            "nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs",
        }
    return False


def _linux_mount_type(path: str) -> str | None:
    best, best_type = "", None
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount = parts[1]
                inside = path == mount or path.startswith(mount.rstrip("/") + "/")
                if inside and len(mount) > len(best):
                    best, best_type = mount, parts[2]
    except OSError:
        return None
    return best_type
