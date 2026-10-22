"""Health, capabilities, and the filesystem browser."""

from __future__ import annotations

import os
import platform
import string
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ... import __version__
from ...core.io.paths import is_network_path
from ...core.sdk import probe_sdk
from ..app import AppState, get_state
from ..schemas import CapabilitiesOut, FsEntry, FsListing

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@router.get("/capabilities", response_model=CapabilitiesOut)
def capabilities(state: AppState = Depends(get_state)):
    return {
        "version": __version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "sdk": probe_sdk().to_dict(),
        "cache_dir": str(state.cache.root),
        "browse_enabled": state.settings.browse_enabled,
        "max_concurrent_jobs": state.settings.max_concurrent_jobs,
    }


def _require_browse(state: AppState) -> None:
    if not state.settings.browse_enabled:
        raise HTTPException(403, detail="filesystem browsing is disabled on this server")


@router.get("/fs/roots", response_model=list[FsEntry])
def fs_roots(state: AppState = Depends(get_state)):
    _require_browse(state)
    roots = [FsEntry(name="Home", path=str(Path.home()), kind="dir")]
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                roots.append(FsEntry(name=f"{letter}:", path=drive, kind="dir"))
    else:
        roots.append(FsEntry(name="/", path="/", kind="dir"))
    return roots


@router.get("/fs/list", response_model=FsListing)
def fs_list(path: str, state: AppState = Depends(get_state)):
    _require_browse(state)
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(404, detail=f"{path} is not a folder")
    entries = []
    try:
        children = sorted(p.iterdir(), key=lambda c: c.name.lower())
    except PermissionError as exc:
        raise HTTPException(403, detail=f"cannot list {path}: {exc}") from exc
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                is_d = child.suffix.lower() == ".d" and (child / "analysis.tdf").is_file()
                entries.append(FsEntry(name=child.name, path=str(child),
                                       kind="d_folder" if is_d else "dir"))
            elif child.name.lower().endswith((".mzml", ".mzml.gz")):
                entries.append(FsEntry(name=child.name, path=str(child), kind="mzml",
                                       size=child.stat().st_size))
            else:
                entries.append(FsEntry(name=child.name, path=str(child), kind="file",
                                       size=child.stat().st_size))
        except OSError:
            continue        # unreadable entry; skip rather than fail the listing
    parent = str(p.parent) if p.parent != p else None
    return FsListing(path=str(p), parent=parent, is_network=is_network_path(p), entries=entries)
