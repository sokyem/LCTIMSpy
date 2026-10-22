"""Untargeted isomer discovery: run, list, inspect candidates, export reports."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..app import AppState, get_state, not_found
from ..schemas import JobOut

router = APIRouter(tags=["discovery"])

Engine = Literal["peaks", "msfragger", "maxquant", "mzidentml", "mztab"]


class DiscoveryOverrides(BaseModel):
    mass_ppm: float | None = Field(None, gt=0, le=100)
    rt_min_delta: float | None = Field(None, ge=0)
    rt_min_resolution: float | None = Field(None, ge=0)
    mobility_min_rel_pct: float | None = Field(None, ge=0)
    mobility_precision_rel: float | None = Field(None, gt=0, lt=0.5)
    similarity_threshold: float | None = Field(None, ge=0, le=1)
    min_annotated_fraction: float | None = Field(None, ge=0, le=1)
    min_snr: float | None = Field(None, gt=0)
    isotope_min_cosine: float | None = Field(None, ge=0, le=1)


class DiscoveryRunIn(BaseModel):
    dataset_id: str
    psm_path: str | None = None
    psm_engine: Engine | None = None
    overrides: DiscoveryOverrides = DiscoveryOverrides()


class PsmPreviewIn(BaseModel):
    path: str = Field(min_length=1)
    engine: Engine | None = None


def _store(state: AppState):
    from ...discovery.store import DiscoveryStore

    return DiscoveryStore(state.settings.state_dir / "discovery")


@router.post("/discovery/psm-preview")
def psm_preview(body: PsmPreviewIn):
    from ...discovery.psms import read_psms

    df = read_psms(body.path, body.engine)
    runs = df["run"].fillna("").astype(str).value_counts()
    return {"engine": df.attrs.get("engine"), "total": len(df),
            "with_mobility": int(df["mobility"].notna().sum()),
            "decoys": int(df["is_decoy"].sum()),
            "runs": [{"run": r, "psms": int(n)} for r, n in runs.items()]}


@router.post("/discovery/runs", response_model=JobOut, status_code=202)
def start_run(body: DiscoveryRunIn, state: AppState = Depends(get_state)):
    from ...discovery.pipeline import new_run_id

    ds = state.registry.get(body.dataset_id)
    if ds is None:
        raise not_found("dataset", body.dataset_id)
    entry = state.registry.cache_entry(ds)
    if entry is None:
        raise HTTPException(409, detail={"error": "not_extracted", "dataset_ids": [ds.id],
                                         "message": "extract this dataset first"})
    spec = {"kind": "discover", "cache_dir": str(state.cache.root), "cache_key": entry.key,
            "out_dir": str(_store(state).root / new_run_id()), "psm_path": body.psm_path,
            "psm_engine": body.psm_engine, "dataset_name": ds.name,
            "overrides": body.overrides.model_dump(exclude_none=True)}
    return state.jobs.submit("discover", spec, ds.id).snapshot()


@router.get("/discovery/runs")
def list_runs(state: AppState = Depends(get_state)):
    return _store(state).list()


def _summary(c: dict) -> dict:
    return {k: v for k, v in c.items() if k != "msms"}


@router.get("/discovery/runs/{run_id}")
def get_run(run_id: str, state: AppState = Depends(get_state)):
    store = _store(state)
    meta = store.meta(run_id)
    if meta is None:
        raise not_found("discovery run", run_id)
    return {"id": run_id, "meta": meta, "candidates": [_summary(c) for c in store.candidates(run_id)]}


@router.get("/discovery/runs/{run_id}/candidates/{candidate_id}")
def get_candidate(run_id: str, candidate_id: int, state: AppState = Depends(get_state)):
    from ...discovery.viewer import candidate_detail

    store = _store(state)
    meta = store.meta(run_id)
    c = store.candidate(run_id, candidate_id) if meta else None
    if c is None:
        raise not_found("candidate", f"{run_id}/{candidate_id}")
    entry = state.cache.get(meta["cache_key"], touch=False)
    detail = candidate_detail(state.cache, entry, c) if entry is not None else None
    msms = c.get("msms") or {}
    return {"candidate": _summary(c), "msms": {k: v for k, v in msms.items() if k != "mirror"},
            "detail": detail,
            "note": None if entry is not None else
            "the dataset's extraction is no longer cached; traces cannot be shown"}


@router.get("/discovery/runs/{run_id}/report")
def report(run_id: str, format: Literal["csv", "html", "json"] = "html",
           state: AppState = Depends(get_state)):
    import json

    from ...discovery.report import to_csv, to_html

    store = _store(state)
    meta = store.meta(run_id)
    if meta is None:
        raise not_found("discovery run", run_id)
    records = store.candidates(run_id)
    name = (meta.get("dataset") or {}).get("name", run_id)
    if format == "csv":
        body, media = to_csv(records), "text/csv"
    elif format == "html":
        body, media = to_html(meta, records), "text/html"
    else:
        body, media = json.dumps({"meta": meta, "candidates": records}), "application/json"
    return Response(body, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="lctimspy_discovery_{name}.{format}"'})


@router.delete("/discovery/runs/{run_id}", status_code=204)
def delete_run(run_id: str, state: AppState = Depends(get_state)):
    if not _store(state).delete(run_id):
        raise not_found("discovery run", run_id)
