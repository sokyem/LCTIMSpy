"""Peptide targets, signals, mobility differences, statistics, and export.

Everything here reads the cache, so it works without the SDK. Endpoints are
plain ``def``: FastAPI runs them in its thread pool, keeping the event loop (and
the SSE stream) responsive during a query.
"""

from __future__ import annotations

import io
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import ValidationError

from ...core.chrom import chromatograms, find_rt_peaks, summed_chromatogram
from ...core.fragments import calculate_precursor_mz, target_list
from ...core.mobil import mobilograms
from ...core.mobility_diff import mobility_difference
from ...core.stats import compare_targets
from ...core.transforms import lttb_downsample
from ..app import AppState, get_state, not_found
from ..schemas import (
    ChromatogramsIn,
    ChromatogramsOut,
    CompareIn,
    CompareOut,
    ExportIn,
    FragmentsIn,
    FragmentsOut,
    MobilityDiffIn,
    MobilityDiffOut,
    MobilogramsIn,
    MobilogramsOut,
    PeaksIn,
    PeaksOut,
    records,
)

router = APIRouter(tags=["analysis"])


def _entries(state: AppState, dataset_ids: list[str]) -> dict:
    """dataset_id -> CacheEntry; 404 for unknown ids, 409 if any is not extracted."""
    entries, missing = {}, []
    for ds_id in dict.fromkeys(dataset_ids):
        ds = state.registry.get(ds_id)
        if ds is None:
            raise not_found("dataset", ds_id)
        entry = state.registry.cache_entry(ds)
        if entry is None:
            missing.append(ds_id)
        else:
            entries[ds_id] = entry
    if missing:
        raise HTTPException(409, detail={"error": "not_extracted", "dataset_ids": missing,
                                         "message": "extract these datasets first"})
    return entries


def _names(state: AppState, ids) -> dict[str, str]:
    return {i: state.registry.get(i).name for i in ids}


@router.post("/peptides/fragments", response_model=FragmentsOut)
def fragments(body: FragmentsIn):
    try:
        targets = target_list(body.peptide, tuple(body.charges), body.ions)
        precursors = calculate_precursor_mz(body.peptide, tuple(body.charges))
    except Exception as exc:          # pyteomics raises PyteomicsError for bad residues
        raise HTTPException(422, detail=f"cannot compute fragments for {body.peptide!r}: {exc}")
    return {"targets": [{"label": label, "mz": mz} for label, mz in targets],
            "precursors": precursors}


def _chromatograms(state: AppState, body: ChromatogramsIn):
    entries = _entries(state, body.dataset_ids)
    mzs = [t.mz for t in body.targets]
    labels = [t.label for t in body.targets]
    out = []
    for ds_id, entry in entries.items():
        for s in chromatograms(state.cache, entry, mzs, body.tolerance.core(), labels):
            x, y = s.retention_time, s.intensity
            if body.max_points:
                x, y = lttb_downsample(x, y, body.max_points)
            out.append({"dataset_id": ds_id, "target_mz": s.target_mz, "label": s.label,
                        "x": x.tolist(), "y": y.tolist()})
    return out


@router.post("/signals/chromatograms", response_model=ChromatogramsOut)
def signal_chromatograms(body: ChromatogramsIn, state: AppState = Depends(get_state)):
    return {"series": _chromatograms(state, body)}


@router.post("/chromatograms/peaks", response_model=PeaksOut)
def chromatogram_peaks(body: PeaksIn, state: AppState = Depends(get_state)):
    entry = _entries(state, [body.dataset_id])[body.dataset_id]
    series = chromatograms(state.cache, entry, [t.mz for t in body.targets], body.tolerance.core())
    peaks = find_rt_peaks(summed_chromatogram(series), n=body.n)
    return {"peaks": [p.__dict__ for p in peaks]}


def _mobilograms(state: AppState, body: MobilogramsIn):
    entries = _entries(state, body.dataset_ids)
    labels = {t.mz: t.label for t in body.targets}
    windows = [w.core() for w in body.windows]
    out = []
    for ds_id, entry in entries.items():
        for s in mobilograms(state.cache, entry, list(labels), windows, body.query.core()):
            out.append({"dataset_id": ds_id, "target_mz": s.target_mz,
                        "label": labels.get(s.target_mz), "x": s.mobility.tolist(),
                        "y": s.intensity.tolist(), "window": s.window, "rt_start": s.rt_start,
                        "rt_end": s.rt_end, "max_intensity_raw": s.max_intensity_raw})
    return out


@router.post("/signals/mobilograms", response_model=MobilogramsOut)
def signal_mobilograms(body: MobilogramsIn, state: AppState = Depends(get_state)):
    return {"series": _mobilograms(state, body)}


def _mobility_difference(state: AppState, body: MobilityDiffIn):
    entries = _entries(state, body.dataset_ids)
    result = mobility_difference(state.cache, entries, [t.mz for t in body.targets],
                                 [w.core() for w in body.windows], body.query.core(),
                                 rt_offsets=body.rt_offsets)
    names = _names(state, entries)
    labels = {t.mz: t.label for t in body.targets}
    for df in (result.per_window, result.differences, result.excluded):
        df.insert(1, "sample_name", df["sample_id"].map(names))
        df.insert(3, "label", df["target_mz"].map(labels))
    return result


def _diff_out(result) -> dict:
    return {"per_window": records(result.per_window),
            "differences": records(result.differences),
            "excluded": records(result.excluded)}


@router.post("/analysis/mobility-difference", response_model=MobilityDiffOut)
def analysis_mobility_difference(body: MobilityDiffIn, state: AppState = Depends(get_state)):
    return _diff_out(_mobility_difference(state, body))


def _compare(state: AppState, body: CompareIn):
    result = _mobility_difference(state, body)
    stats = compare_targets(result.differences, body.stats.core())
    return result, stats


@router.post("/analysis/compare", response_model=CompareOut)
def analysis_compare(body: CompareIn, state: AppState = Depends(get_state)):
    result, stats = _compare(state, body)
    figure = None
    if body.include_figure:
        from ...core.figures import mobility_difference_figure

        figure = json.loads(mobility_difference_figure(stats.data, stats.metric, stats).to_json())
    return {"differences": _diff_out(result), "pairs": records(stats.pairs),
            "data": records(stats.data), "figure": figure}


# ---- export -------------------------------------------------------------------

_MEDIA = {"csv": "text/csv", "html": "text/html", "json": "application/json"}


def _parse(model, data: dict):
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(422, detail=json.loads(exc.json())) from exc


@router.post("/export")
def export(body: ExportIn, state: AppState = Depends(get_state)):
    """Return the requested table or figure as a file download."""
    import pandas as pd

    from ...core import figures

    fmt, kind = body.format, body.kind
    table, fig = None, None
    if kind == "chromatograms":
        req = _parse(ChromatogramsIn, body.request)
        series = _chromatograms(state, req)
        table = pd.DataFrame([{"dataset_id": s["dataset_id"], "target_mz": s["target_mz"],
                               "label": s["label"], "retention_time": x, "intensity": y}
                              for s in series for x, y in zip(s["x"], s["y"])])
        if fmt != "csv":
            fig = figures.go.Figure()
            for s in series:
                fig.add_scatter(x=s["x"], y=s["y"], mode="lines",
                                name=f"{s['dataset_id']} · {s['label'] or s['target_mz']}")
            fig.update_layout(xaxis_title="Retention time (min)", yaxis_title="Intensity")
    elif kind == "mobilograms":
        req = _parse(MobilogramsIn, body.request)
        series = _mobilograms(state, req)
        table = pd.DataFrame([{"dataset_id": s["dataset_id"], "target_mz": s["target_mz"],
                               "label": s["label"], "window": s["window"], "mobility": x,
                               "intensity": y} for s in series for x, y in zip(s["x"], s["y"])])
        if fmt != "csv":
            fig = figures.go.Figure()
            for s in series:
                fig.add_scatter(x=s["x"], y=s["y"], mode="lines",
                                name=f"{s['dataset_id']} · {s['target_mz']:.4f} · W{s['window'] + 1}")
            fig.update_layout(xaxis_title="Mobility 1/K0", yaxis_title="Intensity")
    elif kind == "mobility-difference":
        table = _mobility_difference(state, _parse(MobilityDiffIn, body.request)).differences
        if fmt != "csv":
            raise HTTPException(422, detail="mobility-difference exports as csv; "
                                            "use kind=compare for the figure")
    else:
        result, stats = _compare(state, _parse(CompareIn, body.request))
        table = stats.pairs
        fig = figures.mobility_difference_figure(stats.data, stats.metric, stats)

    if fmt == "csv":
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        content = buf.getvalue()
    elif fmt == "html":
        content = fig.to_html(include_plotlyjs=True, full_html=True)
    else:
        content = fig.to_json()
    filename = f"lctimspy_{kind}.{fmt}"
    return Response(content, media_type=_MEDIA[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
