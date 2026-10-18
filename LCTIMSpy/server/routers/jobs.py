"""Jobs (extraction, copy to local scratch) and the multiplexed SSE event stream."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from ...core.config import ExtractionConfig
from ..app import AppState, get_state, not_found
from ..events import Event
from ..schemas import CopyJobIn, ExtractJobIn, JobOut

router = APIRouter(tags=["jobs"])

HEARTBEAT_S = 15.0


def _dataset(state: AppState, dataset_id: str):
    ds = state.registry.get(dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    return ds


@router.post("/jobs/extract", response_model=JobOut, status_code=202)
def start_extract(body: ExtractJobIn, state: AppState = Depends(get_state)):
    ds = _dataset(state, body.dataset_id)
    active = state.jobs.find_active("extract", ds.id)
    if active is not None:
        return active.snapshot()
    spec = {
        "kind": "extract",
        "path": str(ds.source),
        "cache_dir": str(state.cache.root),
        "config": ExtractionConfig(mobility_bin_width=body.mobility_bin_width).to_dict(),
        "force": body.force,
        **state.settings.worker_spec_extra,
    }
    return state.jobs.submit("extract", spec, ds.id).snapshot()


@router.post("/jobs/copy-local", response_model=JobOut, status_code=202)
def start_copy(body: CopyJobIn, state: AppState = Depends(get_state)):
    ds = _dataset(state, body.dataset_id)
    active = state.jobs.find_active("copy-local", ds.id)
    if active is not None:
        return active.snapshot()
    spec = {"kind": "copy", "src": ds.path, "dest_root": str(state.settings.scratch_dir)}
    return state.jobs.submit("copy-local", spec, ds.id).snapshot()


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(state: AppState = Depends(get_state)):
    return [j.snapshot() for j in state.jobs.list()]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, state: AppState = Depends(get_state)):
    job = state.jobs.get(job_id)
    if job is None:
        raise not_found("job", job_id)
    return job.snapshot(include_log=job.state == "failed")


@router.delete("/jobs/{job_id}", response_model=JobOut)
def cancel_job(job_id: str, state: AppState = Depends(get_state)):
    job = state.jobs.cancel(job_id)
    if job is None:
        raise not_found("job", job_id)
    return job.snapshot()


def format_sse(event: Event) -> str:
    payload = json.dumps(event.data, separators=(",", ":"), default=str)
    head = f"id: {event.id}\n" if event.id else ""
    return f"{head}event: {event.type}\ndata: {payload}\n\n"


@router.get("/events")
async def events(request: Request, since: int | None = None, once: bool = False,
                 state: AppState = Depends(get_state)):
    """Server-sent events: ``job``, ``progress``, ``log``, ``dataset``, ``dataset_removed``,
    ``resync``. Resume with the ``Last-Event-ID`` header or ``?since=``; ``?once=1`` returns
    the backlog and closes (for polling clients and tests)."""
    header = request.headers.get("last-event-id")
    last = int(header) if header and header.isdigit() else (since or 0)

    if once:
        backlog, gap = state.bus.since(last)
        body = "".join(format_sse(e) for e in backlog)
        if gap:
            body = format_sse(Event(0, "resync", {"reason": "events were dropped"})) + body
        return Response(body, media_type="text/event-stream")

    async def stream():
        sub = state.bus.subscribe(last)
        try:
            yield "retry: 2000\n\n"
            while True:
                event = await sub.next(timeout=HEARTBEAT_S)
                if event is None:
                    if await request.is_disconnected():
                        break
                    yield ": ping\n\n"
                else:
                    yield format_sse(event)
        finally:
            sub.close()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
